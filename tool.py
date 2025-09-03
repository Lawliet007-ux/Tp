"""
FIR PII Extractor (Hybrid OCR + NER + Semantic)

Features:
- Extracts PII fields from FIR PDFs (Hindi/English).
- Handles CID-encoded PDFs (OCR fallback).
- Supports 3 OCR backends: Tesseract (default), Google Vision, AWS Textract.
- Uses Stanza (NER) + sentence-transformers for semantic candidate search.
"""

import streamlit as st
import io
import json
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re
from typing import Dict, Any, List, Tuple

# NLP
import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download("punkt")

# Optional: Google & AWS clients
try:
    from google.cloud import vision
except ImportError:
    vision = None

try:
    import boto3
except ImportError:
    boto3 = None


# -------------------------
# Init models
# -------------------------
@st.cache_resource
def init_stanza():
    stanza.download("en", processors="tokenize,ner")
    stanza.download("hi", processors="tokenize,ner")
    return {
        "en": stanza.Pipeline(lang="en", processors="tokenize,ner", use_gpu=False, verbose=False),
        "hi": stanza.Pipeline(lang="hi", processors="tokenize,ner", use_gpu=False, verbose=False),
    }

@st.cache_resource
def init_st_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# OCR Backends
# -------------------------
def ocr_tesseract(pil_img: Image.Image, lang="hin+eng") -> str:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 11)
    pil_for_ocr = Image.fromarray(th)
    return pytesseract.image_to_string(pil_for_ocr, lang=lang)


def ocr_google(pil_img: Image.Image) -> str:
    if vision is None:
        raise RuntimeError("Google Vision not installed")
    client = vision.ImageAnnotatorClient()
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    content = buf.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = [t.description for t in response.text_annotations]
    return texts[0] if texts else ""


def ocr_aws(pil_img: Image.Image) -> str:
    if boto3 is None:
        raise RuntimeError("boto3 not installed")
    client = boto3.client("textract")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    content = buf.getvalue()
    resp = client.detect_document_text(Document={"Bytes": content})
    texts = [b["Text"] for b in resp["Blocks"] if b["BlockType"] == "LINE"]
    return "\n".join(texts)


def ocr_image(pil_img: Image.Image, backend="tesseract") -> str:
    if backend == "google":
        return ocr_google(pil_img)
    elif backend == "aws":
        return ocr_aws(pil_img)
    else:
        return ocr_tesseract(pil_img)


# -------------------------
# PDF Extraction
# -------------------------
def extract_text_pdf(file_bytes: bytes, ocr_backend="tesseract") -> Tuple[str, List[Image.Image]]:
    page_images = []
    text_accum = []

    # Render PDF pages
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc.load_page(i)
        mat = fitz.Matrix(3, 3)  # 300 dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        page_images.append(img)

    # Extract text via pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                text_accum.append(txt)

    text = "\n".join(text_accum)

    # OCR fallback if short text or CID corruption
    if len(text.strip()) < 300 or "(cid:" in text:
        ocr_texts = [ocr_image(img, backend=ocr_backend) for img in page_images]
        text = "\n".join(ocr_texts)

    return text, page_images


# -------------------------
# PII Extraction Helpers
# -------------------------
FIELDS = ["fir_no", "year", "state_name", "dist_name", "police_station",
          "under_acts", "under_sections", "revised_case_category",
          "oparty", "accused_name", "address", "jurisdiction"]

def extract_field(sentence: str, field: str, stanza_pipes: Dict[str, Any]) -> Tuple[Any, float]:
    tokens = sentence.split()
    ents = []
    for pipe in stanza_pipes.values():
        try:
            doc = pipe(sentence)
            for sentn in doc.sentences:
                for ent in sentn.ents:
                    ents.append((ent.text, ent.type))
        except Exception:
            continue

    if field == "fir_no":
        m = re.search(r"\b\d{1,5}/\d{4}\b", sentence)
        if m:
            return m.group(0), 0.95
    if field == "year":
        for t in tokens:
            if t.isdigit() and len(t) == 4 and 1900 <= int(t) <= 2100:
                return t, 0.95
    if field in ("state_name", "dist_name", "police_station"):
        locs = [t for t, label in ents if label in ("GPE", "LOC", "ORG")]
        if locs:
            return locs[0], 0.9
    if field in ("oparty", "accused_name"):
        persons = [t for t, label in ents if label == "PERSON"]
        if persons:
            return persons[0], 0.9
    if field == "under_sections":
        if "IPC" in sentence or "धारा" in sentence:
            return sentence.strip(), 0.85
    if field == "under_acts":
        if any(w.upper() in ["IPC", "CRPC", "NDPS"] for w in tokens):
            return sentence.strip(), 0.85
    if field == "address":
        locs = [t for t, label in ents if label in ("LOC", "GPE")]
        if locs:
            return " ".join(locs), 0.85
        if len(tokens) > 6:
            return sentence.strip(), 0.5
    if field == "jurisdiction":
        return sentence.strip(), 0.4

    return None, 0.2


def extract_all_fields(text: str, stanza_pipes, st_model) -> Dict[str, Any]:
    sents = nltk.sent_tokenize(text)
    results = {}
    for field in FIELDS:
        best_val, best_conf = None, 0
        for sent in sents:
            val, conf = extract_field(sent, field, stanza_pipes)
            if conf > best_conf:
                best_conf = conf
                best_val = val
        results[field] = {"value": best_val, "confidence": round(best_conf, 3)}

    # Add jurisdiction_type
    jur = results.get("jurisdiction", {}).get("value") or ""
    if "pan" in jur.lower() or "all india" in jur.lower():
        results["jurisdiction_type"] = "PAN_INDIA"
    elif "state" in jur.lower() or "राज्य" in jur:
        results["jurisdiction_type"] = "STATE"
    elif "district" in jur.lower() or "जिला" in jur:
        results["jurisdiction_type"] = "DISTRICT"
    else:
        results["jurisdiction_type"] = "UNKNOWN"

    return results


# -------------------------
# Streamlit App
# -------------------------
def app():
    st.title("📄 FIR PII Extractor (Hybrid OCR + NER)")
    st.markdown("Upload FIR PDFs in Hindi/English. Extracts PII using OCR + NLP.")

    ocr_backend = st.selectbox("Choose OCR Backend", ["tesseract", "google", "aws"])
    file = st.file_uploader("Upload FIR PDF", type=["pdf"])

    if file:
        raw = file.read()
        with st.spinner("Extracting text..."):
            text, images = extract_text_pdf(raw, ocr_backend)

        st.subheader("Extracted Text (preview)")
        st.text_area("text", text[:2000], height=300)

        st.subheader("Page Preview")
        st.image(images[0], caption="Page 1", use_column_width=True)

        with st.spinner("Loading NLP models..."):
            stanza_pipes = init_stanza()
            st_model = init_st_model()

        with st.spinner("Extracting PII..."):
            results = extract_all_fields(text, stanza_pipes, st_model)

        st.subheader("Extracted PII Fields")
        st.json(results)

        st.download_button("Download JSON", json.dumps(results, ensure_ascii=False, indent=2),
                           file_name="fir_pii.json", mime="application/json")


if __name__ == "__main__":
    app()
