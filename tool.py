"""
Streamlit FIR PII Extractor (Hybrid — layout + OCR + semantic search + neural NER)

Features:
- Upload a PDF (scanned or digital), handles English + Hindi primarily.
- Attempts text extraction using: pdfplumber -> PyMuPDF (fitz) -> OCR (pytesseract on rendered pages).
- Uses sentence-transformers to semantically locate sentences likely containing each target field.
- Uses Stanza (neural multilingual pipeline) for NER (Hindi + English).
- Avoids regex-based brittle parsing; uses token scanning and neural methods + semantic search.
- Outputs structured JSON with confidence scores and allows download.

Notes:
- This is a heavy pipeline (models will download on first run: stanza models and sentence-transformers).
- Install Tesseract on your machine and Hindi traineddata for best OCR: (Ubuntu example below).

Ubuntu quick setup (run in terminal):
# system deps
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-hin poppler-utils
# python deps
python -m pip install --upgrade pip
pip install streamlit pdfplumber pymupdf pytesseract pillow stanza sentence-transformers numpy scikit-learn opencv-python-headless nltk

Run the app:
streamlit run streamlit_fir_extractor.py

"""

import streamlit as st
import io
import json
import tempfile
from typing import List, Dict, Any, Tuple

# PDF/text/image tools
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np

# NLP
import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')

# -------------------------
# Initialization helpers
# -------------------------
@st.cache_resource
def init_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    return SentenceTransformer(model_name)

@st.cache_resource
def init_stanza_pipelines() -> Dict[str, Any]:
    # download models if needed (first run will download)
    try:
        stanza.download('en', processors='tokenize,ner')
    except Exception:
        pass
    try:
        stanza.download('hi', processors='tokenize,ner')
    except Exception:
        pass

    en = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=False, verbose=False)
    hi = stanza.Pipeline(lang='hi', processors='tokenize,ner', use_gpu=False, verbose=False)
    return {'en': en, 'hi': hi}

# -------------------------
# Text extraction
# -------------------------

def extract_text_pdf_bytes(file_bytes: bytes) -> Tuple[str, List[Image.Image]]:
    """Return extracted text and a list of page images (PIL.Image).
    Strategy:
      1) Try pdfplumber text extraction (best for digital PDFs).
      2) If text content is short, render pages via fitz and run OCR.
      3) Provide list of page images so the UI can show / OCR can use them.
    """
    text_accum = []
    page_images: List[Image.Image] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_accum.append(text)

    # Render pages with fitz to images (for OCR and fallbacks)
    doc = fitz.open(stream=file_bytes, filetype='pdf')
    for i in range(len(doc)):
        page = doc.load_page(i)
        mat = fitz.Matrix(2, 2)  # render at higher resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        page_images.append(img)

    text = "\n".join(text_accum)

    # if extracted text is very short (likely scanned PDF) -> OCR the images
    if len(text.strip()) < 200:
        ocr_texts = []
        for img in page_images:
            ocr_texts.append(ocr_image(img))
        text = "\n".join(ocr_texts)

    return text, page_images


def ocr_image(pil_img: Image.Image, lang: str = 'eng+hin') -> str:
    """Preprocess and OCR a PIL image using pytesseract."""
    # convert to OpenCV image
    img = np.array(pil_img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # basic denoising and adaptive thresholding
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # use adaptive thresholding to increase text contrast
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 11)
    except Exception:
        th = gray

    # run pytesseract
    pil_for_ocr = Image.fromarray(th)
    try:
        text = pytesseract.image_to_string(pil_for_ocr, lang=lang)
    except Exception:
        # fallback to English only
        text = pytesseract.image_to_string(pil_for_ocr, lang='eng')

    return text

# -------------------------
# Semantic search + field prompts
# -------------------------
FIELD_PROMPTS = {
    'fir_no': [
        "FIR number",
        "F.I.R. number",
        "FIR No",
        "First information report number",
        "एफआईआर नंबर",
        "FIR संख्या",
    ],
    'year': [
        "Year of FIR",
        "Year",
        "Date and year",
        "वर्ष",
    ],
    'state_name': [
        "State",
        "State name",
        "राज्य",
    ],
    'dist_name': [
        "District",
        "District name",
        "जिला",
    ],
    'police_station': [
        "Police station",
        "Thana",
        " थाना ",
    ],
    'under_acts': [
        "Acts under which case is registered",
        "Act",
        "अधिनियम",
        "IPC",
    ],
    'under_sections': [
        "Sections",
        "Under sections",
        "धारा",
    ],
    'revised_case_category': [
        "Case category",
        "Offence category",
        "revised case category",
    ],
    'oparty': [
        "Opposite party",
        "Complainant",
        "Plaintiff",
        "पक्ष",
    ],
    'accused_name': [
        "Accused",
        "Accused name",
        "अभियुक्त",
        "विंशेष व्यक्ति",
    ],
    'address': [
        "Address",
        "Address of accused",
        "पता",
    ],
    'jurisdiction': [
        "Jurisdiction",
        "Court jurisdiction",
        "क्षेत्र",
    ],
}

JURISDICTION_TYPES = [
    'PAN_INDIA', 'STATE', 'DISTRICT', 'SPECIAL', 'UNKNOWN'
]

@st.cache_resource
def build_prompt_embeddings(model_name: str = 'all-MiniLM-L6-v2') -> Tuple[SentenceTransformer, Dict[str, np.ndarray]]:
    st_model = init_sentence_transformer(model_name)
    prompt_embeddings = {}
    for field, prompts in FIELD_PROMPTS.items():
        emb = st_model.encode(prompts, convert_to_numpy=True)
        # average embedding for the set of prompts
        prompt_embeddings[field] = emb.mean(axis=0)
    return st_model, prompt_embeddings

# -------------------------
# Extraction helpers (no regex)
# -------------------------

def tokenize_text_into_sentences(text: str) -> List[str]:
    # first split by lines and then NLTK sentence tokenizer
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sents = []
    for ln in lines:
        try:
            sents.extend(nltk.tokenize.sent_tokenize(ln))
        except Exception:
            sents.append(ln)
    return sents


def find_candidate_sentences_for_field(sents: List[str], prompt_emb: np.ndarray, st_model: SentenceTransformer, topk: int = 3) -> List[Tuple[str, float]]:
    # embed sentences and compute cosine similarity
    sent_embs = st_model.encode(sents, convert_to_numpy=True)
    sims = cosine_similarity([prompt_emb], sent_embs)[0]
    idx_sorted = np.argsort(-sims)
    results = []
    for idx in idx_sorted[:topk]:
        results.append((sents[idx], float(sims[idx])))
    return results


def extract_digits_tokens(tokens: List[str]) -> List[str]:
    # returns tokens that contain digits (helps capture things like FIR no: 123/2020)
    out = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            out.append(t)
    return out


def join_adjacent_digit_tokens(tokens: List[str], start_idx: int) -> str:
    # join a run of adjacent tokens that have digits or punctuation often found in ids
    parts = []
    for i in range(start_idx, len(tokens)):
        t = tokens[i]
        if any(ch.isdigit() for ch in t) or any(p in t for p in ['/', '-', '.']):
            parts.append(t)
        else:
            break
    return ' '.join(parts)


def extract_value_from_sentence_by_field(sentence: str, field: str, stanza_pipelines: Dict[str, Any]) -> Tuple[Any, float]:
    """Given a sentence and target field, try to extract the value and a confidence score.
    Uses neural NER (stanza) for names/addresses and token-based heuristics for ids/years/sections.
    """
    # basic tokenization
    tokens = [t.strip() for t in sentence.replace('\t', ' ').split(' ') if t.strip()]

    # try language-specific NER via stanza (we'll try both pipelines to maximize recall)
    def stanza_entities(sent: str) -> List[Tuple[str, str]]:
        ents = []
        for lang, pipe in stanza_pipelines.items():
            try:
                doc = pipe(sent)
            except Exception:
                continue
            for sentn in doc.sentences:
                for ent in sentn.ents:
                    ents.append((ent.text, ent.type))
        return ents

    ents = stanza_entities(sentence)

    # field-specific extraction
    if field == 'fir_no':
        # look for tokens containing digits (e.g., 123/2022 or FIR/12345)
        for i, t in enumerate(tokens):
            if any(ch.isdigit() for ch in t):
                candidate = join_adjacent_digit_tokens(tokens, i)
                return candidate, 0.9
        return None, 0.3

    if field == 'year':
        for t in tokens:
            if t.isdigit() and len(t) == 4:
                y = int(t)
                if 1900 <= y <= 2100:
                    return str(y), 0.95
        # as fallback, look for any 4-digit substring
        for t in tokens:
            digits = ''.join(ch for ch in t if ch.isdigit())
            if len(digits) == 4:
                y = int(digits)
                if 1900 <= y <= 2100:
                    return str(y), 0.8
        return None, 0.2

    if field in ('state_name', 'dist_name', 'police_station'):
        # attempt to extract LOCATION-like entities from stanza
        locs = [text for (text, label) in ents if label in ('GPE', 'LOC', 'ORG')]
        if locs:
            return locs[0], 0.9
        # fallback: pick noun phrases (rough) by heuristics: consecutive titlecase tokens
        title_tokens = [t for t in tokens if any(c.isalpha() for c in t) and t[0].isupper()]
        if title_tokens:
            return ' '.join(title_tokens[:3]), 0.5
        return None, 0.2

    if field in ('oparty', 'accused_name'):
        persons = [text for (text, label) in ents if label in ('PERSON')]
        if persons:
            return persons[0], 0.92
        return None, 0.25

    if field == 'address':
        # addresses are often long; pick LOCATION/ORG entities or whole sentence
        locs = [text for (text, label) in ents if label in ('GPE', 'LOC', 'ORG')]
        if locs:
            return locs[0], 0.9
        # fallback: return entire sentence as probable address
        if any(ch.isdigit() for ch in sentence):
            return sentence.strip(), 0.6
        return None, 0.25

    if field == 'under_acts':
        # look for tokens like IPC or "Indian Penal Code" or Hindi equivalents
        lower = sentence.lower()
        if 'ipc' in lower or 'indian penal code' in lower or 'ipc' in sentence:
            # return the phrase around IPC
            return sentence.strip(), 0.85
        hindi_tokens = ['भारतीय दंड संहिता', 'आईपीसी']
        if any(ht in sentence for ht in hindi_tokens):
            return sentence.strip(), 0.85
        return None, 0.2

    if field == 'under_sections':
        # look for numeric tokens and tokens that contain digits; sections often have digits/commas
        digit_tokens = extract_digits_tokens(tokens)
        if digit_tokens:
            return ', '.join(digit_tokens), 0.85
        return None, 0.2

    if field == 'revised_case_category':
        # semantic classify by comparing against candidate categories via embedding (quick heuristic)
        return sentence.strip(), 0.5

    if field == 'jurisdiction':
        return sentence.strip(), 0.4

    return None, 0.0

# -------------------------
# Main extractor combining steps
# -------------------------

def extract_pi_fields_from_text(text: str, stanza_pipelines: Dict[str, Any], st_model: SentenceTransformer, prompt_embs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    sents = tokenize_text_into_sentences(text)
    results: Dict[str, Any] = {}

    for field in FIELD_PROMPTS.keys():
        candidates = find_candidate_sentences_for_field(sents, prompt_embs[field], st_model, topk=3)
        best_value = None
        best_conf = 0.0
        # try candidates in order
        for sent, score in candidates:
            val, conf = extract_value_from_sentence_by_field(sent, field, stanza_pipelines)
            combined_conf = float(score) * float(conf)
            if val and combined_conf > best_conf:
                best_conf = combined_conf
                best_value = val
        results[field] = {
            'value': best_value,
            'confidence': round(best_conf, 3)
        }

    # small post-processing: map jurisdiction type by heuristic
    jur = results.get('jurisdiction', {}).get('value')
    jtype = 'UNKNOWN'
    if jur:
        low = jur.lower()
        if 'pan' in low or 'all india' in low or 'पैन' in low:
            jtype = 'PAN_INDIA'
        elif 'state' in low or 'राज्य' in low:
            jtype = 'STATE'
        elif 'district' in low or 'जिला' in low:
            jtype = 'DISTRICT'
        else:
            jtype = 'UNKNOWN'
    results['jurisdiction_type'] = jtype

    return results

# -------------------------
# Streamlit UI
# -------------------------

def app():
    st.set_page_config(page_title='FIR PII Extractor', layout='wide')
    st.title('FIR PII Extractor — Hybrid (OCR + Neural NER + Semantic search)')

    st.markdown('Upload an FIR PDF (Hindi/English). Tool will try digital extraction first, then OCR. Uses Stanza (neural NER) and sentence-transformers for semantic matching. No regex.')

    uploaded_file = st.file_uploader('Upload FIR PDF', type=['pdf'], accept_multiple_files=False)
    if not uploaded_file:
        st.info('Upload a PDF to get started')
        st.stop()

    raw_bytes = uploaded_file.read()

    with st.spinner('Extracting text (pdfplumber -> fitz -> OCR fallback) ...'):
        text, page_images = extract_text_pdf_bytes(raw_bytes)

    st.success('Text extraction done')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('Extracted Text (first 1000 chars)')
        st.text_area('extracted_text', value=text[:10000], height=300)

    with col2:
        st.subheader('Preview (page images)')
        for i, img in enumerate(page_images[:3]):
            st.image(img, caption=f'Page {i+1}', use_column_width=True)

    # Initialize models
    with st.spinner('Loading NLP models (stanza + sentence-transformers) ...'):
        stanza_pipes = init_stanza_pipelines()
        st_model, prompt_embs = build_prompt_embeddings()

    with st.spinner('Running field extraction ...'):
        extracted = extract_pi_fields_from_text(text, stanza_pipes, st_model, prompt_embs)

    st.subheader('Extracted structured fields')
    st.json(extracted)

    st.markdown('---')
    st.markdown('### Download Results')
    btn = st.button('Download JSON')
    if btn:
        st.download_button('Download JSON', data=json.dumps(extracted, ensure_ascii=False, indent=2), file_name='fir_extracted.json', mime='application/json')

    st.markdown('---')
    st.header('Notes & next steps')
    st.markdown(
        """
        * This pipeline is hybrid and intentionally avoids brittle regular expressions; it uses semantic search + neural NER.
        * For best results in production: fine-tune a token-classification model (transformer) on labeled FIR data (Hindi+English) and plug into the pipeline.
        * You can improve OCR by installing better Tesseract models or using an enterprise OCR (Google Vision / AWS Textract) if available.
        * If you want, I can help convert this to a Dockerfile + requirements.txt and provide sample labeled data scripts for fine-tuning a model.
        """
    )


if __name__ == '__main__':
    app()
