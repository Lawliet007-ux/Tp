# streamlit_pii_batch.py
"""
Streamlit FIR PII Extractor — Offline batch + uploader
- Fully offline: uses PyMuPDF/pdfplumber + pytesseract + rapidfuzz
- Upload PDFs or point to server folder (path on deployed instance)
- Upload CSV to extend district/police station seeds
- Debug mode shows intermediate candidates
- Download JSON results or one ZIP of all outputs
"""

from __future__ import annotations
import streamlit as st
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64, io, zipfile
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz
from pathlib import Path

st.set_page_config(page_title="FIR PII Extractor (offline)", layout="wide")

# ---------------------- Default seeds (extendable via CSV) -------------------
DEFAULT_DISTRICT_SEED = [
    "Pune","Pune City","Mumbai","Mumbai City","Nagpur","Nashik","Raigad","Meerut","Lucknow","Varanasi","Kanpur","Noida","Ghaziabad"
]
DEFAULT_DISTRICT_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","रायगड","मेरठ","लखनऊ","वाराणसी","कानपुर"]
DEFAULT_PS_SEED = ["Bhosari","Hadapsar","Dadar","Andheri","Colaba","Cyber Crime Cell"]
DEFAULT_PS_SEED_DEV = ["भोसरी","हडपसर","डादर","अंधेरी"]

KNOWN_ACTS = {
    "ipc": "Indian Penal Code 1860",
    "information technology": "Information Technology Act 2000",
    "it act": "Information Technology Act 2000",
    "arms act": "Arms Act 1959",
    "crpc": "Code of Criminal Procedure 1973",
    "ndps": "NDPS Act 1985",
    "pocso": "POCSO Act 2012"
}

SECTION_MAX = 999

# ---------------------- Text normalization helpers ---------------------------
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def devanagari_to_ascii_digits(s: str) -> str:
    return s.translate(DEV_DIGITS)

def remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def strip_nonessential_unicode(s: str) -> str:
    return re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—%₹]", " ", s)

def collapse_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def fix_broken_devanagari_runs(s: str) -> str:
    def once(x): return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", x)
    prev = None; cur = s
    for _ in range(6):
        prev = cur; cur = once(cur)
        if cur == prev: break
    return cur

def canonicalize_text(s: str) -> str:
    if not s: return ""
    s = devanagari_to_ascii_digits(s)
    s = remove_control_chars(s)
    s = strip_nonessential_unicode(s)
    s = collapse_spaces(s)
    s = fix_broken_devanagari_runs(s)
    return s

# ---------------------- PDF -> Text extraction ------------------------------
def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = [p.get_text("text") or "" for p in doc]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_pdfplumber(path: str) -> str:
    try:
        out = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_ocr(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    try:
        doc = fitz.open(path)
        out=[]
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB",[pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=tesseract_langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str="eng+hin+mar") -> str:
    txt = extract_text_pymupdf(path)
    txt = canonicalize_text(txt)
    if len(txt) < 200:
        alt = extract_text_pdfplumber(path); alt = canonicalize_text(alt)
        if len(alt) > len(txt): txt = alt
    if len(txt) < 200:
        ocr = extract_text_ocr(path, tesseract_langs); ocr = canonicalize_text(ocr)
        if len(ocr) > len(txt): txt = ocr
    return canonicalize_text(txt)

# ---------------------- Candidate extractors (multilingual) -----------------
def find_year_candidates(text: str) -> List[str]:
    out=[]
    m = re.search(r"(?:Year|वर्ष|Date of FIR|Date)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m: out.append(m.group(1))
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    if m2: out.append(m2.group(0))
    return list(dict.fromkeys(out))

def find_police_station_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:P\.S\.|P\.S|Police Station|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})",
        r"Name of P\.S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip()
            if v: out.append(v)
    return list(dict.fromkeys(out))

def find_district_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:District|Dist\.|जिला|जिल्हा|District Name)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})",
        r"District\s*\(?([A-Za-z\u0900-\u097F ]{2,80})\)?\s*\("
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            out.append(m.group(1).strip())
    return list(dict.fromkeys(out))

def find_acts_candidates(text: str) -> List[str]:
    found=[]
    low = text.lower()
    for k, v in KNOWN_ACTS.items():
        if k in low and v not in found:
            found.append(v)
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा)[^\n]{0,120}", text, re.IGNORECASE):
        chunk = m.group(0).lower()
        for k,v in KNOWN_ACTS.items():
            if k in chunk and v not in found:
                found.append(v)
    return list(dict.fromkeys(found))

def find_section_candidates(text: str) -> List[str]:
    secs=[]
    for m in re.finditer(r"(?:Section|Sections|U\/s|U\/s\.|धारा|कलम|Sect)\b", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                try:
                    v = int(base.group(1))
                except:
                    continue
                if 1 <= v <= SECTION_MAX:
                    secs.append(str(v))
    if not secs:
        nums = re.findall(r"\b\d{1,3}\b", text)
        for n in nums:
            v = int(n)
            if v >= 10 and v <= SECTION_MAX:
                secs.append(str(v))
    return list(dict.fromkeys(secs))

def find_name_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:Complainant|Informant|तक्रारदार|सूचक|Complainant\/Informant)[^\n]{0,80}[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,160})",
        r"(?:Name|नाव|नाम)\s*[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,160})",
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip(); out.append(v)
    for m in re.finditer(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b", text):
        out.append(m.group(1).strip())
    return list(dict.fromkeys(out))

def find_address_candidates(text: str) -> List[str]:
    out=[]
    for m in re.finditer(r"(?:Address|पत्ता|Address\s*\(|पत्ता\s*\(|Address[:\-\)])\s*[:\-\)]?\s*([A-Za-z0-9\u0900-\u097F,./\- \n]{6,300})", text, re.IGNORECASE):
        v = m.group(1).strip()
        v = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport|Aadhar)", v, flags=re.IGNORECASE)[0].strip()
        out.append(" ".join(v.split()))
    for m in re.finditer(r"[A-Za-z\u0900-\u097F0-9, .\-]{10,200}\b\d{6}\b", text):
        out.append(m.group(0).strip())
    return list(dict.fromkeys(out))

# ---------------------- Fuzzy repair utilities -------------------------------
def fuzzy_repair_to_list(candidate: str, choices: List[str], threshold: int = 70) -> str:
    if not candidate or not choices:
        return candidate
    best = process.extractOne(candidate, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return candidate

# ---------------------- Extraction orchestrator ------------------------------
def extract_all_fields(text: str, district_seed, district_seed_dev, ps_seed, ps_seed_dev, debug: bool=False) -> Dict[str,Any]:
    t = canonicalize_text(text)

    # candidates
    year_c = find_year_candidates(t)
    ps_c = find_police_station_candidates(t)
    dist_c = find_district_candidates(t)
    acts_c = find_acts_candidates(t)
    sections_c = find_section_candidates(t)
    name_c = find_name_candidates(t)
    addr_c = find_address_candidates(t)

    # fuzzy repair
    district_rep = []
    for d in dist_c:
        rep = fuzzy_repair_to_list(d, district_seed + district_seed_dev)
        district_rep.append(rep if rep else d)
    ps_rep = []
    for p in ps_c:
        rep = fuzzy_repair_to_list(p, ps_seed + ps_seed_dev)
        ps_rep.append(rep if rep else p)

    # choose best: simple heuristics (label > fuzzy > fallback)
    def choose_best_label(items: List[str]) -> Optional[str]:
        if not items: return None
        return items[0].strip()

    year_best = choose_best_label(year_c)
    ps_best = choose_best_label(ps_rep or ps_c)
    dist_best = choose_best_label(district_rep or dist_c)
    name_best = choose_best_label(name_c)
    addr_best = choose_best_label(addr_c)

    revised_case_category = "OTHER"
    if acts_c and any("Information Technology" in a or "IT Act" in a for a in acts_c):
        revised_case_category = "CYBER_CRIME"
    if sections_c:
        sset = set(sections_c)
        if any(x in sset for x in ("354","376","509")):
            revised_case_category = "SEXUAL_OFFENCE"
        elif "302" in sset:
            revised_case_category = "MURDER"

    # oparty heuristics
    oparty = None
    if re.search(r"\b(आरोपी|accused|प्रतिवादी)\b", t, re.IGNORECASE):
        oparty = "Accused"
    elif re.search(r"\b(तक्रारदार|complainant|informant|सूचक)\b", t, re.IGNORECASE):
        oparty = "Complainant"

    jurisdiction = dist_best if dist_best else None
    jurisdiction_type = "DISTRICT" if dist_best else None

    out = {
        "year": year_best,
        "state_name": None,
        "dist_name": dist_best,
        "police_station": ps_best,
        "under_acts": acts_c if acts_c else None,
        "under_sections": sections_c if sections_c else None,
        "revised_case_category": revised_case_category,
        "oparty": oparty,
        "name": name_best,
        "address": addr_best,
        "jurisdiction": jurisdiction,
        "jurisdiction_type": jurisdiction_type
    }

    if debug:
        out["_debug"] = {
            "year_candidates": year_c,
            "ps_candidates": ps_c,
            "dist_candidates": dist_c,
            "acts_candidates": acts_c,
            "section_candidates": sections_c,
            "name_candidates": name_c,
            "address_candidates": addr_c
        }
    return out

# ---------------------- Streamlit UI ----------------------------------------
st.title("FIR PII Extractor — Offline (Streamlit)")

st.markdown(
    """
    Upload PDF(s) or point to a folder on the server (if deployed).  
    The extractor will attempt text extraction (PyMuPDF → pdfplumber → Tesseract OCR) and return a JSON per file.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    tesseract_langs = st.text_input("Tesseract languages", value="eng+hin+mar")
    debug = st.checkbox("Show debug candidates", value=False)
    st.markdown("You can upload a CSV (two columns) to extend district or police-station seeds.")

# Seed CSV uploads (optional)
st.subheader("Optional: Upload CSV to extend seeds")
col1, col2 = st.columns(2)
with col1:
    dist_csv = st.file_uploader("Upload district CSV (one district per row)", type=["csv"], key="d_csv")
with col2:
    ps_csv = st.file_uploader("Upload police station CSV (one PS per row)", type=["csv"], key="ps_csv")

# Build seeds
district_seed = list(DEFAULT_DISTRICT_SEED)
district_seed_dev = list(DEFAULT_DISTRICT_SEED_DEV)
ps_seed = list(DEFAULT_PS_SEED)
ps_seed_dev = list(DEFAULT_PS_SEED_DEV)

def load_csv_to_list(uploaded_file) -> List[str]:
    if not uploaded_file:
        return []
    try:
        raw = uploaded_file.read().decode(errors="ignore").splitlines()
        rows = [r.strip().strip('"').strip("'") for r in raw if r.strip()]
        return rows
    except Exception:
        return []

if dist_csv:
    new_d = load_csv_to_list(dist_csv)
    district_seed = new_d + district_seed
if ps_csv:
    new_ps = load_csv_to_list(ps_csv)
    ps_seed = new_ps + ps_seed

# Input: uploaded files OR server folder
st.subheader("Input PDFs")
mode = st.radio("Choose input mode", options=["Upload PDFs (recommended for testing)","Server folder (for bulk on deployed server)"], index=0)

uploaded_files = []
server_folder_path = None
if mode.startswith("Upload"):
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
else:
    server_folder_path = st.text_input("Enter absolute path to folder with PDFs on server (example: /home/appuser/app/pdfs)")
    st.caption("When deployed on Streamlit Cloud, you can place PDFs in your repo and use the repo path here.")

# Action button
run_btn = st.button("Run extraction")

# Results containers
results_container = st.container()
download_container = st.container()

# Processing
if run_btn:
    files_to_process: List[Tuple[str, bytes]] = []

    if mode.startswith("Upload"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            st.stop()
        for f in uploaded_files:
            # read to memory
            files_to_process.append((f.name, f.read()))
    else:
        if not server_folder_path:
            st.warning("Please enter server folder path")
            st.stop()
        p = Path(server_folder_path)
        if not p.exists() or not p.is_dir():
            st.warning("Server folder path does not exist on the server.")
            st.stop()
        # gather PDF file paths
        for fp in sorted(p.glob("*.pdf")):
            # read file bytes
            with open(fp, "rb") as fh:
                files_to_process.append((fp.name, fh.read()))

    total = len(files_to_process)
    st.info(f"Processing {total} file(s). This may take time if OCR kicks in.")

    results = {}
    progress = st.progress(0)
    i = 0

    # temporary directory to store per-file JSONs for zipping
    tmpdir = tempfile.mkdtemp()
    for name, data in files_to_process:
        i += 1
        progress.progress(int(i/total * 100))
        st.text(f"Processing: {name} ({i}/{total})")
        # write bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data); tmp_path = tmp.name
        try:
            # extract text with configured languages
            text = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
            # if extracted text too short, try OCR with stronger fallback
            if len(text) < 60:
                # try OCR explicitly
                text = extract_text_ocr(tmp_path, tesseract_langs)

            # run extractor
            out = extract_all_fields(text, district_seed, district_seed_dev, ps_seed, ps_seed_dev, debug=debug)
            results[name] = out

            # write per-file JSON in tmpdir
            outpath = os.path.join(tmpdir, f"{name}.json")
            with open(outpath, "w", encoding="utf-8") as wf:
                json.dump(out, wf, ensure_ascii=False, indent=2)

            # show per-file result in UI (collapsible)
            with results_container:
                st.markdown(f"**{name}**")
                st.json(out)
                if debug and "_debug" in out:
                    st.markdown("**Debug candidates**")
                    st.json(out["_debug"])
        except Exception as e:
            st.error(f"Error processing {name}: {e}")
            results[name] = {"error": str(e)}
        finally:
            try: os.remove(tmp_path)
            except: pass

    progress.progress(100)
    st.success(f"Processed {total} files.")

    # prepare downloadable ZIP of all JSON outputs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(tmpdir):
            zf.write(os.path.join(tmpdir, fname), arcname=fname)
    zip_buffer.seek(0)

    with download_container:
        st.download_button("Download all JSONs (zip)", data=zip_buffer.getvalue(), file_name="pii_extraction_results.zip")
        st.download_button("Download combined JSON", data=json.dumps(results, ensure_ascii=False, indent=2), file_name="combined_results.json")

    st.info("Tip: use Debug mode to see candidates and extend the seed CSVs to improve fuzzy repairing.")

