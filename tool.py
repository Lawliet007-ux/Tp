# streamlit_pii_humantune.py
"""
Streamlit app: FIR PII Extractor with human-in-the-loop tuning
- Offline: PyMuPDF/pdfplumber -> Tesseract fallback
- Multiple candidate collectors, scoring, debug output
- Interactive correction UI: choose/enter final value per field
- Auto-updates seeds (district / police_station) and writes gold labels (jsonl)
"""

from __future__ import annotations
import streamlit as st
import fitz, pdfplumber, pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, io
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="FIR PII Extractor — Human-in-the-loop", layout="wide")

# -------------------- Config & seeds --------------------
SEEDS_DIR = Path("seeds")
SEEDS_DIR.mkdir(exist_ok=True)
GOLD_FILE = Path("gold_labels.jsonl")  # append corrections here
SEED_DIST = SEEDS_DIR / "districts.txt"
SEED_PS = SEEDS_DIR / "police_stations.txt"

# default seeds (small — extend with CSV or let app update them)
DEFAULT_DIST_SEED = ["Pune","Mumbai","Nagpur","Nashik","Meerut","Lucknow","Varanasi","Kanpur"]
DEFAULT_DIST_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","मेरठ","लखनऊ","वाराणसी","कानपुर"]
DEFAULT_PS_SEED = ["Bhosari","Hadapsar","Dadar","Andheri","Colaba"]
DEFAULT_PS_SEED_DEV = ["भोसरी","हडपसर","डादर","अंधेरी"]

SECTION_MAX = 999

# ensure seed files exist
if not SEED_DIST.exists():
    SEED_DIST.write_text("\n".join(DEFAULT_DIST_SEED + DEFAULT_DIST_SEED_DEV), encoding="utf-8")
if not SEED_PS.exists():
    SEED_PS.write_text("\n".join(DEFAULT_PS_SEED + DEFAULT_PS_SEED_DEV), encoding="utf-8")

def load_seed_list(path: Path) -> List[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except:
        return []

# -------------------- Normalization helpers --------------------
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
def canonicalize_text(s: str) -> str:
    if not s: return ""
    s = s.translate(DEV_DIGITS)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—%₹]", " ", s)
    s = re.sub(r"[ \t]+"," ", s)
    s = re.sub(r"[ \t]*\n[ \t]*","\n", s).strip()
    for _ in range(4):
        s_new = re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", s)
        if s_new == s: break
        s = s_new
    return s

# -------------------- PDF text extraction --------------------
def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = [p.get_text("text") or "" for p in doc]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_pdfplumber(path: str) -> str:
    try:
        out=[]
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_ocr(path: str, langs: str="eng+hin+mar") -> str:
    try:
        doc = fitz.open(path)
        out=[]
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB",[pix.width,pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf_bytes(pdf_bytes: bytes, langs: str="eng+hin+mar") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes); tp = tmp.name
    try:
        txt = extract_text_pymupdf(tp)
        txt = canonicalize_text(txt)
        if len(txt) < 200:
            alt = extract_text_pdfplumber(tp); alt = canonicalize_text(alt)
            if len(alt) > len(txt): txt = alt
        if len(txt) < 200:
            ocr = extract_text_ocr(tp, langs); ocr = canonicalize_text(ocr)
            if len(ocr) > len(txt): txt = ocr
        return canonicalize_text(txt)
    finally:
        try: os.remove(tp)
        except: pass

# -------------------- Candidate collectors (multiple strategies) --------------------
def find_year_candidates(text: str) -> List[str]:
    out=[]
    m = re.search(r"(?:Year|वर्ष|Date of FIR|Date)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m: out.append(m.group(1))
    for m in re.finditer(r"\b(19|20)\d{2}\b", text):
        out.append(m.group(0))
    return list(dict.fromkeys(out))

def find_section_candidates(text: str) -> List[str]:
    secs=[]
    for m in re.finditer(r"(?:धारा|कलम|Section|Sect|U/s|U/s\.)", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}(?:\([^\)]*\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                v = int(base.group(1))
                if 1 <= v <= SECTION_MAX: secs.append(str(v))
    if not secs:
        for n in re.findall(r"\b\d{2,3}\b", text):
            v = int(n)
            if 10 <= v <= SECTION_MAX: secs.append(str(v))
    return list(dict.fromkeys(secs))

def find_acts(text: str) -> List[str]:
    acts=[]
    low = text.lower()
    known = {
        "ipc":"Indian Penal Code 1860",
        "it act":"Information Technology Act 2000",
        "information technology":"Information Technology Act 2000",
        "arms act":"Arms Act 1959",
        "crpc":"Code of Criminal Procedure 1973",
        "pocso":"POCSO Act 2012"
    }
    for k,v in known.items():
        if k in low and v not in acts: acts.append(v)
    # capture "Act" chunks
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा)[^\n]{0,120}", text, re.IGNORECASE):
        ch = m.group(0).strip()
        for k,v in known.items():
            if k in ch.lower() and v not in acts:
                acts.append(v)
    return acts

def find_police_station_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:P\.S\.|Police Station|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,120})",
        r"Name of P\.S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,120})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip()
            if v: out.append(v)
    return list(dict.fromkeys(out))

def find_district_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:District|Dist\.|जिला|जिल्हा|District Name)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,120})",
        r"District\s*\(?([A-Za-z\u0900-\u097F ]{2,80})\)?\s*[,\\)]"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip()
            if v: out.append(v)
    return list(dict.fromkeys(out))

def find_name_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:Complainant|Informant|Complainant\/Informant|तक्रारदार|सूचक|अर्जदार)[^\n]{0,120}[:\-\)]?\s*([A-Za-z\u0900-\u097F .]{2,200})",
        r"(?:Name|नाव|नाम)\s*[:\-\)]\s*([A-Za-z\u0900-\u097F .]{2,200})",
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip(); out.append(v)
    # fallback capitalized english names
    for m in re.finditer(r"\b([A-Z][a-z]{1,25}(?:\s+[A-Z][a-z]{1,25}){1,3})\b", text):
        out.append(m.group(1).strip())
    return list(dict.fromkeys(out))

def find_address_candidates(text: str) -> List[str]:
    out=[]
    for m in re.finditer(r"(?:Address|पत्ता|Address\s*\(|पत्ता\s*\(|Address[:\-\)])\s*[:\-\)]?\s*([A-Za-z0-9\u0900-\u097F,./\-\n ]{8,300})", text, re.IGNORECASE):
        v = m.group(1).strip()
        v = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Aadhar|Passport|PAN)", v, flags=re.IGNORECASE)[0].strip()
        out.append(" ".join(v.split()))
    # fallback: long run containing pin code
    for m in re.finditer(r"([A-Za-z\u0900-\u097F0-9, .\-]{10,200}\b\d{6}\b)", text):
        out.append(m.group(1).strip())
    return list(dict.fromkeys(out))

# -------------------- fuzzy repair & scoring --------------------
def fuzzy_repair(candidate: Optional[str], seed_list: List[str], threshold:int=70) -> Optional[str]:
    if not candidate: return None
    if not seed_list: return candidate
    best = process.extractOne(candidate, seed_list, scorer=fuzz.WRatio)
    if best and best[1] >= threshold: return best[0]
    return candidate

def score_candidate(value: Optional[str], source: str) -> float:
    if not value: return 0.0
    base = 0.0
    if source == "label": base += 0.6
    if source == "fuzzy": base += 0.5
    if source == "ner": base += 0.4
    if source == "fallback": base += 0.2
    # boost for length (not too short)
    ln = len(value.strip())
    base += min(ln, 100) / 200.0
    # penalize if value is placeholder-like
    if re.match(r"^(name|नाव|नाम|type|address)$", value.strip(), re.IGNORECASE): base *= 0.1
    return round(base, 3)

# -------------------- orchestrator: get candidates + scores --------------------
def get_all_candidates(text: str, dist_seeds: List[str], ps_seeds: List[str]) -> Dict[str,List[Tuple[str,str,float]]]:
    t = canonicalize_text(text)
    candidates = {}

    # years
    years = find_year_candidates(t)
    candidates["year"] = [(y, "label", score_candidate(y, "label")) for y in years]

    # district
    dist_c = find_district_candidates(t)
    # add ner/loc fallback: we skip heavy NER for offline but keep placeholders
    dist_entries=[]
    for d in dist_c:
        repaired = fuzzy_repair(d, dist_seeds + dist_seeds)
        src = "fuzzy" if repaired != d else "label"
        dist_entries.append((repaired, src, score_candidate(repaired, src)))
    candidates["dist_name"] = dist_entries

    # police station
    ps_c = find_police_station_candidates(t)
    ps_entries=[]
    for p in ps_c:
        repaired = fuzzy_repair(p, ps_seeds + ps_seeds)
        src = "fuzzy" if repaired != p else "label"
        ps_entries.append((repaired, src, score_candidate(repaired, src)))
    candidates["police_station"] = ps_entries

    # acts and sections
    acts = find_acts(t)
    candidates["under_acts"] = [(a,"label",score_candidate(a,"label")) for a in acts]
    secs = find_section_candidates(t)
    candidates["under_sections"] = [(s,"label",score_candidate(s,"label")) for s in secs]

    # names
    names = find_name_candidates(t)
    candidates["name"] = [(n,"label",score_candidate(n,"label")) for n in names]

    # addresses
    addrs = find_address_candidates(t)
    candidates["address"] = [(a,"label",score_candidate(a,"label")) for a in addrs]

    # oparty: simple detection
    oparty = []
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE): oparty.append(("Accused","heuristic",0.8))
    if re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", t, re.IGNORECASE): oparty.append(("Complainant","heuristic",0.8))
    candidates["oparty"] = oparty

    # jurisdiction: prefer district
    jurisdiction = []
    if candidates.get("dist_name"):
        for v,src,sc in candidates["dist_name"]:
            jurisdiction.append((v,"derived",sc))
    candidates["jurisdiction"] = jurisdiction

    return candidates

# -------------------- Streamlit UI: main ------------------------------------
st.title("FIR PII Extractor — Human-in-the-loop tuning")

st.markdown("""
Use this app to extract PII from FIRs, correct them quickly, and auto-update seeds (districts / police stations).
Corrections are saved to `gold_labels.jsonl` and appended to seed files in `seeds/`.
""")

# sidebar
with st.sidebar:
    st.header("Settings")
    tesseract_langs = st.text_input("Tesseract langs", value="eng+hin+mar")
    debug = st.checkbox("Show debug candidates", value=True)
    st.markdown("You can also upload a CSV to PRE-POPULATE seeds (one entry per line).")

# allow user to upload seed CSVs optionally
seed_dist_upload = st.file_uploader("Upload district seed CSV (optional)", type=["csv","txt"])
seed_ps_upload = st.file_uploader("Upload PS seed CSV (optional)", type=["csv","txt"])

# build seed lists (start from files)
dist_seeds = load_seed_list(SEED_DIST)
ps_seeds = load_seed_list(SEED_PS)

def read_seed_upload(f) -> List[str]:
    try:
        text = f.read().decode(errors="ignore")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines
    except:
        return []

if seed_dist_upload:
    dist_seeds = read_seed_upload(seed_dist_upload) + dist_seeds
if seed_ps_upload:
    ps_seeds = read_seed_upload(seed_ps_upload) + ps_seeds

# input mode
mode = st.radio("Input mode", ["Upload PDFs", "Folder on server (path)"], index=0)
uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True) if mode=="Upload PDFs" else None
server_folder = st.text_input("Server folder path (absolute)") if mode!="Upload PDFs" else None

run_btn = st.button("Run extraction and open reviewer")

# results state in session
if "queue" not in st.session_state:
    st.session_state.queue = []  # list of tuples (filename, pdf_bytes, extracted_candidates, finalized_result or None)

# load queue
if run_btn:
    # clear existing
    st.session_state.queue = []
    files = []
    if mode=="Upload PDFs":
        if not uploaded:
            st.warning("Upload PDFs first.")
            st.stop()
        files = [(f.name, f.read()) for f in uploaded]
    else:
        if not server_folder:
            st.warning("enter server folder")
            st.stop()
        p = Path(server_folder)
        if not p.exists(): st.warning("folder not found"); st.stop()
        for fp in sorted(p.glob("*.pdf")):
            with open(fp,"rb") as fh:
                files.append((fp.name, fh.read()))

    # enqueue with extracted candidates
    for fname, data in files:
        text = extract_text_from_pdf_bytes(data, langs=tesseract_langs)
        candidates = get_all_candidates(text, dist_seeds, ps_seeds)
        # build initial best-guesses (highest score candidate)
        def top(cands):
            return cands[0][0] if cands else None
        initial = {
            "year": top(candidates.get("year",[])),
            "state_name": None,
            "dist_name": top(candidates.get("dist_name",[])),
            "police_station": top(candidates.get("police_station",[])),
            "under_acts": [v for v,_,_ in candidates.get("under_acts",[])],
            "under_sections": [v for v,_,_ in candidates.get("under_sections",[])],
            "revised_case_category": None,
            "oparty": top(candidates.get("oparty",[])),
            "name": top(candidates.get("name",[])),
            "address": top(candidates.get("address",[])),
            "jurisdiction": top(candidates.get("jurisdiction",[])),
            "jurisdiction_type": "DISTRICT" if top(candidates.get("dist_name",[])) else None
        }
        st.session_state.queue.append((fname, data, text, candidates, initial, None))

st.markdown("---")

# Reviewer area: show one doc at a time with navigation
if st.session_state.queue:
    idx = st.number_input("Document index", min_value=0, max_value=max(0,len(st.session_state.queue)-1), value=0, step=1)
    fname, pdf_bytes, text, candidates, initial, finalized = st.session_state.queue[idx]

    st.header(f"Review: {fname}  (doc {idx+1} of {len(st.session_state.queue)})")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Raw extracted text (snippet)")
        st.code(text[:4000])
    with col2:
        st.subheader("Quick stats")
        st.write(f"Candidates found — year: {len(candidates.get('year',[]))}, dist: {len(candidates.get('dist_name',[]))}, ps: {len(candidates.get('police_station',[]))}, names: {len(candidates.get('name',[]))}")

    st.markdown("### Suggested field values — pick or edit (helps train the system)")
    # helper to render dropdowns with candidates
    def render_field(field_name, cand_list, initial_val):
        st.write(f"**{field_name}**")
        options = ["(empty)"]
        # include unique candidates preserving order
        opts = []
        for v,src,sc in cand_list:
            if v and v not in opts: opts.append(v)
        options += opts
        # place initial value first if not in list
        if initial_val and initial_val not in options:
            options = [initial_val] + options
        sel = st.selectbox(f"Choose {field_name} value", options=options, index=0, key=f"{idx}_{field_name}_sel")
        # allow free text edit
        txt = st.text_input(f"Edit {field_name} (final)", value=sel if sel!="(empty)" else "", key=f"{idx}_{field_name}_txt")
        # compute confidence (max candidate score or low if empty)
        max_score = max((sc for _,_,sc in cand_list), default=0.0)
        st.caption(f"candidate_count={len(cand_list)}  top_score={max_score}")
        return txt.strip() if txt.strip()!="" else None

    # render fields
    year_val = render_field("year", candidates.get("year",[]), initial.get("year"))
    dist_val = render_field("dist_name", candidates.get("dist_name",[]), initial.get("dist_name"))
    ps_val = render_field("police_station", candidates.get("police_station",[]), initial.get("police_station"))
    name_val = render_field("name", candidates.get("name",[]), initial.get("name"))
    addr_val = render_field("address", candidates.get("address",[]), initial.get("address"))

    # sections and acts: show lists and allow edit
    st.write("**under_sections** (choose multiple / edit as comma-separated)**")
    secs_opts = [v for v,_,_ in candidates.get("under_sections",[])]
    secs_default = ",".join(secs_opts) if secs_opts else ",".join(initial.get("under_sections") or [])
    secs_free = st.text_input("Edit sections (comma separated)", value=secs_default, key=f"{idx}_sections")
    sections_final = [s.strip() for s in secs_free.split(",") if s.strip()]

    st.write("**under_acts** (list)**")
    acts_opts = [v for v,_,_ in candidates.get("under_acts",[])]
    acts_default = ",".join(acts_opts) if acts_opts else ",".join(initial.get("under_acts") or [])
    acts_free = st.text_input("Edit acts (comma separated)", value=acts_default, key=f"{idx}_acts")
    acts_final = [a.strip() for a in acts_free.split(",") if a.strip()]

    # oparty
    oparty_opts = [v for v,_,_ in candidates.get("oparty",[])]
    oparty_sel = st.selectbox("oparty (if known)", options=["(unknown)"] + oparty_opts, index=0, key=f"{idx}_oparty")
    oparty_val = None if oparty_sel=="(unknown)" else oparty_sel

    # final action buttons
    st.markdown("### Actions")
    cola, colb, colc = st.columns(3)
    with cola:
        if st.button("Save correction / Accept", key=f"{idx}_save"):
            # build final dict
            final = {
                "file": fname,
                "year": year_val,
                "state_name": None,
                "dist_name": dist_val,
                "police_station": ps_val,
                "under_acts": acts_final or None,
                "under_sections": sections_final or None,
                "revised_case_category": None,
                "oparty": oparty_val,
                "name": name_val,
                "address": addr_val,
                "jurisdiction": dist_val or None,
                "jurisdiction_type": "DISTRICT" if dist_val else None
            }
            # append to gold file
            try:
                with GOLD_FILE.open("a", encoding="utf-8") as gf:
                    gf.write(json.dumps(final, ensure_ascii=False) + "\n")
                st.success("Saved to gold_labels.jsonl")
            except Exception as e:
                st.error(f"Failed to save gold label: {e}")

            # auto-append new seeds (if present and not already included)
            updated = False
            if dist_val:
                ds = load_seed_list(SEED_DIST)
                if dist_val not in ds:
                    ds.append(dist_val); SEED_DIST.write_text("\n".join(ds), encoding="utf-8"); updated=True
            if ps_val:
                ps = load_seed_list(SEED_PS)
                if ps_val not in ps:
                    ps.append(ps_val); SEED_PS.write_text("\n".join(ps), encoding="utf-8"); updated=True
            if updated:
                st.info("Updated seed files with your corrections (districts / police_stations).")

            # store finalized result in session queue
            st.session_state.queue[idx] = (fname, pdf_bytes, text, candidates, initial, final)
    with colb:
        if st.button("Mark as SKIP (no change)", key=f"{idx}_skip"):
            st.info("Skipped (no gold label saved).")
            st.session_state.queue[idx] = (fname, pdf_bytes, text, candidates, initial, None)
    with colc:
        if st.button("Mark as BAD (needs full manual review)", key=f"{idx}_bad"):
            st.warning("Marked BAD — saved to gold with error flag")
            final = {"file": fname, "error": "BAD_DOCUMENT"}
            with GOLD_FILE.open("a", encoding="utf-8") as gf: gf.write(json.dumps(final, ensure_ascii=False) + "\n")
            st.session_state.queue[idx] = (fname, pdf_bytes, text, candidates, initial, final)

    st.markdown("---")
    st.write("Navigation: change Document index at top to move through queue")

    # export seeds / gold
    st.markdown("### Export / Utilities")
    if st.button("Download current seeds (districts + ps)"):
        z = {"districts": load_seed_list(SEED_DIST), "police_stations": load_seed_list(SEED_PS)}
        st.download_button("Download seeds JSON", json.dumps(z, ensure_ascii=False, indent=2), file_name="seeds_export.json")
    if GOLD_FILE.exists():
        if st.button("Download gold labels (jsonl)"):
            st.download_button("Download gold labels", GOLD_FILE.read_text(encoding="utf-8"), file_name="gold_labels.jsonl")

else:
    st.info("No documents in queue. Upload PDFs or point to a server folder and click 'Run extraction and open reviewer'.")

# -------------------- end of file --------------------
