# fir_pii_extractor_gpt_hybrid.py
"""
FIR PII Extractor — GPT-hybrid approach (best-effort ChatGPT-style extraction)

Usage:
    streamlit run fir_pii_extractor_gpt_hybrid.py

Notes:
 - For ChatGPT-style output, provide an OpenAI API key (sidebar). The app will send the cleaned FIR text with a strict prompt that returns JSON only.
 - If you do not provide an API key, a local fallback extractor runs (regex + fuzzy + optional local NER).
"""

from __future__ import annotations
import streamlit as st
import fitz, pdfplumber, pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64, textwrap
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz

# OpenAI client
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — GPT Hybrid", layout="wide")

# -------------------- Configuration & seeds --------------------
SECTION_MAX = 999
PLACEHOLDERS = set(["name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps"])
DISTRICT_SEED = ["Pune","Mumbai","Nagpur","Nashik","Meerut","Lucknow","Varanasi","Kanpur","Noida","Ghaziabad"]
DISTRICT_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","मेरठ","लखनऊ","वाराणसी","कानपुर"]
POLICE_PS_SEED = ["Bhosari","Hadapsar","Dadar","Andheri","Colaba","Cyber Crime Cell"]
POLICE_PS_SEED_DEV = ["भोसरी","हडपसर","डादर","अंधेरी"]
KNOWN_ACTS = {
    "ipc":"Indian Penal Code 1860",
    "it act":"Information Technology Act 2000",
    "arms act":"Arms Act 1959",
    "crpc":"Code of Criminal Procedure 1973",
    "ndps":"NDPS Act 1985",
    "pocso":"POCSO Act 2012"
}

# -------------------- Text cleaning helpers --------------------
DEV_DIGITS = str.maketrans("०१२३४५६७८९","0123456789")

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

# -------------------- PDF -> text extraction --------------------
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
        out = []
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=tesseract_langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str="eng+hin+mar") -> str:
    text = extract_text_pymupdf(path)
    text = canonicalize_text(text)
    if len(text) < 200:
        alt = extract_text_pdfplumber(path); alt = canonicalize_text(alt)
        if len(alt) > len(text): text = alt
    if len(text) < 200:
        ocr = extract_text_ocr(path, tesseract_langs); ocr = canonicalize_text(ocr)
        if len(ocr) > len(text): text = ocr
    return canonicalize_text(text)

# -------------------- Local fallback extraction (regex + fuzzy) -----------
# This is used when OpenAI key is not provided. Good, but less flexible.

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out=[]
    for it in items:
        if it is None: continue
        s = it.strip()
        if not s: continue
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def filter_placeholder_candidate(c: Optional[str]) -> Optional[str]:
    if not c: return None
    v = c.strip()
    if not v: return None
    low = v.lower()
    if low in PLACEHOLDERS or re.search(r"\b(of p\.?s\.?|then name of p\.?s\.?)\b", low): return None
    if len(re.sub(r"\W+","", v)) < 2: return None
    return v

def fuzzy_repair(candidate: str, choices: List[str], threshold: int=65) -> str:
    if not candidate or not choices: return candidate
    best = process.extractOne(candidate, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold: return best[0]
    return candidate

def local_extract(text: str) -> Dict[str,Any]:
    t = canonicalize_text(text)
    out = {k: None for k in ["year","state_name","dist_name","police_station","under_acts","under_sections","revised_case_category","oparty","name","address","jurisdiction","jurisdiction_type"]}

    # Year
    m = re.search(r"\b(19|20)\d{2}\b", t)
    if m: out["year"] = m.group(0)

    # District (labelled)
    m = re.search(r"(?:District|Dist\.|जिला|जिल्हा)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", t, re.IGNORECASE)
    if m: out["dist_name"] = fuzzy_repair(m.group(1).strip(), DISTRICT_SEED + DISTRICT_SEED_DEV)

    # Police Station
    m = re.search(r"(?:Police Station|P\.S\.|PS|पोलीस ठाणे|थाना)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", t, re.IGNORECASE)
    if m: out["police_station"] = fuzzy_repair(m.group(1).strip(), POLICE_PS_SEED + POLICE_PS_SEED_DEV)

    # Acts
    acts = []
    for k, v in KNOWN_ACTS.items():
        if k in t.lower(): acts.append(v)
    out["under_acts"] = acts if acts else None

    # Sections: near labels
    sections=[]
    for m in re.finditer(r"(?:Section|धारा|कलम|U/s|U/s\.)", t, re.IGNORECASE):
        window = t[m.start():m.start()+200]
        nums = re.findall(r"\b\d{1,3}\b", window)
        for n in nums:
            n_int = int(n)
            if 1 <= n_int <= SECTION_MAX:
                sections.append(str(n_int))
    sections = dedupe_preserve_order(sections)
    out["under_sections"] = sections if sections else None

    # Name (labelled)
    m = re.search(r"(?:Name|नाव|नाम|Complainant|Informant|तक्रारदार|सूचक)\s*[:\-\)]\s*([A-Za-z\u0900-\u097F .]{2,160})", t, re.IGNORECASE)
    if m: out["name"] = m.group(1).strip()
    # Address (labelled)
    m = re.search(r"(?:Address|पत्ता)\s*[:\-\)]\s*([A-Za-z0-9\u0900-\u097F,./\-\n ]{6,200})", t, re.IGNORECASE)
    if m:
        addr = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport)", m.group(1), flags=re.IGNORECASE)[0].strip()
        out["address"] = " ".join(addr.split())

    # Oparty heuristics
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE): out["oparty"] = "Accused"
    elif re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", t, re.IGNORECASE): out["oparty"] = "Complainant"

    # Jurisdiction
    if out["dist_name"]:
        out["jurisdiction"] = out["dist_name"]; out["jurisdiction_type"] = "DISTRICT"
    else:
        out["jurisdiction_type"] = None

    # Case category simple mapping
    cat="OTHER"
    if out["under_acts"] and any("Information Technology" in a for a in out["under_acts"]): cat="CYBER_CRIME"
    out["revised_case_category"]=cat

    # sanitize placeholders
    for k in ["name","address","dist_name","police_station","state_name"]:
        out[k] = filter_placeholder_candidate(out.get(k))

    return out

# -------------------- OpenAI prompt & call --------------------

EXTRACTION_FIELDS = ["year","state_name","dist_name","police_station","under_acts","under_sections","revised_case_category","oparty","name","address","jurisdiction","jurisdiction_type"]

PROMPT_SYSTEM = """You are an extraction assistant that MUST return exactly one JSON object (no surrounding text).
You will be given the plain text of a police FIR (which may be in English, Hindi, Marathi, or a mixture).
Extract the following fields exactly as keys in JSON: year,state_name,dist_name,police_station,under_acts,under_sections,revised_case_category,oparty,name,address,jurisdiction,jurisdiction_type.

Rules:
- Return ONLY a single valid JSON object. No commentary, no markdown.
- Use null (not string) for missing fields.
- For lists (under_acts, under_sections) return JSON arrays or null.
- Sections should be numeric strings (e.g. "323","506"), no text labels.
- Do NOT invent data. If you're unsure, use null.
- Keep names and addresses in the original script if present (Hindi/Marathi), do not transliterate unless both scripts appear.
- revised_case_category: choose one of [CYBER_CRIME,WEAPONS,SEXUAL_OFFENCE,MURDER,FRAUD,OTHER]. Pick the best match based on acts/sections.
- jurisdiction_type should be "DISTRICT" or "STATE" or "PAN_INDIA" or null.
- If multiple plausible values exist, pick the most confident one.
- Do minimal normalization: strip obvious prefixes like 'District:' or 'Police Station:' from values.
- Ensure JSON validates (use null, not "null").

Example output format (exact keys, even if values null):
{"year":"2025","state_name":"Maharashtra","dist_name":"Pune","police_station":"Bhosari","under_acts":["Information Technology Act 2000"],"under_sections":["173"],"revised_case_category":"CYBER_CRIME","oparty":"Complainant","name":"राम कुमार","address":"Hadapsar, Pune","jurisdiction":"Pune","jurisdiction_type":"DISTRICT"}
"""

def build_extraction_prompt(text: str, instruction_extra: Optional[str]=None) -> str:
    # shorten or chunk if extremely long: keep top, mid, bottom areas where labels appear
    txt = text.strip()
    if len(txt) > 30000:
        # keep first 15000 and last 15000 (heuristic)
        txt = txt[:15000] + "\n\n... (skipped middle) ...\n\n" + txt[-15000:]
    # wrap safely
    prompt = PROMPT_SYSTEM + "\n\nFIR_TEXT_BEGIN\n" + txt + "\nFIR_TEXT_END\n"
    if instruction_extra:
        prompt += "\n\n" + instruction_extra
    return prompt

def call_openai_extract(text: str, api_key: str, model: str="gpt-4", temperature: float=0.0) -> Optional[Dict[str,Any]]:
    if not OPENAI_AVAILABLE:
        st.error("OpenAI python package not installed.")
        return None
    openai.api_key = api_key
    prompt = build_extraction_prompt(text)
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":PROMPT_SYSTEM},
                {"role":"user","content":text}
            ],
            temperature=temperature,
            max_tokens=1500
        )
        # fetch assistant content
        content = resp["choices"][0]["message"]["content"].strip()
        # try to locate JSON in content
        json_text = None
        # if content is pure JSON, parse
        try:
            json_text = content
            parsed = json.loads(json_text)
            return parsed
        except Exception:
            # try to extract first {...} block
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    return parsed
                except Exception:
                    # fallthrough
                    pass
        # nothing parseable
        st.error("OpenAI returned unparseable output. See debug output.")
        st.code(content)
        return None
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# -------------------- Post-process & validate --------------------------------

def post_process_model_output(obj: Dict[str,Any]) -> Dict[str,Any]:
    out = {k: None for k in EXTRACTION_FIELDS}
    if not isinstance(obj, dict):
        return out
    # copy keys if present
    for k in EXTRACTION_FIELDS:
        v = obj.get(k, None)
        # normalize under_sections to list of numeric strings
        if k == "under_sections" and v:
            cleaned=[]
            if isinstance(v, list):
                for s in v:
                    sstr = str(s)
                    m = re.search(r"(\d{1,3})", sstr)
                    if m: cleaned.append(m.group(1))
            elif isinstance(v, str):
                for s in re.findall(r"\d{1,3}", v):
                    cleaned.append(s)
            out[k] = cleaned if cleaned else None
            continue
        # under_acts: ensure list
        if k == "under_acts" and v:
            if isinstance(v, list):
                out[k] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                out[k] = [x.strip() for x in re.split(r"[;,/]\s*", v) if x.strip()]
            else:
                out[k] = None
            continue
        # simple copy with placeholder filtering
        if isinstance(v, str):
            vv = v.strip()
            if vv.lower() in ("null","none","n/a",""):
                out[k] = None
            else:
                out[k] = vv
        else:
            out[k] = v
    # basic fuzzy repairs: district and police_station
    if out.get("dist_name"):
        out["dist_name"] = fuzzy_repair(out["dist_name"], DISTRICT_SEED + DISTRICT_SEED_DEV)
    if out.get("police_station"):
        out["police_station"] = fuzzy_repair(out["police_station"], POLICE_PS_SEED + POLICE_PS_SEED_DEV)
    # jurisdiction inference
    if not out.get("jurisdiction"):
        if out.get("dist_name"):
            out["jurisdiction"] = out["dist_name"]
            out["jurisdiction_type"] = "DISTRICT"
        elif out.get("state_name"):
            out["jurisdiction"] = out["state_name"]
            out["jurisdiction_type"] = "STATE"
    return out

# -------------------- Streamlit UI ------------------------------------------

st.title("FIR PII Extractor — GPT-hybrid (ChatGPT-style JSON)")

with st.sidebar:
    st.header("Settings")
    use_openai = st.checkbox("Use OpenAI (recommended for ChatGPT-style)", value=True)
    api_key_input = st.text_input("OpenAI API key (or set OPENAI_API_KEY env var)", type="password")
    model_choice = st.selectbox("Model to use (OpenAI)", options=["gpt-4","gpt-4o","gpt-4o-mini","gpt-3.5-turbo"], index=0)
    tesseract_langs = st.text_input("Tesseract langs", value="eng+hin+mar")
    debug = st.checkbox("Debug: show intermediate text and candidates", value=False)
    st.markdown("If no OpenAI key is provided, the app will run a local fallback extractor.")

uploaded = st.file_uploader("Upload FIR PDF(s)", type=["pdf"], accept_multiple_files=True)
pasted = st.text_area("Or paste FIR text here", height=300, placeholder="Paste FIR text or OCR content here...")

if st.button("Extract PII"):
    results = {}
    # prepare API key
    api_key = api_key_input or os.environ.get("OPENAI_API_KEY") or None
    # process uploaded files first
    if uploaded:
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            try:
                text = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
                if debug:
                    st.subheader(f"Raw extracted text for {f.name} (first 2000 chars):")
                    st.code(text[:2000])
                if use_openai and api_key:
                    parsed = call_openai_extract(text, api_key=api_key, model=model_choice)
                    if parsed:
                        final = post_process_model_output(parsed)
                    else:
                        final = local_extract(text)
                else:
                    final = local_extract(text)
                results[f.name] = final
            finally:
                try: os.remove(tmp_path)
                except: pass

    # process pasted text
    if pasted and pasted.strip():
        text = canonicalize_text(pasted)
        if debug:
            st.subheader("Raw pasted text (first 2000 chars)")
            st.code(text[:2000])
        if use_openai and api_key:
            parsed = call_openai_extract(text, api_key=api_key, model=model_choice)
            if parsed:
                final = post_process_model_output(parsed)
            else:
                final = local_extract(text)
        else:
            final = local_extract(text)
        results["pasted_text"] = final

    if not results:
        st.warning("Please upload one or more PDFs or paste text.")
    else:
        st.subheader("Extracted PII (final)")
        st.json(results, expanded=True)
        out_json = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(out_json.encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
        st.success("Extraction done. If some fields are wrong, enable Debug and examine the extracted text + adjust seed lists.")

