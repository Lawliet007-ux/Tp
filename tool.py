# batch_pii_extractor_offline.py
"""
Batch offline FIR PII extractor.
- Drop-in runnable script.
- Inputs: folder of PDFs.
- Outputs: JSON file per PDF and overall CSV summary.
"""
import os, re, json, tempfile, unicodedata, argparse, multiprocessing, math
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz, pdfplumber, pytesseract
from PIL import Image
from rapidfuzz import process, fuzz
from tqdm import tqdm
import pandas as pd

# Optional transformers NER (if installed and model available)
try:
    from transformers import pipeline
    HUGGING_AVAILABLE = True
except Exception:
    HUGGING_AVAILABLE = False

# ------------------ Config/Seeds ------------------
SECTION_MAX = 999
PLACEHOLDERS = set(["name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps"])

# Minimal seed lists -- replace with your full CSVs for best results
DISTRICT_SEED = ["Pune","Mumbai","Nagpur","Nashik","Meerut","Lucknow","Varanasi","Kanpur","Noida","Ghaziabad"]
DISTRICT_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","मेरठ","लखनऊ","वाराणसी","कानपुर"]
POLICE_PS_SEED = ["Bhosari","Hadapsar","Dadar","Andheri","Colaba"]
POLICE_PS_SEED_DEV = ["भोसरी","हडपसर","डादर","अंधेरी","काळाब"]

KNOWN_ACTS = {
    "ipc":"Indian Penal Code 1860",
    "it act":"Information Technology Act 2000",
    "arms act":"Arms Act 1959",
    "crpc":"Code of Criminal Procedure 1973",
    "ndps":"NDPS Act 1985",
    "pocso":"POCSO Act 2012"
}

# Tesseract languages
TESS_LANGS = "eng+hin+mar"

# ------------------ Utility functions ------------------
def canonicalize_text(s: str) -> str:
    if not s: return ""
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—%₹]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s).strip()
    # collapse spaces between devanagari runs
    for _ in range(3):
        s_new = re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", s)
        if s_new == s: break
        s = s_new
    # convert devanagari digits
    s = s.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return s

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

def extract_text_ocr(path: str, tesseract_langs: str = TESS_LANGS) -> str:
    try:
        doc = fitz.open(path)
        out=[]
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=tesseract_langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str) -> str:
    txt = extract_text_pymupdf(path)
    txt = canonicalize_text(txt)
    if len(txt) < 200:
        alt = extract_text_pdfplumber(path)
        alt = canonicalize_text(alt)
        if len(alt) > len(txt): txt = alt
    if len(txt) < 200:
        ocr = extract_text_ocr(path)
        ocr = canonicalize_text(ocr)
        if len(ocr) > len(txt): txt = ocr
    return canonicalize_text(txt)

# ------------------ Local candidate extractors ------------------
def find_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

def find_district(text: str) -> List[str]:
    cands=[]
    for m in re.finditer(r"(?:District|Dist\.|जिला|जिल्हा|District Name)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", text, re.IGNORECASE):
        cands.append(m.group(1).strip())
    return list(dict.fromkeys(cands))

def find_police_station(text: str) -> List[str]:
    c=[]
    for m in re.finditer(r"(?:Police Station|P\.S\.|PS|पोलीस ठाणे|थाना)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", text, re.IGNORECASE):
        c.append(m.group(1).strip())
    return list(dict.fromkeys(c))

def find_acts(text: str) -> List[str]:
    found=[]
    low = text.lower()
    for k,v in KNOWN_ACTS.items():
        if k in low and v not in found:
            found.append(v)
    return found

def find_sections(text: str) -> List[str]:
    secs=[]
    for m in re.finditer(r"(?:Section|धारा|कलम|U/s|U/s\.)", text, re.IGNORECASE):
        window = text[m.start():m.start()+200]
        nums = re.findall(r"\b\d{1,3}\b", window)
        for n in nums:
            ni=int(n)
            if 1<=ni<=SECTION_MAX:
                secs.append(str(ni))
    if not secs:
        nums = re.findall(r"\b\d{2,3}\b", text)
        for n in nums:
            ni=int(n)
            if 10<=ni<=SECTION_MAX: secs.append(str(ni))
    return list(dict.fromkeys(secs))

def find_names_addresses_via_regex(text: str) -> (List[str], List[str]):
    names=[]
    addrs=[]
    for m in re.finditer(r"(?:Name|नाव|नाम|Complainant|Informant|तक्रारदार|सूचक)\s*[:\-\)]\s*([A-Za-z\u0900-\u097F .]{2,160})", text, re.IGNORECASE):
        names.append(m.group(1).strip())
    for m in re.finditer(r"(?:Address|पत्ता)\s*[:\-\)]\s*([A-Za-z0-9\u0900-\u097F,./\-\n ]{8,300})", text, re.IGNORECASE):
        v = m.group(1).strip()
        v = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport)", v, flags=re.IGNORECASE)[0].strip()
        addrs.append(" ".join(v.split()))
    return list(dict.fromkeys(names)), list(dict.fromkeys(addrs))

def fuzzy_repair_candidate(val: str, choices: List[str], threshold:int=70) -> str:
    if not val or not choices: return val
    best = process.extractOne(val, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return val

# ------------------ NER wrapper (optional) ------------------
def load_hf_ner(model_name: str = "ai4bharat/indic-ner"):
    if not HUGGING_AVAILABLE:
        return None
    try:
        return pipeline("ner", model=model_name, aggregation_strategy="simple")
    except Exception as e:
        print("NER load failed:", e)
        return None

def ner_extract(text: str, ner_pipe) -> Dict[str,List[str]]:
    if ner_pipe is None: return {"PER":[], "LOC":[], "ORG":[]}
    res={"PER":[], "LOC":[], "ORG":[]}
    try:
        ents = ner_pipe(text[:8000])
        for e in ents:
            grp = e.get("entity_group") or e.get("entity")
            word = e.get("word") or ""
            if not word: continue
            word = word.strip()
            if grp in ("PER","PERSON"): res["PER"].append(word)
            elif grp in ("LOC","LOCATION","GPE"): res["LOC"].append(word)
            elif grp in ("ORG",): res["ORG"].append(word)
    except Exception as e:
        # ignore errors
        pass
    # dedupe
    for k in res: res[k]=list(dict.fromkeys(res[k]))
    return res

# ------------------ Single-document pipeline ------------------
def extract_document(path: str, ner_pipe=None, debug=False) -> Dict[str,Any]:
    txt = extract_text_from_pdf(path)
    if debug:
        print("Extracted text (first 400 chars):", txt[:400])
    out = {
        "year": None, "state_name": None, "dist_name": None, "police_station": None,
        "under_acts": None, "under_sections": None, "revised_case_category": None,
        "oparty": None, "name": None, "address": None, "jurisdiction": None, "jurisdiction_type": None,
        "_candidates": {}
    }

    # regex candidates
    year = find_year(txt)
    dists = find_district(txt)
    pss = find_police_station(txt)
    acts = find_acts(txt)
    secs = find_sections(txt)
    names, addrs = find_names_addresses_via_regex(txt)

    out["_candidates"]["year"]=year
    out["_candidates"]["dists"]=dists
    out["_candidates"]["pss"]=pss
    out["_candidates"]["acts"]=acts
    out["_candidates"]["secs"]=secs
    out["_candidates"]["names"]=names
    out["_candidates"]["addrs"]=addrs

    # ner candidates
    if ner_pipe:
        ner_c = ner_extract(txt, ner_pipe)
        out["_candidates"]["ner"]=ner_c
    else:
        out["_candidates"]["ner"]={"PER":[],"LOC":[],"ORG":[]}

    # choose best candidates (simple scoring)
    # Year
    out["year"] = year

    # name: prefer regex label candidate, else NER PER
    candidate_name = None
    if names:
        candidate_name = names[0]
    elif out["_candidates"]["ner"]["PER"]:
        candidate_name = out["_candidates"]["ner"]["PER"][0]
    out["name"] = candidate_name

    # address: regex then NER LOC
    candidate_addr = addrs[0] if addrs else (out["_candidates"]["ner"]["LOC"][0] if out["_candidates"]["ner"]["LOC"] else None)
    out["address"] = candidate_addr

    # dist: labeled regex, else ner loc; then fuzzy repair to seed lists
    cand_dist = dists[0] if dists else (out["_candidates"]["ner"]["LOC"][0] if out["_candidates"]["ner"]["LOC"] else None)
    cand_dist = fuzzy_repair_candidate(cand_dist, DISTRICT_SEED + DISTRICT_SEED_DEV) if cand_dist else None
    out["dist_name"] = cand_dist

    # police station: labeled then ner then fuzzy repair
    cand_ps = pss[0] if pss else (out["_candidates"]["ner"]["ORG"][0] if out["_candidates"]["ner"]["ORG"] else None)
    cand_ps = fuzzy_repair_candidate(cand_ps, POLICE_PS_SEED + POLICE_PS_SEED_DEV) if cand_ps else None
    out["police_station"] = cand_ps

    # acts and sections
    out["under_acts"] = acts if acts else None
    out["under_sections"] = secs if secs else None

    # revised category heuristics
    rc = "OTHER"
    if acts and any("Information Technology" in a or "IT Act" in a for a in acts): rc="CYBER_CRIME"
    if secs and any(s in secs for s in ["354","376","509"]): rc="SEXUAL_OFFENCE"
    if acts and any("Arms" in a for a in acts): rc="WEAPONS"
    out["revised_case_category"]=rc

    # oparty detection
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", txt, re.IGNORECASE):
        out["oparty"]="Accused"
    elif re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", txt, re.IGNORECASE):
        out["oparty"]="Complainant"

    # jurisdiction inference
    if out["dist_name"]:
        out["jurisdiction"]=out["dist_name"]; out["jurisdiction_type"]="DISTRICT"
    else:
        out["jurisdiction_type"]=None

    # Clean placeholders
    for k in ["name","address","dist_name","police_station"]:
        v = out.get(k)
        if v and (v.strip().lower() in PLACEHOLDERS or len(re.sub(r"\W+","",v))<2):
            out[k]=None

    return out

# ------------------ Batch runner ------------------
def process_one(args):
    path, ner_name, debug = args
    ner_pipe = None
    if ner_name:
        ner_pipe = load_ner(ner_name)  # lazy loader; implement outside if heavy
    try:
        result = extract_document(path, ner_pipe=ner_pipe, debug=debug)
    except Exception as e:
        result = {"error": str(e)}
    return path, result

# lightweight ner loader function (to avoid reloading in each process)
_NER_CACHE={}
def load_ner(name):
    global _NER_CACHE
    if name in _NER_CACHE:
        return _NER_CACHE[name]
    try:
        pipe = pipeline("ner", model=name, aggregation_strategy="simple")
        _NER_CACHE[name]=pipe
        return pipe
    except Exception as e:
        print("NER load failed:", e)
        return None

def batch_process(input_folder: str, output_folder: str, processes: int = 4, ner_model: Optional[str] = None, debug=False):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder); output_folder.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([str(p) for p in input_folder.glob("*.pdf")])
    args = [(p, ner_model, debug) for p in pdfs]

    results = {}
    with multiprocessing.Pool(processes) as pool:
        for path, res in tqdm(pool.imap_unordered(process_one, args), total=len(args)):
            fname = Path(path).name
            out_path = output_folder / (fname + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            results[fname] = res

    # produce CSV summary
    rows=[]
    for fname,res in results.items():
        if "error" in res:
            rows.append({"file":fname,"error":res["error"]})
        else:
            rows.append({
                "file":fname,
                "year":res.get("year"),
                "state_name":res.get("state_name"),
                "dist_name":res.get("dist_name"),
                "police_station":res.get("police_station"),
                "under_acts": ";".join(res.get("under_acts") or []),
                "under_sections": ";".join(res.get("under_sections") or []),
                "revised_case_category": res.get("revised_case_category"),
                "oparty": res.get("oparty"),
                "name": res.get("name"),
                "address": res.get("address"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_folder)/"summary.csv", index=False)
    print(f"Done. Results written to {output_folder}")

# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder with PDFs")
    parser.add_argument("--output_folder", required=True, help="Folder to write JSON outputs")
    parser.add_argument("--processes", type=int, default=max(1, multiprocessing.cpu_count()-1))
    parser.add_argument("--ner_model", type=str, default=None, help="Optional huggingface model name for NER (e.g. ai4bharat/indic-ner)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # if ner model specified but transformers not available, warn
    if args.ner_model and not HUGGING_AVAILABLE:
        print("transformers not installed; running without NER.")
        args.ner_model = None

    batch_process(args.input_folder, args.output_folder, processes=args.processes, ner_model=args.ner_model, debug=args.debug)
