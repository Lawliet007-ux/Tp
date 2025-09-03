import streamlit as st
import re
import json
import pandas as pd
from typing import Dict, List

# ---------- Extraction Functions ---------- #

def extract_fir_no(text: str) -> str:
    match = re.search(r'FIR\s*No[^\d]*(\d+)', text, re.IGNORECASE)
    return match.group(1) if match else ""

def extract_year(text: str) -> str:
    match = re.search(r'Year\s*\(.*?\)\s*[:\-]?\s*(\d{4})', text, re.IGNORECASE)
    return match.group(1) if match else ""

def extract_police_station(text: str) -> str:
    match = re.search(r'P\.?S\.?\s*\(.*?\):\s*([^\n]+)', text)
    return match.group(1).strip() if match else ""

def extract_district(text: str) -> str:
    match = re.search(r'District\s*\(.*?\):\s*([^\n]+)', text)
    return match.group(1).strip() if match else ""

def extract_state(text: str) -> str:
    match = re.search(r'State\s*\(.*?\):\s*([^\n]+)', text)
    return match.group(1).strip() if match else ""

def extract_under_acts_sections(text: str) -> Dict[str, List[str]]:
    acts, sections = [], []
    for line in text.splitlines():
        if re.search(r'धिननयम|Act', line, re.IGNORECASE):
            continue
        if re.search(r'\d+\s*[,;]?', line) and not line.strip().isdigit():
            if re.search(r'\d{3,}', line):
                acts.append(re.sub(r'\s+', ' ', line).strip())
            if re.search(r'\d+(\(\d+\))?', line):
                sections.append(re.sub(r'\s+', ' ', line).strip())
    return {"under_acts": list(set(acts)), "under_sections": list(set(sections))}

def extract_names(text: str) -> List[str]:
    matches = re.findall(r'Name\s*\(.*?\):\s*([^\n]+)', text)
    return [m.strip() for m in matches if m.strip()]

def extract_address(text: str) -> List[str]:
    matches = re.findall(r'Address\s*\(.*?\):\s*([^\n]+)', text)
    return [m.strip() for m in matches if m.strip()]

def extract_oparty(text: str) -> str:
    if "Accused" in text:
        return "Accused"
    elif "Complainant" in text or "Informant" in text:
        return "Complainant"
    return ""

def extract_jurisdiction(text: str) -> Dict[str, str]:
    if "outside the limit" in text:
        return {"jurisdiction": "Outside PS limit", "jurisdiction_type": "EXTERNAL"}
    return {"jurisdiction": "Within PS limit", "jurisdiction_type": "LOCAL"}

def extract_case_category(acts: List[str]) -> str:
    if any("IPC" in a for a in acts):
        return "Criminal"
    if any("IT Act" in a for a in acts):
        return "Cyber Crime"
    return "General"

# ---------- Main Extraction Pipeline ---------- #

def extract_pii(text: str) -> Dict:
    acts_sections = extract_under_acts_sections(text)
    data = {
        "fir_no": extract_fir_no(text),
        "year": extract_year(text),
        "state_name": extract_state(text),
        "dist_name": extract_district(text),
        "police_station": extract_police_station(text),
        "under_acts": acts_sections["under_acts"],
        "under_sections": acts_sections["under_sections"],
        "revised_case_category": extract_case_category(acts_sections["under_acts"]),
        "oparty": extract_oparty(text),
        "name": extract_names(text),
        "address": extract_address(text),
    }
    data.update(extract_jurisdiction(text))
    return data

# ---------- Streamlit App ---------- #

st.title("FIR PII Extraction Tool 🚔")

input_mode = st.radio("Select Input Mode", ["Single FIR Text", "Multiple FIR Texts"])

if input_mode == "Single FIR Text":
    text_input = st.text_area("Paste FIR Text Here", height=400)
    if st.button("Extract"):
        if text_input.strip():
            result = extract_pii(text_input)
            st.json(result)

elif input_mode == "Multiple FIR Texts":
    text_input = st.text_area("Paste Multiple FIR Texts (separated by ----)", height=400)
    if st.button("Extract All"):
        if text_input.strip():
            fir_texts = text_input.split("----")
            results = [extract_pii(txt) for txt in fir_texts if txt.strip()]
            df = pd.DataFrame(results)
            st.dataframe(df)
            st.download_button("Download JSON", json.dumps(results, indent=2, ensure_ascii=False), "fir_results.json")
