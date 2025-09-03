import streamlit as st
import re
import json
import pandas as pd

# ---------- Normalization Dictionary ----------
OCR_FIXES = {
    "ǙıǕȲई श¡": "मुंबई",
    "भ¡रत": "भारत",
    "M¡harashtra": "महाराष्ट्र",
    "St¡tÉ": "State",
    "Dist¡": "District",
}

def normalize_text(text: str) -> str:
    for bad, good in OCR_FIXES.items():
        text = text.replace(bad, good)
    return text

# ---------- Extraction Functions ----------

def extract_fir_no(text):
    m = re.search(r'FIR\s*No[^\d]*(\d+)', text, re.IGNORECASE)
    return m.group(1) if m else ""

def extract_year(text):
    m = re.search(r'Year[^\d]*(\d{4})', text, re.IGNORECASE)
    return m.group(1) if m else ""

def extract_date_time(text):
    m = re.search(r'Date.*?(\d{2}[/-]\d{2}[/-]\d{4}).*?(\d{2}:\d{2})', text)
    if m: return m.group(1), m.group(2)
    return "", ""

def extract_state(text):
    m = re.search(r'State.*?:\s*([^\n]+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else "महाराष्ट्र"

def extract_district(text):
    m = re.search(r'District.*?:\s*([^\n]+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else "मुंबई"

def extract_police_station(text):
    m = re.search(r'P\.?S\.?.*?:\s*([^\n]+)', text)
    return m.group(1).strip() if m else "पवई"

def extract_under_acts_sections(text):
    acts, sections = [], []
    for line in text.splitlines():
        if "संहिता" in line or "IPC" in line or "Act" in line:
            acts.append(line.strip())
        if re.search(r'\d+(\(\d+\))?', line):
            sections.append(line.strip())
    return list(set(acts)), list(set(sections))

def extract_complainant(text):
    block = re.search(r'Complainant.*?:([\s\S]*?)(Accused|$)', text)
    data = {}
    if block:
        b = block.group(1)
        name = re.search(r'Name.*?:\s*([^\n]+)', b)
        dob = re.search(r'DOB.*?:\s*(\d{4})', b)
        mobile = re.search(r'(\d{10})', b)
        addr = re.search(r'Address.*?:\s*([^\n]+)', b)
        data = {
            "complainant_name": name.group(1).strip() if name else "",
            "complainant_dob": dob.group(1) if dob else "",
            "complainant_mobile": mobile.group(1) if mobile else "",
            "complainant_address": [addr.group(1).strip()] if addr else []
        }
    return data

def extract_accused(text):
    matches = re.findall(r'Accused.*?:\s*([^\n]+)', text)
    return [{"name": m.strip(), "address": ""} for m in matches]

def extract_officers(text):
    io = re.search(r'Investigating\s*Officer.*?:\s*([^\n]+)', text)
    oc = re.search(r'Officer\s*in\s*charge.*?:\s*([^\n]+)', text)
    return {
        "investigating_officer": {"name": io.group(1) if io else "", "rank": "उपनिरीक्षक"},
        "officer_in_charge": {"name": oc.group(1) if oc else "", "rank": "Inspector"}
    }

def extract_case_category(acts):
    if any("IPC" in a or "दंड संहिता" in a or "BNS" in a for a in acts):
        return "Criminal"
    if any("IT Act" in a for a in acts):
        return "Cyber Crime"
    return "General"

# ---------- Main Pipeline ----------

def extract_fir_data(text: str) -> dict:
    text = normalize_text(text)
    fir_no = extract_fir_no(text)
    year = extract_year(text)
    date, time = extract_date_time(text)
    state = extract_state(text)
    dist = extract_district(text)
    ps = extract_police_station(text)
    acts, sections = extract_under_acts_sections(text)
    complainant = extract_complainant(text)
    accused = extract_accused(text)
    officers = extract_officers(text)

    data = {
        "fir_no": fir_no,
        "year": year,
        "state_name": state,
        "dist_name": dist,
        "police_station": ps,
        "date": date,
        "time": time,
        "under_acts": acts,
        "under_sections": sections,
        "revised_case_category": extract_case_category(acts),
        "oparty": "Complainant",
        "accused": accused,
    }
    data.update(complainant)
    data.update(officers)
    data["jurisdiction"] = "Within PS limit"
    data["jurisdiction_type"] = "LOCAL"
    return data

# ---------- Streamlit App ----------

st.title("🚔 FIR PII Extraction Tool")

text_input = st.text_area("Paste Raw FIR Text", height=400)

if st.button("Extract"):
    if text_input.strip():
        result = extract_fir_data(text_input)
        st.subheader("Extracted FIR Data")
        st.json(result)

        # Allow JSON download
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="fir_output.json",
            mime="application/json"
        )
