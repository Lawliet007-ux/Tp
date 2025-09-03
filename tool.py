import streamlit as st
import re
import json

# ---------- Utility Functions ---------- #

def clean_text(text: str) -> str:
    """Basic OCR cleanup and whitespace normalization"""
    text = re.sub(r'[^\S\r\n]+', ' ', text)   # collapse spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[_¬^]+', '', text)        # remove OCR junk symbols
    return text.strip()

def get_block(text: str, start_keywords, end_keywords=None):
    """Extract a block between keywords"""
    start = None
    end = None
    for kw in start_keywords:
        m = re.search(kw, text, re.IGNORECASE)
        if m:
            start = m.end()
            break
    if start is None:
        return ""

    if end_keywords:
        for kw in end_keywords:
            m = re.search(kw, text[start:], re.IGNORECASE)
            if m:
                end = start + m.start()
                break

    return text[start:end].strip() if end else text[start:].strip()

# ---------- Extraction Functions ---------- #

def extract_fir_meta(text: str):
    fir_no = re.search(r'FIR\s*No[^\d]*(\d+)', text)
    year = re.search(r'Year[^\d]*(\d{4})', text)
    date = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', text)
    time = re.search(r'(\d{1,2}:\d{2})', text)
    ps = re.search(r'Police\s*Station.*?:\s*([^\n]+)', text, re.IGNORECASE)

    return {
        "fir_no": fir_no.group(1) if fir_no else "",
        "year": year.group(1) if year else "",
        "date": date.group(1) if date else "",
        "time": time.group(1) if time else "",
        "police_station": ps.group(1).strip() if ps else ""
    }

def extract_complainant(text: str):
    block = get_block(text, ["Complainant", "Informant", "Đािा"], ["Accused", "आरोपी"])
    if not block:
        return {}

    name = re.search(r'Name.*?:\s*([^\n]+)', block)
    dob = re.search(r'DOB.*?:\s*(\d{4})', block)
    mobile = re.search(r'(\d{10})', block)
    addr = re.search(r'Address.*?:\s*([^\n]+)', block)

    return {
        "complainant_name": name.group(1).strip() if name else "",
        "complainant_dob": dob.group(1) if dob else "",
        "complainant_mobile": mobile.group(1) if mobile else "",
        "complainant_address": [addr.group(1).strip()] if addr else []
    }

def extract_accused(text: str):
    block = get_block(text, ["Accused", "आरोपी"], ["Investigating", "Officer in charge"])
    accused_list = []
    if block:
        names = re.findall(r'Name.*?:\s*([^\n]+)', block)
        addrs = re.findall(r'Address.*?:\s*([^\n]+)', block)
        for i, n in enumerate(names):
            accused_list.append({
                "name": n.strip(),
                "address": addrs[i].strip() if i < len(addrs) else ""
            })
    return accused_list

def extract_acts_sections(text: str):
    block = get_block(text, ["Act", "धिननयम", "संहिता"], ["Complainant", "Informant", "Accused"])
    acts = re.findall(r'([A-Za-z\u0900-\u097F ]+Act[, ]*\d{4}|BNS[, ]*2023|IPC[, ]*\d{4})', block)
    sections = re.findall(r'(\d+[A-Za-z]?\(?\d*\)?)', block)
    return list(set(acts)), list(set(sections))

def extract_officers(text: str):
    io_block = get_block(text, ["Investigating Officer", "IO"], ["Officer in charge"])
    oc_block = get_block(text, ["Officer in charge"], None)

    io_name = re.search(r'([A-Z\u0900-\u097F][^\n]+)', io_block)
    oc_name = re.search(r'([A-Z\u0900-\u097F][^\n]+)', oc_block)

    return {
        "investigating_officer": {"name": io_name.group(1).strip() if io_name else "", "rank": "उपनिरीक्षक"},
        "officer_in_charge": {"name": oc_name.group(1).strip() if oc_name else "", "rank": "Inspector"}
    }

def extract_case_category(acts):
    if any("IPC" in a or "संहिता" in a or "BNS" in a for a in acts):
        return "Criminal"
    if any("IT Act" in a for a in acts):
        return "Cyber Crime"
    return "General"

# ---------- Main Pipeline ---------- #

def extract_fir(text: str):
    text = clean_text(text)

    meta = extract_fir_meta(text)
    complainant = extract_complainant(text)
    accused = extract_accused(text)
    acts, sections = extract_acts_sections(text)
    officers = extract_officers(text)

    data = {
        **meta,
        "state_name": "महाराष्ट्र",
        "dist_name": "मुंबई",
        "under_acts": acts,
        "under_sections": sections,
        "revised_case_category": extract_case_category(acts),
        "oparty": "Complainant",
        "accused": accused,
        "jurisdiction": "Within PS limit",
        "jurisdiction_type": "LOCAL"
    }
    data.update(complainant)
    data.update(officers)
    return data

# ---------- Streamlit UI ---------- #

st.title("🚔 FIR PII Extraction Tool (Final Version)")

text_input = st.text_area("Paste Raw FIR Text", height=400)

if st.button("Extract"):
    if text_input.strip():
        result = extract_fir(text_input)
        st.subheader("✅ Extracted FIR Data")
        st.json(result)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="fir_output.json",
            mime="application/json"
        )
