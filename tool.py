import streamlit as st
import fitz  # PyMuPDF for accurate text extraction with Unicode support (handles Hindi/English well)
import re
import json
from typing import Dict, Any
from io import BytesIO

# Function to extract text from PDF using PyMuPDF (fitz) - this preserves Unicode for Hindi/English
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        # Read the uploaded file into bytes
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"  # Extract text block by block to preserve structure
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
    return text

# Function to parse PII from extracted text using regex patterns
# Patterns are designed based on common FIR structures in the samples (Hindi/English mix)
# We focus on exact matches, no hallucinations, only extract if present
def parse_pii(text: str) -> Dict[str, Any]:
    pii = {
        "year": None,
        "state_name": None,
        "dist_name": None,
        "police_station": None,
        "under_acts": [],
        "under_sections": [],
        "revised_case_category": None,
        "oparty": {"complainant": None, "accused": []},
        "address": None,
        "jurisdiction": None,
        "jurisdiction_type": "LOCAL"  # Default to LOCAL unless PAN_INDIA detected (rare in FIRs)
    }

    # Year: From "Year (वष�):" or date
    year_match = re.search(r'Year \(वष�\): (\d{4})', text, re.IGNORECASE)
    if year_match:
        pii["year"] = year_match.group(1)
    else:
        # Fallback to FIR date year
        date_match = re.search(r'Date and Time of FIR.*(\d{4})', text, re.IGNORECASE)
        if date_match:
            pii["year"] = date_match.group(1)

    # State: From address or fixed patterns (e.g., Maharashtra from samples)
    state_match = re.search(r'(महाराष्ट्र|Maharashtra|महारा��|other states)', text, re.IGNORECASE)
    if state_match:
        pii["state_name"] = state_match.group(1).strip()

    # District: From "District (�ज��ा):"
    dist_match = re.search(r'District \(�ज��ा\): ([\w\s]+)', text, re.IGNORECASE | re.UNICODE)
    if dist_match:
        pii["dist_name"] = dist_match.group(1).strip()

    # Police Station: From "P.S. (पोलीस ठाणे):"
    ps_match = re.search(r'P.S. \(पोलीस ठाणे\): ([\w\s]+)', text, re.IGNORECASE | re.UNICODE)
    if ps_match:
        pii["police_station"] = ps_match.group(1).strip()

    # Under Acts and Sections: From table-like structure
    acts_sections = re.findall(r'Acts \(�धिननयम\)\s*Sections \(कलम\)\s*([\w\s\d,()�]+)\s*(\d+)', text, re.UNICODE)
    for act, section in acts_sections:
        clean_act = act.strip().replace('�', '')  # Clean garbled chars if any
        pii["under_acts"].append(clean_act)
        pii["under_sections"].append(section.strip())

    # Deduplicate and join sections
    pii["under_acts"] = list(set(pii["under_acts"]))
    pii["under_sections"] = list(set(pii["under_sections"]))

    # Revised Case Category: Normalize based on sections/acts (rule-based mapping)
    # Add more mappings as needed for accuracy
    category_map = {
        "25": "Arms Act Violation",
        "3": "Arms Act Violation",
        "135": "Police Act Violation",
        "37(1)": "Police Act Violation",
        "303(2)": "Theft",
        "285": "Rash Driving/Negligence"
    }
    categories = set()
    for sec in pii["under_sections"]:
        if sec in category_map:
            categories.add(category_map[sec])
    if categories:
        pii["revised_case_category"] = ", ".join(categories)
    else:
        pii["revised_case_category"] = "Unknown"

    # Oparty: Complainant from "Name (नाव):"
    complainant_match = re.search(r'Name \(नाव\): ([\w\s�]+)', text, re.IGNORECASE | re.UNICODE)
    if complainant_match:
        pii["oparty"]["complainant"] = complainant_match.group(1).strip()

    # Accused: From accused details table
    accused_matches = re.findall(r'Name \(नाव\)\s*([\w\s�]+)', text, re.UNICODE)
    if accused_matches:
        pii["oparty"]["accused"] = [name.strip() for name in accused_matches if name.strip()]

    # Address: From address fields
    address_match = re.search(r'Address \(प�ा\): ([\w\s�,�/]+)', text, re.IGNORECASE | re.UNICODE | re.MULTILINE)
    if address_match:
        pii["address"] = address_match.group(1).strip()

    # Jurisdiction: Same as district or police station limit
    if pii["dist_name"]:
        pii["jurisdiction"] = pii["dist_name"]

    # Jurisdiction Type: Check if "PAN_INDIA" or similar mentioned, else LOCAL
    if "PAN_INDIA" in text.upper() or "ALL INDIA" in text.upper():
        pii["jurisdiction_type"] = "PAN_INDIA"

    # Ensure no unnecessary data: Only populate if found
    return {k: v for k, v in pii.items() if v is not None and v != [] and v != {}}

# Streamlit App
st.title("FIR PDF PII Extractor Tool")
st.markdown("Upload a PDF (FIR document) to extract PII. Supports Hindi/English mixed text.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Extracting text and PII...")
    text = extract_text_from_pdf(uploaded_file)
    
    # Optional: Display extracted text for verification
    with st.expander("View Extracted Text"):
        st.text_area("Raw Text", text, height=300)
    
    pii_data = parse_pii(text)
    
    if pii_data:
        st.subheader("Extracted PII")
        st.json(pii_data)
    else:
        st.warning("No PII extracted. Check if the PDF matches the expected FIR format.")
else:
    st.info("Upload a PDF to begin.")

# Instructions for running: pip install streamlit pymupdf
# Run with: streamlit run this_file.py
