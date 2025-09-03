import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

# Load model (choose any instruct model from HF)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

SYSTEM_PROMPT = """
You are a legal FIR data extractor.
Extract structured JSON from the given FIR text with these fields:
- fir_no
- year
- state_name
- dist_name
- police_station
- under_acts
- under_sections
- revised_case_category
- oparty
- name
- address
- jurisdiction
- jurisdiction_type
- complainant (name, dob, mobile, address)
- accused (list of {name, address})
- investigating_officer (name, rank)
- officer_in_charge (name, rank)
Return ONLY valid JSON, no explanation.
"""

def extract_with_llm(fir_text: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\nFIR Text:\n{fir_text}\n\nJSON:"
    output = pipe(prompt, max_new_tokens=800, do_sample=False, temperature=0.0)
    raw = output[0]["generated_text"].split("JSON:")[-1].strip()
    try:
        return json.loads(raw)
    except:
        cleaned = raw.strip("```json").strip("```")
        return json.loads(cleaned)

# ---------- Streamlit UI ----------
st.title("🚔 FIR Extractor (HuggingFace LLM)")

text_input = st.text_area("Paste FIR text", height=400)

if st.button("Extract"):
    if text_input.strip():
        result = extract_with_llm(text_input)
        st.json(result)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="fir_output.json",
            mime="application/json"
        )
