# tool.py
import streamlit as st
import fitz  # PyMuPDF
import io
import base64
import os
import html
from PIL import Image

# Optional OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Legal Judgment PDF → HTML ", layout="wide")

# ---------- Helpers ----------
def to_data_url(pix):
    img_bytes = pix.tobytes("png")
    b64 = base64.b64encode(img_bytes).decode('ascii')
    return f"data:image/png;base64,{b64}"

def extract_layout_pages(pdf_bytes, render_dpi=150):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = render_dpi / 72.0
    pages = []
    for p in doc:
        mat = fitz.Matrix(scale, scale)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        img_url = to_data_url(pix)
        pw, ph = pix.width, pix.height

        page_dict = p.get_text("dict")
        spans_list = []
        for block in page_dict.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    bbox = span.get('bbox', [0,0,0,0])
                    x0, y0, x1, y1 = bbox
                    x_px = x0 * scale
                    y_px = y0 * scale
                    w_px = max(1, (x1 - x0) * scale)
                    h_px = max(1, (y1 - y0) * scale)
                    text = span.get('text', '')
                    size = span.get('size', 0)
                    font = span.get('font', '')
                    flags = span.get('flags', 0)
                    spans_list.append({
                        'x': x_px, 'y': y_px, 'w': w_px, 'h': h_px,
                        'text': text, 'font': font, 'size': size, 'flags': flags
                    })
        pages.append({'width_px': pw, 'height_px': ph, 'img': img_url, 'spans': spans_list})
    doc.close()
    return pages

def generate_high_fidelity_html(pages, include_image=True, fonts_dict=None):
    pages_html = []
    font_faces = []
    if fonts_dict:
        for fname, b64 in fonts_dict.items():
            safe_name = fname.replace(' ', '_')
            face = (
                "@font-face {"
                f"font-family: '{safe_name}';"
                f"src: url(data:font/ttf;charset=utf-8;base64,{b64}) format('truetype');"
                "font-weight: normal;"
                "font-style: normal;"
                "}"
            )
            font_faces.append(face)
    font_css = "\n".join(font_faces)

    for p in pages:
        w = p['width_px']
        h = p['height_px']
        bg_style = ''
        if include_image:
            bg_style = f"background-image:url('{p['img']}'); background-size: {w}px {h}px; background-repeat:no-repeat;"
        spans_html = []
        for s in p['spans']:
            if not s['text']:
                continue
            content = html.escape(s['text']).replace('\n', '<br/>')
            font_px = max(6, s['h'] * 0.9)
            font_family = s['font'].split('+')[-1].split('-')[0] if s['font'] else 'serif'
            font_family_css = f"font-family: '{font_family}', serif;"

            span_style = (
                f"position:absolute; left:{s['x']:.2f}px; top:{s['y']:.2f}px; "
                f"width:{s['w']:.2f}px; height:{s['h']:.2f}px; "
                f"font-size:{font_px:.2f}px; line-height:1; {font_family_css} "
                "white-space:pre; overflow:hidden;"
            )
            span_html = f"<div class=\"text-span\" style=\"{span_style}\">{content}</div>"
            spans_html.append(span_html)

        page_html = (
            f"<div class='pdf-page' style='position:relative; width:{w}px; height:{h}px; {bg_style}'>\n"
            f"{''.join(spans_html)}\n"
            f"</div>\n"
        )
        pages_html.append(page_html)

    css = (
        "<style>\n"
        f"{font_css}\n"
        "body { background:#ececec; margin:0; font-family: Georgia, 'Times New Roman', serif; }\n"
        ".viewer { display:flex; flex-direction:column; align-items:center; gap:20px; padding:20px; }\n"
        ".pdf-page { box-shadow:0 6px 18px rgba(0,0,0,0.12); background-color:white; }\n"
        ".text-span { color: rgba(0,0,0,0.98); }\n"
        "</style>\n"
    )

    html_full = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "<meta charset='utf-8'/>\n"
        "<title>High-fidelity Judgment Export</title>\n"
        f"{css}\n"
        "</head>\n"
        "<body>\n"
        "<div class='viewer'>\n"
        f"{''.join(pages_html)}\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
    return html_full

# ---------- Streamlit UI ----------
st.title("High-fidelity Legal Judgment PDF → HTML")

uploaded = st.file_uploader("Upload judgment PDF", type=["pdf"])
render_dpi = st.slider("Render DPI (increase for higher fidelity)", min_value=72, max_value=300, value=150, step=10)
include_image = st.checkbox("Include original rendered page images (recommended)", value=True)
use_ocr = st.checkbox("Force OCR (if PDF is scanned)", value=False)

uploaded_fonts = st.file_uploader("Upload .ttf font files (optional, multiple)", type=["ttf"], accept_multiple_files=True)

if uploaded is not None:
    pdf_bytes = uploaded.read()
    st.info(f"Processing {uploaded.name} — {len(pdf_bytes):,} bytes")

    with st.spinner("Extracting pages and layout (PyMuPDF)..."):
        try:
            pages = extract_layout_pages(pdf_bytes, render_dpi=render_dpi)
            total_spans = sum(len(p['spans']) for p in pages)
            if total_spans == 0 and (use_ocr or (not pages)) and OCR_AVAILABLE:
                st.warning("No text spans found — falling back to OCR per page.")
                ocr_pages = []
                for p in pages:
                    header, b64 = p['img'].split(',', 1)
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    spans = []
                    n = len(data['text'])
                    for i in range(n):
                        txt = data['text'][i]
                        if not txt.strip():
                            continue
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        spans.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': txt, 'font': 'OCR', 'size': h})
                    ocr_pages.append({'width_px': p['width_px'], 'height_px': p['height_px'], 'img': p['img'], 'spans': spans})
                pages = ocr_pages
            elif total_spans == 0 and use_ocr and not OCR_AVAILABLE:
                st.error('OCR requested but pytesseract not available in this environment.')
        except Exception as e:
            st.error(f"Error while extracting layout: {e}")
            st.stop()

    fonts_dict = {}
    if uploaded_fonts:
        for f in uploaded_fonts:
            name = os.path.splitext(f.name)[0]
            b64 = base64.b64encode(f.read()).decode('ascii')
            fonts_dict[name] = b64

    with st.spinner("Generating high-fidelity HTML..."):
        html_out = generate_high_fidelity_html(pages, include_image=include_image, fonts_dict=fonts_dict if fonts_dict else None)

    st.subheader("Preview")
    st.components.v1.html(html_out, height=900, scrolling=True)

    st.download_button("Download HTML", data=html_out.encode('utf-8'),
                       file_name=os.path.splitext(uploaded.name)[0] + "_export.html", mime='text/html')

    st.subheader("Raw HTML Output")
    st.code(html_out, language="html")

    st.markdown("---")
    st.markdown("**If you want a closer match:**\n\n- Upload original TTF fonts used by the court (if available).\n- Increase Render DPI to 200-300.\n- If you need absolute pixel perfection for a small set of courts, share sample PDFs and I'll tune CSS and font mappings specifically for those templates.")
else:
    st.info("Upload a PDF to begin. For best results, upload a sample judgment that you're trying to match and optionally upload its fonts.")
