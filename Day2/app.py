import os
import uuid
import base64
import traceback
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageSequence
import numpy as np
import cv2
from pdf2image import convert_from_path
import json


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from ocr_engines.tesseract_ocr import run_tesseract
from ocr_engines.easyocr_ocr import run_easyocr
from ocr_engines.paddleocr_ocr import run_paddleocr
from ocr_engines.core import load_image_bgr, clean_ocr_text, remove_duplicate_lines

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

# Output directories
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selection of ocr engine
ENGINE_MAP = {
    0: ("tesseract", run_tesseract),
    1: ("easyocr", run_easyocr),
    2: ("paddleocr", run_paddleocr),
}

def pil_to_bgr_numpy(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def iterate_pages_from_path(path: str):
    ext = Path(path).suffix.lower()

    # PDF path
    if ext == ".pdf":
        pages = convert_from_path(path, dpi=300)
        for i, pil_page in enumerate(pages):
            bgr = pil_to_bgr_numpy(pil_page)
            yield i, bgr, None

    # TIFF
    elif ext in (".tif", ".tiff"):
        pil_img = Image.open(path)
        for i, page in enumerate(ImageSequence.Iterator(pil_img)):
            bgr = pil_to_bgr_numpy(page)
            yield i, bgr, None

    # Single image formats supported by cv2
    else:
        # For webp etc, cv2.imread may handle, or PIL fallback
        img = load_image_bgr(path)
        if img is not None:
            yield 0, img, None
        else:
            try:
                pil_img = Image.open(path)
                bgr = pil_to_bgr_numpy(pil_img)
                yield 0, bgr, None
            except Exception as e:
                raise RuntimeError(f"Unsupported image format or cannot read file: {e}")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
      - JSON mode: {"filename":"C:/path/to/file.png", "typeofocr": 0}
    """
    data = request.get_json(silent=True)
    uploaded_file = None
    file_path = None

    # Default engine is tesseract
    typeofocr = 0
    if data:
        file_b64 = data.get("file_b64") 
        file_path = data.get("filename")
        typeofocr = int(data.get("typeofocr", 0))
        if file_b64:
            try:
                if "," in file_b64 and file_b64.startswith("data:"):
                    file_b64 = file_b64.split(",", 1)[1]

                raw = base64.b64decode(file_b64)
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"Invalid base64 data: {e}"}), 400

            try:
                uid = uuid.uuid4().hex
                tmp_name = f"{uid}_upload.png"   # use png by default
                temp_path = os.path.join(OUTPUT_DIR, tmp_name)
                with open(temp_path, "wb") as f:
                    f.write(raw)
                file_path = temp_path
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"Failed to write decoded base64 to temp file: {e}"}), 500

        # If no base64 but filename provided in JSON -> use that path (dev/test only)
        elif file_path:
            file_path = file_path
    else:
        if "file" not in request.files:
            return jsonify({"error": "No file provided (use JSON 'filename' or multipart 'file')" }), 400
        uploaded_file = request.files["file"]
        typeofocr = int(request.form.get("typeofocr", 0))
    try:
        typeofcr_int = int(typeofocr)
    except Exception:
        return jsonify({"error": "typeofocr must be integer 0/1/2"}), 400

    if typeofcr_int not in ENGINE_MAP:
        return jsonify({"error": f"typeofocr must be one of {list(ENGINE_MAP.keys())}"}), 400

    if uploaded_file:
        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename in upload"}), 400

        safe_name = secure_filename(uploaded_file.filename)
        uid = uuid.uuid4().hex  # unique id upload
        unique_filename = f"{uid}_{safe_name}"
        temp_path = os.path.join(OUTPUT_DIR, unique_filename)

        try:
            uploaded_file.save(temp_path)
            file_path = temp_path
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500

    if not file_path:
        return jsonify({"error": "No filename provided"}), 400

    #base64
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        base64_info = {"base64_length": len(b64)}
        # print(b64) 
    except Exception as e:
        #base64 computation failed but we can still attempt OCR.
        traceback.print_exc()
        base64_info = {"base64_error": str(e)}

    engine_name, engine_func = ENGINE_MAP[typeofcr_int]

    # Iterate pages and OCR each page. Collect per-page outputs and errors.
    page_texts = []
    page_errors = []
    page_count = 0

    try:
        for page_index, bgr_img, _temp in iterate_pages_from_path(file_path):
            page_count += 1
            page_id = f"p{page_index+1}"
            try:
                if engine_name == "paddleocr":
                # pass numpy BGR image directly (run_paddleocr will convert to RGB internally)
                    page_raw = engine_func(bgr_img, lang="en")
                elif engine_name == "tesseract":
                    page_raw = engine_func(bgr_img, lang="eng", is_scanned=True)
                elif engine_name == "easyocr":
                    page_raw = engine_func(bgr_img, is_scanned=True)
                else:
                    page_raw = ""
                # clean per page, keep duplicates removal
                page_clean = clean_ocr_text(page_raw)
                page_texts.append((page_index+1, page_clean))
            except Exception as e_page:
                traceback.print_exc()
                page_errors.append({"page": page_index+1, "error": str(e_page)})
                page_texts.append((page_index+1, f"[error on page {page_index+1}: {e_page}]"))

    except Exception as e_iter:
        traceback.print_exc()
        return jsonify({"error": f"Failed to iterate document pages: {e_iter}"}), 500

    combined_pages = []
    for pnum, ptxt in page_texts:
        combined_pages.append(f"--- PAGE {pnum} ---\n{ptxt}\n")
    combined_text = "\n".join(combined_pages)

    # Remove duplicate lines across the whole document
    try:
        final_text = remove_duplicate_lines(combined_text)
    except Exception:
        final_text = combined_text

    #cleaning text 
    try:
        cleaned = clean_ocr_text(final_text)
        final_text = remove_duplicate_lines(cleaned)
    except Exception as e:
        traceback.print_exc()
        final_text = final_text or ""
        final_text = f"[cleaning_failed] {str(e)}\n\n{final_text}"

    # Save output .txt 
    try:
        base_name = f"{Path(file_path).stem}"
        out_base = f"{base_name}_{engine_name}_{uuid.uuid4().hex[:8]}"
        out_txt_path = os.path.join(OUTPUT_DIR, f"{out_base}.txt")
        with open(out_txt_path, "w", encoding="utf-8") as fout:
            fout.write(final_text or "")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to write output file: {str(e)}"}), 500

    response = {
        "input_path": file_path,
        "engine": engine_name,
        "pages_processed": page_count,
        "page_errors": page_errors,
        "text_preview": final_text,
        "txt_file": out_txt_path,
        # "base64_info": base64_info
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
