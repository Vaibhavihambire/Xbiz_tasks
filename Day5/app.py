# app.py
from io import BytesIO
import os
import uuid
import base64
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import fitz
import numpy as np
from PIL import Image, ImageSequence

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ocr_engines.tesseract_ocr import run_tesseract
from ocr_engines.easyocr_ocr import run_easyocr
from ocr_engines.paddleocr_ocr import run_paddleocr
from ocr_engines.ocr_preprocess import ocr_preprocess
from ocr_engines.detect import detect_document_types
from ocr_engines.core import start_transaction, close_transaction

# ---------------- App setup ----------------

app = Flask(__name__)
CORS(app, resources={r"/ocr": {"origins": "*"}})

# OCR engines
ENGINE_MAP = {
    0: ("tesseract", "binary", run_tesseract),
    1: ("easyocr", "gray", run_easyocr),
    2: ("paddleocr", "gray", run_paddleocr),
}

# ---------------- Helpers ----------------

def pil_to_bgr_numpy(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def iterate_pages_from_path(path):
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            mat = fitz.Matrix(2.0, 2.0)
            for i in range(len(doc)):
                pix = doc[i].get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                yield i, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), None
        finally:
            doc.close()
        return

    if ext in (".tif", ".tiff"):
        pil_img = Image.open(path)
        try:
            for i, page in enumerate(ImageSequence.Iterator(pil_img)):
                yield i, pil_to_bgr_numpy(page), None
        finally:
            pil_img.close()
        return

    img = cv2.imread(path)
    if img is not None:
        yield 0, img, None
        return

    pil_img = Image.open(path)
    try:
        yield 0, pil_to_bgr_numpy(pil_img), None
    finally:
        pil_img.close()


# ---------------- UI ----------------

@app.route("/")
def serve_ui():
    return send_from_directory("static", "index.html")


# ---------------- OCR API ----------------

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json(silent=True)
    uploaded_file = None
    file_path = None

    typeofocr = int((data or {}).get("typeofocr", request.form.get("typeofocr", 0)))

    if typeofocr not in ENGINE_MAP:
        return jsonify({"error": "Invalid typeofocr"}), 400

    engine_name, input_type, engine_func = ENGINE_MAP[typeofocr]

    # Input handling
    if data and data.get("file_b64"):
        raw = base64.b64decode(data["file_b64"].split(",", 1)[-1])
        txn_name = data.get("transaction_name", "txn")
    else:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        uploaded_file = request.files["file"]
        txn_name = request.form.get("transaction_name") or uploaded_file.filename

    txn_id, txn_base = start_transaction(txn_name, data)
    txn_req_dir = Path(txn_base) / "Request"
    txn_res_dir = Path(txn_base) / "Response"
    txn_out_dir = Path(txn_base) / "Output"
    txn_req_dir.mkdir(parents=True, exist_ok=True)
    txn_res_dir.mkdir(parents=True, exist_ok=True)
    txn_out_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    if data and data.get("file_b64"):
        file_path = str(txn_req_dir / "input.png")
        with open(file_path, "wb") as f:
            f.write(raw)
    else:
        safe_name = secure_filename(uploaded_file.filename)
        file_path = str(txn_req_dir / f"{uuid.uuid4().hex}_{safe_name}")
        uploaded_file.save(file_path)

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    page_results = []
    page_errors = []
    overall_types = set()


    for page_index, bgr_img, _ in iterate_pages_from_path(file_path):
        page_no = page_index + 1

        try:
            if engine_name == "paddleocr":
                gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                binary = None
            else:
                gray, binary = ocr_preprocess(bgr_img)

            Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)).save(
                txn_res_dir / f"page_{page_no}_original.png"
            )
            cv2.imwrite(str(txn_res_dir / f"page_{page_no}_gray.png"), gray)
            if binary is not None:
                cv2.imwrite(str(txn_res_dir / f"page_{page_no}_binary.png"), binary)

            # OCR
            if engine_name == "tesseract":
                page_text = run_tesseract(binary, lang="eng")
            elif engine_name == "easyocr":
                page_text = run_easyocr(gray)
            else: 
                page_text = run_paddleocr(gray)

        except Exception as e:
            traceback.print_exc()
            page_errors.append({"page": page_no, "error": str(e)})
            page_text = ""

        detected = detect_document_types(page_text) or ["unknown"]
        overall_types.update(detected)

        page_results.append({
            "page": page_no,
            "document_types": detected,
            "text": page_text
        })

    combined_text = "\n\n".join(
        f"--- PAGE {p['page']} ---\n{p['text']}" for p in page_results
    )

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta = close_transaction(
        transaction_id=txn_id,
        base_path=txn_base,
        extracted_text=combined_text,
        engine_name=engine_name,
        input_path=file_path,
        started_at=started_at,
        finished_at=finished_at,
        extra_info={
            "pages_processed": len(page_results),
            "page_errors": page_errors,
            "document_types": list(overall_types),
        }
    )

    
    return jsonify({
        "input_path": file_path,
        "engine": engine_name,
        "txt_file": meta["output_file"],
        "pages": page_results,
        "overall_document_types": list(overall_types),
    }), 200


# ---------------- Run ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=False, use_reloader=False)
