# app.py
import uuid
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import fitz
import numpy as np
from PIL import Image, ImageSequence
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from paddleocr_ocr import run_paddleocr
from core import start_transaction, close_transaction

app = Flask(__name__)
CORS(app)

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def iterate_pages(path):
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        mat = fitz.Matrix(2.0, 2.0)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, pil_to_bgr(img)
        doc.close()
        return

    if ext in (".tif", ".tiff"):
        pil = Image.open(path)
        for i, p in enumerate(ImageSequence.Iterator(pil)):
            yield i + 1, pil_to_bgr(p)
        pil.close()
        return

    img = cv2.imread(path)
    if img is not None:
        yield 1, img


@app.route("/ocr", methods=["POST"])
def ocr_api():
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    uploaded = request.files["file"]
    safe = secure_filename(uploaded.filename)

    txn_id, txn_base = start_transaction(
        Path(safe).stem,
        {"engine": "paddleocr"}
    )

    req_dir = Path(txn_base) / "Request"
    res_dir = Path(txn_base) / "Response"
    out_dir = Path(txn_base) / "Output"

    for d in (req_dir, res_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    file_path = req_dir / f"{uuid.uuid4().hex}_{safe}"
    uploaded.save(file_path)

    pages = []
    texts = []

    started_at = datetime.now().isoformat()

    try:
        for page_no, img in iterate_pages(file_path):
            cv2.imwrite(str(res_dir / f"page_{page_no}.png"), img)

            text = run_paddleocr(img)
            texts.append(f"--- PAGE {page_no} ---\n{text}")

            pages.append({
                "page": page_no,
                "text": text
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    final_text = "\n\n".join(texts)
    finished_at = datetime.now().isoformat()

    meta = close_transaction(
        transaction_id=txn_id,
        base_path=txn_base,
        extracted_text=final_text,
        engine_name="paddleocr",
        input_path=str(file_path),
        started_at=started_at,
        finished_at=finished_at
    )

    return jsonify({
        "engine": "paddleocr",
        "pages": pages,
        "txt_file": meta["output_file"]
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=False)

