from io import BytesIO
import os
import uuid
import base64
import traceback
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np
import cv2
import fitz

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ocr_engines.tesseract_ocr import run_tesseract
from ocr_engines.easyocr_ocr import run_easyocr
from ocr_engines.paddleocr_ocr import run_paddleocr
from ocr_engines.core import clean_ocr_text, remove_duplicate_lines, start_transaction, close_transaction, smart_resize_for_ocr
from datetime import datetime

app = Flask(__name__)

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


def iterate_pages_from_path(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            # matrix scale (1.5-2.0) to control DPI/resolution
            mat = fitz.Matrix(2.0, 2.0)
            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                yield i, bgr, None
        finally:
            doc.close()
        return

    # TIFF (multi-page)
    if ext in (".tif", ".tiff"):
        pil_img = Image.open(path)
        try:
            for i, page in enumerate(ImageSequence.Iterator(pil_img)):
                bgr = pil_to_bgr_numpy(page)
                yield i, bgr, None
        finally:
            try:
                pil_img.close()
            except Exception:
                pass
        return

    img = cv2.imread(path)
    if img is not None:
        yield 0, img, None
        return

    # PIL fallback (handles webp, uncommon types)
    try:
        pil_img = Image.open(path)
        try:
            bgr = pil_to_bgr_numpy(pil_img)
            yield 0, bgr, None
        finally:
            try:
                pil_img.close()
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Unsupported image format or cannot read file: {e}")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():

    data = request.get_json(silent=True)
    uploaded_file = None
    file_path = None

    typeofocr = 0
    if data:
        file_b64 = data.get("file_b64")
        file_path = data.get("filename")
        typeofocr = int(data.get("typeofocr", 0))
    else:
        if "file" not in request.files:
            return jsonify({"error": "No file provided (use JSON 'filename' or multipart 'file')"}), 400
        uploaded_file = request.files["file"]
        typeofocr = int(request.form.get("typeofocr", 0))

    # Validate typeofocr
    try:
        typeofcr_int = int(typeofocr)
    except Exception:
        return jsonify({"error": "typeofocr must be integer 0/1/2"}), 400

    if typeofcr_int not in ENGINE_MAP:
        return jsonify({"error": f"typeofocr must be one of {list(ENGINE_MAP.keys())}"}), 400

    # transaction
    if data:
        request_transaction_name = data.get("transaction_name") or (
            uploaded_file.filename if uploaded_file and getattr(uploaded_file, "filename", None) else (Path(file_path).stem if file_path else "txn")
        )
    else:
        request_transaction_name = request.form.get("transaction_name") or (
            uploaded_file.filename if uploaded_file and getattr(uploaded_file, "filename", None) else (Path(file_path).stem if file_path else "txn")
        )

    request_payload = data if data else {"filename": str(file_path) if file_path else None, "typeofocr": typeofocr}
    txn_id, txn_base = start_transaction(request_transaction_name, request_payload)

    txn_req_dir = Path(txn_base) / "Request"
    txn_res_dir = Path(txn_base) / "Response"
    txn_out_dir = Path(txn_base) / "Output"
    txn_req_dir.mkdir(parents=True, exist_ok=True)
    txn_res_dir.mkdir(parents=True, exist_ok=True)
    txn_out_dir.mkdir(parents=True, exist_ok=True)

    if data and data.get("file_b64"):
        file_b64 = data.get("file_b64")
        try:
            if "," in file_b64 and file_b64.startswith("data:"):
                file_b64 = file_b64.split(",", 1)[1]
            raw = base64.b64decode(file_b64)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Invalid base64 data: {e}"}), 400

        try:
            # try to decide extension from provided filename first
            provided_name = None
            if data and data.get("filename"):
                provided_name = Path(data.get("filename")).name

            save_name = None
            if provided_name and Path(provided_name).suffix:
                save_name = provided_name
            else:
                if raw[:4] == b"%PDF":
                    save_name = "requested.pdf"
                else:
                    try:
                        img_try = Image.open(BytesIO(raw))
                        save_name = "requested.png"
                        img_try.close()
                    except Exception:
                        save_name = "requested.bin"

            temp_path = str(txn_req_dir / save_name)
            with open(temp_path, "wb") as f:
                f.write(raw)
            file_path = temp_path
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Failed to write decoded base64 to transaction request file: {e}"}), 500
    
    if uploaded_file:
        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename in upload"}), 400

        safe_name = secure_filename(uploaded_file.filename)
        uid = uuid.uuid4().hex
        unique_filename = f"{uid}_{safe_name}"
        temp_path = str(txn_req_dir / unique_filename)

        try:
            uploaded_file.save(temp_path)
            file_path = temp_path
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Failed to save uploaded file into transaction request folder: {str(e)}"}), 500

    if not file_path:
        return jsonify({"error": "No filename provided"}), 400

    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        base64_info = {"base64_length": len(b64)}
    except Exception as e:
        traceback.print_exc()
        base64_info = {"base64_error": str(e)}

    engine_name, engine_func = ENGINE_MAP[typeofcr_int]

    overall_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    page_texts = []
    page_errors = []
    page_count = 0

    try:
        for page_index, bgr_img, _temp in iterate_pages_from_path(file_path):
            page_count += 1
            page_id = f"p{page_index+1}"
            try:
                pre_img = smart_resize_for_ocr(bgr_img, min_h=700, max_h=1200)
                pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
                pre_save_path = txn_res_dir / f"preprocessed_p{page_index+1}.png"
                Image.fromarray(pre_rgb).save(pre_save_path, format="PNG")
            except Exception:
                traceback.print_exc()

            try:
                if engine_name == "paddleocr":
                    tmp_page_name = f"{uuid.uuid4().hex}_{Path(file_path).stem}_page{page_index+1}.png"
                    tmp_page_path = str(txn_res_dir / tmp_page_name)
                    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    Image.fromarray(rgb).save(tmp_page_path)
                    try:
                        page_raw = engine_func(tmp_page_path, lang="en")
                    finally:
                        try:
                            Path(tmp_page_path).unlink()
                        except Exception:
                            pass
                elif engine_name == "tesseract":
                    page_raw = engine_func(bgr_img, lang="eng", is_scanned=True)
                elif engine_name == "easyocr":
                    page_raw = engine_func(bgr_img, is_scanned=True)
                else:
                    page_raw = ""
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

    # Remove duplicate lines
    try:
        final_text = remove_duplicate_lines(combined_text)
    except Exception:
        final_text = combined_text

    # cleaning text
    try:
        cleaned = clean_ocr_text(final_text)
        final_text = remove_duplicate_lines(cleaned)
    except Exception as e:
        traceback.print_exc()
        final_text = final_text or ""
        final_text = f"[cleaning_failed] {str(e)}\n\n{final_text}"


    try:
        overall_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall_meta = close_transaction(
            transaction_id=txn_id,
            base_path=txn_base,
            extracted_text=final_text,
            engine_name=engine_name,
            input_path=file_path,
            file_index=None,
            started_at=overall_started_at,
            finished_at=overall_finished_at,
            extra_info={
                "pages_processed": page_count,
                "page_errors": page_errors,
                "base64_info": base64_info,
            },
        )

        try:
            for p in (txn_res_dir.glob("response_p*.txt")):
                try:
                    p.unlink()
                except Exception:
                    pass
            for p in (txn_out_dir.glob("output_p*.txt")):
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            traceback.print_exc()

        out_txt_path = overall_meta.get("output_file") or str(txn_out_dir / f"{Path(file_path).stem}_{engine_name}_{uuid.uuid4().hex[:8]}.txt")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to write output file: {str(e)}"}), 500

    response = {
        "input_path": file_path,
        "engine": engine_name,
        "text_preview": final_text,
        "txt_file": out_txt_path,
        "document_type": doc_type
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
