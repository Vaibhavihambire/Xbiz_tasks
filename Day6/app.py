# app.py (final patched: short filenames, safe txn names, reuse txn, paddle tmp verification)
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
from ocr_engines.core import (
    clean_ocr_text,
    remove_duplicate_lines,
    start_transaction,
    close_transaction,
    smart_resize_for_ocr,
)
from ocr_engines.detect import detect_document_type, detect_all_document_types

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


def _safe_txn_name(s: str, max_chars: int = 20) -> str:
    s2 = secure_filename(s)[:max_chars]
    return s2 if s2 else uuid.uuid4().hex[:8]


def _short_unique_filename(original_name: str) -> str:
    ext = Path(original_name).suffix or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"


def process_file(
    file_path: str,
    original_name: str,
    typeofcr_int: int,
    batch_name: str = None,
    existing_txn_id: str = None,
    existing_txn_base: str = None,
):
    """
    Process a single file. Reuse existing transaction if provided.
    Returns a result dict.
    """
    base_stem = Path(original_name).stem if original_name else Path(file_path).stem
    txn_display_name = f"{batch_name}_{base_stem}" if batch_name else base_stem
    request_payload = {"filename": original_name, "typeofocr": typeofcr_int}

    # Reuse provided txn if available
    if existing_txn_id and existing_txn_base:
        txn_id = existing_txn_id
        txn_base = existing_txn_base
    else:
        txn_name_safe = _safe_txn_name(txn_display_name)
        txn_id, txn_base = start_transaction(txn_name_safe, request_payload)

    txn_req_dir = Path(txn_base) / "Request"
    txn_res_dir = Path(txn_base) / "Response"
    txn_out_dir = Path(txn_base) / "Output"
    txn_req_dir.mkdir(parents=True, exist_ok=True)
    txn_res_dir.mkdir(parents=True, exist_ok=True)
    txn_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[txn] using: {txn_base}")

    # If file already inside the Request folder, do NOT copy; otherwise copy into Request
    try:
        src = Path(file_path)
        dest = txn_req_dir / src.name
        try:
            if src.resolve() != dest.resolve():
                # copy into request folder (binary copy)
                with open(src, "rb") as rf, open(dest, "wb") as wf:
                    wf.write(rf.read())
        except Exception:
            # fallback: try PIL save (useful if src is not a plain file)
            try:
                img = Image.open(src)
                img.convert("RGB").save(dest)
                img.close()
            except Exception:
                pass
        file_path = str(dest)
    except Exception:
        # keep original if anything goes wrong
        file_path = file_path

    engine_name, engine_func = ENGINE_MAP[typeofcr_int]

    overall_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    page_texts = []
    page_errors = []
    page_count = 0

    try:
        for page_index, bgr_img, _temp in iterate_pages_from_path(file_path):
            page_count += 1
            # save preprocessed image per page
            try:
                pre_img = smart_resize_for_ocr(bgr_img, min_h=700, max_h=1200)
                pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
                pre_save_path = txn_res_dir / f"preprocessed_p{page_index+1}.png"
                Image.fromarray(pre_rgb).save(pre_save_path, format="PNG")
            except Exception:
                traceback.print_exc()

            try:
                if engine_name == "paddleocr":
                    tmp_page_name = f"{uuid.uuid4().hex}_page{page_index+1}.png"
                    tmp_page_path = str(txn_res_dir / tmp_page_name)
                    txn_res_dir.mkdir(parents=True, exist_ok=True)

                    # save and verify temporary file
                    try:
                        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                        Image.fromarray(rgb).save(tmp_page_path)
                        print(f"[tmp] saved: {tmp_page_path}")
                    except Exception as e_save:
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to save temporary page image: {e_save}")

                    if not Path(tmp_page_path).exists():
                        raise RuntimeError(f"Temporary page file not found after save: {tmp_page_path}")

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
                page_texts.append((page_index + 1, page_clean))
            except Exception as e_page:
                traceback.print_exc()
                page_errors.append({"page": page_index + 1, "error": str(e_page)})
                page_texts.append((page_index + 1, f"[error on page {page_index+1}: {e_page}]"))
    except Exception as e_iter:
        traceback.print_exc()
        return {"input_file": original_name, "error": f"Failed to iterate document pages: {e_iter}"}

    # combine pages into final text
    combined_pages = []
    for pnum, ptxt in page_texts:
        combined_pages.append(f"--- PAGE {pnum} ---\n{ptxt}\n")
    combined_text = "\n".join(combined_pages)

    try:
        final_text = remove_duplicate_lines(combined_text)
    except Exception:
        final_text = combined_text

    try:
        cleaned = clean_ocr_text(final_text)
        final_text = remove_duplicate_lines(cleaned)
    except Exception as e:
        traceback.print_exc()
        final_text = final_text or ""
        final_text = f"[cleaning_failed] {str(e)}\n\n{final_text}"

    # detect document types (per-page + whole doc fallback)
    detected_types = set()
    for pnum, ptxt in page_texts:
        types_here = detect_all_document_types(ptxt)
        for t in types_here:
            detected_types.add(t)
    if not detected_types:
        whole_types = detect_all_document_types(final_text)
        for t in whole_types:
            detected_types.add(t)
    doc_type = ", ".join(sorted(detected_types)) if detected_types else detect_document_type(final_text)

    # write aggregated overall Response + Output into transaction folder
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
                "document_type": doc_type,
            },
        )

        # Defensive cleanup: remove old per-page text files if any
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

        out_txt_path = overall_meta.get("output_file") or str(
            txn_out_dir / f"{Path(file_path).stem}_{engine_name}_{uuid.uuid4().hex[:8]}.txt"
        )
    except Exception as e:
        traceback.print_exc()
        return {"input_file": original_name, "error": f"Failed to write output file: {str(e)}"}

    result = {
        "input_file": original_name,
        "transaction_id": txn_id,
        "engine": engine_name,
        "pages_processed": page_count,
        "page_errors": page_errors,
        "txt_file": out_txt_path,
        "document_type": doc_type,
        "request_folder": str(txn_req_dir),
        "response_folder": str(txn_res_dir),
        "output_folder": str(txn_out_dir),
    }
    return result


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    - JSON mode: {"filename":"C:/path/to/file.png", "typeofocr": 0, "transaction_name": "vvvv"}
    - Or multipart/form-data with 'file' and 'typeofocr' and optional 'transaction_name'
    """
    data = request.get_json(silent=True)
    file_path = None

    typeofocr = 0
    uploaded_files = []
    if data:
        file_b64 = data.get("file_b64")
        file_path = data.get("filename")
        typeofocr = int(data.get("typeofocr", 0))
    else:
        if "file" not in request.files:
            return jsonify({"error": "No file provided (use JSON 'filename' or multipart 'file')"}), 400
        uploaded_files = request.files.getlist("file")
        typeofocr = int(request.form.get("typeofocr", 0))

    try:
        typeofcr_int = int(typeofocr)
    except Exception:
        return jsonify({"error": "typeofocr must be integer 0/1/2"}), 400

    if typeofcr_int not in ENGINE_MAP:
        return jsonify({"error": f"typeofocr must be one of {list(ENGINE_MAP.keys())}"}), 400

    results = []

    # handle base64 single-file input
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
            provided_name = None
            if data and data.get("filename"):
                provided_name = Path(data.get("filename")).name

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

            base_name_for_txn = data.get("transaction_name") or (Path(save_name).stem if save_name else "txn")
            tmp_txn_id, tmp_txn_base = start_transaction(_safe_txn_name(base_name_for_txn), data)
            tmp_req_dir = Path(tmp_txn_base) / "Request"
            tmp_req_dir.mkdir(parents=True, exist_ok=True)
            temp_path = str(tmp_req_dir / save_name)
            with open(temp_path, "wb") as f:
                f.write(raw)

            # pass existing txn info so process_file won't create a new txn
            res = process_file(
                temp_path,
                save_name,
                typeofcr_int,
                batch_name=data.get("transaction_name"),
                existing_txn_id=tmp_txn_id,
                existing_txn_base=tmp_txn_base,
            )
            results.append(res)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Failed to write decoded base64 to transaction request file: {e}"}), 500

        return jsonify({"results": results}), 200

    # multipart uploads (multiple files allowed)
    for uploaded in (uploaded_files or []):
        if uploaded.filename == "":
            results.append({"input_file": "", "error": "Empty filename in upload"})
            continue

        safe_name = secure_filename(uploaded.filename)
        short_saved_name = _short_unique_filename(safe_name)  # short file name to avoid long paths

        batch_name = request.form.get("transaction_name")
        file_stem = Path(safe_name).stem
        base_txn = f"{batch_name}_{file_stem}" if batch_name else file_stem
        txn_name_single = _safe_txn_name(base_txn)

        try:
            tx_id, tx_base = start_transaction(txn_name_single, {"filename": safe_name, "typeofocr": typeofcr_int})
            req_dir = Path(tx_base) / "Request"
            req_dir.mkdir(parents=True, exist_ok=True)
            temp_path = str(req_dir / short_saved_name)
            uploaded.save(temp_path)

            # Reuse transaction we created above when calling process_file
            res = process_file(
                temp_path,
                safe_name,
                typeofcr_int,
                batch_name=batch_name,
                existing_txn_id=tx_id,
                existing_txn_base=tx_base,
            )
            results.append(res)
        except Exception as e:
            traceback.print_exc()
            results.append({"input_file": safe_name, "error": str(e)})

    return jsonify({"results": results}), 200


if __name__ == "__main__":
    app.run(debug=True)
