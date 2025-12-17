
import re
import cv2
import numpy as np
import json
import uuid
import os

from pathlib import Path
from datetime import datetime

def load_image_bgr(path: str):
    img = cv2.imread(path)
    return img


def smart_resize_for_ocr(img, min_h=700, max_h=1200):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if min_h <= h <= max_h:
        return img
    if h < min_h:
        target_h = min_h
    else:
        target_h = max_h

    scale = target_h / float(h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized


def clean_ocr_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars = []

    for ch in text:
        if ch == "\n":
            cleaned_chars.append(ch)
        elif ch.isprintable():
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)

    # allow:
    # - digits 0-9
    # - letters a-z and A-Z
    # - basic punctuation: colon (:), slash (/), hyphen (-)
    # - spaces and newlines
    cleaned = re.sub(r"[^0-9A-Za-z:/\-\n ]", " ", cleaned)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    lines = cleaned.split("\n")

    final_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(ch.isalnum() for ch in stripped):
            final_lines.append(stripped)

    result = "\n".join(final_lines)

    return result

# remove duplicate lines
def remove_duplicate_lines(text: str) -> str:
    if text is None:
        return ""
    lines = text.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        normalized = " ".join(stripped.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(stripped)
    result = "\n".join(unique_lines)
    return result

def _safe_name(s: str) -> str:
    if s is None:
        return ""
    return "".join(c for c in s if c.isalnum() or c in (" ", "_", "-")).rstrip().replace(" ", "_")


def start_transaction(transaction_name: str, request_obj=None, base_root: str = "transactions"):
    txn_id = uuid.uuid4().hex[:6]
    safe = _safe_name(transaction_name) or "txn"
    folder_name = f"{safe}_{txn_id}"
    base_path = Path(base_root) / folder_name

    req_dir = base_path / "Request"
    res_dir = base_path / "Response"
    out_dir = base_path / "Output"
    req_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    request_file = req_dir / "request.txt"
    with open(request_file, "w", encoding="utf-8") as f:
        f.write(f"transaction_name: {transaction_name}\n")
        f.write(f"transaction_id: {txn_id}\n")
        f.write(f"request_time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
        f.write("\n--- request payload ---\n")
        if request_obj is None:
            f.write("<no payload provided>\n")
        else:
            try:
                f.write(json.dumps(request_obj, indent=2, ensure_ascii=False))
            except Exception:
                f.write(repr(request_obj))

    return txn_id, base_path


def close_transaction(
    transaction_id: str,
    base_path: str or Path, # type: ignore
    extracted_text: str,
    engine_name: str,
    input_path: str,
    file_index: int = None,
    started_at: str = None,
    finished_at: str = None,
    extra_info: dict = None
):
  
    base = Path(base_path)
    res_dir = base / "Response"
    out_dir = base / "Output"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if finished_at is None:
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if started_at is None:
        started_at = "<unknown>"

    input_path = str(input_path or "")
    input_name = Path(input_path).name
    input_ext = Path(input_path).suffix.lower().lstrip(".")

    rname = "response"
    if file_index is not None:
        rname += f"_p{file_index}"
    rfile = res_dir / f"{rname}.txt"

    with open(rfile, "w", encoding="utf-8") as f:
        f.write(f"transaction_id: {transaction_id}\n")
        f.write(f"engine: {engine_name}\n")
        f.write(f"started_at: {started_at}\n")
        f.write(f"finished_at: {finished_at}\n")
        f.write(f"input_path: {input_path}\n")
        f.write(f"input_name: {input_name}\n")
        f.write(f"input_extension: {input_ext}\n")
        if file_index is not None:
            f.write(f"page_index: {file_index}\n")
        f.write("\n--- extracted text (preview) ---\n")
        preview = (extracted_text or "").strip()[:500]
        f.write(preview + ("\n" if preview else ""))

        if extra_info:
            f.write("\n--- extra info ---\n")
            try:
                f.write(json.dumps(extra_info, indent=2, ensure_ascii=False))
            except Exception:
                f.write(repr(extra_info))

    oname = "output"
    if file_index is not None:
        oname += f"_p{file_index}"
    ofile = out_dir / f"{oname}.txt"
    with open(ofile, "w", encoding="utf-8") as f:
        f.write((extracted_text or "").strip() + "\n")

    return {"response_file": str(rfile), "output_file": str(ofile)}