import json
import uuid

from pathlib import Path
from datetime import datetime

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
        f.write(f"request_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    base_path: str or Path,
    extracted_text: str,
    engine_name: str,
    input_path: str,
    file_index: int = None,
    started_at: str = None,
    finished_at: str = None,
    extra_info: dict = None,
    time_taken: str = None
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
    # if time_taken is None: 
    #     time_taken = round( (datetime.fromisoformat(finished_at) -
    #          datetime.fromisoformat(started_at)).total_seconds(), 5)
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
        # f.write(f"Duration_of_time_taken: {time_taken}\n")
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