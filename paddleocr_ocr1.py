import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Tuple

# -------------------------------------------------
# 1. Initialize PaddleOCR (CPU, English only)
# -------------------------------------------------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=16
)

# -------------------------------------------------
# 2. Normalize PaddleOCR (PaddleX-style) result
# -------------------------------------------------
def normalize_paddlex_result(ocr_result) -> List[Dict]:
    """
    Converts PaddleOCR / PaddleX OCRResult into
    flat box objects with geometry
    """

    texts = ocr_result["rec_texts"]
    scores = ocr_result["rec_scores"]
    polys = ocr_result["rec_polys"]

    boxes = []

    for text, conf, poly in zip(texts, scores, polys):
        if not text or not text.strip():
            continue

        poly = np.array(poly)

        xs = poly[:, 0]
        ys = poly[:, 1]

        boxes.append({
            "text": text.strip(),
            "conf": float(conf),
            "x_min": float(xs.min()),
            "y_min": float(ys.min()),
            "x_max": float(xs.max()),
            "y_max": float(ys.max()),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "height": float(ys.max() - ys.min())
        })

    return boxes

# -------------------------------------------------
# 3. Group OCR boxes into visual text lines
# -------------------------------------------------
def group_boxes_into_lines(
    boxes: List[Dict],
    y_threshold_ratio: float = 0.6
) -> List[Dict]:
    """
    Groups OCR boxes into visual text lines
    """

    boxes = sorted(boxes, key=lambda b: b["cy"])

    lines = []
    current_line = []
    current_y = None

    for box in boxes:
        if current_y is None:
            current_line = [box]
            current_y = box["cy"]
            continue

        y_threshold = box["height"] * y_threshold_ratio

        if abs(box["cy"] - current_y) <= y_threshold:
            current_line.append(box)
            current_y = (current_y + box["cy"]) / 2
        else:
            current_line = sorted(current_line, key=lambda b: b["x_min"])
            lines.append({
                "text": " ".join(b["text"] for b in current_line),
                "boxes": current_line
            })
            current_line = [box]
            current_y = box["cy"]

    if current_line:
        current_line = sorted(current_line, key=lambda b: b["x_min"])
        lines.append({
            "text": " ".join(b["text"] for b in current_line),
            "boxes": current_line
        })

    return lines

# -------------------------------------------------
# 4. Classify reconstructed lines
# -------------------------------------------------
def classify_lines(reconstructed_lines: List[Dict]) -> Tuple[List[str], List[Dict], List[Dict]]:
    """
    Separates:
    - headers
    - key-value lines
    - table lines
    """

    headers = []
    key_value = []
    table_lines = []

    for line in reconstructed_lines:
        boxes = line["boxes"]
        xs = [b["x_min"] for b in boxes]
        x_range = max(xs) - min(xs)

        if len(boxes) == 1:
            headers.append(line["text"])

        elif x_range < 220:
            key_value.append(line)

        else:
            table_lines.append(line)

    return headers, key_value, table_lines

# -------------------------------------------------
# 5. Extract key-value pairs using geometry
# -------------------------------------------------
def extract_key_values(
    kv_lines: List[Dict],
    value_x_threshold: float = 100
) -> Dict[str, str]:
    """
    Extracts Key : Value pairs using x-position
    """

    data = {}

    for line in kv_lines:
        boxes = sorted(line["boxes"], key=lambda b: b["x_min"])

        key_parts = []
        value_parts = []

        for b in boxes:
            if b["x_min"] < value_x_threshold:
                key_parts.append(b["text"])
            else:
                value_parts.append(b["text"])

        key = " ".join(key_parts).replace(":", "").strip()
        value = " ".join(value_parts).strip()

        if key:
            data[key] = value

    return data

# -------------------------------------------------
# 6. Extract table rows (raw, no alignment yet)
# -------------------------------------------------
def extract_table_rows(table_lines: List[Dict]) -> List[List[str]]:
    """
    Extracts table rows as ordered text lists
    """

    rows = []

    for line in table_lines:
        row = [b["text"] for b in sorted(line["boxes"], key=lambda b: b["x_min"])]
        rows.append(row)

    return rows

# -------------------------------------------------
# 7. MAIN PaddleOCR pipeline (single image)
# -------------------------------------------------
def run_paddleocr(bgr_img: np.ndarray) -> Dict:
    """
    Complete OCR → structure pipeline
    """

    if hasattr(ocr, "predict"):
        ocr_raw = ocr.predict(bgr_img)
    else:
        ocr_raw = ocr.ocr(bgr_img)

    ocr_result = ocr_raw[0]

    boxes = normalize_paddlex_result(ocr_result)
    reconstructed_lines = group_boxes_into_lines(boxes)

    headers, kv_lines, table_lines = classify_lines(reconstructed_lines)

    kv_data = extract_key_values(kv_lines)
    table_rows = extract_table_rows(table_lines)

    return {
        "headers": headers,
        "key_values": kv_data,
        "tables": table_rows,
        "raw_lines": [l["text"] for l in reconstructed_lines]
    }
