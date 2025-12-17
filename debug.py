import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict


ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=16
)
def normalize_paddlex_result(ocr_result):
    """
    Correct parser for PaddleX OCRResult
    """

    # OCRResult behaves like a dict
    texts = ocr_result["rec_texts"]
    scores = ocr_result["rec_scores"]
    polys = ocr_result["rec_polys"]

    boxes = []

    for text, conf, poly in zip(texts, scores, polys):
        if not text.strip():
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


def group_boxes_into_lines(
    boxes: List[Dict],
    y_threshold_ratio: float = 0.6
):

    boxes = sorted(boxes, key=lambda b: b["cy"])

    lines = []
    current_line = []
    current_y = None

    for box in boxes:
        if not box["text"]:
            continue

        if current_y is None:
            current_line = [box]
            current_y = box["cy"]
            continue

        y_threshold = box["height"] * y_threshold_ratio

        if abs(box["cy"] - current_y) <= y_threshold:
            current_line.append(box)
            current_y = (current_y + box["cy"]) / 2
        else:
            # finalize line
            current_line = sorted(current_line, key=lambda b: b["x_min"])
            lines.append({
                "text": " ".join(b["text"] for b in current_line),
                "boxes": current_line
            })

            current_line = [box]
            current_y = box["cy"]

    # flush last line
    if current_line:
        current_line = sorted(current_line, key=lambda b: b["x_min"])
        lines.append({
            "text": " ".join(b["text"] for b in current_line),
            "boxes": current_line
        })

    return lines



# def debug_print_ocr(ocr_raw):
#     print("\n========== RAW OCR TYPE ==========")
#     print(type(ocr_raw))

#     print("\n========== RAW OCR LENGTH ==========")
#     try:
#         print(len(ocr_raw))
#     except:
#         print("No length")

#     print("\n========== RAW OCR CONTENT ==========")
#     for i, item in enumerate(ocr_raw):
#         print(f"\n--- ITEM {i} ---")
#         print(type(item))
#         print(item)
#         if i == 2:
#             break

if __name__ == "__main__":

    IMAGE_PATH = "C:\\Users\\vaibh\\Documents\\Xbiz_tasks\\image1.png"  # <-- change path

    img = cv2.imread(IMAGE_PATH)
    assert img is not None, "Image not found"

    # Run OCR
    # ocr_raw = ocr.ocr(img, cls=True)
    if hasattr(ocr, "predict"):
        ocr_raw = ocr.predict(img)
    else:
        ocr_raw = ocr.ocr(img)
    # debug_print_ocr(ocr_raw)
    # Normalize

    boxes = normalize_paddlex_result(ocr_raw[0])

    print("\n========== NORMALIZED BOXES ==========")
    for b in boxes[:15]:
        print(f"{b['text']} | y={int(b['y_min'])} | x={int(b['x_min'])}")



    # Group into lines
    reconstructed_lines = group_boxes_into_lines(boxes)
    print("\n========== LINE WITH X-RANGES ==========")
    for line in reconstructed_lines:
        xs = [b["x_min"] for b in line["boxes"]]
        print(
            line["text"]
            # " | x-range:",
            # int(min(xs)), "→", int(max(xs))
        )
    # print("\n========== RECONSTRUCTED LINES ==========\n")
    # for i, line in enumerate(lines, 1):
    #     print(f"{i:02d}: {line}")
