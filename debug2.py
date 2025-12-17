import cv2
from paddleocr_ocr1 import (
    normalize_paddlex_result,
    group_boxes_into_lines,
    classify_lines,
    extract_key_values,
    extract_table_rows,
    run_paddleocr,
    ocr
)

IMAGE_PATH = "image.png"   # <-- your test image

img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found"

# ---------- RAW OCR ----------
if hasattr(ocr, "predict"):
    raw = ocr.predict(img)
else:
    raw = ocr.ocr(img)

print("\n===== RAW OCR TYPE =====")
print(type(raw), "length:", len(raw))

# ---------- NORMALIZATION ----------
boxes = normalize_paddlex_result(raw[0])

print("\n===== NORMALIZED BOXES =====")
for b in boxes[:15]:
    print(f"{b['text']} | x={int(b['x_min'])} y={int(b['y_min'])}")

# ---------- LINE RECONSTRUCTION ----------
lines = group_boxes_into_lines(boxes)

print("\n===== RECONSTRUCTED LINES =====")
for i, l in enumerate(lines, 1):
    print(f"{i:02d}: {l['text']}")

# ---------- CLASSIFICATION ----------
headers, kv_lines, table_lines = classify_lines(lines)

print("\n===== HEADERS =====")
for h in headers:
    print(h)

print("\n===== KEY-VALUE LINES =====")
for l in kv_lines:
    print(l["text"])

print("\n===== TABLE LINES =====")
for l in table_lines[:5]:
    print(l["text"])

# ---------- KEY-VALUE EXTRACTION ----------
kv = extract_key_values(kv_lines)

print("\n===== KEY-VALUE PAIRS =====")
for k, v in kv.items():
    print(f"{k} : {v}")

# ---------- TABLE EXTRACTION ----------
rows = extract_table_rows(table_lines)

print("\n===== TABLE ROWS =====")
for r in rows[:5]:
    print(r)

# ---------- FULL PIPELINE ----------
result = run_paddleocr(img)

print("\n===== FINAL PIPELINE OUTPUT =====")
print(result.keys())
