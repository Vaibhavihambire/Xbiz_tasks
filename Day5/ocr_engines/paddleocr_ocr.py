# paddleocr_ocr.py
from typing import Any, List
from paddleocr import PaddleOCR
import cv2
import numpy as np

_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=16
)


def extract_texts_from_ocr_result(result: Any) -> List[str]:
    def extract(obj: Any):
        texts = []

        if isinstance(obj, str) and obj.strip():
            texts.append(obj.strip())

        elif isinstance(obj, dict):
            for key in ("text", "rec_text", "transcription", "sentence", "text_line"):
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    texts.append(obj[key].strip())
            for v in obj.values():
                texts.extend(extract(v))

        elif isinstance(obj, (list, tuple)):
            for el in obj:
                texts.extend(extract(el))

        return texts

    all_texts = extract(result)

    filtered = []
    for t in all_texts:
        s = t.lower()
        if s.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if "\\" in s:
            continue
        filtered.append(t)

    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out


def run_paddleocr(gray_img: np.ndarray) -> str:

    if len(gray_img.shape) == 2:
        img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    else:
        img = gray_img

    # Run OCR
    if hasattr(_ocr, "predict"):
        ocr_raw = _ocr.predict(img)
    else:
        ocr_raw = _ocr.ocr(img)

    lines = extract_texts_from_ocr_result(ocr_raw)

    return "\n".join(lines)
