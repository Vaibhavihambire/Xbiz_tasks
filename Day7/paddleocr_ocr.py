from typing import Any, List
import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict

_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=8
)

def extract_texts(obj: Any) -> List[str]:
    texts = []

    if obj is None:
        return texts

    # string
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            texts.append(s)

    # tuple/list
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            texts.extend(extract_texts(v))

    # dict
    elif isinstance(obj, dict):
        for k in ("text", "rec_text", "transcription"):
            if k in obj and isinstance(obj[k], str):
                if obj[k].strip():
                    texts.append(obj[k].strip())
        for v in obj.values():
            texts.extend(extract_texts(v))

    return texts


def run_paddleocr(img_bgr: np.ndarray) -> str:
    if img_bgr is None:
        return ""

    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    h, w = img_bgr.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    # Run OCR
    if hasattr(_ocr, "predict"):
        result = _ocr.predict(img_bgr)
    else:
        result = _ocr.ocr(img_bgr)

    texts = extract_texts(result)

    clean = []
    for t in texts:
        low = t.lower()
        if low.endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue
        if "\\" in t:
            continue
        clean.append(t)

    seen = set()
    final = []
    for t in clean:
        if t not in seen:
            seen.add(t)
            final.append(t)

    return "\n".join(final)
