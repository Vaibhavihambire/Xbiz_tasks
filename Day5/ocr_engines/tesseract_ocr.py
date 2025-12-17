# tesseract_ocr.py
import pytesseract
import cv2
import numpy as np

def run_tesseract(binary_img: np.ndarray, lang="eng") -> str:
    rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

    config = "--oem 3 --psm 6 -c dpi=300"

    return pytesseract.image_to_string(
        rgb,
        lang=lang,
        config=config
    )
