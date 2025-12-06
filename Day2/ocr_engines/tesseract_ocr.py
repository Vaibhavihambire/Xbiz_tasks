# For cleaning extracted text (regular expression)
import re
import cv2
import pytesseract
from .core import smart_resize_for_ocr


def preprocess_for_tesseract(img_bgr, is_scanned=True):
    resized = smart_resize_for_ocr(img_bgr, min_h=900, max_h=1400)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

# run Tesseract OCRs
def run_tesseract(img_bgr, lang="eng", is_scanned=True):
    preprocessed_gray = preprocess_for_tesseract(img_bgr, is_scanned=is_scanned)
    img_rgb = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2RGB)
    # --oem 3 : use LSTM engine
    # --psm 6 : assume a block of text (multi-line)
    # -c dpi=300 : treat image as 300 DPI
    config = "--oem 3 --psm 6 -c dpi=300"
    text = pytesseract.image_to_string(
        img_rgb,   
        lang=lang, 
        config=config
    )

    return text
