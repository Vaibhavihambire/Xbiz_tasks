import cv2
import easyocr
from .core import smart_resize_for_ocr

_easyocr_reader = easyocr.Reader(['en'])


# preprocess image for EasyOCR
def preprocess_for_easyocr(img_bgr, is_scanned=True):
    img = smart_resize_for_ocr(img_bgr)
    if is_scanned:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))

        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return img


#run EasyOCR
def run_easyocr(img_bgr, is_scanned=True):
    img = preprocess_for_easyocr(img_bgr, is_scanned=is_scanned)

    result_text_list = _easyocr_reader.readtext(img, detail=0)
    text = "\n".join(result_text_list)

    return text
