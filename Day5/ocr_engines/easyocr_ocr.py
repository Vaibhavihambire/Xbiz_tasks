# easyocr_ocr.py
import easyocr
import numpy as np

_reader = easyocr.Reader(['en'])

def run_easyocr(gray_img: np.ndarray) -> str:
    result = _reader.readtext(gray_img, detail=0)
    return "\n".join(result)
