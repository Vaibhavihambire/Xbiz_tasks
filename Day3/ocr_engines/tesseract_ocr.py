# tesseract_ocr.py
# For cleaning extracted text (regular expression)
import re
# Import OpenCV for image operations
import cv2
# Import pytesseract for OCR
import pytesseract

# Import shared resize helper from core.py
from .core import smart_resize_for_ocr


# Define a function to preprocess an image for Tesseract OCR
def preprocess_for_tesseract(img_bgr, is_scanned=True):
    # Step 1: Resize image into a good height range (to simulate 300 DPI)
    # We do NOT want tiny text, so we ensure height is at least ~900 pixels.
    resized = smart_resize_for_ocr(img_bgr, min_h=900, max_h=1400)

    # Step 2: Convert color image (BGR) to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply a small Gaussian blur to reduce tiny noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

# Define a function to run Tesseract OCR and return plain text
def run_tesseract(img_bgr, lang="eng", is_scanned=True):
    # Preprocess the image using the function above
    preprocessed_gray = preprocess_for_tesseract(img_bgr, is_scanned=is_scanned)

    # Convert grayscale image to RGB because Tesseract expects RGB order
    img_rgb = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2RGB)

    # Build Tesseract configuration string
    # --oem 3 : use LSTM engine
    # --psm 6 : assume a block of text (multi-line)
    # -c dpi=300 : hint Tesseract to treat image as 300 DPI
    config = "--oem 3 --psm 6 -c dpi=300"

    # Run Tesseract OCR with the given image, language, and config
    text = pytesseract.image_to_string(
        img_rgb,   # preprocessed RGB image
        lang=lang, # language, for now we use only 'eng'
        config=config
    )

    

    # Return recognized clean text
    return text
