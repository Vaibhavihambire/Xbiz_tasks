# easyocr_ocr.py

# Import OpenCV for image operations
import cv2
# Import easyocr library for OCR
import easyocr

# Import shared resize helper from core.py
from .core import smart_resize_for_ocr


# Create an EasyOCR reader instance once (global) for performance
_easyocr_reader = easyocr.Reader(['en'])


# Define a function to preprocess image for EasyOCR
def preprocess_for_easyocr(img_bgr, is_scanned=True):
    # First, resize image to a good range for OCR
    img = smart_resize_for_ocr(img_bgr)

    # If the document is scanned or low-quality, we improve contrast
    if is_scanned:
        # Convert image from BGR to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split LAB image into L (lightness), A, and B channels
        l, a, b = cv2.split(lab)

        # Create a CLAHE object to enhance contrast on the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Apply CLAHE to the L channel
        cl = clahe.apply(l)

        # Merge enhanced L channel back with A and B channels
        merged = cv2.merge((cl, a, b))

        # Convert LAB image back to BGR color space
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Return the preprocessed color image (no binarization)
    return img


# Define a function to run EasyOCR
def run_easyocr(img_bgr, is_scanned=True):
    # First, preprocess the image for EasyOCR
    img = preprocess_for_easyocr(img_bgr, is_scanned=is_scanned)

    # Use EasyOCR reader to read text from the image
    # detail=0 returns only text strings (no boxes or probabilities)
    result_text_list = _easyocr_reader.readtext(img, detail=0)

    # Join all lines of text into a single string separated by newlines
    text = "\n".join(result_text_list)

    # Return the extracted text as a string
    return text
