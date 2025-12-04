# app.py
# This file defines a simple Flask API with one endpoint /ocr
# It accepts an image file, runs three OCR engines (Tesseract, EasyOCR, PaddleOCR),
# saves results as .txt files, and returns a JSON response.

# Import os to handle file paths and directories
import os

# Import Flask, request, jsonify to build the API
from flask import Flask, request, jsonify
# Import secure_filename to safely handle uploaded filenames
from werkzeug.utils import secure_filename

# Import OpenCV to read image files
import cv2

# Import OCR functions from ocr_engines package
from ocr_engines.tesseract_ocr import run_tesseract
from ocr_engines.easyocr_ocr import run_easyocr
from ocr_engines.paddleocr_ocr import run_paddleocr
from ocr_engines.core import load_image_bgr, clean_ocr_text, remove_duplicate_lines

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Create a Flask application instance
app = Flask(__name__)

# Define output directory for saving .txt files
OUTPUT_DIR = "outputs"

# Ensure that the output directory exists (create if it does not exist)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define a route for OCR at path "/ocr" with POST method
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # Check if any file part named "file" is present in the request
    if "file" not in request.files:
        # If not present, return error response with status code 400
        return jsonify({"error": "No file uploaded"}), 400

    # Get the uploaded file from request
    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        # If empty, return error response with status code 400
        return jsonify({"error": "Empty filename"}), 400

    # Use secure_filename to sanitize the uploaded filename
    filename = secure_filename(file.filename)

    # Build a temporary path to save the uploaded image file
    temp_path = os.path.join(OUTPUT_DIR, filename)

    # Save the uploaded file to the temporary path
    file.save(temp_path)

    # Default: treat every file as scanned document for better OCR accuracy
    is_scanned = True
    doc_type = "scanned"

    # Load image from disk using helper function (returns BGR image)
    img = load_image_bgr(temp_path)

    # If image could not be read, return error
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Run Tesseract OCR on the image
    tesseract_raw = run_tesseract(img, lang="eng", is_scanned=is_scanned)
    # First clean unwanted characters from Tesseract text
    tesseract_clean = clean_ocr_text(tesseract_raw)
    # Then remove duplicate lines from Tesseract text
    tesseract_text = remove_duplicate_lines(tesseract_clean)

    # Run EasyOCR on the image
    easyocr_raw = run_easyocr(img, is_scanned=is_scanned)
    # First clean unwanted characters from EasyOCR text
    easyocr_clean = clean_ocr_text(easyocr_raw)
    # Then remove duplicate lines from EasyOCR text
    easyocr_text = remove_duplicate_lines(easyocr_clean)

    # Run PaddleOCR on the image
    # paddle_text = run_paddleocr(img, is_scanned=is_scanned)
    paddle_raw = run_paddleocr(img, lang="en")
    paddle_text = clean_ocr_text(paddle_raw)

    # Split the filename into base name and extension
    base, _ = os.path.splitext(filename)

    # Build file paths for three .txt output files
    tesseract_path = os.path.join(OUTPUT_DIR, f"{base}_tesseract.txt")
    easyocr_path   = os.path.join(OUTPUT_DIR, f"{base}_easyocr.txt")
    paddle_path    = os.path.join(OUTPUT_DIR, f"{base}_paddleocr.txt")

    # Save Tesseract result into its .txt file (UTF-8 encoding)
    with open(tesseract_path, "w", encoding="utf-8") as f:
        f.write(tesseract_text or "")

    # Save EasyOCR result into its .txt file
    with open(easyocr_path, "w", encoding="utf-8") as f:
        f.write(easyocr_text or "")

    # Save PaddleOCR result into its .txt file
    with open(paddle_path, "w", encoding="utf-8") as f:
        f.write(paddle_text or "")

    # Build JSON response object
    response = {
        "filename": filename,
        "doc_type": doc_type,
        "results": {
            "tesseract": {
                "text": tesseract_text,
                "txt_file": tesseract_path
            },
            "easyocr": {
                "text": easyocr_text,
                "txt_file": easyocr_path
            },
            "paddleocr": {
                "text": paddle_text,
                "txt_file": paddle_path
            }
        }
    }

    # Return JSON response with status 200
    return jsonify(response)


# Main block to run the Flask app if this file is executed directly
if __name__ == "__main__":
    # Run Flask app in debug mode on default port 5000
    app.run(debug=True)
