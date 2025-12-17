import cv2
import numpy as np
from pytesseract import image_to_osd
import re
from deskew import determine_skew
from PIL import Image
import math

# ---------------- Orientation ----------------

def get_coarse_rotation_angle(image):
    try:
        osd_data = image_to_osd(image)
        match = re.search(r'Rotate:\s*(\d+)', osd_data)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0

def apply_osd_rotation(image, coarse_angle):
    angle = coarse_angle % 360
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

# ---------------- Preprocessing ----------------

def resize_for_ocr(pil_image, target_width):
    if not target_width or target_width <= 0:
        return pil_image
    w, h = pil_image.size
    if w >= target_width:
        return pil_image
    scale = target_width / w
    return pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def remove_illumination(gray, blur_size=51):
    bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0).astype(np.float32) + 1e-6
    norm = cv2.divide(gray.astype(np.float32), bg, scale=255)
    return np.clip(norm, 0, 255).astype(np.uint8)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(2.0, (8,8))
    return clahe.apply(gray)

def denoise_gray(gray):
    return cv2.medianBlur(gray, 3)

def adaptive_threshold(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 10
    )

def morphological_clean(bin_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

# ---------------- Main ----------------

def ocr_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = resize_for_ocr(pil, 1200)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = remove_illumination(gray)
    # gray = apply_clahe(gray)
    gray = denoise_gray(gray)

    skew_angle = determine_skew(gray)
    img = rotate_image(img, skew_angle)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coarse_angle = get_coarse_rotation_angle(img)
    img = apply_osd_rotation(img, coarse_angle)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = adaptive_threshold(gray)
    binary = morphological_clean(binary)

    return gray, binary

# ---------------- Run ----------------

image_path = r"C:\Users\vaibh\Documents\Xbiz_tasks\Day4\samples\adhar1.jpeg"

gray, binary = ocr_preprocess(image_path)

cv2.imwrite(r"C:\Users\vaibh\Documents\Xbiz_tasks\finalimage.png", gray)

cv2.imshow("Final Preprocessed Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
