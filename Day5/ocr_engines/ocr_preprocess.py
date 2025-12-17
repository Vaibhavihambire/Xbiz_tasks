# ocr_preprocess.py
import cv2
import numpy as np
from pytesseract import image_to_osd
import re
from deskew import determine_skew
from PIL import Image

# ---------------- Orientation ----------------

def get_coarse_rotation_angle(image):
    try:
        osd_data = image_to_osd(image)
        match = re.search(r'Rotate:\s*(\d+)', osd_data)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


def apply_osd_rotation(image, coarse_angle):
    if image is None:
        return image
    angle = coarse_angle % 360
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # For arbitrary angle
        return rotate_image(image, -angle)


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

def resize_for_ocr(img_bgr, target_width=1200):
    h, w = img_bgr.shape[:2]
    if w >= target_width:
        return img_bgr
    scale = target_width / w
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def remove_illumination(gray, blur_size=51):
    bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0).astype(np.float32) + 1e-6
    norm = cv2.divide(gray.astype(np.float32), bg, scale=255)
    return np.clip(norm, 0, 255).astype(np.uint8)


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

# ---------------- Main ----------------

def ocr_preprocess(img_bgr):
    if img_bgr is None:
        raise ValueError("Invalid image")

    img = resize_for_ocr(img_bgr)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = remove_illumination(gray)
    gray = denoise_gray(gray)

    skew_angle = determine_skew(gray)
    img = rotate_image(img, skew_angle)

    # skew_angle = determine_skew(gray)
    # if abs(skew_angle) > 1.5:
    #     img = rotate_image(img, skew_angle)

    gray_for_osd = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coarse_angle = get_coarse_rotation_angle(gray_for_osd)

    # if coarse_angle in (90, 180, 270) and abs(skew_angle) < 1.0:
    img = apply_osd_rotation(img, coarse_angle)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = adaptive_threshold(gray)
    binary = morphological_clean(binary)

    return gray, binary
