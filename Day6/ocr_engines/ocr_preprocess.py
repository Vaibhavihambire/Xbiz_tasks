#!/usr/bin/env python3
"""
ocr_preprocess.py

Preprocessing utilities for OCR:
 - preprocess_image(bgr_img, ...) -> (color_resized_bgr, cleaned_binary_gray, saved_path_or_None, deskew_angle)
 - preprocess_pipeline(input_path, ...) for path convenience
"""
import cv2
import numpy as np
from PIL import Image
import math
from pathlib import Path

def rotate_image(cv_image, angle: float):
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(cv_image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def get_skew_angle(cv_image, debug=False):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print("No contours found; assuming angle 0")
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -angle

def deskew(bgr_img, max_abs_angle=10):
    angle = get_skew_angle(bgr_img)
    if abs(angle) < 0.5 or abs(angle) > max_abs_angle:
        return bgr_img, 0.0
    return rotate_image(bgr_img, angle), angle

def pil_resize_only_upscale(pil_img, target_w=None):
    if target_w is None or target_w <= 0:
        return pil_img
    w, h = pil_img.size
    if w >= target_w:
        return pil_img
    scale = target_w / w
    return pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def adaptive_binarize_gray(gray, method='gaussian', block_size=25, C=10):
    if block_size % 2 == 0:
        block_size += 1
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    method_flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
    binar = cv2.adaptiveThreshold(gray, 255, method_flag, cv2.THRESH_BINARY, block_size, C)
    return binar

def preprocess_image(bgr_img,
                     target_width: int = 1600,
                     do_deskew: bool = True,
                     binar_method: str = 'gaussian',
                     block_size: int = 25,
                     C: int = 10,
                     save_to: str = None):

    img = bgr_img.copy()
    angle = 0.0
    if do_deskew:
        img, angle = deskew(img)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = pil_resize_only_upscale(pil, target_width)
    color_resized = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(color_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    binar = adaptive_binarize_gray(gray, method=binar_method, block_size=block_size, C=C)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kernel)

    saved_path = None
    if save_to:
        p = Path(save_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cleaned)
        saved_path = str(p)

    return color_resized, cleaned, saved_path, angle

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img

def preprocess_pipeline(input_path,
                        target_width=None,
                        do_deskew=True,
                        thresh_method='gaussian',
                        block_size=25,
                        C=10,
                        show=False,
                        save_path=None):
    img = load_image(input_path)
    color_resized, cleaned, saved_path, angle = preprocess_image(
        img,
        target_width=target_width,
        do_deskew=do_deskew,
        binar_method=thresh_method,
        block_size=block_size,
        C=C,
        save_to=save_path
    )
    outputs = {
        'original': img,
        'resized': color_resized,
        'binar': cleaned,
        'deskew_angle': angle
    }
    return outputs
