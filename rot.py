import cv2
import numpy as np
from pytesseract import image_to_string, image_to_osd
import re
from deskew import determine_skew

def get_coarse_rotation_angle(image_path):
    try:
        osd_data = image_to_osd(image_path)
        match = re.search(r'Rotate:\s*(\d+)', osd_data)
        if match:
            coarse_angle = int(match.group(1))
            return coarse_angle
        else:
            print("Warning: Tesseract OSD 'Rotate:' angle not found in output.")
            return 0
    except Exception as e:
        print(f"Error determining OSD angle (Tesseract issue or image read error): {e}")
        return 0

def apply_osd_rotation(image, coarse_angle):
    if image is None:
        return image

    angle = int(coarse_angle) % 360

    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        
        return rotate_image(image, -angle)

# ----------------- Skew -----------------
def calculate_skew_angle(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image}")

    angle = determine_skew(image)
    return angle

def rotate_image(image, angle):

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ----------------- Combined pipeline -----------------
def correct_orientation_and_skew(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # ---------- STEP 1: DESKEW FIRST ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    skew_angle = determine_skew(gray)

    img_after_skew = rotate_image(img, skew_angle)

    # ---------- STEP 2: OSD AFTER DESKEW ----------
    # Save temporary image for OSD (required because OSD expects a path)
    temp_path = "temp_for_osd.png"
    cv2.imwrite(temp_path, img_after_skew)

    coarse_angle = get_coarse_rotation_angle(temp_path)

    # ---------- STEP 3: APPLY COARSE ROTATION ----------
    final_img = apply_osd_rotation(img_after_skew, coarse_angle)

    return final_img, coarse_angle, skew_angle

final, coarse, skew = correct_orientation_and_skew("C://Users//vaibh//Documents//Xbiz_tasks//Day4//samples//cd2.jpg")
cv2.imwrite(r"C:\Users\vaibh\Documents\Xbiz_tasks\corrected.png", final)
