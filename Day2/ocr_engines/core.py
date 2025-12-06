
import re
import cv2
import numpy as np

def load_image_bgr(path: str):
    img = cv2.imread(path)
    return img


def smart_resize_for_ocr(img, min_h=700, max_h=1200):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    if min_h <= h <= max_h:
        return img

    if h < min_h:
        target_h = min_h
    else:
        target_h = max_h

    scale = target_h / float(h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized


# function to estimate the skew angle of a grayscale or binary image
# def estimate_skew_angle(gray):
#     coords = np.column_stack(np.where(gray > 0))
#     if coords.size == 0:
#         return 0.0
#     rect = cv2.minAreaRect(coords)
#     angle = rect[-1
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     return angle


# def smart_deskew(gray, angle_threshold=1.5):
#     angle = estimate_skew_angle(gray)

#     if abs(angle) < angle_threshold:
#         return gray

#     (h, w) = gray.shape[:2]

#     center = (w // 2, h // 2)

#     M = cv2.getRotationMatrix2D(center, angle, 1.0)

#     rotated = cv2.warpAffine(
#         gray,           # input image
#         M,              # rotation matrix
#         (w, h),         # output image size
#         flags=cv2.INTER_CUBIC,            # interpolation method
#         borderMode=cv2.BORDER_REPLICATE   # fill border by replicating edge
#     )
#     return rotated

def clean_ocr_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars = []

    for ch in text:
        if ch == "\n":
            cleaned_chars.append(ch)
        elif ch.isprintable():
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    # We allow:
    # - digits 0-9
    # - letters a-z and A-Z
    # - basic punctuation: colon (:), slash (/), hyphen (-)
    # - spaces and newlines
    cleaned = re.sub(r"[^0-9A-Za-z:/\-\n ]", " ", cleaned)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    lines = cleaned.split("\n")

    final_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if any(ch.isalnum() for ch in stripped):
            final_lines.append(stripped)

    result = "\n".join(final_lines)
    return result

#remove duplicate lines from OCR text
def remove_duplicate_lines(text: str) -> str:    
    if text is None:
        return ""

    lines = text.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        normalized = " ".join(stripped.lower().split())

        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(stripped)

    result = "\n".join(unique_lines)

    return result
