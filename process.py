import cv2
import numpy as np
from PIL import Image
import math
import re
from pytesseract import image_to_osd
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
        # For arbitrary angle
        return rotate_image(image, -angle)

# ----------------- Skew  -----------------
def calculate_skew_angle(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    angle = determine_skew(image)
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ----------------- Resize helper -----------------
def resize_for_ocr(pil_image, target_width):
    w, h = pil_image.size
    if target_width is None or target_width <= 0:
        return pil_image
    if w >= target_width:
        return pil_image
    scale = target_width / w
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))
    return pil_image.resize((new_w, new_h), Image.LANCZOS)

# ----------------- Adaptive threshold -----------------
def adaptive_threshold_opencv(gray, method='gaussian', block_size=25, C=10):
    if block_size % 2 == 0:
        block_size += 1
    if method == 'gaussian':
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    binar = cv2.adaptiveThreshold(gray, 255, adaptiveMethod,
                                  cv2.THRESH_BINARY, block_size, C)
    return binar

# -----------------  prespective correction -----------------
def perspective_correction_if_needed(img, area_thresh_ratio=0.2):
    orig = img.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h*w*area_thresh_ratio):
            # skip small areas
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
            return warped
    return orig

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def remove_illumination(gray, blur_size=51):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0).astype(np.float32)
    bg[bg==0] = 1.0
    norm = cv2.divide(gray.astype(np.float32), bg, scale=255.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm

def sharpen_unsharp(gray):
    gaussian = cv2.GaussianBlur(gray, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

# ----------------- Main preprocess function -----------------
def ocr_preprocess(image_path,
                   target_width=1200,
                   apply_perspective=True,
                   use_fastnlm=True,
                   denoise_h=10,
                   thresh_block=25,
                   thresh_C=10,
                   return_images=False):

    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Image not found: {image_path}")
    img = orig.copy()

    if apply_perspective:
        img = perspective_correction_if_needed(img)

    coarse_angle = get_coarse_rotation_angle(image_path)
    img = apply_osd_rotation(img, coarse_angle)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_resized = resize_for_ocr(pil, target_width)
    img = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = remove_illumination(gray, blur_size=51)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    try:
        if use_fastnlm:
            gray = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
        else:
            gray = cv2.medianBlur(gray, 3)
    except Exception:
        gray = cv2.medianBlur(gray, 3)

    gray = sharpen_unsharp(gray)

    try:
        skew = determine_skew(gray)
    except Exception as e:
        skew = 0.0
    if abs(skew) > 0.2:
        img = rotate_image(img, skew)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_img = adaptive_threshold_opencv(gray, method='gaussian', block_size=thresh_block, C=thresh_C)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin_clean, connectivity=8)
    sizes = stats[1:, -1]  
    min_size = 10  
    bin_clean2 = np.zeros(output.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        if sizes[i-1] >= min_size:
            bin_clean2[output == i] = 255

    gray_clean = gray  

    cv2.imwrite("pre_gray.png", gray_clean)
    cv2.imwrite("pre_bin.png", bin_clean2)

    if return_images:
        return gray_clean, bin_clean2
    else:
        print("Saved: pre_gray.png and pre_bin.png")
        return "pre_gray.png", "pre_bin.png"

# ----------------- test UI -----------------
if __name__ == "__main__":
    import argparse, os, sys, subprocess, platform
    parser = argparse.ArgumentParser(description="OCR preprocessing quick tester")
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--width", type=int, default=1200, help="target width for upscaling")
    parser.add_argument("--no-perspective", dest="perspective", action="store_false",
                        help="disable perspective correction")
    parser.add_argument("--no-fastnlm", dest="fastnlm", action="store_false",
                        help="disable fastNlMeansDenoising (use median blur instead)")
    args = parser.parse_args()

    gray_path, bin_path = ocr_preprocess(args.image,
                                        target_width=args.width,
                                        apply_perspective=args.perspective,
                                        use_fastnlm=args.fastnlm)

    def try_open(path):
        try:
            if platform.system() == "Darwin":
                subprocess.call(["open", path])
            elif platform.system() == "Windows":
                os.startfile(path)
            else:
                subprocess.call(["xdg-open", path])
        except Exception:
            pass

    try_open(gray_path)
    try_open(bin_path)
