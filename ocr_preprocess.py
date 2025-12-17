
import cv2
import numpy as np
from PIL import Image
import math
import argparse
import os
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def rotate_image(cv_image, angle: float):
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_image, m, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_skew_angle(cv_image, debug=False):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # small blur to smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use OTSU to binarize, then invert for white on black
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # combine text into lines/blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug: print("No contours found; assuming angle 0")
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    # cv2.minAreaRect returns angle in [-90,0)
    if angle < -45:
        angle = 90 + angle
    return -angle  

def deskew(cv_image, max_abs_angle=10):
    angle = get_skew_angle(cv_image)
    if abs(angle) < 0.5 or abs(angle) > max_abs_angle:
        return cv_image, 0.0
    deskewed = rotate_image(cv_image, angle)
    return deskewed, angle


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


def preprocess_pipeline(input_path,
                        target_width=None,
                        do_deskew=True,
                        thresh_method='gaussian',
                        block_size=25,
                        C=10,
                        show=True,
                        save_path=None):
    img = load_image(input_path)
    original = img.copy()

    # Deskew
    if do_deskew:
        img, angle = deskew(img)
    else:
        angle = 0.0

    # Convert to PIL,resize
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_resized = resize_for_ocr(pil, target_width)
    img_resized = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # light Gaussian blur to reduce noise (small kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    binar = adaptive_threshold_opencv(gray, method=thresh_method, block_size=block_size, C=C)

    # morphological opening to remove very small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kernel)

    outputs = {
        'original': original,
        'deskewed': img,
        'resized': img_resized,
        'gray': gray,
        'binar': binar,
        'cleaned': cleaned,
        'deskew_angle': angle
    }

    if show:
        _show_preprocessed(outputs)

    if save_path:
        cv2.imwrite(save_path, outputs['cleaned'])
        print(f"Saved preprocessed image to: {save_path}")

    return outputs


def _show_preprocessed(outputs):
    figs = [
        # ('Original', cv2.cvtColor(outputs['original'], cv2.COLOR_BGR2RGB)),
        # (f"Deskewed (angle={outputs['deskew_angle']:.2f})", cv2.cvtColor(outputs['deskewed'], cv2.COLOR_BGR2RGB)),
        # ('Resized (for OCR)', cv2.cvtColor(outputs['resized'], cv2.COLOR_BGR2RGB)),
        ('Final - Binary (cleaned)', outputs['cleaned'])
    ]

    n = len(figs)
    plt.figure(figsize=(12, 3 * n))
    for i, (title, img) in enumerate(figs, 1):
        plt.subplot(n, 1, i)
        # Binary images are 2D arrays
        if img.ndim == 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Simple OCR preprocessing tool")
    p.add_argument('--input', '-i', required=True, help='Path to input image')
    p.add_argument('--width', '-w', type=int, default=1600, help='Target width to upscale to (do not downscale)')
    p.add_argument('--no-deskew', dest='deskew', action='store_false', help='Disable deskew step')
    p.add_argument('--method', choices=['gaussian', 'mean'], default='gaussian', help='Adaptive threshold method')
    p.add_argument('--block', type=int, default=25, help='Adaptive threshold block size (odd integer)')
    p.add_argument('--C', type=int, default=10, help='Adaptive threshold constant to subtract')
    p.add_argument('--save', '-s', action='store_true', help='Save preprocessed binary image (cleaned) to disk')
    p.add_argument('--out', default=None, help='Output path for saved preprocessed image')
    p.add_argument('--no-show', dest='show', action='store_false', help='Do not display images (useful in headless mode)')
    return p.parse_args()


def main():
    args = parse_args()
    save_path = args.out if args.out else None
    if args.save and save_path is None:
        # derive save path
        base, ext = os.path.splitext(args.input)
        save_path = f"{base}_prep.png"

    outputs = preprocess_pipeline(
        input_path=args.input,
        target_width=args.width,
        do_deskew=args.deskew,
        thresh_method=args.method,
        block_size=args.block,
        C=args.C,
        show=args.show,
        save_path=save_path
    )


if __name__ == '__main__':
    main()
