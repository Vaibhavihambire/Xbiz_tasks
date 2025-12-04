# core.py
# This file contains common helper functions for:
# - loading images
# - resizing images safely
# - estimating skew angle
# - deskewing images when needed
# Import re for regular expressions
import re
# Import OpenCV for image operations
import cv2
# Import numpy for array and coordinate operations
import numpy as np


# Define a function to load an image in BGR format from a given path
def load_image_bgr(path: str):
    # Use cv2.imread to load the image from disk
    img = cv2.imread(path)
    # Return the loaded 
    return img


# Define a function to resize image height within a safe range for OCR
def smart_resize_for_ocr(img, min_h=700, max_h=1200):
    # Get the current height and width of the image
    h, w = img.shape[:2]

    # If height or width is zero, return the image as it is
    if h == 0 or w == 0:
        return img

    # If image height is already within the desired range, do not resize
    if min_h <= h <= max_h:
        return img

    # If image height is smaller than minimum height, we want to upscale it
    if h < min_h:
        target_h = min_h
    # If image height is larger than maximum height, we want to downscale it
    else:
        target_h = max_h

    # Calculate the scale factor based on target height
    scale = target_h / float(h)

    # Calculate new width while keeping aspect ratio same
    new_w = int(w * scale)
    # Calculate new height (should be equal to target_h)
    new_h = int(h * scale)

    # Resize the image using cubic interpolation (good for text)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Return the resized image
    return resized


# Define a function to estimate the skew angle of a grayscale or binary image
def estimate_skew_angle(gray):
    # Find coordinates of all pixels which are greater than 0 (non-background)
    coords = np.column_stack(np.where(gray > 0))

    # If there are no such pixels, return 0.0 angle (no information)
    if coords.size == 0:
        return 0.0

    # Use minAreaRect to get the minimum area bounding box for these points
    rect = cv2.minAreaRect(coords)

    # The third value of rect is the angle
    angle = rect[-1]

    # OpenCV returns angle in a specific range; we adjust it to a more natural value
    # If angle is less than -45 degrees, we adjust it using -(90 + angle)
    if angle < -45:
        angle = -(90 + angle)
    # Otherwise we simply take negative of the angle
    else:
        angle = -angle

    # Return the final estimated skew angle
    return angle


# Define a function to deskew image only when skew is actually present
def smart_deskew(gray, angle_threshold=1.5):
    # First, estimate the skew angle of the given grayscale image
    angle = estimate_skew_angle(gray)

    # If the absolute value of angle is very small (less than threshold),
    # we consider the image already straight and return it as is
    if abs(angle) < angle_threshold:
        return gray

    # Get the height and width of the image
    (h, w) = gray.shape[:2]

    # Calculate the center of the image (needed for rotation)
    center = (w // 2, h // 2)

    # Get the rotation matrix using the center and angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation using warpAffine to get rotated (deskewed) image
    rotated = cv2.warpAffine(
        gray,           # input image
        M,              # rotation matrix
        (w, h),         # output image size
        flags=cv2.INTER_CUBIC,            # interpolation method
        borderMode=cv2.BORDER_REPLICATE   # fill border by replicating edge
    )

    # Return the deskewed image
    return rotated

# Define a helper function to clean OCR text by removing unwanted characters
def clean_ocr_text(text: str) -> str:
    # If text is None, return an empty string
    if text is None:
        return ""

    # Convert any Windows-style (\r\n) or old-style (\r) newlines into simple \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Create a list to collect only printable characters and newlines
    cleaned_chars = []

    # Loop over every character in the text
    for ch in text:
        # If character is a newline, we keep it to separate lines
        if ch == "\n":
            cleaned_chars.append(ch)
        # If character is printable (letters, digits, basic punctuation, space), we keep it
        elif ch.isprintable():
            cleaned_chars.append(ch)
        # All other control characters are skipped

    # Join the kept characters back into a string
    cleaned = "".join(cleaned_chars)

    # Now we remove any unusual symbols we do not want.
    # We allow:
    # - digits 0-9
    # - letters a-z and A-Z
    # - basic punctuation: colon (:), slash (/), hyphen (-)
    # - spaces and newlines
    cleaned = re.sub(r"[^0-9A-Za-z:/\-\n ]", " ", cleaned)

    # Replace multiple spaces or tabs with a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Split the cleaned text into lines for further processing
    lines = cleaned.split("\n")

    # Create a list to collect final lines
    final_lines = []

    # Loop through each line
    for line in lines:
        # Remove leading and trailing spaces from the line
        stripped = line.strip()

        # If line is empty after stripping, skip it
        if not stripped:
            continue

        # If the line has at least one alphanumeric character, we keep it
        if any(ch.isalnum() for ch in stripped):
            final_lines.append(stripped)

    # Join all final lines back together with newline characters
    result = "\n".join(final_lines)

    # Return the final cleaned OCR text
    return result

# Define a helper function to remove duplicate lines from OCR text
def remove_duplicate_lines(text: str) -> str:
    """
    This function removes duplicate lines from a given text.
    It keeps the first occurrence of each unique line and preserves the order.
    """

    # If text is None, return an empty string
    if text is None:
        return ""

    # Split the full text into individual lines using newline character
    lines = text.split("\n")

    # Create a set to remember which normalized lines we have already seen
    seen = set()

    # Create a list to store final unique lines in order
    unique_lines = []

    # Loop over each line from the original text
    for line in lines:
        # Remove leading and trailing spaces from the line
        stripped = line.strip()

        # If the stripped line is empty, we skip it
        if not stripped:
            continue

        # Create a normalized version for comparison
        # We lower-case and collapse spaces so that minor differences in spacing
        # do not cause lines to be treated as different.
        normalized = " ".join(stripped.lower().split())

        # If this normalized line has not been seen before, we keep it
        if normalized not in seen:
            # Add normalized version to the set to mark it as seen
            seen.add(normalized)
            # Add the original stripped line (with its original case) to output
            unique_lines.append(stripped)

    # Join all unique lines back together with newline characters
    result = "\n".join(unique_lines)

    # Return the final text with duplicate lines removed
    return result
