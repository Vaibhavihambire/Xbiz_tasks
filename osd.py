import cv2
import numpy as np
from deskew import determine_skew
from pytesseract import image_to_string
from skimage.transform import rotate

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

image_path = 'C://Users//vaibh//Documents//Xbiz_tasks//Day4//samples//cd.jpg'

original_image = cv2.imread(image_path) 

if original_image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    angle = calculate_skew_angle(image_path)
    print(f"Detected skew angle: {angle:.2f} degrees")

    corrected_image = rotate_image(original_image, angle)

    cv2.imshow("Original Image", original_image)
    cv2.imshow(f"Corrected Image (Rotated {angle:.2f} degrees)", corrected_image)
    cv2.imwrite(r'C:\Users\vaibh\Documents\Xbiz_tasks\preprocessed_document_v3.png',corrected_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text = image_to_string(corrected_image)
    print(f"OCR Result:\n{text}")

