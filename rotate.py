import cv2
import numpy as np
from pytesseract import image_to_string, image_to_osd
import re

def rotate_image_no_crop(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
    return rotated_mat

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
        return rotate_image_no_crop(image, -angle)

image_path = 'C://Users//vaibh//Documents//Xbiz_tasks//Day4//samples//cd.jpg' 

original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Could not load image at {image_path}. Please check the path.")
else:
    print("--- Starting OSD Rotation Process ---")
    
    coarse_angle = get_coarse_rotation_angle(image_path)
    
    print(f"Detected Coarse OSD Angle: {coarse_angle} degrees")
    
    final_rotated_image = apply_osd_rotation(original_image, coarse_angle)
    total_angle_applied = coarse_angle 
    # cv2.imshow("Original Image", original_image)
    # cv2.imshow(f"Corrected Image (OSD Angle Applied: {total_angle_applied} deg)", final_rotated_image)
    cv2.imwrite(r'C:\Users\vaibh\Documents\Xbiz_tasks\preprocessed_document_v4.png',final_rotated_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text = image_to_string(final_rotated_image)
    print(f"OCR Result:\n{text}")
