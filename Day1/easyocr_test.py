import time
import cv2
import easyocr
#

# load image (or use path: cv2.imread('input.png'))
image = cv2.imread('input.png')

# create reader for Marathi, Hindi, English
reader = easyocr.Reader(['mr','hi','en'], gpu=False)

results = reader.readtext(image, rotation_info=[90, 180, 270])

# bbox
for text in results:
    print(f'Text: {text}')

#  | Prob: {prob:.3f}
