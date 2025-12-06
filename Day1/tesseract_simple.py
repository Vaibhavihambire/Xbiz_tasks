
import cv2
from PIL import Image
import pytesseract

# img = Image.open("vaibhavi_birthcertificate.jpg")
# text = pytesseract.image_to_string(img, lang="hin+mar+eng")
# print("=== OCR RESULT ===")
# print(text)

# img = cv2.imread("input.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # simple preprocessing
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# # save or pass to PIL
# text = pytesseract.image_to_string(Image.fromarray(gray), lang="hin+mar+eng")
# print(text)

from skimage.filters import threshold_local

image = cv2.imread('input.png')

# We get the Value component from the HSV color space 
# then we apply adaptive thresholdingto 
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
V = cv2.GaussianBlur(V, (3,3), 0) #good for clean images and disadv for documents like aadharcard
T = threshold_local(V, 25, offset=10, method="gaussian")

# Apply the threshold operation 
thresh = cv2.normalize(V - T, None, 0, 255, cv2.NORM_MINMAX)
thresh = thresh.astype("uint8")
# show threshold result (convert single channel to 3-channel for display)
# thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
# convert to PIL Image and run pytesseract with languages
pil = Image.fromarray(thresh)

output_txt = pytesseract.image_to_string(pil, lang='mar+hin+eng')
print("PyTesseract Extracted: {}".format(output_txt))