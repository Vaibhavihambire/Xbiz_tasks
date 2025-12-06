from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(lang='multilingual')  # or lang='en' or specific model configs
result = ocr.ocr('input.png', cls=True)
for line in result:
    print(line)
