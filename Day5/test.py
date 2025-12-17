# import requests

# url = "https://worthwhile-cayden-compensational.ngrok-free.dev/ocr"

# files = {
#     "file": open(r"C:\Users\vaibh\Documents\Xbiz_tasks\Day4c\samples\adhar3.png", "rb")
# }
# data = {"typeofocr": 2}

# res = requests.post(url, files=files, data=data)
# print(res.status_code)
# print(res.json())

import requests

url = "http://127.0.0.1:5500/ocr"

files = {
    "file": open(r"C:\Users\vaibh\Documents\Xbiz_tasks\Day4c\samples\adhar3.png", "rb")
}
data = {"typeofocr": 2}

res = requests.post(url, files=files, data=data)
print(res.status_code)
print(res.json())

