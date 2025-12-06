
from typing import Any, List
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import cv2

_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_batch_num=16
)


def extract_texts_from_ocr_result(result: Any) -> List[str]:
    def extract(obj: Any):
        texts = []

        if isinstance(obj, str) and obj.strip():
            texts.append(obj.strip())

        elif isinstance(obj, dict):
            for key in ("text", "rec_text", "transcription", "sentence", "text_line"):
                # If such a key exists and is a non-empty string, add it
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    texts.append(obj[key].strip())

            # Then, recursively process all values in the dictionary
            for v in obj.values():
                texts.extend(extract(v))

        # Case 3: if object is a list or tuple
        elif isinstance(obj, (list, tuple)):
            # Recursively process each element
            for el in obj:
                texts.extend(extract(el))

        # Return all collected texts at this level
        return texts

    # Call the inner recursive function on the whole result
    all_texts = extract(result)

    # Filter: keep only strings that have at least one alphanumeric character
    # filtered = [t for t in all_texts if any(ch.isalnum() for ch in t)]
    filtered = []
    for t in all_texts:
        s = t.lower()
        # Skip strings that look like file paths or filenames
        if s.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if "\\" in s: #"/" in s or 
            continue
        # keep only strings that have at least one alphanumeric character
        # if any(ch.isalnum() for ch in s):
        filtered.append(t)

    # Remove duplicates while keeping first occurrence order
    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)

    # Return the final list of unique text strings
    return out


# Define a function to run PaddleOCR on an image path
def run_paddleocr(image_input, lang: str = "en") -> str:
    """
    No manual preprocessing is done here; PaddleOCR handles detection and recognition.
    """
    # Helper: if numpy array in BGR, convert to RGB for Paddle if needed
    try:
        # Case: filesystem path (keep original behavior)
        if isinstance(image_input, str):
            if hasattr(_ocr, "predict"):
                ocr_raw = _ocr.predict(image_input)
            else:
                ocr_raw = _ocr.ocr(image_input)

        # Case: PIL Image -> convert to numpy RGB
        elif isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))

            # Ensure contiguous uint8
            arr_rgb = np.ascontiguousarray(arr, dtype=np.uint8)

            if hasattr(_ocr, "predict"):
                # NOTE: pass a list (batch) to predict
                ocr_raw = _ocr.predict([arr_rgb])
            else:
                ocr_raw = _ocr.ocr(arr_rgb)

        # Case: numpy array (likely BGR from OpenCV)
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            # If it's BGR (OpenCV style), convert to RGB
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                # grayscale or other -> convert to 3-channel RGB
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                arr_rgb = arr

            # Ensure contiguous and uint8
            arr_rgb = np.ascontiguousarray(arr_rgb, dtype=np.uint8)

            if hasattr(_ocr, "predict"):
                ocr_raw = _ocr.predict([arr_rgb])   # pass as batch list
            else:
                ocr_raw = _ocr.ocr(arr_rgb)

        else:
            raise ValueError("Unsupported input type for PaddleOCR: path, PIL.Image, or numpy array expected")

        # extract lines
        lines = extract_texts_from_ocr_result(ocr_raw)
        text = "\n".join(lines)
        return text

    except Exception:
        # If an in-memory approach failed for any reason, try the safe path-based fallback (if input was array)
        # This fallback will save to a temp PNG and call the old path-based method.
        try:
            # if input is array or PIL, write to a temp file and call the original file-path logic
            uid = np.random.randint(0, 2**31)
            tmp_name = f"paddle_tmp_{uid}.png"
            tmp_path = tmp_name
            if isinstance(image_input, np.ndarray):
                rgb = image_input
                if rgb.ndim == 3 and rgb.shape[2] == 3:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                Image.fromarray(np.ascontiguousarray(rgb)).save(tmp_path)
            elif isinstance(image_input, Image.Image):
                image_input.convert("RGB").save(tmp_path)
            else:
                # nothing to fallback to
                raise

            # call original path-based API
            if hasattr(_ocr, "predict"):
                ocr_raw = _ocr.predict(tmp_path)
            else:
                ocr_raw = _ocr.ocr(tmp_path)

            os.remove(tmp_path)
            lines = extract_texts_from_ocr_result(ocr_raw)
            return "\n".join(lines)
        except Exception:
            # re-raise the original exception to let caller handle/log it
            raise