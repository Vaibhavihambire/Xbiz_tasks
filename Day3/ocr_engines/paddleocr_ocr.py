
from typing import Any, List
from paddleocr import PaddleOCR


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
def run_paddleocr(image_path: str, lang: str = "en") -> str:
    """
    No manual preprocessing is done here; PaddleOCR handles detection and recognition.
    """
    # If this PaddleOCR object supports predict(), use it
    if hasattr(_ocr, "predict"):
        # Call predict() using the image path
        ocr_raw = _ocr.predict(image_path)
    else:
        # Otherwise fall back to ocr() method
        ocr_raw = _ocr.ocr(image_path)

    # Use the helper to extract text strings from raw result
    lines = extract_texts_from_ocr_result(ocr_raw)

    # Join all lines with newline characters to form a single big string
    text = "\n".join(lines)

    # Return the final text extracted by PaddleOCR
    return text
