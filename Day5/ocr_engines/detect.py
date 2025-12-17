import re

def detect_document_types(ocr_text: str):
    if not ocr_text:
        return ["unknown"]

    txt = ocr_text
    lower = txt.lower()
    found = []

    # PAN (India)
    try:
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", ocr_text):
            found.append("pan")
    except re.error:
        pass
    if any(k in lower for k in ["permanent account number", "income tax department", "pan card", "pancard"]):
        if "pan" not in found:
            found.append("pan")

    # Aadhaar
    if re.search(r"\b(?:\d{12}|\d{4}\s\d{4}\s\d{4})\b", ocr_text):
        found.append("aadhaar")
    if any(k in lower for k in ["aadhaar", "aadhar", "uidai", "unique identification authority"]):
        if "aadhaar" not in found:
            found.append("aadhaar")

    # Election / Voter ID
    if any(k in lower for k in ["election commission", "voter id", "elector", "epic", "voter card"]):
        found.append("election")

    # Driving License
    if any(k in lower for k in ["driving licence", "driving license", "dl no", "license no", "driving licence no", "driving lic"]):
        found.append("driving_license")

   
    if re.search(r"\b[A-Z]\d{7}\b", ocr_text) or any(k in lower for k in ["passport", "passport no", "passport number"]):
        found.append("passport")


    if re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", ocr_text):
        found.append("bank_account")
    if any(k in lower for k in ["account number", "account no", "a/c no", "a/c", "account no.", "account#"]):
        if "bank_account" not in found:
            found.append("bank_account")
    # Cheque detection
    if any(k in lower for k in ["cheque", "check", "payable to", "micr"]):
        if "cheque" not in found:
            found.append("cheque")

    # If nothing clear found, return unknown
    if not found:
        return ["unknown"]

    seen = []
    for f in found:
        if f not in seen:
            seen.append(f)
    return seen


def detect_document_type(ocr_text: str):

    types = detect_document_types(ocr_text)
    if not types:
        return "unknown"
    return types[0] if isinstance(types, (list, tuple)) else types
