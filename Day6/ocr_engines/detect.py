import re

_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"\b(\d{4}\s\d{4}\s\d{4}|\d{12})\b")
_DL_RE = re.compile(r"\b([A-Z]{2}\d{2}\s?\d{11,13}|\w{2}\s?\d{2}\s?\d{11,13})\b")  # loose match
_PASSPORT_RE = re.compile(r"\b([A-Z][0-9]{7})\b")  # e.g. A1234567 (common format)
_BANK_ACC_RE = re.compile(r"\b\d{9,18}\b")  # generic long number (9-18 digits) for bank accounts

PAN_KEYWORDS = ["permanent account number", "income tax department", "pan card", "pancard"]
AADHAAR_KEYWORDS = ["aadhaar", "aadhar", "uidai", "unique identification authority"]
ELECTION_KEYWORDS = ["election commission", "voter id", "elector", "epic", "electoral"]
DL_KEYWORDS = ["driving licence", "driving license", "dl no", "driving licence of", "driver licence"]
PASSPORT_KEYWORDS = ["passport", "passport no", "passport number"]
BANK_KEYWORDS = ["account number", "ifsc", "bank", "branch", "account no", "account number"]


def detect_document_type(ocr_text: str):
    if not ocr_text:
        return "unknown"

    txt = ocr_text.strip()
    txt_lower = txt.lower()

    scores = {
        "pan": 0,
        "aadhaar": 0,
        "election": 0,
        "driving_license": 0,
        "passport": 0,
        "bank_account": 0,
    }

    # PAN: regex (case-sensitive) + keywords
    if _PAN_RE.search(txt):
        scores["pan"] += 2
    if any(k in txt_lower for k in PAN_KEYWORDS):
        scores["pan"] += 1

    # Aadhaar: regex + keywords
    if _AADHAAR_RE.search(txt):
        scores["aadhaar"] += 2
    if any(k in txt_lower for k in AADHAAR_KEYWORDS):
        scores["aadhaar"] += 1

    # Election / Voter
    if any(k in txt_lower for k in ELECTION_KEYWORDS):
        scores["election"] += 2

    # Driving licence: pattern + keywords
    if _DL_RE.search(txt):
        scores["driving_license"] += 2
    if any(k in txt_lower for k in DL_KEYWORDS):
        scores["driving_license"] += 1

    # Passport: regex + keywords
    if _PASSPORT_RE.search(txt):
        scores["passport"] += 2
    if any(k in txt_lower for k in PASSPORT_KEYWORDS):
        scores["passport"] += 1

    # Bank account: long digits + keywords 
    bank_digit = _BANK_ACC_RE.search(txt)
    bank_kw = any(k in txt_lower for k in BANK_KEYWORDS)
    if bank_digit and bank_kw:
        scores["bank_account"] += 3
    elif bank_digit or bank_kw:
        scores["bank_account"] += 1

    # require score >= 2 to accept; pick highest score
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] >= 2:
        return best[0]
    return "unknown"


def detect_all_document_types(ocr_text: str):

    if not ocr_text:
        return []

    types = []
    if "--- PAGE" in ocr_text:
        parts = [p for p in ocr_text.split("--- PAGE") if p.strip()]
    else:
        parts = [p for p in ocr_text.split("\n\n") if p.strip()]

    checked = set()
    for part in parts:
        # PAN
        if (_PAN_RE.search(part) or any(k in part.lower() for k in PAN_KEYWORDS)) and "pan" not in checked:
            if _score_for_part(part, "pan") >= 2:
                types.append("pan"); checked.add("pan")
        # Aadhaar
        if (_AADHAAR_RE.search(part) or any(k in part.lower() for k in AADHAAR_KEYWORDS)) and "aadhaar" not in checked:
            if _score_for_part(part, "aadhaar") >= 2:
                types.append("aadhaar"); checked.add("aadhaar")
        # Election
        if (any(k in part.lower() for k in ELECTION_KEYWORDS)) and "election" not in checked:
            if _score_for_part(part, "election") >= 2:
                types.append("election"); checked.add("election")
        # Driving licence
        if (_DL_RE.search(part) or any(k in part.lower() for k in DL_KEYWORDS)) and "driving_license" not in checked:
            if _score_for_part(part, "driving_license") >= 2:
                types.append("driving_license"); checked.add("driving_license")
        # Passport
        if (_PASSPORT_RE.search(part) or any(k in part.lower() for k in PASSPORT_KEYWORDS)) and "passport" not in checked:
            if _score_for_part(part, "passport") >= 2:
                types.append("passport"); checked.add("passport")
        # Bank account
        bank_digit = _BANK_ACC_RE.search(part)
        bank_kw = any(k in part.lower() for k in BANK_KEYWORDS)
        if bank_digit and bank_kw and "bank_account" not in checked:
            if _score_for_part(part, "bank_account") >= 2:
                types.append("bank_account"); checked.add("bank_account")

    whole = detect_document_type(ocr_text)
    if whole != "unknown" and whole not in checked:
        types.append(whole)

    return types


def _score_for_part(text: str, label: str):

    t = text
    t_l = t.lower()
    s = 0
    if label == "pan":
        if _PAN_RE.search(t):
            s += 2
        if any(k in t_l for k in PAN_KEYWORDS):
            s += 1
    elif label == "aadhaar":
        if _AADHAAR_RE.search(t):
            s += 2
        if any(k in t_l for k in AADHAAR_KEYWORDS):
            s += 1
    elif label == "election":
        if any(k in t_l for k in ELECTION_KEYWORDS):
            s += 2
    elif label == "driving_license":
        if _DL_RE.search(t):
            s += 2
        if any(k in t_l for k in DL_KEYWORDS):
            s += 1
    elif label == "passport":
        if _PASSPORT_RE.search(t):
            s += 2
        if any(k in t_l for k in PASSPORT_KEYWORDS):
            s += 1
    elif label == "bank_account":
        bank_digit = _BANK_ACC_RE.search(t)
        bank_kw = any(k in t_l for k in BANK_KEYWORDS)
        if bank_digit and bank_kw:
            s += 3
        elif bank_digit or bank_kw:
            s += 1
    return s
