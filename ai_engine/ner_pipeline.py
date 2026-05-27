import re
import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
NER_MODEL_PATH  = os.path.join(BASE_DIR, "..", "model_training", "model")
RISK_MODEL_PATH = os.path.join(BASE_DIR, "..", "model_training", "risk_model")

# ── Regex patterns ─────────────────────────────────────────────────────────────
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\+?[\d][\d\s\-().]{6,}\d"),
    "SSN":   re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "DOB":   re.compile(
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"
        r"|(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})"
        r"|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE
    ),
}

MIN_CONFIDENCE = 0.60
MIN_VALUE_LEN  = 2

# ── Load NER model ─────────────────────────────────────────────────────────────
def _load_ner():
    try:
        tok   = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
        pipe  = pipeline("ner", model=model, tokenizer=tok,
                         aggregation_strategy="simple")
        print("✅ Custom RoBERTa NER model loaded.")
        return pipe
    except Exception as e:
        print(f"⚠️  Custom NER model not found ({e}). Using base model.")
        return pipeline("ner", model="dslim/bert-base-NER",
                        aggregation_strategy="simple")

# ── Load Risk Classifier ───────────────────────────────────────────────────────
def _load_risk():
    try:
        tok   = AutoTokenizer.from_pretrained(RISK_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(RISK_MODEL_PATH)
        pipe  = pipeline("text-classification", model=model, tokenizer=tok)
        print("✅ Custom DistilBERT Risk Classifier loaded.")
        return pipe
    except Exception as e:
        print(f"⚠️  Risk model not found ({e}). Using heuristics.")
        return None

ner_pipeline  = _load_ner()
risk_pipeline = _load_risk()

PII_ENTITY_TYPES = {
    "PER", "EMAIL", "PHONE", "ADDR", "LOC",
    "DOB", "SSN", "PER-NAME", "PERSON"
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_fragment(value: str, label: str) -> bool:
    v = value.strip()
    if len(v) <= MIN_VALUE_LEN:
        return True
    if re.fullmatch(r"[\s\-.,@#/\\]+", v):
        return True
    if label in ("EMAIL", "PHONE", "SSN") and not re.search(r"[a-zA-Z0-9]", v):
        return True
    if label == "EMAIL" and "@" not in v:
        return True
    if label == "PHONE" and len(re.sub(r"\D", "", v)) < 5:
        return True
    return False

def _already_covered(new_val: str, existing: list) -> bool:
    nv = re.sub(r"\s", "", new_val.lower())
    for e in existing:
        ev = re.sub(r"\s", "", e["value"].lower())
        if nv in ev or ev in nv:
            return True
    return False

# ── Regex extraction ───────────────────────────────────────────────────────────
def _regex_extract(text: str) -> dict:
    found = {}
    for label, pattern in REGEX_PATTERNS.items():
        matches = pattern.findall(text)
        flat = []
        for m in matches:
            val = (max(m, key=len) if isinstance(m, tuple) else m).strip()
            if val and len(val) > MIN_VALUE_LEN:
                flat.append(val)
        if flat:
            found[label] = [
                {"value": v, "confidence": 0.90, "source": "regex"}
                for v in flat
            ]
    return found

# ── Main PII extraction ────────────────────────────────────────────────────────
def extract_pii(text: str) -> dict:
    pii = {}

    # 1) NER model
    try:
        ner_results = ner_pipeline(text[:512])
        for entity in ner_results:
            label = entity["entity_group"].upper()
            word  = entity["word"].strip()
            score = round(float(entity["score"]), 3)

            if not any(t in label for t in PII_ENTITY_TYPES):
                continue
            if score < MIN_CONFIDENCE:
                continue
            if _is_fragment(word, label):
                continue

            pii.setdefault(label, []).append({
                "value":      word,
                "confidence": score,
                "source":     "model"
            })
    except Exception as e:
        print(f"NER error: {e}")

    # 2) Regex fallback — add only what model missed
    regex_found = _regex_extract(text)
    for label, entities in regex_found.items():
        existing = pii.get(label, [])
        for e in entities:
            if not _already_covered(e["value"], existing):
                pii.setdefault(label, []).append(e)

    return pii

# ── Risk classification ────────────────────────────────────────────────────────
def _heuristic_risk(pii: dict) -> str:
    if any(k in pii for k in ["SSN", "DOB"]):
        return "High"
    if any(k in pii for k in ["EMAIL", "PHONE"]):
        return "Medium"
    return "Low"

def classify_risk(text: str, pii: dict) -> dict:
    if risk_pipeline:
        try:
            result = risk_pipeline(text[:512])[0]
            return {
                "label":      result["label"],
                "confidence": round(result["score"], 3)
            }
        except Exception:
            pass
    return {"label": _heuristic_risk(pii), "confidence": 0.75}

# ── Identity match scoring ─────────────────────────────────────────────────────
def score_identity_match(scraped_pii: dict, user_profile: dict) -> float:
    score = 0.0

    # Name — 0.35
    name_entities = (scraped_pii.get("PER") or
                     scraped_pii.get("PER-NAME") or
                     scraped_pii.get("PERSON") or [])
    if name_entities and user_profile.get("name"):
        user_name = user_profile["name"].lower()
        parts     = [p for p in user_name.split() if len(p) > 2]
        for e in name_entities:
            val = e["value"].lower()
            if user_name in val or val in user_name:
                score += 0.35
                break
            if any(p in val for p in parts):
                score += 0.20
                break

    # Email — 0.25
    for e in scraped_pii.get("EMAIL", []):
        if user_profile.get("email", "").lower() in e["value"].lower():
            score += 0.25
            break

    # Phone — 0.20 (digit-only comparison)
    user_phone_digits = re.sub(r"\D", "", user_profile.get("phone", ""))
    for e in scraped_pii.get("PHONE", []):
        scraped_digits = re.sub(r"\D", "", e["value"])
        if user_phone_digits and len(user_phone_digits) >= 5:
            if user_phone_digits in scraped_digits or scraped_digits in user_phone_digits:
                score += 0.20
                break

    # Location — 0.10
    loc_entities = scraped_pii.get("LOC", scraped_pii.get("ADDR", []))
    if user_profile.get("city"):
        for e in loc_entities:
            if user_profile["city"].lower() in e["value"].lower():
                score += 0.10
                break

    # SSN — 0.05
    user_ssn = re.sub(r"\D", "", user_profile.get("ssn", ""))
    for e in scraped_pii.get("SSN", []):
        if user_ssn and user_ssn in re.sub(r"\D", "", e["value"]):
            score += 0.05
            break

    # DOB — 0.05
    for e in scraped_pii.get("DOB", []):
        if user_profile.get("dob", "") in e["value"]:
            score += 0.05
            break

    return round(min(score, 1.0), 2)
