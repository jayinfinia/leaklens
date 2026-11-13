"""Central settings for LeakLens.

Everything tunable lives here so the pipeline modules stay small.
Environment variables override the defaults.
"""

import os

# Set LEAKLENS_MOCK=1 to skip all model downloads and return canned results.
# Useful for frontend work and for tests.
MOCK_MODE = os.getenv("LEAKLENS_MOCK", "0") == "1"

# "cuda", "mps", or "cpu". Leave unset to auto-detect.
DEVICE = os.getenv("LEAKLENS_DEVICE")

# Images are downscaled to this longest side before analysis (speed).
MAX_ANALYSIS_SIDE = int(os.getenv("LEAKLENS_MAX_SIDE", "1280"))

# ---------------------------------------------------------------- models
OWL_MODEL = os.getenv("LEAKLENS_OWL_MODEL", "google/owlv2-base-patch16-ensemble")
STREETCLIP_MODEL = os.getenv("LEAKLENS_STREETCLIP_MODEL", "geolocal/StreetCLIP")
USE_GEOCLIP = os.getenv("LEAKLENS_USE_GEOCLIP", "1") == "1"

DETECT_THRESHOLD = float(os.getenv("LEAKLENS_DETECT_THRESHOLD", "0.2"))
OCR_LANGS = ["en"]
OCR_MIN_CONF = float(os.getenv("LEAKLENS_OCR_MIN_CONF", "0.4"))

# ---------------------------------------------------------------- clues
# category -> prompts (OWLv2 text queries), weight (risk points at
# confidence 1.0), and the plain-language reason shown to the user.
CLUE_CATEGORIES = {
    "street_sign": {
        "prompts": ["a street sign", "a road sign with a street name"],
        "weight": 3.0,
        "reason": "Street signs can name the exact road.",
    },
    "house_number": {
        "prompts": ["a house number", "an address number on a building"],
        "weight": 3.0,
        "reason": "House numbers can narrow a photo to a single building.",
    },
    "license_plate": {
        "prompts": ["a car license plate"],
        "weight": 2.0,
        "reason": "Plate designs reveal the state or country.",
    },
    "storefront": {
        "prompts": ["a storefront sign", "a business sign"],
        "weight": 2.5,
        "reason": "Business names can be looked up on a map.",
    },
    "landmark": {
        "prompts": ["a famous landmark", "a monument", "a distinctive tower"],
        "weight": 2.5,
        "reason": "Landmarks are easy to match against public photos.",
    },
    "transit": {
        "prompts": ["a bus stop sign", "a train station sign"],
        "weight": 2.5,
        "reason": "Transit signs name stops and routes.",
    },
    "institution_logo": {
        "prompts": ["a school or university logo", "a sports team jersey"],
        "weight": 1.5,
        "reason": "Logos and uniforms tie a photo to an institution or city.",
    },
    "window_view": {
        "prompts": ["a view out of a window"],
        "weight": 1.0,
        "reason": "Window views reveal the skyline and floor height.",
    },
}

# Readable text found by OCR.
TEXT_CLUE_WEIGHT = 2.5
TEXT_CLUE_REASON = "Readable text (names, addresses, phone numbers) can be searched."

# Risk level cutoffs on the 0-100 score.
RISK_MEDIUM = 30
RISK_HIGH = 60
