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

DETECT_THRESHOLD = float(os.getenv("LEAKLENS_DETECT_THRESHOLD", "0.3"))
MAX_OBJECT_CLUES = int(os.getenv("LEAKLENS_MAX_OBJECT_CLUES", "12"))
# weaker boxes still get a zoomed OCR read (not shown as clues)
CROP_THRESHOLD = float(os.getenv("LEAKLENS_CROP_THRESHOLD", "0.15"))
OCR_LANGS = ["en"]
OCR_ENGINE = os.getenv("LEAKLENS_OCR_ENGINE", "auto")  # auto | apple | easyocr
OCR_APPLE_LANGS = ["en-US"]
OCR_MAX_CROPS = int(os.getenv("LEAKLENS_OCR_MAX_CROPS", "12"))
OCR_CROP_MAX_AREA = 0.25  # skip crops bigger than this share of the image
OCR_MAX_SIDE = int(os.getenv("LEAKLENS_OCR_MAX_SIDE", "2048"))
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

# Scene-level signals: no single box to blur, but they still narrow the
# location. Scored with StreetCLIP against neutral "nothing here" prompts.
SCENE_CUES = {
    "skyline": {"prompt": "a city skyline in the background", "weight": 1.5,
                "reason": "Skylines are distinctive and easy to match."},
    "mountains": {"prompt": "mountains on the horizon", "weight": 1.0,
                  "reason": "Ridgelines can be matched against terrain maps."},
    "coastline": {"prompt": "a coastline or beach", "weight": 1.0,
                  "reason": "Coastline shapes and beaches narrow the region."},
    "road_markings": {"prompt": "a road with painted lane markings", "weight": 1.0,
                      "reason": "Lane line colors and styles differ by country."},
    "power_lines": {"prompt": "utility poles and power lines", "weight": 0.8,
                    "reason": "Utility pole designs vary by region."},
    "vegetation": {"prompt": "palm trees or tropical plants", "weight": 0.8,
                   "reason": "Plants narrow down the climate zone."},
    "architecture": {"prompt": "houses with a distinctive regional architectural style", "weight": 1.0,
                     "reason": "Building styles are strongly regional."},
    "snow": {"prompt": "snow on the ground", "weight": 0.5,
             "reason": "Weather narrows the region and the date."},
}
NEUTRAL_SCENES = [
    "a close-up photo of an object", "a photo of a person indoors",
    "a plain wall", "a photo of food on a table",
]
SCENE_MIN_PROB = float(os.getenv("LEAKLENS_SCENE_MIN_PROB", "0.15"))

# Attribution: blur each clue alone and measure how much the top location
# guess weakens. Costs one extra image embedding per clue.
ATTRIBUTION = os.getenv("LEAKLENS_ATTRIBUTION", "1") == "1"
ATTRIBUTION_MAX_CLUES = int(os.getenv("LEAKLENS_ATTRIBUTION_MAX", "8"))

# Optional: search detected sign text on OpenStreetMap (Nominatim).
# Off by default because it sends the text off this machine.
TEXT_LOOKUP = os.getenv("LEAKLENS_TEXT_LOOKUP", "0") == "1"
LOOKUP_MAX_QUERIES = int(os.getenv("LEAKLENS_LOOKUP_MAX", "12"))  # Nominatim calls per photo (1/s)
LOOKUP_MAX_PER_TEXT = int(os.getenv("LEAKLENS_LOOKUP_MAX_PER_TEXT", "6"))  # so one sign can't use them all
# A result counts only if the sign's words are in the place's *name* (not its
# address). RapidFuzz similarity 0-100 between sign and name.
NAME_MATCH_MIN = float(os.getenv("LEAKLENS_NAME_MATCH_MIN", "85"))
NAME_TOKEN_MIN = 80  # per-word fuzzy match (OCR slips like PARLAMENT); words of 4 letters or less must be exact
# Searches near the AI model's guessed city first: a box this many km around it.
LOOKUP_VIEWBOX_KM = float(os.getenv("LEAKLENS_LOOKUP_VIEWBOX_KM", "30"))
# A US state needs this much evidence (StreetCLIP probability, AI confidence,
# phone area code, state named on a sign) before a match there can be strong.
LOOKUP_PRIOR_MIN = float(os.getenv("LEAKLENS_LOOKUP_PRIOR_MIN", "0.1"))
# How much each kind of evidence adds to a state's prior.
PRIOR_PHONE = 0.7  # area codes travel with mobile phones and ported numbers
PRIOR_STATE_TEXT = 0.8
PRIOR_CITY_TEXT = 0.3  # a well-known city named alone on a sign ("DOWNTOWN PHOENIX")
# Chains and brands (app/data/brands.txt, from the name-suggestion-index) and
# names that many far-apart places share are never strong matches on their own.
CHAIN_MIN_PLACES = 3  # this many separate places (>50 km apart) in 2+ states: a chain or common name
# Common street names ("16th St N", "E Lake St") are strong only at an
# intersection, or when evidence other than the AI model's guess puts this much
# weight on one state and that state has a single such street.
GENERIC_STREET_PRIOR_MIN = 0.6
# "1000 OSAGE" on a sign: compare with OSM's house number and street, and if
# OSM has no address for a candidate, search "<number> <street>" around it.
LOOKUP_MAX_ADDRESS = 2  # address searches per photo
ADDRESS_NEAR_M = 150
# Readings two sources (OCR, AI model) agree on, compared with RapidFuzz.
READING_AGREE_MIN = 85
OVERPASS_URL = os.getenv("LEAKLENS_OVERPASS_URL", "https://overpass-api.de/api/interpreter")
NEARBY_M = 300  # two clues this close together corroborate each other
NOMINATIM_URL = os.getenv("LEAKLENS_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")

# Risk level cutoffs on the 0-100 score.
RISK_MEDIUM = 30
RISK_HIGH = 60

# Local AI vision model through Ollama (#1). auto: used when available.
VLM_MODE = os.getenv("LEAKLENS_VLM", "auto")  # auto | off
OLLAMA_URL = os.getenv("LEAKLENS_OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("LEAKLENS_VLM_MODEL", "qwen3-vl:8b-instruct")  # non-thinking build answers directly
VLM_THINK_BUDGET = int(os.getenv("LEAKLENS_VLM_THINK_BUDGET", "8192"))
VLM_MAX_SIDE = int(os.getenv("LEAKLENS_VLM_MAX_SIDE", "1024"))
# Ollama defaults to a small context; the image alone can fill it and cut the answer off
VLM_NUM_CTX = int(os.getenv("LEAKLENS_VLM_NUM_CTX", "16384"))
VLM_TIMEOUT = float(os.getenv("LEAKLENS_VLM_TIMEOUT", "180"))
AI_AGREE_KM = 40  # a sign-text candidate this close to the AI's guess counts as agreement

# ---------------------------------------------------------------- one final answer
# GeoCLIP candidate list: many raw pins, grouped into ~city-sized clusters and
# named offline with the bundled gazetteer (app/data/us_cities.csv).
GEOCLIP_TOP_K = int(os.getenv("LEAKLENS_GEOCLIP_TOP_K", "40"))
GEOCLIP_PINS_SHOWN = 5  # raw pins kept in Geolocation.points for the map
CLUSTER_KM = float(os.getenv("LEAKLENS_CLUSTER_KM", "25"))
MAX_CLUSTERS = 8

# Second AI pass: the model picks among named candidate places (or "other").
VLM_CHOOSE = os.getenv("LEAKLENS_VLM_CHOOSE", "1") == "1"
CHOOSE_MAX_OPTIONS = int(os.getenv("LEAKLENS_CHOOSE_MAX_OPTIONS", "8"))
CHOOSE_MIN_OPTIONS = 2
# Skip the chooser when at least this share of GeoCLIP's pin mass is already
# near the first guess (fuse.first_pass_confirmed). Above 1 never skips.
CHOOSE_SKIP_MASS = float(os.getenv("LEAKLENS_CHOOSE_SKIP_MASS", "0.25"))

# Fusion (app/pipeline/fuse.py). Every candidate is a place on the map; it
# collects evidence from every source within FUSE_AGREE_KM. A strong sign-text
# match always wins outright. Weights were set from 35 benchmark photos
# (scripts/fuse_replay.py; leave-one-out checked), so treat them as rough:
# - the chooser almost always repeats the first guess (same model), so it adds
#   little on top of it;
# - a dominant GeoCLIP cluster (half its pins or more) was right 19 of 24
#   times, about as often as a 90%+ confident AI guess, so pin mass weighs
#   about as much as AI confidence;
# - GeoCLIP's verification score separates right from wrong spots only weakly;
# - when StreetCLIP's top state is >= 50%, the true state was in its top 3
#   every time (20 of 20).
FUSE_AGREE_KM = float(os.getenv("LEAKLENS_FUSE_AGREE_KM", "40"))
FUSE_W_CHOOSER = 0.3    # x the chooser's confidence (on top of the first pass)
FUSE_W_AI = 1.0         # x the best first-pass AI confidence nearby
FUSE_W_TEXT_WEAK = 0.4  # a possible sign-text match nearby
FUSE_W_CLUSTER = 1.5    # x share of GeoCLIP's top-k mass in the nearby clusters
FUSE_W_GEOCLIP = 0.2    # x GeoCLIP verification score at the spot (0-1)
FUSE_W_STATE = 0.5      # x StreetCLIP prob of the spot's state / top state prob
FUSE_W_AI_STATE = 0.3   # the spot is in the state the AI model named
FUSE_W_AGREE = 0.4      # per extra independent source nearby (AI, sign text, GeoCLIP)
FUSE_W_OFF_STATE = 0.5  # penalty: the spot is outside StreetCLIP's top 3 states...
FUSE_STATE_SURE = 0.5   # ...and StreetCLIP's top state has at least this probability
FUSE_SCORE_FULL = 4.0   # a score this high is reported as ~90% confidence
FUSE_MAX_CONF = 0.9
