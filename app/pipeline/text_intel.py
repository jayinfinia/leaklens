"""What does a piece of sign text actually reveal?

"CLASS OF 2024" tells an investigator almost nothing; "UVA MED SCHOOL",
"Main St" or a phone number can pin a photo to a town. Each OCR string
gets a kind, a weight (risk points at full confidence) and a reason.
"""

import re
from dataclasses import dataclass

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
}

# A small gazetteer is enough to catch the obvious cases; the optional
# OpenStreetMap lookup handles everything else.
US_CITIES = {
    "new york", "nyc", "brooklyn", "manhattan", "los angeles", "chicago", "houston",
    "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "austin",
    "san jose", "jacksonville", "fort worth", "columbus", "charlotte", "indianapolis",
    "san francisco", "seattle", "denver", "nashville", "oklahoma city", "boston",
    "el paso", "portland", "las vegas", "detroit", "memphis", "louisville", "baltimore",
    "milwaukee", "albuquerque", "tucson", "fresno", "sacramento", "atlanta", "miami",
    "raleigh", "omaha", "minneapolis", "tulsa", "cleveland", "new orleans", "tampa",
    "pittsburgh", "cincinnati", "st louis", "saint louis", "orlando", "honolulu",
    "anchorage", "birmingham", "huntsville", "montgomery", "mobile", "richmond",
    "norfolk", "charlottesville", "arlington", "alexandria", "durham", "savannah",
    "charleston", "knoxville", "chattanooga", "salt lake city", "boise", "spokane",
    "buffalo", "rochester", "albany", "hartford", "providence", "burlington",
    "madison", "des moines", "kansas city", "wichita", "little rock", "jackson",
    "baton rouge", "birmingham", "lexington", "annapolis", "washington dc",
    "cville", "nola", "philly", "atl", "chi-town", "big apple",
}

INSTITUTIONS = re.compile(
    r"\b(university|univ|college|school|academy|hospital|medical|clinic|church|chapel|"
    r"cathedral|temple|mosque|synagogue|county|city of|town of|museum|theat(?:er|re)|"
    r"library|station|airport|stadium|arena|park|zoo|courthouse|city hall|post office|"
    r"fire dept|police|campus|hotel|inn|mall|market)\b", re.I)
STREET = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|pkwy|parkway|"
    r"hwy|highway|ct|court|pl|place|sq|square|ter|terrace|cir|circle|trl|trail|pike|"
    r"route|rte|interstate)\b\.?", re.I)
PHONE = re.compile(r"\(?\b\d{3}\)?[\s.\-]?\d{3}[\s.\-]\d{4}\b")
ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
YEAR = re.compile(r"^(?:19|20)\d{2}$")
ACRONYM = re.compile(r"^[A-Z]{2,4}$")
URL = re.compile(r"(www\.|\.com\b|\.org\b|\.net\b|\.edu\b|\.gov\b|@)", re.I)

GENERIC = {
    "congratulations", "class", "of", "the", "and", "open", "closed", "welcome", "sale",
    "exit", "stop", "enter", "entrance", "push", "pull", "no", "parking", "one", "way",
    "only", "free", "hours", "today", "now", "new", "hot", "cold", "menu", "coffee",
    "bar", "grill", "cafe", "restaurant", "shop", "store", "food", "drinks", "beer",
    "wine", "pizza", "eat", "sale", "off", "lunch", "dinner", "breakfast", "happy",
    "hour", "special", "specials", "thank", "you", "for", "to", "in", "on", "at", "a",
    "an", "is", "we", "are", "our", "your", "with", "danger", "caution", "warning",
    "pedestrian", "crossing", "slow", "yield", "speed", "limit", "mph", "do", "not",
    "telephone", "phone", "taxi", "bus", "restrooms", "restroom", "atm", "toilets",
    "information", "parking", "garage", "fire", "hydrant", "loading", "zone", "towing",
    "tow", "away", "private", "property", "keep", "clear", "emergency", "shop", "shoppe",
}


def _near(a: str, b: str) -> bool:
    """Within one edit: OCR slips like TELEPHONM for TELEPHONE."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
        else:
            diff += 1
            if diff > 1:
                return False
        j += 1
    return True


GARBLED = re.compile(r"^[a-z]{2,}([A-Z]+[a-z]|[A-Z]{2,}$)")  # "choRe", "shOPPE": case flips mid-word


def looks_garbled(text: str) -> bool:
    """OCR misreads often mix case oddly. Real names rarely start with two
    lowercase letters and then switch to capitals (eBay, iPhone, McDonald
    are fine)."""
    words = re.findall(r"[A-Za-z]+", text)
    return any(GARBLED.match(w) for w in words) and len(words) <= 2


def is_generic(word: str) -> bool:
    w = word.lower()
    return w in GENERIC or (len(w) >= 6 and any(len(g) >= 6 and _near(w, g) for g in GENERIC))


@dataclass
class TextIntel:
    kind: str
    weight: float
    reason: str


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def clean_query(text: str) -> str:
    """Drop words that only add noise to a search ("CONGRATULATIONS", "2024")."""
    kept = [w for w in re.findall(r"[A-Za-z0-9&'.-]+", text.replace("_", " "))
            if not is_generic(w.strip(".'")) and not YEAR.match(w)]
    return " ".join(kept)


def classify(text: str) -> TextIntel:
    t = " ".join(text.replace("_", " ").split())
    low = t.lower()
    words = _words(t)
    joined = " ".join(words)

    if PHONE.search(t):
        return TextIntel("phone", 3.5, "Phone number: the area code alone reveals the region.")
    if URL.search(t):
        return TextIntel("web", 3.0, "Website or handle: one search finds the business and its address.")
    if any(re.search(rf"\b{re.escape(c)}\b", joined) for c in US_CITIES):
        return TextIntel("place", 4.0, "Names a city outright.")
    if any(re.search(rf"\b{re.escape(s)}\b", joined) for s in US_STATES):
        return TextIntel("place", 3.0, "Names a state.")
    if STREET.search(t) and len(words) >= 2:
        return TextIntel("street", 3.5, "Street name: narrows the photo to one road.")
    if INSTITUTIONS.search(low):
        return TextIntel("institution", 3.0, "Names an institution or public place you can look up.")
    if ZIP.search(t):
        return TextIntel("zip", 3.5, "ZIP code: pins the photo to a neighborhood.")

    if looks_garbled(t):
        return TextIntel("garbled", 0.3, "Unclear reading; probably an OCR error.")

    meaningful = [w for w in words if not is_generic(w) and not YEAR.match(w)]
    if not meaningful:
        return TextIntel("generic", 0.4, "Generic text; reveals little on its own.")
    if all(w.isdigit() for w in meaningful):
        if len(meaningful) == 1 and len(meaningful[0]) <= 5:
            return TextIntel("number", 1.5, "Could be an address number.")
        return TextIntel("generic", 0.6, "Numbers without context; reveal little.")
    if any(ACRONYM.match(w) for w in t.split()) and len(meaningful) <= 2:
        return TextIntel("acronym", 2.0, "Acronyms often name a local school, team or business.")
    return TextIntel("name", 2.5, "Business or place name: one search can find it.")
