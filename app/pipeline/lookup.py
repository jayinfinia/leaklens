"""Opt-in: search sign text on OpenStreetMap, the way an investigator would.

Vision models don't read. A person who sees "Poppy & Parliament" or
"PARAMOUNT" on a sign just searches it, and adds the region the scene
suggests. This layer does the same with Nominatim (OpenStreetMap's free
geocoder): first near the city the AI model guessed (a ~30 km box), then
"<text>, <region the models guessed>", then the text alone.

Search engines return *something* for almost any text, so every result is
checked before it counts:
  - its name (or alt/old/brand/official name) must contain the sign's
    distinctive words; a "Poppy" café on "Parliament Street" is not
    "POPPY & PARLIAMENT" (words like "SALOON" or an address on the same
    sign may be missing from the name);
  - it must sit in a US state that other evidence points to (StreetCLIP's
    region, the AI model's guesses, a phone number's area code, a state
    named on a sign) to be called strong without further corroboration;
  - chains and brands ("Budweiser", "Yellow Cab"), names that many far-apart
    places share and common street names ("16th St N") are never strong on
    their own: only an address on the sign that matches OSM's, another sign
    nearby, or (for streets) one clear state can make them strong;
  - a phone number's area code outweighs the AI model's guess: a match in
    another state is only possible;
  - a city and state printed on a sign ("NASHVILLE TENNESSEE") are a strong
    clue by themselves (offline, from the bundled city list) and beat a match
    elsewhere.

One sign is often ambiguous ("POPPY" matches businesses in many cities),
so, like an investigator, we use the clues together:
  1. co-location: if two different signs in the photo both have a
     candidate within ~300 m of each other, that spot is the answer;
  2. nearby names: for each candidate of an ambiguous sign, ask Overpass
     (OpenStreetMap's query API) whether any *other* sign's text names a
     place within ~300 m.

Off by default because it sends detected text (never the photo, never phone
numbers) off this machine. A house number and street read on a sign
("1000 OSAGE") is searched only inside a ~1 km box around a candidate place. Follows Nominatim's usage policy
(https://operations.osmfoundation.org/policies/nominatim/): identifying
User-Agent, at most one request per second, results cached, a hard cap on
requests per photo.
"""

import json
import math
import re
import time
import urllib.parse
import urllib.request
from functools import lru_cache

from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..schemas import Clue, Geolocation, TextMatch
from . import gazetteer as gz
from .text_intel import (STATE_NAMES, STREET, US_CITIES, classify, clean_query, is_generic, phone_places,
                         state_code, states_in, strip_phones)

UA = "LeakLens/0.3 (photo privacy checker; https://github.com/jayinfinia/leaklens)"
AMBIGUOUS_KM = 50
SEARCHABLE = {"name", "institution", "street", "place", "acronym", "web", "zip"}
NAME_KEYS = ("name", "alt_name", "old_name", "brand", "official_name", "short_name")
_MIN_INTERVAL = 1.1  # seconds between Nominatim requests
_last_call = 0.0


# ------------------------------------------------------------ Nominatim
# A hit is a tuple: (display_name, lat, lon, type, importance, names, state,
# country, road, house_number, branded). Tests (and older callers) may pass
# only the first 4 or 5 fields; the helpers below fall back to the display
# name for the rest.
def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read())


def _hit(d: dict) -> tuple | None:
    try:
        lat, lon = float(d["lat"]), float(d["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    names = []
    if d.get("name"):
        names.append(d["name"])
    for k, v in (d.get("namedetails") or {}).items():
        if v and k.split(":")[0] in NAME_KEYS and v not in names:
            names.append(v)
    addr = d.get("address") or {}
    country = (addr.get("country_code") or "").lower()
    iso = addr.get("ISO3166-2-lvl4") or ""
    state = iso[3:] if iso.startswith("US-") else (state_code(addr.get("state", "")) if country == "us" else "")
    road = addr.get("road") or addr.get("pedestrian") or addr.get("footway") or ""
    branded = bool((d.get("namedetails") or {}).get("brand"))
    return (d.get("display_name", ""), lat, lon, d.get("type", ""), float(d.get("importance") or 0.0),
            tuple(names), state, country, road, addr.get("house_number") or "", branded)


@lru_cache(maxsize=512)
def _nominatim(params: tuple) -> tuple:
    """One rate-limited, cached Nominatim /search call. `params` is a tuple of
    (key, value) pairs so it can be a cache key."""
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    q = {k: v for k, v in params if v not in ("", None)}
    q.update(format="jsonv2", limit=10, addressdetails=1, namedetails=1)
    try:
        data = _get_json(f"{config.NOMINATIM_URL}?{urllib.parse.urlencode(q)}")
    except Exception as exc:
        print(f"[leaklens] lookup failed for {dict(params)}: {exc}")
        return ()
    finally:
        _last_call = time.monotonic()
    return tuple(h for h in (_hit(d) for d in (data or []) if isinstance(d, dict)) if h)


def _search(query: str, countrycodes: str = "") -> tuple:
    """Free-text search."""
    return _nominatim((("q", query), ("countrycodes", countrycodes)))


def _search_bounded(query: str, countrycodes: str, viewbox: str) -> tuple:
    """Free-text search restricted to a box (lon1,lat1,lon2,lat2)."""
    return _nominatim((("q", query), ("countrycodes", countrycodes), ("viewbox", viewbox), ("bounded", 1)))


def _search_structured(name: str, city: str, state: str, countrycodes: str) -> tuple:
    """Structured search: a named place ("amenity" in Nominatim's terms) in a city."""
    return _nominatim((("amenity", name), ("city", city), ("state", state), ("countrycodes", countrycodes)))


def _names(h) -> tuple:
    if len(h) > 5:
        return h[5]
    return (h[0].split(",")[0],)  # legacy hit: the display name starts with the name


def _state(h) -> str:
    if len(h) > 6:
        return h[6]
    for part in h[0].split(",")[1:]:
        code = state_code(part.strip())
        if code and len(part.strip()) > 2:
            return code
    return ""


def _country(h) -> str:
    if len(h) > 7:
        return h[7]
    return "us" if _state(h) else ""


def _road(h) -> str:
    return h[8] if len(h) > 8 else ""


def _housenumber(h) -> str:
    return h[9] if len(h) > 9 else ""


# ------------------------------------------------------------ name check
_ABBREV = {"st": "street", "ave": "avenue", "av": "avenue", "rd": "road", "blvd": "boulevard",
           "dr": "drive", "ln": "lane", "pkwy": "parkway", "hwy": "highway", "ct": "court",
           "pl": "place", "sq": "square", "mt": "mount", "ft": "fort", "ctr": "center"}
_STOP = {"and", "the", "of", "a", "an", "at", "on", "in"}


def norm_name(text: str) -> str:
    """'The Poppy & Parliament' -> 'poppy and parliament'; 'Main St.' -> 'main street'."""
    t = text.lower().replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    words = re.sub(r"[^a-z0-9]+", " ", t).split()
    if words and words[0] == "the":
        words = words[1:]
    return " ".join(_ABBREV.get(w, w) for w in words)


def _distinctive(text: str) -> list[str]:
    words = [w for w in norm_name(strip_addresses(strip_phones(text))).split() if w not in _STOP]
    strong = [w for w in words if len(w) >= 3 and not is_generic(w)]
    return strong or words


def _token_found(tok: str, name_words: list[str]) -> bool:
    from rapidfuzz import fuzz

    if len(tok) <= 4:  # short words: exact (no "illy" in "Lilly's")
        return tok in name_words
    return any(fuzz.ratio(tok, w) >= config.NAME_TOKEN_MIN for w in name_words)


def name_score(text: str, name: str) -> float:
    """How well a place's name matches the sign, 0-100. 0 unless the sign's
    distinctive words are in the name (one may be missing when there are 3+).
    Words that only describe the business ("SALOON", "CO.") may be missing,
    and an address on the sign ("1000 OSAGE") is not part of the name."""
    from rapidfuzz import fuzz

    toks, n = _distinctive(text), norm_name(name)
    if not toks or not n:
        return 0.0
    words = n.split()
    need = [t for t in toks if t not in DESCRIPTORS] or toks
    missing = sum(not _token_found(t, words) for t in need)
    if missing and (len(need) < 3 or missing > 1):
        return 0.0
    q = " ".join(t for t in toks if _token_found(t, words)) or " ".join(toks)
    return float(max(fuzz.token_set_ratio(q, n), fuzz.partial_ratio(q, n)))


def verify(text: str, hits: tuple) -> tuple:
    """Only the hits whose own name matches the sign. Matching through the
    address ("Poppy" on "Parliament Street") doesn't count."""
    return tuple(h for h in hits
                 if max((name_score(text, nm) for nm in _names(h)), default=0.0) >= config.NAME_MATCH_MIN)


# ------------------------------------------------------------ sign anatomy
# Words that describe a business rather than name it: optional in the name
# check and dropped from the ends of a query ("SALOON BUCKHORN EXCHANGE
# STEAKHOUSE" -> "BUCKHORN EXCHANGE").
DESCRIPTORS = {
    "saloon", "steakhouse", "steak", "tavern", "pub", "diner", "bistro", "eatery", "kitchen", "brewery",
    "brewing", "taproom", "lounge", "bakery", "deli", "cantina", "grill", "grille", "bar", "cafe",
    "restaurant", "co", "company", "inc", "llc", "corp", "corporation", "ltd", "est", "established",
    "since", "the",
}
_SUFFIXES = {"street", "avenue", "road", "boulevard", "drive", "lane", "way", "place", "court", "parkway",
             "highway", "terrace", "circle", "trail", "pike", "square", "av", "ter", "cir", "trl"}
_DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
               "northeast", "northwest", "southeast", "southwest"}
# "1000 OSAGE", "215 E Main St": a house number and a street on a sign.
_ADDR = re.compile(r"\b(\d{1,5})\s+((?:[NSEW]\.?\s+)?[A-Za-z]{3,}(?:\s+(?:st|street|ave|avenue|av|rd|road|"
                   r"blvd|boulevard|dr|drive|ln|lane|way|pl|place|ct|court|pkwy|parkway|hwy|highway|ter|"
                   r"terrace|cir|circle)\b\.?)?)", re.I)
_NOT_STREET = {"years", "year", "yrs", "miles", "mile", "feet", "foot", "off", "percent", "calories", "people",
               "days", "day", "hours", "hrs", "min", "mins", "minutes", "lbs", "pack", "for", "and", "cents",
               "dollars", "usd", "acres", "units", "rooms", "seats", "beds", "baths", "spaces", "only"}
# Street names most towns have: never strong alone.
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)?$")
COMMON_STREETS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "main", "lake", "park", "oak", "pine", "maple", "elm", "cedar", "walnut",
    "cherry", "chestnut", "hickory", "dogwood", "magnolia", "birch", "spruce", "willow", "poplar", "locust",
    "washington", "lincoln", "jefferson", "madison", "franklin", "jackson", "adams", "monroe", "church",
    "high", "center", "centre", "market", "water", "mill", "spring", "ridge", "hill", "river", "union",
    "broad", "broadway", "front", "college", "school", "state", "prospect", "pleasant", "liberty", "green",
    "valley", "forest", "highland", "meadow", "sunset", "railroad", "bridge", "grove", "view", "lakeview",
    "university", "commerce", "north", "south", "east", "west", "central", "court", "cross", "academy",
}


def _street_core(street: str) -> str:
    """'E Lake St' -> 'lake'; 'Osage Street' -> 'osage'."""
    words = norm_name(street).split()
    return " ".join(w for w in words if w not in _SUFFIXES and w not in _DIRECTIONS)


def _addresses(text: str) -> list[tuple[str, str, str, tuple[int, int]]]:
    """(house number, street core, as read, span) for each address on a sign."""
    out = []
    for m in _ADDR.finditer(text):
        core = _street_core(m.group(2))
        first = core.split()[0] if core else ""
        if not first or first in _NOT_STREET or first in DESCRIPTORS or is_generic(first):
            continue
        out.append((m.group(1), core, f"{m.group(1)} {m.group(2).strip()}", m.span()))
    return out


def strip_addresses(text: str) -> str:
    """'1000 OSAGE • BUCKHORN EXCHANGE' -> '• BUCKHORN EXCHANGE'."""
    for *_, (a, b) in reversed(_addresses(text)):
        text = text[:a] + " " + text[b:]
    return " ".join(text.split())


def sign_addresses(clues: list[Clue], ai=None) -> list[tuple[str, str, str]]:
    """Addresses read anywhere in the photo, the ones most readings agree on
    first: (house number, street core, as read)."""
    from rapidfuzz import fuzz

    texts = [c.label for c in clues if c.kind == "text"] + list(ai.sign_texts if ai else [])
    groups: list[tuple[str, dict]] = []  # (number, {(core, as read): count})
    for t in texts:
        for num, core, read, _ in _addresses(strip_phones(t)):
            g = next((g for n, g in groups if n == num and any(fuzz.ratio(c, core) >= 80 for c, _ in g)), None)
            if g is None:
                groups.append((num, g := {}))
            g[(core, read)] = g.get((core, read), 0) + 1
    out = []
    for num, g in sorted(groups, key=lambda x: -sum(x[1].values())):
        (core, read), _ = max(g.items(), key=lambda kv: kv[1])  # the reading most engines agree on
        out.append((num, core, read))
    return out


def _street_like(a: str, b: str) -> bool:
    from rapidfuzz import fuzz

    return bool(a and b) and (fuzz.ratio(a, b) >= 80 or a in b.split() or b in a.split())


def _addr_agrees(h, addrs) -> str:
    """The sign's address that matches the hit's OSM address ('' if none)."""
    m = re.match(r"\d+", _housenumber(h) or "")
    road = _street_core(_road(h))
    if not m or not road:
        return ""
    return next((read for num, core, read in addrs if num == m.group() and _street_like(core, road)), "")


def core_name(text: str) -> str:
    """The part of a sign that names the place: no phone numbers, no address,
    no descriptive words at either end. Falls back to the whole reading."""
    words = strip_addresses(strip_phones(text)).split()

    def key(w):
        return re.sub(r"[^a-z0-9]", "", w.lower())

    while words and (key(words[0]) in DESCRIPTORS or not key(words[0])):
        words.pop(0)
    while words and (key(words[-1]) in DESCRIPTORS or not key(words[-1])):
        words.pop()
    if any(len(key(w)) >= 3 and not is_generic(key(w)) for w in words):
        return " ".join(words)
    return strip_phones(text)


BRANDS_FILE = Path(__file__).resolve().parent.parent / "data" / "brands.txt"


@lru_cache(maxsize=1)
def brands() -> frozenset:
    """US chain and brand names (name-suggestion-index, BSD-3), normalized."""
    try:
        lines = BRANDS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    names = (norm_name(line) for line in lines if line.strip() and not line.startswith("#"))
    return frozenset(n for n in names if len(n.replace(" ", "")) >= 3)


def brand_of(text: str) -> str:
    """The chain or brand a sign names ('' if none). One OCR slip is allowed
    in names of 7+ letters ("Budweisen")."""
    core = norm_name(core_name(text))
    if not core:
        return ""
    plain = " ".join(w for w in core.split() if not is_generic(w))  # "starbucks coffee" -> "starbucks"
    for name in (core, plain):
        if name and name in brands():
            return name
    if len(core.replace(" ", "")) >= 7:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein

        best = process.extractOne(core, brands(), scorer=Levenshtein.distance, score_cutoff=1)
        if best is not None:
            return best[0]
    return ""


def _branded(h) -> bool:
    """OSM tags the place with a brand, or its name is a known brand."""
    return (len(h) > 10 and bool(h[10])) or any(norm_name(n) in brands() for n in _names(h))


def generic_street(text: str) -> bool:
    """A street name most towns have: numbered ("16th St N") or common ("E Lake St")."""
    if not STREET.search(text):
        return False
    words = _street_core(strip_phones(text)).split()
    return not words or all(_ORDINAL.match(w) or w in COMMON_STREETS for w in words)


def _places_count(hits: tuple) -> tuple[int, int]:
    """Separate places (more than AMBIGUOUS_KM apart) and states among hits."""
    centers: list = []
    for h in hits:
        if not any(_km(h[1], h[2], c[1], c[2]) <= AMBIGUOUS_KM for c in centers):
            centers.append(h)
    return len(centers), len({_state(h) for h in hits} - {""})


def chainlike(hits: tuple) -> bool:
    places, states = _places_count(hits)
    return places >= config.CHAIN_MIN_PLACES and states >= 2


# ------------------------------------------------------------ evidence
# Cities that are also everyday words or names: never a place clue alone.
_CITY_WORDS = {"mobile", "jackson", "austin", "charlotte", "madison", "lincoln", "savannah", "aurora",
               "independence", "florence", "phoenix", "columbus", "arlington", "lexington", "richmond"}


@dataclass
class Evidence:
    """What the photo says about where it is, before any map search."""
    prior: dict = field(default_factory=dict)       # state -> weight, all evidence
    indep: dict = field(default_factory=dict)       # the same without the AI model's guesses
    places: list = field(default_factory=list)      # "City, State" from signs and phone numbers
    phone_states: set = field(default_factory=set)
    text_states: set = field(default_factory=set)   # states named on signs


def _city_alone(m) -> bool:
    return (m.city is not None and not m.paired and m.city.name.lower() in US_CITIES
            and m.city.name.lower() not in _CITY_WORDS)


def evidence(geo: Geolocation, ai=None, clues: list[Clue] | None = None) -> Evidence:
    ev = Evidence()

    def add(code, w, independent=True):
        if code:
            ev.prior[code] = ev.prior.get(code, 0.0) + w
            if independent:
                ev.indep[code] = ev.indep.get(code, 0.0) + w

    for r in geo.region:
        add(state_code(r.name), r.prob)
    for g in (ai.guesses if ai else []):
        if g.confidence >= 0.2:
            found = states_in(g.place)
            add(found[0] if found else "", g.confidence, independent=False)
    texts = [c.label for c in (clues or []) if c.kind == "text"] + list(ai.sign_texts if ai else [])
    signed, phoned = [], []
    for t in dict.fromkeys(texts):
        for city, code in phone_places(t):  # offline; the number is never searched
            add(code, config.PRIOR_PHONE)
            ev.phone_states.add(code)
            if city:
                phoned.append(f"{city}, {STATE_NAMES[code]}")
        named = set(states_in(t)) if classify(t).kind == "place" else set()
        alone = set()
        for m in gz.named_places(t):
            if m.paired or m.city is None:
                named.add(m.state)
                if m.paired:
                    signed.append(f"{m.city.name}, {STATE_NAMES.get(m.state, m.state)}")
            elif _city_alone(m):
                alone.add(m.state)
        for code in named:
            add(code, config.PRIOR_STATE_TEXT)
            ev.text_states.add(code)
        for code in alone - named:
            add(code, config.PRIOR_CITY_TEXT)
    ev.places = list(dict.fromkeys(signed + phoned))
    return ev


def build_prior(geo: Geolocation, ai=None, clues: list[Clue] | None = None) -> tuple[dict, list[str]]:
    """US states the photo's evidence points to ({code: weight}), and
    "City, State" hints from that evidence (signs, phone exchanges)."""
    ev = evidence(geo, ai, clues)
    return ev.prior, ev.places


def _supported(h, prior: dict):
    """True: the hit's state has evidence. False: it doesn't (or it's outside
    the US when the evidence says US). None: nothing to judge by."""
    if not prior:
        return None
    st, cc = _state(h), _country(h)
    if st:
        return prior.get(st, 0.0) >= config.LOOKUP_PRIOR_MIN
    if cc and cc != "us":
        return False
    return None


def _where(h) -> str:
    st = _state(h)
    return STATE_NAMES.get(st, st) if st else "a place"


# ------------------------------------------------------------ which text to search
def _searchable(c: Clue) -> bool:
    if c.kind != "text" or c.category.split(":")[-1] not in SEARCHABLE:
        return False
    words = clean_query(strip_phones(c.label)).split()
    # one short word ("illy", "choR") matches too much by chance
    return len(words) >= 2 or sum(ch.isalnum() for ch in "".join(words)) >= 5


def _candidates(clues: list[Clue]) -> list[Clue]:
    texts = [c for c in clues if _searchable(c)]
    return sorted(texts, key=lambda c: c.weight * c.score, reverse=True)


def _messy(text: str) -> int:
    """Words that look misread ("KexcHaNG", "55T"): clean readings are searched first."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    return sum(bool(re.search(r"[a-z]{2}[A-Z]", w) or re.fullmatch(r"\d+[A-Za-z]+", w)
                    and not _ORDINAL.match(w.lower())) for w in words)


def _agrees(a: str, b: str) -> bool:
    from rapidfuzz import fuzz

    a, b = norm_name(a), norm_name(b)
    return bool(a and b) and max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b)) >= config.READING_AGREE_MIN


def agreed(candidates: list[Clue], clues: list[Clue], ai=None) -> list[Clue]:
    """With more than one reader (OCR and the AI model), search only readings
    two of them agree on: a reading only one engine saw ("Pa-pints") is
    usually a misreading or a hallucination. With a single reader, keep all."""
    if ai is None or not ai.sign_texts:
        return candidates
    ocr = [c for c in clues if c.kind == "text" and c.box is not None]
    out = []
    for c in candidates:
        sources = {s for s in c.source.split("+") if s}
        if c.box is not None:
            sources.add("ocr")
        if "ai" not in sources and any(_agrees(c.label, t) for t in ai.sign_texts):
            sources.add("ai")
        if "ocr" not in sources and any(_agrees(c.label, o.label) for o in ocr if o.id != c.id):
            sources.add("ocr")
        if len(sources) >= 2:
            out.append(c)
    return out


# ------------------------------------------------------------ queries
def _place_hint(place: str) -> str:
    """'Huntsville, Alabama, USA' -> 'Huntsville, Alabama' (country is implied by countrycodes)."""
    parts = [p.strip() for p in place.split(",") if p.strip()]
    if len(parts) >= 3 and parts[-1].lower() in ("usa", "us", "united states", "united states of america"):
        parts = parts[:-1]
    return ", ".join(parts[:2])


def variants(text: str) -> list[str]:
    """Spellings to try: the sign's name part (no address, no "SALOON"), with
    '&' as 'and', and without it. clean_query already drops a leading 'The'
    and a spelled-out 'and' (OSM names often have them; Nominatim only needs
    the query's words to be in the name)."""
    base = clean_query(core_name(text))
    out = [base]
    if "&" in base:
        out += [" ".join(base.replace("&", " and ").split()), " ".join(base.replace("&", " ").split())]
    return [v for v in dict.fromkeys(out) if len(v) >= 3]


def _viewbox(lat: float, lon: float, km: float) -> str:
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return f"{lon - dlon:.4f},{lat + dlat:.4f},{lon + dlon:.4f},{lat - dlat:.4f}"


def _queries(text: str, geo: Geolocation, ai_places: list[str] | None = None,
             points: list[tuple[str, float, float]] | None = None,
             states: list[str] | None = None) -> list[tuple]:
    """(kind, args, label, hinted) plans, most specific first: near the
    likeliest city (from a sign, a phone number or the AI model), structured
    city search, '<text>, <hint>', '<text>, <state a sign or phone names>',
    '<text>, <StreetCLIP's state>', text alone."""
    vs = variants(text)
    if not vs:
        return []
    plans = []
    for i, (label, lat, lon) in enumerate((points or [])[:2]):
        box = _viewbox(lat, lon, config.LOOKUP_VIEWBOX_KM)
        for v in (vs if i == 0 else vs[:1]):
            plans.append(("bounded", (v, box), f"{v} (within {config.LOOKUP_VIEWBOX_KM:.0f} km of "
                                                  f"{_place_hint(label) or label})", True))
    hints = [h for h in dict.fromkeys(_place_hint(p) for p in (ai_places or [])) if h]
    for h in hints[:2]:
        city, _, st = h.partition(",")
        if st.strip() and state_code(st.strip()):
            plans.append(("structured", (vs[-1], city.strip(), st.strip()), f"{vs[-1]} in {h}", True))
            break
    for h in hints[:1]:
        plans.append(("text", (f"{vs[0]}, {h}",), f"{vs[0]}, {h}", True))
    for st in (states or [])[:1]:
        if not hints or st not in hints[0]:
            plans.append(("text", (f"{vs[0]}, {st}",), f"{vs[0]}, {st}", True))
    if geo.region:
        plans.append(("text", (f"{vs[0]}, {geo.region[0].name}",), f"{vs[0]}, {geo.region[0].name}", True))
    elif geo.country and geo.country[0].prob >= 0.3:
        plans.append(("text", (f"{vs[0]}, {geo.country[0].name}",), f"{vs[0]}, {geo.country[0].name}", True))
    plans += [("text", (v,), v, False) for v in vs[:2]]
    seen, out = set(), []
    for p in plans:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            out.append(p)
    return out


def _run(kind: str, args: tuple, countrycodes: str) -> tuple:
    if kind == "bounded":
        return _search_bounded(args[0], countrycodes, args[1])
    if kind == "structured":
        return _search_structured(*args, countrycodes)
    return _search(args[0], countrycodes)


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def _spread_km(hits: tuple) -> float:
    return max((_km(a[1], a[2], b[1], b[2]) for a in hits for b in hits), default=0.0)


def _importance(h) -> float:
    return h[4] if len(h) > 4 else 0.0


def _dominant(hits: tuple) -> bool:
    """OpenStreetMap ranks the first result far above the others (a famous
    theater vs. a road with the same name)."""
    top = _importance(hits[0])
    second = max((_importance(h) for h in hits[1:]), default=0.0)
    return top >= 0.3 and top >= 1.5 * second


def _judge_spread(text: str, hits: tuple, hinted: bool, geo: Geolocation):
    """Strong: every result sits in one place (within 50 km), and either the
    search was narrowed by a region hint or the text is specific (2+
    meaningful words). Everything else is a possible match: a single common
    word like "POPPY" can match businesses in many cities, and we must not
    let the model's (possibly wrong) pin pick one for us."""
    spread = _spread_km(hits)
    together = spread <= AMBIGUOUS_KM
    specific = len(clean_query(text).split()) >= 2
    if together and (hinted or specific):
        if len(hits) > 1 and geo.points:  # same area: the model's pin can break ties
            pin = geo.points[0]
            best = min(hits, key=lambda h: _km(h[1], h[2], pin.lat, pin.lon))
        else:
            best = hits[0]
        return best, "strong", ""
    if not together:
        n = f"{len(hits)}+" if len(hits) >= 5 else str(len(hits))
        if hinted and _dominant(hits):
            return hits[0], "strong", (f"The best-known match in that area ({n} places share the name; "
                                       "OpenStreetMap ranks this one far above the rest).")
        # Never let the vision model's pin choose between far-apart places: it can be
        # confidently wrong. Keep OpenStreetMap's own ranking.
        return hits[0], "weak", f"{n} places share this name, up to {spread:,.0f} km apart."
    return hits[0], "weak", "A single word: could be a coincidence."


def judge(text: str, hits: tuple, hinted: bool, geo: Geolocation, prior: dict | None = None):
    """Pick a (name-verified) hit and decide how much to trust it.

    On top of the spread rules above, the state must agree with the photo's
    other evidence: a match in a state nothing else points to is only a
    possible match (corroboration can still upgrade it), and a search that
    wasn't narrowed by any hint is never strong on its own unless its state
    has support.
    """
    prior = prior or {}
    support = [_supported(h, prior) for h in hits]
    backed = tuple(h for h, s in zip(hits, support) if s is not False)
    pool = backed or hits
    best, confidence, note = _judge_spread(text, pool, hinted, geo)
    if pool is not hits and len(pool) < len(hits) and confidence == "strong" and not note:
        note = (f"{len(hits) - len(pool)} other place(s) share the name, but only this one is "
                "where the rest of the photo points.")
    if confidence == "strong":
        s = _supported(best, prior)
        if s is False:
            confidence, note = "weak", (f"In {_where(best)}, which nothing else in the photo points to.")
        elif not hinted and s is not True:
            confidence, note = "weak", ("Found by name alone; nothing else in the photo points to "
                                        "this area yet.")
    elif not backed and note:
        note += " None is in a state the photo points to."
    return best, confidence, note


# ------------------------------------------------------------ corroboration
STREET_SUFFIX = {
    "st": "st(reet)?", "street": "st(reet)?", "ave": "av(enue|e)?", "avenue": "av(enue|e)?",
    "rd": "r(oa)?d", "road": "r(oa)?d", "blvd": "b(oulevar)?d", "dr": "dr(ive)?",
    "drive": "dr(ive)?", "ln": "l(a)?ne?", "lane": "l(a)?ne?", "pkwy": "p(ar)?kw(a)?y",
}


def term_regex(text: str) -> str | None:
    """Loose, case-insensitive OSM name pattern for a sign's text.

    Only letters and digits survive (no escaping problems); words may be
    separated by spaces, '&', 'and' or punctuation; whole words only.
    """
    words = [w.lower() for w in clean_query(text).split() if any(ch.isalnum() for ch in w)]
    words = ["".join(ch for ch in w if ch.isalnum()) for w in words]
    words = [w for w in words if w]
    if not words or sum(len(w) for w in words) < 4:
        return None
    parts = words[:-1] + [STREET_SUFFIX.get(words[-1], words[-1])]
    return "(^|[^a-z0-9])" + "[^a-z0-9]{1,6}(and[^a-z0-9]{1,3})?".join(parts) + "([^a-z0-9]|$)"


def _overpass(points: list[tuple[float, float]], regexes: list[str]) -> list[tuple[str, float, float]]:
    """Named OSM features matching any regex within NEARBY_M of any point."""
    pattern = "|".join(f"({r})" for r in regexes)
    clauses = "".join(f'nwr(around:{config.NEARBY_M},{lat:.6f},{lon:.6f})["name"~"{pattern}",i];'
                      for lat, lon in points)
    query = f"[out:json][timeout:20];({clauses});out center 60;"
    req = urllib.request.Request(config.OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"[leaklens] overpass failed: {exc}")
        return []
    out = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        name = (el.get("tags") or {}).get("name")
        if lat is not None and lon is not None and name:
            out.append((name, float(lat), float(lon)))
    return out


def _closest_pair(a: tuple, b: tuple):
    best = None
    for ha in a:
        for hb in b:
            d = _km(ha[1], ha[2], hb[1], hb[2])
            if best is None or d < best[2]:
                best = (ha, hb, d)
    return best


def _short(name: str) -> str:
    return name.split(",")[0]


def _upgrade(m: TextMatch, hit, note: str) -> TextMatch:
    name, lat, lon, kind = hit[:4]
    return m.model_copy(update={"name": name, "lat": lat, "lon": lon, "kind": kind,
                                "confidence": "strong", "note": note})


def corroborate(matches: dict, pools: dict, clues: list[Clue], use_overpass: bool = True,
                ai_points: list[tuple[str, float, float]] | None = None,
                feature_keys: list[str] | None = None, locked: set | None = None) -> dict:
    """Upgrade weak matches that other evidence in the same photo agrees with.
    `locked` matches (chains, common street names, a phone number elsewhere)
    can only be upgraded by another sign next to them, never by the AI
    model's guess or by nearby objects."""
    near_km = config.NEARBY_M / 1000
    by_id = {c.id: c for c in clues}
    locked = locked or set()

    # 0. The local AI model's own guess lands near one of the candidates.
    for cid, m in list(matches.items()):
        if m.confidence == "strong" or not ai_points or cid in locked:
            continue
        best = None
        for place, alat, alon in ai_points:
            for h in pools[cid]:
                d = _km(h[1], h[2], alat, alon)
                if d <= config.AI_AGREE_KM and (best is None or d < best[0]):
                    best = (d, h, place)
        if best:
            matches[cid] = _upgrade(m, best[1], f"Agrees with the local AI model's guess ({best[2]}).")

    # 1. Two different signs with candidates next to each other.
    for cid, m in list(matches.items()):
        if m.confidence == "strong":
            continue
        for oid, ohits in pools.items():
            if oid == cid:
                continue
            pair = _closest_pair(pools[cid], ohits)
            if pair and pair[2] <= near_km:
                matches[cid] = _upgrade(m, pair[0], f"Next to {_short(pair[1][0])}, which matches "
                                                    f"“{by_id[oid].label}”, also in the photo.")
                break

    # 2. Ask OpenStreetMap what else is named near each candidate.
    if not use_overpass:
        return matches
    calls = 0
    for cid, m in list(matches.items()):
        if m.confidence == "strong" or calls >= 2 or len(pools[cid]) > 10:
            continue
        others = [r for r in (term_regex(c.label) for c in clues
                              if c.kind == "text" and c.id != cid
                              and c.category.split(":")[-1] not in ("generic", "number", "garbled")) if r]
        if not others:
            continue
        calls += 1
        found = _overpass([(h[1], h[2]) for h in pools[cid]], others)
        if not found:
            continue
        support = []
        for h in pools[cid]:
            names = sorted({n for n, lat, lon in found if _km(h[1], h[2], lat, lon) <= near_km
                            and n.lower() != _short(h[0]).lower()})
            support.append((len(names), h, names))
        n, hit, names = max(support, key=lambda x: x[0])
        if n:
            matches[cid] = _upgrade(m, hit, f"Nearby on the map: {', '.join(names[:3])}, "
                                            f"matching other text in the photo.")

    # 3. Bellingcat-style: which candidate has the photo's objects around it?
    if feature_keys:
        from . import features

        fcalls = 0
        for cid, m in list(matches.items()):
            if m.confidence == "strong" or fcalls >= 2 or len(pools[cid]) > 10 or cid in locked:
                continue
            fcalls += 1
            hits = pools[cid][: features.MAX_POINTS]
            scores = features.score_points([(h[1], h[2]) for h in hits], feature_keys)
            win = features.pick(scores)
            if win is not None:
                matches[cid] = _upgrade(m, hits[win], "Nearby on the map: "
                                        f"{features.labels(scores[win][1])}, also seen in the photo.")
    return matches


def geocode_guesses(ai, us: bool):
    """Put the AI model's top guesses on the map (2 lookups at most)."""
    if ai is None or not ai.guesses:
        return ai
    out = []
    for i, g in enumerate(ai.guesses):
        if i < 2 and not config.MOCK_MODE:
            hits = _search(g.place, "us" if us else "")
            if hits:
                g = g.model_copy(update={"lat": hits[0][1], "lon": hits[0][2]})
        elif config.MOCK_MODE and i == 0:
            g = g.model_copy(update={"lat": 38.0293, "lon": -78.4767})
        out.append(g)
    return ai.model_copy(update={"guesses": out})


# ------------------------------------------------------------ vetting a match
def _address_search(hits: tuple, addrs: list, run) -> tuple:
    """For candidates OSM has no address for, search the sign's "<number>
    <street>" in a small box around each (one per place, two at most) and
    keep the candidates the address lands on."""
    near_km = config.ADDRESS_NEAR_M / 1000
    num, _, read = addrs[0]
    seen: list = []
    for h in hits:
        if _housenumber(h) or any(_km(h[1], h[2], o[1], o[2]) <= AMBIGUOUS_KM for o in seen):
            continue
        if len(seen) >= config.LOOKUP_MAX_ADDRESS:
            break
        seen.append(h)
        found = run("bounded", (read, _viewbox(h[1], h[2], 1.0)))
        if found is None:
            break
        if any(_km(h[1], h[2], a[1], a[2]) <= near_km or _addr_agrees(a, addrs[:1]) for a in found):
            return (h,)
    return ()


def assess(text: str, hits: tuple, hinted: bool, geo: Geolocation, ev: Evidence, addrs: list,
           run) -> tuple:
    """judge() plus the checks that keep one sign from overriding everything
    else: (best hit, confidence, note, locked, address read)."""
    best, conf, note, lock, everywhere = _vet(text, hits, hinted, geo, ev, run)
    # A house number and street on the sign that match OSM's address settle it,
    # chain or not. If OSM has no address for the candidates, search it there.
    if addrs:
        pool = tuple(dict.fromkeys(hits + everywhere))
        agree = tuple(h for h in pool if _addr_agrees(h, addrs))
        if not agree and conf != "strong":
            agree = _address_search(pool, addrs, run)
        if agree:
            hit = agree[0]
            read = _addr_agrees(hit, addrs) or addrs[0][2]
            return hit, "strong", (f"The name and the address on the sign ({read}) match this place on "
                                   "OpenStreetMap."), False, read
    return best, conf, note, lock, ""


def _vet(text: str, hits: tuple, hinted: bool, geo: Geolocation, ev: Evidence, run) -> tuple:
    """(best, confidence, note, locked, every same-named place seen)."""
    best, conf, note = judge(text, hits, hinted, geo, ev.prior)
    st = _state(best)
    if brand_of(text) or _branded(best):
        why = (f"“{text}” names a chain or brand found in many places; it can't place the photo by itself "
               "unless another sign next to it agrees.")
        return best, "weak", (why + (" " + note if note and conf != "strong" else "")).strip(), True, ()
    generic = generic_street(text)
    if conf != "strong":
        return best, conf, note, generic, ()
    if generic:
        in_state = tuple(h for h in hits if _state(h) == st)
        if st and ev.indep.get(st, 0.0) >= config.GENERIC_STREET_PRIOR_MIN and _spread_km(in_state) <= AMBIGUOUS_KM:
            return best, "strong", (f"A common street name, but the only one found in {_where(best)}, "
                                    "where the rest of the photo points."), False, ()
        return best, "weak", ("A common street name that many towns have; it needs a cross street "
                              "or other evidence."), True, ()
    if ev.phone_states and st and st not in ev.phone_states and st not in ev.text_states:
        phones = " or ".join(sorted(STATE_NAMES.get(p, p) for p in ev.phone_states))
        return best, "weak", (f"A phone number in the photo has a{'n' if phones[0] in 'AEIOU' else ''} "
                              f"{phones} area code, but this place is in {_where(best)}."), True, ()
    # Is this one of many same-named places (a chain, a common name)?
    everywhere = hits
    if hinted:
        v = variants(text)
        found = run("text", (v[0],)) if v else ()
        if found is None:
            return best, "weak", ("Couldn't check whether other places share this name (search limit "
                                  "reached)."), False, ()
        everywhere = verify(text, found)
    if chainlike(everywhere):
        places, states = _places_count(everywhere)
        return best, "weak", (f"{places} places in {states} states share this name (a chain or a common "
                              "name), so it can't place the photo by itself."), True, everywhere
    return best, conf, note, False, everywhere


def place_matches(clues: list[Clue], ai=None) -> dict[str, TextMatch]:
    """City and state printed on a sign, matched offline. Strong when OCR read
    it (the AI model alone can echo its own guess); a well-known city named
    alone is a possible match."""
    out = {}
    for c in clues:
        if c.kind != "text":
            continue
        mentions = gz.named_places(c.label)
        paired = [m for m in mentions if m.paired and m.city is not None]
        alone = [m for m in mentions if _city_alone(m)]
        if not paired and not alone:
            continue
        m = (paired or alone)[0]
        located = c.box is not None or "ocr" in c.source.split("+")
        strong = bool(paired) and located
        state = STATE_NAMES.get(m.state, m.state)
        if paired:
            note = f"The sign names {m.city.name}, {state}." + ("" if located else
                                                               " Read only by the AI model, so not certain.")
        else:
            note = f"The sign names {m.city.name}; it may only be a name."
        out[c.id] = TextMatch(clue_id=c.id, text=c.label, query=f"{m.city.name}, {state} (offline city list)",
                              name=f"{m.city.name}, {state}, United States", lat=m.city.lat, lon=m.city.lon,
                              kind="city", confidence="strong" if strong else "weak", n_results=1, note=note)
    return out


def lookup(clues: list[Clue], geo: Geolocation, ai=None, feature_keys: list[str] | None = None) -> list[TextMatch]:
    if config.MOCK_MODE:
        return [TextMatch(clue_id=c.id, text=c.label, query=f"{c.label}, Virginia",
                          name=f"{c.label.title()}, Charlottesville, Virginia, United States",
                          lat=38.0313, lon=-78.4826, kind="road", confidence="strong")
                for c in _candidates(clues)[:1]]

    us = bool(geo.country) and geo.country[0].name == "United States" and geo.country[0].prob >= 0.5
    cc = "us" if us else ""
    ev = evidence(geo, ai, clues)
    # States the photo itself names (a sign, a phone number) outweigh the AI
    # model: its guesses elsewhere are not used to search or to corroborate.
    sure = ev.phone_states | ev.text_states

    def trusted(g) -> bool:
        found = states_in(g.place)
        return not sure or not found or found[0] in sure

    guesses = [g for g in (ai.guesses if ai else []) if g.confidence >= 0.2 and trusted(g)]
    local = [(p, gz.find(p)) for p in ev.places]
    points = ([(p, c.lat, c.lon) for p, c in local if c is not None] +
              [(g.place, g.lat, g.lon) for g in guesses if g.lat is not None and g.lon is not None])
    hint_places = ev.places + [g.place for g in guesses]
    states = [STATE_NAMES[s] for s in sorted(sure, key=lambda s: -ev.indep.get(s, 0.0)) if s in STATE_NAMES]
    ai_points = [(g.place, g.lat, g.lon) for g in (ai.guesses if ai else []) if g.lat is not None and trusted(g)]
    addrs = sign_addresses(clues, ai)

    budget = [config.LOOKUP_MAX_QUERIES]
    memo: dict = {}

    def run(kind, args):
        """One Nominatim call; the same call twice in a photo is free. None: no budget left."""
        if (kind, args) not in memo:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            memo[(kind, args)] = _run(kind, args, cc)
        return memo[(kind, args)]

    matches: dict[str, TextMatch] = {}
    pools: dict[str, tuple] = {}
    locked: set = set()
    addressed: set = set()
    searched: set = set()
    # The AI model doubles as a proofreader: text only one reader saw
    # ("Pa-pints") is usually a misreading, so don't search it.
    candidates = sorted(agreed(_candidates(clues), clues, ai), key=lambda c: _messy(c.label))
    for c in candidates[:5]:
        vs = variants(c.label)
        if not vs or vs[0].lower() in searched:
            continue
        searched.add(vs[0].lower())
        brand = bool(brand_of(c.label))
        per_text = config.LOOKUP_MAX_PER_TEXT
        for kind, args, label, hinted in _queries(c.label, geo, hint_places, points, states):
            if brand and not hinted:
                continue  # a chain searched nationwide returns an arbitrary 10 branches
            if per_text <= 0:
                break
            if (kind, args) not in memo:
                per_text -= 1
            found = run(kind, args)
            if found is None:
                break
            hits = verify(c.label, found)
            if not hits:
                continue
            best, confidence, note, lock, read = assess(c.label, hits, hinted, geo, ev, addrs, run)
            name, lat, lon, kind_ = best[:4]
            matches[c.id] = TextMatch(clue_id=c.id, text=c.label, query=label, name=name, lat=lat,
                                      lon=lon, kind=kind_, confidence=confidence,
                                      n_results=len(hits), note=note)
            pools[c.id] = hits
            if lock:
                locked.add(c.id)
            if read:
                addressed.add(c.id)
            break
    matches = corroborate(matches, pools, clues, ai_points=ai_points, feature_keys=feature_keys, locked=locked)
    return _with_places(matches, place_matches(clues, ai), addressed)


def _with_places(matches: dict, places: dict, addressed: set) -> list[TextMatch]:
    """Add city-and-state signs; a strong one outranks a match elsewhere."""
    near = 80.0
    for cid, pm in places.items():
        m = matches.get(cid)
        if m is None or (pm.confidence == "strong" and not (
                m.confidence == "strong" and _km(m.lat, m.lon, pm.lat, pm.lon) <= near)):
            matches[cid] = pm
    signs = [m for m in matches.values() if m.kind == "city" and m.confidence == "strong"]
    for cid, m in list(matches.items()):
        if m.confidence != "strong" or m.kind == "city" or cid in addressed or not signs:
            continue
        if all(_km(m.lat, m.lon, p.lat, p.lon) > near for p in signs):
            matches[cid] = m.model_copy(update={
                "confidence": "weak",
                "note": f"Another sign names {_short(signs[0].name)}, far from here. {m.note}".strip()})
    return sorted(matches.values(), key=lambda m: (m.confidence != "strong", m.kind == "city"))
