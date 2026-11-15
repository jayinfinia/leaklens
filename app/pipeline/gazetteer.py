"""Offline US place names: nearest city to a coordinate, and coordinates for
a "City, State" string. No network.

Data: app/data/us_cities.csv, the 2,620 US places with population >= 20,000
from GeoNames (cities15000, CC BY 4.0, via the MIT-licensed geonamescache
package). Columns: name, state (USPS code), lat, lon, population.
"""

import csv
import re
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "us_cities.csv"

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "Washington, D.C.",
}
_STATE_CODE = {v.lower(): k for k, v in STATES.items()} | {k.lower(): k for k in STATES} | {
    "d.c.": "DC", "dc": "DC", "district of columbia": "DC", "washington dc": "DC",
    "washington d.c.": "DC"}
_US = {"usa", "us", "u.s.", "u.s.a.", "united states", "united states of america", "america"}


@dataclass(frozen=True)
class City:
    name: str
    state: str  # USPS code
    lat: float
    lon: float
    population: int

    @property
    def label(self) -> str:
        return f"{self.name}, {self.state}"

    @property
    def state_name(self) -> str:
        return STATES.get(self.state, self.state)


def _key(name: str) -> str:
    s = name.lower().replace("saint ", "st ").replace("st. ", "st ").replace("fort ", "ft ")
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    return " ".join(s.split())


@lru_cache(maxsize=1)
def cities() -> tuple[City, ...]:
    with DATA.open(newline="") as f:
        return tuple(City(r["name"], r["state"], float(r["lat"]), float(r["lon"]), int(r["population"]))
                     for r in csv.DictReader(f))


@lru_cache(maxsize=1)
def _arrays():
    import numpy as np

    cs = cities()
    lat = np.radians([c.lat for c in cs])
    lon = np.radians([c.lon for c in cs])
    return lat, lon, np.cos(lat)


@lru_cache(maxsize=1)
def _by_name() -> dict[str, list[City]]:
    out: dict[str, list[City]] = {}
    for c in cities():  # most populous first (file order)
        keys = {_key(c.name)}
        if c.name.endswith(" City"):  # "New York City" -> "New York"
            keys.add(_key(c.name[:-5]))
        for k in keys:
            out.setdefault(k, []).append(c)
    return out


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def nearest(lat: float, lon: float) -> tuple[City, float]:
    """Closest listed city and its distance in km (vectorized, well under 1 ms)."""
    import numpy as np

    la, lo, cosla = _arrays()
    p, q = math.radians(lat), math.radians(lon)
    a = np.sin((la - p) / 2) ** 2 + math.cos(p) * cosla * np.sin((lo - q) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(1.0, a)))
    i = int(d.argmin())
    return cities()[i], float(d[i])


def state_at(lat: float, lon: float, max_km: float = 80.0) -> str:
    """USPS code of the state a point is (probably) in: that of the nearest
    listed city, or "" when nothing is close (outside the US)."""
    c, d = nearest(lat, lon)
    return c.state if d <= max_km else ""


def state_code(name: str) -> str:
    return _STATE_CODE.get(" ".join((name or "").lower().replace(",", " ").split()), "") or \
        _STATE_CODE.get((name or "").strip().lower(), "")


def find(place: str) -> City | None:
    """'Charlottesville, Virginia, USA' -> City. Tries each comma part as the
    city name, constrained by any state part. None for places outside the US
    or not in the list."""
    parts = [p.strip() for p in (place or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) > 1 and parts[-1].lower() not in _US and not state_code(parts[-1]):
        return None  # "Toronto, Ontario, Canada"
    state = ""
    for p in parts[1:]:
        code = state_code(p)
        if code:
            state = code
    if not state and parts[0].lower().startswith("washington") and "d.c" in place.lower():
        state = "DC"
    for p in parts:
        hits = _by_name().get(_key(p), [])
        if state:
            hits = [c for c in hits if c.state == state]
        if hits:
            return hits[0]
    return None


def state_center(code: str) -> City | None:
    """Population-weighted center of a state's listed cities, as a pseudo-city."""
    cs = [c for c in cities() if c.state == code]
    if not cs:
        return None
    w = sum(c.population for c in cs)
    lat = sum(c.lat * c.population for c in cs) / w
    lon = sum(c.lon * c.population for c in cs) / w
    return City(STATES.get(code, code), code, round(lat, 4), round(lon, 4), w)


def describe(lat: float, lon: float) -> tuple[str, float | None]:
    """Readable name for a point: 'Charlottesville, VA' when within 15 km of a
    listed city, 'near Charlottesville, VA' within 100 km, else ''."""
    c, d = nearest(lat, lon)
    if d <= 15:
        return c.label, round(d, 1)
    if d <= 100:
        return f"near {c.label}", round(d, 1)
    return "", None


# ------------------------------------------------------------ places named in sign text
# "NASHVILLE TEENESSEE" on a sign, "WELCOME TO Fabulous LAS VEGAS NEVADA":
# a city and its state printed together are a place clue on their own, no
# map search needed. OCR slips are tolerated in longer words only.
_STREET_WORDS = {
    "st", "street", "ave", "av", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive", "ln", "lane",
    "way", "pkwy", "parkway", "hwy", "highway", "ct", "court", "pl", "place", "sq", "square", "ter",
    "terrace", "cir", "circle", "trl", "trail", "pike", "expy", "fwy", "freeway", "turnpike", "aly", "alley",
}
_STATE_WORDS = {name.lower(): code for code, name in STATES.items() if code != "DC"}
STATE_FUZZY_MIN = 85  # RapidFuzz ratio; only for names of 6+ letters ("TEENESSEE" -> Tennessee)
_DC = re.compile(r"\bwashington,?\s*d\.?\s?c\b\.?|\bdistrict of columbia\b", re.I)
CITY_FUZZY_MIN = 88   # cities are only fuzzy-matched within a state the same text names


@dataclass(frozen=True)
class Mention:
    """A place named in a piece of text: a city with its state ("paired"),
    a city alone, or a state alone (city is None)."""
    city: City | None
    state: str
    paired: bool = False


def _state_of(words: str) -> str:
    from rapidfuzz import fuzz

    if words in _STATE_WORDS:
        return _STATE_WORDS[words]
    if len(words.replace(" ", "")) < 6:
        return ""
    size = len(words.split())
    pool = [s for s in _STATE_WORDS if len(s.split()) == size]  # "washington st" is not a state
    if not pool:
        return ""
    best = max(pool, key=lambda s: fuzz.ratio(words, s))
    ok = fuzz.ratio(words, best) >= STATE_FUZZY_MIN and words[0] == best[0]
    return _STATE_WORDS[best] if ok else ""


def named_places(text: str) -> list[Mention]:
    """Cities and states a sign names. A state or city followed by a street
    word ("Georgia Ave", "Denver St") is a street, not a place."""
    from rapidfuzz import fuzz

    text = text or ""
    dc = []
    if _DC.search(text):
        city = find("Washington, D.C.")
        dc.append(Mention(city, "DC", city is not None))
        text = _DC.sub(" ", text)
    orig = re.findall(r"[A-Za-z]+", text)
    toks = [w.lower() for w in orig]
    n = len(toks)
    used = [False] * n

    def street_after(j: int) -> bool:
        return j < n and toks[j] in _STREET_WORDS

    def free(i: int, j: int) -> bool:
        return j <= n and not any(used[i:j])

    # 1. full state names (longest first); "Kansas City" stays a city
    states: list[tuple[int, int, str]] = []
    for size in (3, 2, 1):
        for i in range(n - size + 1):
            if not free(i, i + size):
                continue
            code = _state_of(" ".join(toks[i:i + size]))
            if not code or street_after(i + size):
                continue
            if any(_key(" ".join(toks[i:i + size + k])) in _by_name() for k in (1, 2) if i + size + k <= n):
                continue
            states.append((i, i + size, code))
            for k in range(i, i + size):
                used[k] = True
    codes = {s for _, _, s in states}

    # 2. cities: exact names, or near-misses inside a state the text names
    out: list[Mention] = []
    for size in (3, 2, 1):
        for i in range(n - size + 1):
            if not free(i, i + size) or street_after(i + size):
                continue
            words = " ".join(toks[i:i + size])
            if len(words.replace(" ", "")) < 4:
                continue
            hits = _by_name().get(_key(words), [])
            if not hits and codes and len(words.replace(" ", "")) >= 6:
                pool = [c for c in cities() if c.state in codes]
                best = max(pool, key=lambda c: fuzz.ratio(words, _key(c.name)), default=None)
                if best is not None and fuzz.ratio(words, _key(best.name)) >= CITY_FUZZY_MIN:
                    hits = [best]
            if not hits:
                continue
            # paired: the state follows the city (one word between at most), or "City, ST"
            after = {s for a, _, s in states if i + size <= a <= i + size + 1}
            if i + size < n and len(orig[i + size]) == 2 and orig[i + size].isupper():
                after.add(orig[i + size])
            pair = next((c for c in hits if c.state in after), None)
            if pair is not None:
                out.append(Mention(pair, pair.state, True))
            elif len({c.state for c in hits}) == 1:
                out.append(Mention(hits[0], hits[0].state, False))
            else:
                continue
            for k in range(i, i + size):
                used[k] = True
    paired_states = {m.state for m in out if m.paired}
    out += [Mention(None, s) for _, _, s in states if s not in paired_states]
    return dc + out
