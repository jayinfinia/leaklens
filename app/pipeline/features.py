"""#2: Bellingcat-style proximity search over objects, not just names.

Bellingcat's OpenStreetMap Search finds places where several features sit
within a short distance of each other. One "Poppy" is ambiguous; a Poppy
with a phone box, a butcher's shop and a bus stop all within 150 m is not.

The vocabulary below is shared with the local AI model (vlm.py): the model
may only answer with these keys, so every feature it names maps to real
OpenStreetMap tags. Weights reflect rarity in US streetscapes: a phone box
is rare evidence, a bus stop is everywhere.
"""

import json
import urllib.parse
import urllib.request

from .. import config

# key -> (OSM tag filters (any matches), rarity weight, plain label)
FEATURES: dict[str, tuple[list[tuple[str, str]], float, str]] = {
    "telephone box": ([("amenity", "telephone")], 3.0, "phone box"),
    "fountain": ([("amenity", "fountain")], 2.0, "fountain"),
    "statue or monument": ([("historic", "monument"), ("historic", "memorial"), ("tourism", "artwork")], 1.5, "monument or artwork"),
    "theatre or cinema": ([("amenity", "theatre"), ("amenity", "cinema")], 2.0, "theatre or cinema"),
    "butcher shop": ([("shop", "butcher")], 2.0, "butcher"),
    "bakery": ([("shop", "bakery")], 1.0, "bakery"),
    "pub or bar": ([("amenity", "pub"), ("amenity", "bar")], 1.0, "pub or bar"),
    "cafe": ([("amenity", "cafe")], 0.6, "cafe"),
    "restaurant": ([("amenity", "restaurant")], 0.3, "restaurant"),
    "pharmacy": ([("amenity", "pharmacy")], 0.8, "pharmacy"),
    "bank": ([("amenity", "bank")], 0.8, "bank"),
    "church": ([("amenity", "place_of_worship")], 0.8, "place of worship"),
    "post office": ([("amenity", "post_office")], 1.2, "post office"),
    "library": ([("amenity", "library")], 1.5, "library"),
    "fire station": ([("amenity", "fire_station")], 1.5, "fire station"),
    "hotel": ([("tourism", "hotel")], 0.8, "hotel"),
    "bus stop": ([("highway", "bus_stop")], 0.5, "bus stop"),
    "train or tram station": ([("railway", "station"), ("railway", "tram_stop")], 1.5, "rail or tram stop"),
    "bike share dock": ([("amenity", "bicycle_rental")], 1.5, "bike share dock"),
    "park": ([("leisure", "park")], 0.5, "park"),
    "playground": ([("leisure", "playground")], 0.8, "playground"),
    "water tower": ([("man_made", "water_tower")], 2.0, "water tower"),
    "lighthouse": ([("man_made", "lighthouse")], 3.0, "lighthouse"),
    "gas station": ([("amenity", "fuel")], 0.6, "gas station"),
    "parking garage": ([("amenity", "parking"), ("parking", "multi-storey")], 0.4, "parking"),
    "public toilets": ([("amenity", "toilets")], 1.0, "public toilets"),
    "railway crossing": ([("railway", "level_crossing")], 1.2, "railway crossing"),
}

# detector categories that imply a feature
DETECTOR_FEATURES = {"transit": "bus stop", "landmark": "statue or monument"}

FEATURE_M = 150
MAX_POINTS = 6
MAX_FEATURES = 6


def features_in(clues, ai_features: list[str] | None = None) -> list[str]:
    """Feature keys seen in the photo, rarest first."""
    keys = {DETECTOR_FEATURES[c.category] for c in clues
            if c.kind == "object" and c.category in DETECTOR_FEATURES}
    keys |= {k for k in (ai_features or []) if k in FEATURES}
    return sorted(keys, key=lambda k: FEATURES[k][1], reverse=True)[:MAX_FEATURES]


def _query(points, keys) -> str:
    clauses = []
    for lat, lon in points:
        for k in keys:
            for tk, tv in FEATURES[k][0]:
                clauses.append(f'nwr(around:{FEATURE_M},{lat:.6f},{lon:.6f})["{tk}"="{tv}"];')
    return f"[out:json][timeout:25];({''.join(clauses)});out center tags 400;"


def _fetch(query: str) -> list[dict]:
    from .lookup import UA

    req = urllib.request.Request(config.OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("elements", [])
    except Exception as exc:
        print(f"[leaklens] overpass (features) failed: {exc}")
        return []


def _key_for(tags: dict, keys) -> str | None:
    for k in keys:
        if any(tags.get(tk) == tv for tk, tv in FEATURES[k][0]):
            return k
    return None


def score_points(points: list[tuple[float, float]], keys: list[str]) -> list[tuple[float, list[str]]]:
    """For each point: (rarity-weighted score, feature keys found within FEATURE_M)."""
    from .lookup import _km

    points, keys = points[:MAX_POINTS], keys[:MAX_FEATURES]
    if not points or not keys:
        return [(0.0, []) for _ in points]
    found: list[set] = [set() for _ in points]
    for el in _fetch(_query(points, keys)):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        key = _key_for(el.get("tags") or {}, keys)
        if lat is None or lon is None or key is None:
            continue
        for i, (plat, plon) in enumerate(points):
            if _km(plat, plon, float(lat), float(lon)) <= FEATURE_M / 1000 + 0.02:
                found[i].add(key)
    return [(sum(FEATURES[k][1] for k in f), sorted(f, key=lambda k: -FEATURES[k][1])) for f in found]


def pick(scores: list[tuple[float, list[str]]], min_score: float = 2.5, margin: float = 2.0):
    """Index of a clear winner, or None. A winner needs real evidence (a rare
    feature or several common ones) and must beat the runner-up by `margin`x."""
    if not scores:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i][0], reverse=True)
    best = scores[order[0]][0]
    second = scores[order[1]][0] if len(order) > 1 else 0.0
    if best >= min_score and best >= margin * max(second, 0.01):
        return order[0]
    return None


def labels(keys: list[str]) -> str:
    return ", ".join(FEATURES[k][2] for k in keys)
