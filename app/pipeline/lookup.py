"""Opt-in: search sign text on OpenStreetMap, the way an investigator would.

Vision models don't read. A person who sees "Poppy & Parliament" or
"PARAMOUNT" on a sign just searches it, and adds the region the scene
suggests. This layer does the same with Nominatim (OpenStreetMap's free
geocoder): first "<text>, <region the models guessed>", then the text
alone.

Off by default because it sends detected text off this machine. Follows
Nominatim's usage policy: identifying User-Agent, at most one request per
second, results cached.
"""

import json
import math
import time
import urllib.parse
import urllib.request
from functools import lru_cache

from .. import config
from ..schemas import Clue, Geolocation, TextMatch
from .text_intel import clean_query

UA = "LeakLens/0.3 (photo privacy checker; https://github.com/jayinfinia/leaklens)"
AMBIGUOUS_KM = 50
SEARCHABLE = {"name", "institution", "street", "place", "acronym", "web", "zip"}
_last_call = 0.0


@lru_cache(maxsize=512)
def _search(query: str, countrycodes: str = "") -> tuple:
    global _last_call
    wait = 1.1 - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    q = {"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 0}
    if countrycodes:
        q["countrycodes"] = countrycodes
    params = urllib.parse.urlencode(q)
    req = urllib.request.Request(f"{config.NOMINATIM_URL}?{params}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"[leaklens] lookup failed for {query!r}: {exc}")
        return ()
    finally:
        _last_call = time.monotonic()
    return tuple((d["display_name"], float(d["lat"]), float(d["lon"]), d.get("type", "")) for d in data)


def _candidates(clues: list[Clue]) -> list[Clue]:
    texts = [c for c in clues if c.kind == "text" and c.category.split(":")[-1] in SEARCHABLE]
    return sorted(texts, key=lambda c: c.weight * c.score, reverse=True)


def _queries(text: str, geo: Geolocation) -> list[tuple[str, bool]]:
    """(query, uses_region_hint) pairs, most specific first."""
    text = clean_query(text)
    if len(text) < 3:
        return []
    hints = []
    if geo.region:
        hints.append((f"{text}, {geo.region[0].name}", True))
    elif geo.country and geo.country[0].prob >= 0.3:
        hints.append((f"{text}, {geo.country[0].name}", True))
    return hints + [(text, False)]


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def _spread_km(hits: tuple) -> float:
    return max((_km(a[1], a[2], b[1], b[2]) for a in hits for b in hits), default=0.0)


def judge(text: str, hits: tuple, hinted: bool, geo: Geolocation):
    """Pick a hit and decide how much to trust it.

    Strong: every result sits in one place (within 50 km), and either the
    search was narrowed by the model's region or the text is specific
    (2+ meaningful words). Everything else is a possible match: a single
    common word like "POPPY" can match businesses in many cities, and we
    must not let the model's (possibly wrong) pin pick one for us.
    """
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
        if hinted and geo.points:  # within the model's region, show the one nearest its guess
            pin = geo.points[0]
            best = min(hits, key=lambda h: _km(h[1], h[2], pin.lat, pin.lon))
            return best, "weak", f"{n} places in that area share this name; showing the one nearest the model's guess."
        return hits[0], "weak", f"{n} places share this name, up to {spread:,.0f} km apart."
    return hits[0], "weak", "A single word: could be a coincidence."


def lookup(clues: list[Clue], geo: Geolocation) -> list[TextMatch]:
    if config.MOCK_MODE:
        return [TextMatch(clue_id=c.id, text=c.label, query=f"{c.label}, Virginia",
                          name=f"{c.label.title()}, Charlottesville, Virginia, United States",
                          lat=38.0313, lon=-78.4826, kind="road", confidence="strong")
                for c in _candidates(clues)[:1]]

    us = bool(geo.country) and geo.country[0].name == "United States" and geo.country[0].prob >= 0.5
    matches, budget = [], config.LOOKUP_MAX_QUERIES
    for c in _candidates(clues):
        for q, hinted in _queries(c.label, geo):
            if budget <= 0:
                return matches
            budget -= 1
            hits = _search(q, "us" if us else "")
            if hits:
                (name, lat, lon, kind), confidence, note = judge(c.label, hits, hinted, geo)
                matches.append(TextMatch(clue_id=c.id, text=c.label, query=q, name=name, lat=lat,
                                         lon=lon, kind=kind, confidence=confidence,
                                         n_results=len(hits), note=note))
                break
    # strongest first
    return sorted(matches, key=lambda m: m.confidence != "strong")
