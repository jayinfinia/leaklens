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


def _queries(text: str, geo: Geolocation) -> list[str]:
    text = clean_query(text)
    if len(text) < 3:
        return []
    hints = []
    if geo.region:
        hints.append(f"{text}, {geo.region[0].name}")
    elif geo.country and geo.country[0].prob >= 0.3:
        hints.append(f"{text}, {geo.country[0].name}")
    return hints + [text]


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def _pick(hits: tuple, geo: Geolocation):
    """Several places share a name: take the one nearest the model's own pin."""
    if len(hits) > 1 and geo.points:
        pin = geo.points[0]
        return min(hits, key=lambda h: _km(h[1], h[2], pin.lat, pin.lon))
    return hits[0]


def lookup(clues: list[Clue], geo: Geolocation) -> list[TextMatch]:
    if config.MOCK_MODE:
        return [TextMatch(clue_id=c.id, text=c.label, query=f"{c.label}, Virginia",
                          name=f"{c.label.title()}, Charlottesville, Virginia, United States",
                          lat=38.0313, lon=-78.4826, kind="road")
                for c in _candidates(clues)[:1]]

    us = bool(geo.country) and geo.country[0].name == "United States" and geo.country[0].prob >= 0.5
    matches, budget = [], config.LOOKUP_MAX_QUERIES
    for c in _candidates(clues):
        for q in _queries(c.label, geo):
            if budget <= 0:
                return matches
            budget -= 1
            hits = _search(q, "us" if us else "")
            if hits:
                name, lat, lon, kind = _pick(hits, geo)
                matches.append(TextMatch(clue_id=c.id, text=c.label, query=q,
                                         name=name, lat=lat, lon=lon, kind=kind))
                break
    return matches
