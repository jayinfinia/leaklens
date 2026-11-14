"""Opt-in: search sign text on OpenStreetMap, the way an investigator would.

Vision models don't read. A person who sees "Poppy & Parliament" or
"PARAMOUNT" on a sign just searches it, and adds the region the scene
suggests. This layer does the same with Nominatim (OpenStreetMap's free
geocoder): first "<text>, <region the models guessed>", then the text alone.

One sign is often ambiguous ("POPPY" matches businesses in many cities),
so, like an investigator, we use the clues together:
  1. co-location: if two different signs in the photo both have a
     candidate within ~300 m of each other, that spot is the answer;
  2. nearby names: for each candidate of an ambiguous sign, ask Overpass
     (OpenStreetMap's query API) whether any *other* sign's text names a
     place within ~300 m.

Off by default because it sends detected text (never the photo) off this
machine. Follows Nominatim's usage policy: identifying User-Agent, at most
one request per second, results cached.
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
    q = {"q": query, "format": "jsonv2", "limit": 10, "addressdetails": 0}
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
    return tuple((d["display_name"], float(d["lat"]), float(d["lon"]), d.get("type", ""),
                  float(d.get("importance") or 0.0)) for d in data)


def _searchable(c: Clue) -> bool:
    if c.kind != "text" or c.category.split(":")[-1] not in SEARCHABLE:
        return False
    words = clean_query(c.label).split()
    # one short word ("illy", "choR") matches too much by chance
    return len(words) >= 2 or sum(ch.isalnum() for ch in "".join(words)) >= 5


def _candidates(clues: list[Clue]) -> list[Clue]:
    texts = [c for c in clues if _searchable(c)]
    return sorted(texts, key=lambda c: c.weight * c.score, reverse=True)


def _place_hint(place: str) -> str:
    """'Huntsville, Alabama, USA' -> 'Huntsville, Alabama' (country is implied by countrycodes)."""
    parts = [p.strip() for p in place.split(",") if p.strip()]
    if len(parts) >= 3 and parts[-1].lower() in ("usa", "us", "united states", "united states of america"):
        parts = parts[:-1]
    return ", ".join(parts[:2])


def _queries(text: str, geo: Geolocation, ai_places: list[str] | None = None) -> list[tuple[str, bool]]:
    """(query, uses_region_hint) pairs, most specific first."""
    text = clean_query(text)
    if len(text) < 3:
        return []
    hints = [(f"{text}, {_place_hint(p)}", True) for p in (ai_places or [])[:1] if _place_hint(p)]
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


def _importance(h) -> float:
    return h[4] if len(h) > 4 else 0.0


def _dominant(hits: tuple) -> bool:
    """OpenStreetMap ranks the first result far above the others (a famous
    theater vs. a road with the same name)."""
    top = _importance(hits[0])
    second = max((_importance(h) for h in hits[1:]), default=0.0)
    return top >= 0.3 and top >= 1.5 * second


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
        if hinted and _dominant(hits):
            return hits[0], "strong", (f"The best-known match in that area ({n} places share the name; "
                                       "OpenStreetMap ranks this one far above the rest).")
        # Never let the vision model's pin choose between far-apart places: it can be
        # confidently wrong. Keep OpenStreetMap's own ranking.
        return hits[0], "weak", f"{n} places share this name, up to {spread:,.0f} km apart."
    return hits[0], "weak", "A single word: could be a coincidence."


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
                feature_keys: list[str] | None = None) -> dict:
    """Upgrade weak matches that other evidence in the same photo agrees with."""
    near_km = config.NEARBY_M / 1000
    by_id = {c.id: c for c in clues}

    # 0. The local AI model's own guess lands near one of the candidates.
    for cid, m in list(matches.items()):
        if m.confidence == "strong" or not ai_points:
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
            if m.confidence == "strong" or fcalls >= 2 or len(pools[cid]) > 10:
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


def lookup(clues: list[Clue], geo: Geolocation, ai=None, feature_keys: list[str] | None = None) -> list[TextMatch]:
    if config.MOCK_MODE:
        return [TextMatch(clue_id=c.id, text=c.label, query=f"{c.label}, Virginia",
                          name=f"{c.label.title()}, Charlottesville, Virginia, United States",
                          lat=38.0313, lon=-78.4826, kind="road", confidence="strong")
                for c in _candidates(clues)[:1]]

    us = bool(geo.country) and geo.country[0].name == "United States" and geo.country[0].prob >= 0.5
    ai_places = [g.place for g in (ai.guesses if ai else []) if g.confidence >= 0.2]
    ai_points = [(g.place, g.lat, g.lon) for g in (ai.guesses if ai else []) if g.lat is not None]
    matches: dict[str, TextMatch] = {}
    pools: dict[str, tuple] = {}
    budget = config.LOOKUP_MAX_QUERIES
    candidates = _candidates(clues)
    if ai is not None and ai.sign_texts:
        # The AI model doubles as a proofreader: OCR text it didn't also read is
        # usually a misreading ("Pa-pints"), so don't search it.
        candidates = [c for c in candidates if c.source in ("ai", "ocr+ai")]
    for c in candidates[:5]:
        for q, hinted in _queries(c.label, geo, ai_places):
            if budget <= 0:
                break
            budget -= 1
            hits = _search(q, "us" if us else "")
            if hits:
                best, confidence, note = judge(c.label, hits, hinted, geo)
                name, lat, lon, kind = best[:4]
                matches[c.id] = TextMatch(clue_id=c.id, text=c.label, query=q, name=name, lat=lat,
                                          lon=lon, kind=kind, confidence=confidence,
                                          n_results=len(hits), note=note)
                pools[c.id] = hits
                break
    matches = corroborate(matches, pools, clues, ai_points=ai_points, feature_keys=feature_keys)
    return sorted(matches.values(), key=lambda m: m.confidence != "strong")
