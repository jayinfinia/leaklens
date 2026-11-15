"""One final answer: fuse every location signal into a single best guess.

Order of trust:
  1. GPS in the file's metadata (exact).
  2. A strong sign-text match on OpenStreetMap (verified on the map).
  3. Otherwise every candidate place is scored by the evidence within
     FUSE_AGREE_KM of it: the AI chooser's pick and confidence, the first-pass
     AI guesses, possible sign-text matches, the share of GeoCLIP's pins
     nearby, GeoCLIP's own similarity between the photo and that spot
     (geo-verification, as in G3, NeurIPS 2024), whether StreetCLIP and the AI
     put the photo in that state, and how many independent sources agree. A
     spot outside StreetCLIP's top 3 states loses points when StreetCLIP is
     confident.
  4. Nothing else: the most likely state's population center, at low confidence.

Weights live in config (FUSE_*), checked offline on saved benchmark evidence
with scripts/fuse_replay.py.
This module also builds the candidate list for the AI chooser (vlm.choose)
and puts places on the map offline (gazetteer.py).
"""

import random
from dataclasses import dataclass, field

from .. import config
from ..schemas import AIChoice, AIReading, BestGuess, ExifReport, Geolocation, TextMatch
from . import gazetteer as gz

PRIORITY = {"chooser": 0, "ai": 1, "text": 2, "geoclip": 3}
# Both AI passes are the same model, so they count as one independent source.
FAMILY = {"chooser": "ai", "ai": "ai", "text": "text", "geoclip": "geoclip"}
SOURCE_NAMES = {"chooser": "AI pick from candidates", "ai": "AI first guess", "text": "possible sign-text match",
                "geoclip": "GeoCLIP pins"}


# ------------------------------------------------------------ places on the map
def _label(place: str, lat: float | None = None, lon: float | None = None) -> str:
    """A tidy 'City, ST' for a free-form place, when the gazetteer knows it."""
    c = gz.find(place)
    if c is not None and (lat is None or gz.km(c.lat, c.lon, lat, lon) <= 50):
        return c.label
    return place


def place_guesses(ai: AIReading | None) -> AIReading | None:
    """Give AI guesses that have no coordinates yet (no map search, or it found
    nothing) the coordinates of the named city from the offline gazetteer."""
    if ai is None or ai.error or not ai.guesses:
        return ai
    out = []
    for g in ai.guesses:
        if g.lat is None:
            c = gz.find(g.place)
            if c is not None:
                g = g.model_copy(update={"lat": c.lat, "lon": c.lon})
        out.append(g)
    return ai.model_copy(update={"guesses": out})


# ------------------------------------------------------------ chooser options
@dataclass
class Option:
    name: str
    lat: float | None
    lon: float | None
    source: str


def _short(name: str, parts: int = 3) -> str:
    return ", ".join(p.strip() for p in name.split(",")[:parts] if p.strip())


def _key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def options(geo: Geolocation, ai: AIReading | None, matches: list[TextMatch],
            limit: int | None = None) -> list[Option]:
    """5-10 named candidate places for the AI chooser, shuffled (order biases
    small models). Mixes sources so no single one fills the list."""
    limit = limit or config.CHOOSE_MAX_OPTIONS
    pools: dict[str, list[Option]] = {"ai": [], "text": [], "geoclip": [], "state": []}
    for g in (ai.guesses if ai and not ai.error else []):
        if g.lat is None and "," not in g.place:
            continue  # "Somewhere in the eastern US": not a place to choose
        pools["ai"].append(Option(_label(g.place, g.lat, g.lon), g.lat, g.lon, "ai"))
    for m in matches:
        if m.confidence != "strong":
            name, _ = gz.describe(m.lat, m.lon)
            if name:
                pools["text"].append(Option(name.removeprefix("near "), m.lat, m.lon, "text"))
    for c in geo.clusters:
        if c.place:
            city = gz.find(c.place.removeprefix("near "))
            exact = city is not None and not c.place.startswith("near ")
            pools["geoclip"].append(Option(c.place.removeprefix("near "), city.lat if exact else c.lat,
                                           city.lon if exact else c.lon, "geoclip"))
    us = bool(geo.country) and geo.country[0].name == "United States"
    if us:  # a big city in each likely state, so the list is never just one place
        for r in geo.region[:2]:
            code = gz.state_code(r.name)
            city = next((c for c in gz.cities() if c.state == code), None)
            if city is not None:
                pools["state"].append(Option(city.label, city.lat, city.lon, "state"))

    picked: dict[str, Option] = {}

    def take(pool: list[Option], n: int):
        for o in pool:
            if n <= 0 or len(picked) >= limit:
                return
            if _key(o.name) not in picked:
                picked[_key(o.name)] = o
                n -= 1

    take(pools["ai"], 3)
    take(pools["text"], 2)
    take(pools["geoclip"], limit)
    if len(picked) < 4:
        take(pools["state"], 2)
    out = list(picked.values())
    random.Random("|".join(sorted(picked))).shuffle(out)  # deterministic per candidate set
    return out


def first_pass_confirmed(geo: Geolocation, ai: AIReading | None) -> bool:
    """True when GeoCLIP already backs the AI's first guess: at least
    CHOOSE_SKIP_MASS of its pin mass lies within FUSE_AGREE_KM of it. On the
    benchmark the chooser then always repeated the first guess, so the second
    AI call is skipped (it costs several seconds)."""
    if ai is None or ai.error or not geo.clusters:
        return False
    g = next((g for g in ai.guesses if g.lat is not None), None)
    if g is None:
        return False
    total = sum(c.prob for c in geo.clusters) or 1.0
    near = sum(c.prob for c in geo.clusters if gz.km(c.lat, c.lon, g.lat, g.lon) <= config.FUSE_AGREE_KM)
    return near / total >= config.CHOOSE_SKIP_MASS


def locate_choice(choice: AIChoice, opts: list[Option], lookup: bool = False, us: bool = True) -> AIChoice:
    """Coordinates for the chooser's pick: the option's own, else the gazetteer,
    else (only when the user allowed map search) OpenStreetMap."""
    if choice.error or not choice.place:
        return choice
    if not choice.other:
        o = next((o for o in opts if o.name == choice.place), None)
        if o is not None and o.lat is not None:
            return choice.model_copy(update={"lat": o.lat, "lon": o.lon})
    c = gz.find(choice.place)
    if c is not None:
        return choice.model_copy(update={"lat": c.lat, "lon": c.lon})
    if lookup and not config.MOCK_MODE:
        from .lookup import _search

        try:
            hits = _search(choice.place, "us" if us else "")
        except Exception:
            hits = ()
        if hits:
            return choice.model_copy(update={"lat": hits[0][1], "lon": hits[0][2]})
    return choice


# ------------------------------------------------------------ fusion
@dataclass
class Candidate:
    lat: float
    lon: float
    place: str
    source: str  # chooser | ai | text | geoclip
    value: float  # confidence (chooser, ai), 1 (text) or share of GeoCLIP mass
    support: dict = field(default_factory=dict)
    score: float = 0.0
    parts: dict = field(default_factory=dict)


def candidates(geo: Geolocation, ai: AIReading | None, matches: list[TextMatch]) -> list[Candidate]:
    out = []
    ok = ai is not None and not ai.error
    ch = ai.choice if ok else None
    if ch is not None and not ch.error and ch.lat is not None:
        out.append(Candidate(ch.lat, ch.lon, _label(ch.place, ch.lat, ch.lon), "chooser", ch.confidence))
    for g in (ai.guesses if ok else []):
        if g.lat is not None:
            out.append(Candidate(g.lat, g.lon, _label(g.place, g.lat, g.lon), "ai", g.confidence))
    for m in matches:
        # Ambiguous matches (chains, common streets) were often found by searching near the AI
        # model's own guess, so counting them would echo that guess as independent agreement.
        if m.confidence != "strong" and not m.ambiguous:
            out.append(Candidate(m.lat, m.lon, _short(m.name), "text", 1.0))
    total = sum(c.prob for c in geo.clusters) or 1.0
    for c in geo.clusters:
        out.append(Candidate(c.lat, c.lon, c.place or f"{c.lat:.3f}, {c.lon:.3f}", "geoclip", c.prob / total))
    return out


def _state_scores(geo: Geolocation) -> dict[str, float]:
    """USPS code -> StreetCLIP prob of that state relative to its top state."""
    if not geo.region or not geo.country or geo.country[0].name != "United States":
        return {}
    top = geo.region[0].prob or 1e-9
    return {gz.state_code(r.name): min(1.0, r.prob / top) for r in geo.region if gz.state_code(r.name)}


def score_candidates(cands: list[Candidate], geo: Geolocation, ai: AIReading | None,
                     verify: list[float] | None = None) -> list[Candidate]:
    """Score each candidate with the evidence near it. `verify` holds GeoCLIP's
    verification score for each candidate's spot (or None)."""
    states = _state_scores(geo)
    top3 = set(list(states)[:3])
    sure = bool(states) and geo.region[0].prob >= config.FUSE_STATE_SURE
    ai_states = set()
    if ai is not None and not ai.error:
        for s in (ai.state, ai.choice.state if ai.choice else ""):
            if gz.state_code(s):
                ai_states.add(gz.state_code(s))
    for i, c in enumerate(cands):
        near = [o for o in cands if gz.km(c.lat, c.lon, o.lat, o.lon) <= config.FUSE_AGREE_KM]
        sup: dict[str, float] = {}
        for o in near:
            if o.source == "geoclip":  # pin mass adds up
                sup["geoclip"] = min(1.0, sup.get("geoclip", 0.0) + o.value)
            else:
                sup[o.source] = max(sup.get(o.source, 0.0), o.value)
        agreeing = {FAMILY[s] for s, v in sup.items() if v >= 0.05}
        code = gz.state_at(c.lat, c.lon)
        parts = {
            "chooser": config.FUSE_W_CHOOSER * sup.get("chooser", 0.0),
            "ai": config.FUSE_W_AI * sup.get("ai", 0.0),
            "text": config.FUSE_W_TEXT_WEAK * sup.get("text", 0.0),
            "cluster": config.FUSE_W_CLUSTER * sup.get("geoclip", 0.0),
            "verify": config.FUSE_W_GEOCLIP * (verify[i] if verify else 0.0),
            "state": config.FUSE_W_STATE * states.get(code, 0.0),
            "ai_state": config.FUSE_W_AI_STATE * (code in ai_states),
            "agree": config.FUSE_W_AGREE * max(0, len(agreeing) - 1),
            # StreetCLIP is rarely wrong about the state when it is confident.
            # A rural spot far from any listed city takes the nearest one's state.
            "off_state": -config.FUSE_W_OFF_STATE * (sure and (code or gz.state_at(c.lat, c.lon, 300)) not in top3),
        }
        c.support, c.parts, c.score = sup, parts, sum(parts.values())
    return cands


def _why(c: Candidate, geo: Geolocation, verify: float | None) -> str:
    bits = []
    if "chooser" in c.support:
        bits.append(f"the AI model picked it from the candidates ({c.support['chooser']:.0%})")
    if "ai" in c.support:
        bits.append(f"the AI model's first guess agrees ({c.support['ai']:.0%})")
    if "text" in c.support:
        bits.append("a sign's text possibly matches a place here")
    if "geoclip" in c.support:
        bits.append(f"{c.support['geoclip']:.0%} of GeoCLIP's top pins are nearby")
    if verify is not None and verify >= 0.6:
        bits.append("GeoCLIP rates the spot highly for this photo")
    if c.parts.get("state"):
        top = geo.region[0].name if geo.region else ""
        st = gz.STATES.get(gz.state_at(c.lat, c.lon), "")
        bits.append(f"StreetCLIP's top state is {top}" if st == top else f"StreetCLIP also considers {st}")
    if c.parts.get("ai_state"):
        bits.append("in the state the AI model named")
    return ("Because " + "; ".join(bits) + ".") if bits else "Best of weak signals."


# The last call's candidates and GeoCLIP scores, for the benchmark's evidence dump.
LAST: dict = {}


def fuse(geo: Geolocation, ai: AIReading | None = None, matches: list[TextMatch] | None = None,
         scorer=None, exif: ExifReport | None = None) -> BestGuess | None:
    matches = matches or []
    LAST.clear()
    if exif is not None and exif.gps:
        name, _ = gz.describe(exif.gps.lat, exif.gps.lon)
        return BestGuess(lat=exif.gps.lat, lon=exif.gps.lon, place=name or "GPS position",
                         source="metadata", confidence=1.0, supports=["metadata"],
                         why="The file's metadata contains the exact GPS position.")
    strong = next((m for m in matches if m.confidence == "strong"), None)
    if strong is not None:
        return BestGuess(lat=strong.lat, lon=strong.lon, place=_short(strong.name),
                         source="text", confidence=0.95, supports=["text"],
                         why=f"Sign text “{strong.text}” was found on the map: {strong.name}. {strong.note}".strip())

    cands = candidates(geo, ai, matches)
    if not cands:
        return _state_fallback(geo)
    verify = None
    if scorer is not None:
        try:
            verify = scorer.score([(c.lat, c.lon) for c in cands])
        except Exception as exc:
            print(f"[leaklens] GeoCLIP verification failed: {exc}")
    score_candidates(cands, geo, ai, verify)
    LAST.clear()
    LAST.update(candidates=[dict(lat=c.lat, lon=c.lon, place=c.place, source=c.source, value=c.value,
                                 score=c.score, parts=c.parts) for c in cands], verify=verify)
    order = sorted(range(len(cands)), key=lambda i: (-round(cands[i].score, 6), PRIORITY[cands[i].source]))
    i = order[0]
    best = cands[i]
    conf = min(config.FUSE_MAX_CONF, best.score / config.FUSE_SCORE_FULL)
    return BestGuess(lat=round(best.lat, 5), lon=round(best.lon, 5), place=best.place, source=best.source,
                     confidence=round(conf, 2),
                     supports=sorted((k for k, v in best.support.items() if v >= 0.05), key=PRIORITY.get),
                     why=_why(best, geo, verify[i] if verify else None))


def _state_fallback(geo: Geolocation) -> BestGuess | None:
    states = _state_scores(geo)
    if not states:
        return None
    top = geo.region[0]
    c = gz.state_center(gz.state_code(top.name))
    if c is None:
        return None
    return BestGuess(lat=c.lat, lon=c.lon, place=f"somewhere in {top.name}", source="state",
                     confidence=round(min(0.3, 0.5 * top.prob), 2), supports=["state"],
                     why=f"Only the state is known: StreetCLIP's top state is {top.name} ({top.prob:.0%}). "
                         "The pin is the state's population center, not a place.")
