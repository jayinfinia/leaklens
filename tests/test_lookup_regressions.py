"""Regressions from the 35-photo US street benchmark (real models, real
Nominatim). Each test replays one photo's OCR and AI readings against a fake
Nominatim built from real responses (tests/fixtures/nominatim_us.json:
jsonv2, namedetails, addressdetails, countrycodes=us) plus a few records
written in the same shape for places the real run returned but the fixture
file doesn't hold.

The fake answers like Nominatim's free-text search: a record matches when
every query word is in its name or address (directions and single letters
are optional), `viewbox`+`bounded` keep records inside the box, and the
structured search matches name, city and state. At most 10 records."""

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

from app import config
from app.pipeline import features, fuse, gazetteer as gz, lookup as lookup_mod, vlm
from app.pipeline.text_intel import classify
from app.schemas import AIGuess, AIReading, Clue, Geolocation, GuessLabel

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "nominatim_us.json").read_text())


def record(name, lat, lon, city, state, iso, road=None, house=None, kind="yes", category="amenity",
           importance=0.05):
    """A Nominatim jsonv2 record in the fixture file's shape."""
    parts = [p for p in (name, house, road, city, state, "United States") if p]
    addr = {"road": road, "house_number": house, "city": city, "state": state, "ISO3166-2-lvl4": iso,
            "country": "United States", "country_code": "us"}
    return {"lat": str(lat), "lon": str(lon), "name": name, "display_name": ", ".join(parts),
            "category": category, "type": kind, "importance": importance,
            "namedetails": {"name": name}, "address": {k: v for k, v in addr.items() if v}}


# Places the real run returned (or that exist in OSM) beyond the fixture file.
EXTRA = [
    # phoenix-1's wrong strong answer, 3,449 km from the truth
    record("Yellow Cab Management Corporation", 40.7447, -73.9485, "Long Island City", "New York", "US-NY",
           road="Skillman Avenue", kind="company", category="office", importance=0.06),
    # birmingham-1's wrong strong answer, 1,004 km from the truth
    record("Northeast 16th Street", 35.4895, -97.5006, "Oklahoma City", "Oklahoma", "US-OK",
           road="Northeast 16th Street", kind="residential", category="highway", importance=0.05),
    record("Sixteenth Street Baptist Church", 33.5165, -86.8147, "Birmingham", "Alabama", "US-AL",
           road="6th Avenue North", house="1530", kind="place_of_worship", importance=0.42),
    record("Dorchester Avenue", 42.3301, -71.0569, "Boston", "Massachusetts", "US-MA",
           road="Dorchester Avenue", kind="primary", category="highway", importance=0.2),
    record("Dorchester Avenue", 42.3112, -71.0573, "Boston", "Massachusetts", "US-MA",
           road="Dorchester Avenue", kind="primary", category="highway", importance=0.2),
    record("Welcome to Fabulous Las Vegas", 36.0820, -115.1728, "Las Vegas", "Nevada", "US-NV",
           road="Las Vegas Boulevard South", kind="attraction", category="tourism", importance=0.45),
]

_DIRS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
         "se": "southeast", "sw": "southwest"}
_OPTIONAL = set(_DIRS.values()) | {"and"}


def _words(text):
    out = []
    for w in lookup_mod.norm_name(text).split():
        out.append(_DIRS.get(w, w))
    return out


def _db():
    seen, out = set(), []
    for recs in list(FIXTURES.values()) + [EXTRA]:
        for r in recs:
            key = (r["lat"], r["lon"], r.get("name"))
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


DB = _db()


def _record_words(r) -> set:
    a = r.get("address") or {}
    vals = [r.get("name") or ""] + [str(v) for k, v in a.items() if k not in ("country_code",)]
    words = set()
    for v in vals:
        words.update(_words(v))
    iso = a.get("ISO3166-2-lvl4", "")
    if iso.startswith("US-"):
        words.add(iso[3:].lower())
    return words


def _matches(query, r) -> bool:
    have = _record_words(r)
    return all(w in have or w in _OPTIONAL or len(w) == 1 for w in _words(query.replace(",", " ")))


def _in_box(r, viewbox) -> bool:
    x1, y1, x2, y2 = map(float, viewbox.split(","))
    lat, lon = float(r["lat"]), float(r["lon"])
    return min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2)


def fake_nominatim(params: dict) -> list:
    if "amenity" in params:
        city, state = params.get("city", "").lower(), params.get("state", "")
        code = gz.state_code(state)
        out = [r for r in DB if _matches(params["amenity"], r)
               and (r.get("address") or {}).get("city", "").lower() == city
               and (r.get("address") or {}).get("ISO3166-2-lvl4", "") == f"US-{code}"]
        return out[:10]
    out = [r for r in DB if _matches(params.get("q", ""), r)]
    if params.get("bounded") == "1":
        out = [r for r in out if _in_box(r, params["viewbox"])]
    return out[:10]


@pytest.fixture
def osm(monkeypatch):
    """Real mode with the fixture-backed Nominatim; Overpass and the object
    search find nothing. Returns the list of request parameters."""
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(lookup_mod, "_overpass", lambda points, regexes: [])
    monkeypatch.setattr(features, "_fetch", lambda q: [])
    requests = []

    def fake(url):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        requests.append(params)
        return fake_nominatim(params)

    monkeypatch.setattr(lookup_mod, "_get_json", fake)
    return requests


# ----------------------------------------------------------- building a photo's evidence
def ocr_clues(texts):
    out = []
    for k, t in enumerate(texts):
        intel = classify(t)
        out.append(Clue(id=f"txt-{k}", kind="text", category=f"text:{intel.kind}", label=t, score=0.8,
                        box=[k * 10, 0, k * 10 + 9, 9], reason=intel.reason, weight=intel.weight))
    return out


def ai_reading(signs, place, conf):
    c = gz.find(place)
    return AIReading(model="m", sign_texts=signs,
                     guesses=[AIGuess(place=place, confidence=conf, why="w",
                                      lat=c.lat if c else None, lon=c.lon if c else None)])


def geo_with(*states):
    return Geolocation(country=[GuessLabel(name="United States", prob=0.95)],
                       region=[GuessLabel(name=n, prob=p) for n, p in states])


def run_photo(ocr, signs, place, conf, geo):
    ai = ai_reading(signs, place, conf)
    clues = vlm.merge_texts(ocr_clues(ocr), ai)
    matches = lookup_mod.lookup(clues, geo, ai)
    return matches, fuse.fuse(geo, ai, matches)


def strong(matches):
    return [m for m in matches if m.confidence == "strong"]


def near(m, place, km=40):
    c = gz.find(place)
    return gz.km(m.lat, m.lon, c.lat, c.lon) <= km


# ----------------------------------------------------------- the photos
DENVER_OCR = ["KexcHaNG SUCKTTORN 55T 1893 1000 OSACE : BUCKHORN ESCHANCE", "1000 OSAGE • BUCKHORN EXCHANGE",
              "SALOON BUCKHORN EXCHANGE S"]
DENVER_AI = ["SALOON BUCKHORN EXCHANGE STEAKHOUSE", "EST. 1893", "1000 OSAGE • BUCKHORN EXCHANGE", "Orange St",
             "NO PARKING"]


def test_denver_buckhorn_exchange_is_strong_in_denver_by_its_address(osm):
    """The AI model says Colorado Springs and StreetCLIP leans Utah. The name
    alone matches five Buckhorn Exchanges in five states; the "1000 OSAGE" on
    the sign matches OSM's address for the Denver one."""
    matches, best = run_photo(DENVER_OCR, DENVER_AI, "Colorado Springs, CO", 0.9,
                              geo_with(("Utah", 0.4), ("Colorado", 0.2)))
    (m, *_) = strong(matches)
    assert "Buckhorn Exchange" in m.name and "Denver" in m.name
    assert "1000 OSAGE" in m.note
    assert best.source == "text" and near(best, "Denver, CO", 5)


def test_denver_found_in_the_state_is_strong_too(osm):
    matches, _ = run_photo(DENVER_OCR, DENVER_AI, "Colorado Springs, CO", 0.9, geo_with(("Colorado", 0.5)))
    (m, *_) = strong(matches)
    assert "Denver" in m.name and "1000 OSAGE" in m.note


def test_denver_address_search_when_osm_has_no_address_for_the_name(osm, monkeypatch):
    """Same photo, but the OSM objects named Buckhorn Exchange carry no
    address: "1000 OSAGE" is searched around each candidate (two at most)
    and lands on the Denver one."""
    bare = [dict(r, address={k: v for k, v in r["address"].items() if k not in ("house_number", "road")})
            if r.get("name") == "Buckhorn Exchange" else r for r in DB]
    monkeypatch.setattr(sys.modules[__name__], "DB", bare)
    matches, _ = run_photo(DENVER_OCR, DENVER_AI, "Denver, CO", 0.5, geo_with(("Utah", 0.4)))
    (m, *_) = strong(matches)
    assert "Denver" in m.name
    addr = [r for r in osm if r.get("bounded") == "1" and r["q"].startswith("1000")]
    assert 1 <= len(addr) <= config.LOOKUP_MAX_ADDRESS


PHOENIX_OCR = ["480-889-1630", "Yellow Cab Co.", "AAA CAB"]
PHOENIX_AI = ["Yellow Cab Co.", "AAA CAB", "480-889-1630"]


def test_phoenix_yellow_cab_is_never_strong_in_new_york(osm):
    """The AI model says New York (0.95); the cab's phone number has an
    Arizona area code. Yellow Cab is a brand: weak, and found in Arizona."""
    matches, best = run_photo(PHOENIX_OCR, PHOENIX_AI, "New York, NY", 0.95,
                              geo_with(("New York", 0.4), ("New Jersey", 0.1)))
    assert not [m for m in strong(matches) if "New York" in m.name]
    cab = next(m for m in matches if "Yellow Cab" in m.name)
    assert cab.confidence == "weak" and "Phoenix" in cab.name and "brand" in cab.note
    sent = " ".join(" ".join(r.values()) for r in osm)
    assert "480" not in sent and "1630" not in sent  # the phone number never leaves


def test_phoenix_without_the_phone_number_still_no_strong_new_york(osm):
    """Even with nothing pointing to Arizona, a brand found next to the AI
    model's guess is not upgraded by that guess."""
    matches, best = run_photo(["Yellow Cab Co.", "AAA CAB"], ["Yellow Cab Co.", "AAA CAB"], "New York, NY", 0.95,
                              geo_with(("New York", 0.4)))
    assert not strong(matches)
    cab = next(m for m in matches if "Yellow Cab" in m.name)
    assert "Long Island City" in cab.name and cab.confidence == "weak"
    assert best.source != "text"


NASHVILLE_OCR = ["Budweisen", "The Groab American Lager", "SKY BEIN @SALOONO NASHVILLE TEENESSEE"]
NASHVILLE_AI = ["Budweiser", "The Great American Lager", "WHISKEY BENT SALOON", "MUSICIAN LOADING AREA"]


def test_nashville_place_text_beats_budweiser_in_utah(osm):
    matches, best = run_photo(NASHVILLE_OCR, NASHVILLE_AI, "Bend, Oregon", 0.6,
                              geo_with(("Utah", 0.35), ("Oregon", 0.2)))
    assert not [m for m in strong(matches) if "Utah" in m.name]
    top = matches[0]
    assert top.confidence == "strong" and "Nashville" in top.name and top.kind == "city"
    assert best.source == "text" and near(best, "Nashville, TN")
    assert not any(m.confidence == "strong" and "Budweiser" in m.name for m in matches)


def test_budweiser_searched_in_utah_is_only_a_weak_brand_match(osm):
    """Without the city sign, the brand is still never strong."""
    matches, _ = run_photo(["Budweisen"], ["Budweiser"], "Bend, Oregon", 0.6, geo_with(("Utah", 0.35)))
    assert not strong(matches)
    assert all("brand" in m.note for m in matches)


def test_birmingham_16th_st_n_is_not_strong_in_oklahoma_city(osm):
    # two OCR engines: EasyOCR "16th Si N", Apple Vision "16th St N"
    matches, best = run_photo(["16th Si N", "16th St N"], ["16th St N"], "Oklahoma City, OK", 0.7,
                              geo_with(("Oklahoma", 0.3), ("Alabama", 0.2)))
    assert not strong(matches)
    m = matches[0]
    assert "Oklahoma City" in m.name and "common street name" in m.note
    assert best.source != "text"


def test_common_street_stays_weak_on_image_evidence_alone(osm):
    """Image models alone can't make a common street name strong: on the real
    benchmark StreetCLIP put birmingham-1 in Oklahoma. It stays a possible match
    that the fusion can still use."""
    matches, _ = run_photo(["E Lake St"], ["E Lake St"], "Seattle, WA", 0.6, geo_with(("Minnesota", 0.7)))
    (m,) = matches
    assert m.confidence == "weak" and "Minneapolis" in m.name


def test_birmingham_16th_st_n_with_confident_wrong_streetclip(osm):
    # Real run: StreetCLIP said Oklahoma with high confidence and the AI said Oklahoma City.
    matches, best = run_photo(["16th Si N", "16th St N"], ["16th St N"], "Oklahoma City, OK", 0.8,
                              geo_with(("Oklahoma", 0.8), ("Alabama", 0.05)))
    assert not strong(matches)
    assert matches[0].ambiguous  # the fusion must not count it as agreement with the AI guess


def test_minneapolis_e_lake_st_is_only_weak_support(osm):
    matches, _ = run_photo(["E Lake St"], ["E Lake St"], "Seattle, WA", 0.6,
                           geo_with(("Minnesota", 0.35), ("Washington", 0.2)))
    (m,) = matches
    assert m.confidence == "weak" and "Minneapolis" in m.name


def test_huntsville_poppy_and_parliament_still_strong(osm):
    matches, best = run_photo(["POPPY"], ["POPPY & PARLIAMENT"], "Huntsville, Alabama, USA", 0.6,
                              Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                                          region=[GuessLabel(name="Washington, D.C.", prob=0.82)]))
    (m,) = matches
    assert m.confidence == "strong" and "Poppy and Parliament" in m.name and "Huntsville" in m.name
    assert near(best, "Huntsville, AL", 2)


def test_boston_street_sign_still_strong_in_dorchester(osm):
    matches, _ = run_photo(["DORCHESTER AVE"], ["Dorchester Ave"], "Boston, MA", 0.7,
                           geo_with(("Massachusetts", 0.5)))
    (m,) = matches
    assert m.confidence == "strong" and "Dorchester Avenue" in m.name and "Boston" in m.name


def test_sixteenth_street_baptist_church_still_strong(osm):
    matches, best = run_photo(["SIXTEENTH STREET BAPTIST CHURCH"], ["SIXTEENTH STREET BAPTIST CHURCH"],
                              "Birmingham, AL", 0.7, geo_with(("Alabama", 0.3)))
    (m,) = strong(matches)
    assert "Sixteenth Street Baptist Church" in m.name and near(best, "Birmingham, AL", 5)


def test_las_vegas_welcome_sign_still_strong(osm):
    text = "WELCOME TO Fabulous LAS VEGAS NEVADA"
    matches, best = run_photo([text], [text], "Las Vegas, NV", 0.9, geo_with(("Nevada", 0.5)))
    m = matches[0]
    assert m.confidence == "strong" and "Las Vegas" in m.name and m.kind != "city"  # the sign itself
    assert near(best, "Las Vegas, NV")


def test_las_vegas_welcome_sign_offline_is_strong_by_place_text(osm, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "DB", [r for r in DB if "Welcome" not in (r.get("name") or "")])
    text = "WELCOME TO Fabulous LAS VEGAS NEVADA"
    matches, best = run_photo([text], [text], "Phoenix, AZ", 0.5, geo_with(("Arizona", 0.5)))
    assert matches[0].confidence == "strong" and matches[0].kind == "city" and near(best, "Las Vegas, NV")


# ----------------------------------------------------------- the parts
@pytest.mark.parametrize("text,brand", [
    ("Yellow Cab Co.", True), ("AAA CAB", True), ("Budweisen", True), ("Budweiser", True),
    ("Starbucks Coffee", True), ("POPPY & PARLIAMENT", False), ("BUCKHORN EXCHANGE", False),
    ("SIXTEENTH STREET BAPTIST CHURCH", False), ("PARAMOUNT", False),
])
def test_brand_detection(text, brand):
    assert bool(lookup_mod.brand_of(text)) is brand


@pytest.mark.parametrize("text,generic", [
    ("16th St N", True), ("E Lake St", True), ("Main Street", True), ("W 3rd Ave", True),
    ("Dorchester Ave", False), ("Osage St", False), ("SIXTEENTH STREET BAPTIST CHURCH", False),
    ("POPPY & PARLIAMENT", False),
])
def test_generic_streets(text, generic):
    assert lookup_mod.generic_street(text) is generic


def test_sign_anatomy():
    assert lookup_mod.core_name("1000 OSAGE • BUCKHORN EXCHANGE") == "BUCKHORN EXCHANGE"
    assert lookup_mod.core_name("SALOON BUCKHORN EXCHANGE STEAKHOUSE") == "BUCKHORN EXCHANGE"
    assert lookup_mod.core_name("Yellow Cab Co.") == "Yellow Cab"
    assert lookup_mod.variants("SALOON BUCKHORN EXCHANGE STEAKHOUSE") == ["BUCKHORN EXCHANGE"]
    assert lookup_mod.name_score("1000 OSAGE • BUCKHORN EXCHANGE", "Buckhorn Exchange") >= config.NAME_MATCH_MIN
    assert lookup_mod.name_score("SALOON BUCKHORN EXCHANGE STEAKHOUSE", "Buckhorn Exchange") >= config.NAME_MATCH_MIN
    clues = ocr_clues(DENVER_OCR)
    addrs = lookup_mod.sign_addresses(clues, AIReading(model="m", sign_texts=DENVER_AI))
    assert addrs[0][:2] == ("1000", "osage") and len(addrs) == 1  # OSACE is the same address, misread
    assert lookup_mod._addresses("EST. 1893") == [] and lookup_mod._addresses("2 FOR 1 SPECIALS") == []


@pytest.mark.parametrize("text,expect", [
    ("SKY BEIN @SALOONO NASHVILLE TEENESSEE", ("Nashville", "TN", True)),
    ("WELCOME TO Fabulous LAS VEGAS NEVADA", ("Las Vegas", "NV", True)),
    ("Huntsville, AL 35801", ("Huntsville", "AL", True)),
    ("Kansas City Missouri", ("Kansas City", "MO", True)),
    ("Welcome to Georgia", (None, "GA", False)),
])
def test_place_names_on_signs(text, expect):
    m = gz.named_places(text)[0]
    assert (m.city.name if m.city else None, m.state, m.paired) == expect


@pytest.mark.parametrize("text", ["Georgia Ave", "Washington St", "Denver St", "POPPY & PARLIAMENT", "16th St N"])
def test_streets_named_after_places_are_not_places(text):
    assert gz.named_places(text) == []


def test_phone_area_code_outweighs_the_ai_guess_for_a_plain_name(osm, monkeypatch):
    """A non-brand name found in New York (the AI model's city) while a phone
    number on the same photo is from Arizona: only a possible match."""
    extra = record("Desert Rose Tavern", 40.7300, -73.9900, "New York", "New York", "US-NY", kind="pub")
    monkeypatch.setattr(sys.modules[__name__], "DB", DB + [extra])
    matches, _ = run_photo(["DESERT ROSE TAVERN", "(480) 889-1630"], ["DESERT ROSE TAVERN"],
                           "New York, NY", 0.95, geo_with(("New York", 0.5)))
    m = next(m for m in matches if "Desert Rose" in m.name)
    assert m.confidence == "weak" and "Arizona" in m.note
