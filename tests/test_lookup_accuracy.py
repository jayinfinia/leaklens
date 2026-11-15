"""Sign-text lookup accuracy: name verification, state prior, offline phone
numbers, hinted (bounded) searches and reading agreement. Nominatim is faked
at the HTTP layer, so the real response parsing is exercised."""

import urllib.parse

import pytest

from app import config
from app.pipeline import features, lookup as lookup_mod
from app.pipeline.text_intel import phone_places, states_in
from app.schemas import AIGuess, AIReading, Clue, Geolocation, GuessLabel, GuessPoint

HSV = (34.7304, -86.5861)


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(lookup_mod, "_overpass", lambda points, regexes: [])
    monkeypatch.setattr(features, "_fetch", lambda q: [])


@pytest.fixture
def nominatim(monkeypatch):
    """Install a fake Nominatim: `nominatim(responder)` where responder gets
    the request's query parameters (dict) and returns JSON records."""
    requests = []

    def install(responder):
        def fake(url):
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
            requests.append(params)
            return responder(params)

        monkeypatch.setattr(lookup_mod, "_get_json", fake)
        return requests

    return install


def rec(name, lat, lon, city, state, iso, road=None, kind="pub", alt=None, importance=0.2):
    """A Nominatim jsonv2 record with namedetails and addressdetails."""
    parts = [p for p in (name, road, city, state, "United States") if p]
    nd = {"name": name} if name else None
    if alt:
        nd["alt_name"] = alt
    return {"display_name": ", ".join(parts), "lat": str(lat), "lon": str(lon), "type": kind,
            "importance": importance, "name": name or "", "namedetails": nd,
            "address": {"road": road, "city": city, "state": state, "ISO3166-2-lvl4": iso,
                        "country": "United States", "country_code": "us"}}


# Places a nationwide search for "POPPY & PARLIAMENT" might plausibly return:
# each has one of the words, none has both in its name.
DECOYS = [
    rec("Poppy", 37.5407, -77.4360, "Richmond", "Virginia", "US-VA", road="Parliament Street", kind="cafe"),
    rec("Parliament Pub", 38.9072, -77.0369, "Washington", "District of Columbia", "US-DC"),
    rec(None, 37.55, -77.44, "Richmond", "Virginia", "US-VA", road="Poppy Parliament Lane",
        kind="house"),  # an address point: no name at all
]
THE_PUB = rec("The Poppy and Parliament", HSV[0] + 0.0002, HSV[1], "Huntsville", "Alabama", "US-AL",
              road="Jefferson Street", alt="Poppy & Parliament")


def _sign(label="POPPY & PARLIAMENT", source="ocr+ai", box=(0, 0, 10, 10), score=0.6, kind="name"):
    return Clue(id="sign", kind="text", category=f"text:{kind}", label=label, score=score,
                box=list(box) if box else None, reason="r", weight=2.5, source=source)


def _geo_dc():
    return Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                       region=[GuessLabel(name="Washington, D.C.", prob=0.82),
                               GuessLabel(name="Virginia", prob=0.05)],
                       points=[GuessPoint(lat=38.9, lon=-77.0, prob=0.1)])


def _in_box(params, lat, lon) -> bool:
    if params.get("bounded") != "1" or "viewbox" not in params:
        return False
    x1, y1, x2, y2 = map(float, params["viewbox"].split(","))
    return min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2)


# ----------------------------------------------------------- Huntsville regression
def test_huntsville_poppy_and_parliament_is_found_in_huntsville_not_richmond(real_mode, nominatim):
    """The real failure: StreetCLIP says D.C., a nationwide search returns
    places in Richmond and D.C. that share a word with the sign. The AI model
    read the sign and guessed Huntsville; the bounded search there finds
    "The Poppy and Parliament". Result: Huntsville, strong."""

    def responder(p):
        if "viewbox" in p:
            # Nominatim free text can miss "&"; the "and" spelling finds it.
            if _in_box(p, *HSV) and "&" not in p["q"]:
                return [THE_PUB]
            return []
        return DECOYS

    requests = nominatim(responder)
    ai = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT"],
                   guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6, why="sign",
                                    lat=HSV[0], lon=HSV[1])])
    (m,) = lookup_mod.lookup([_sign()], _geo_dc(), ai)
    assert m.confidence == "strong"
    assert "Huntsville" in m.name and "Poppy and Parliament" in m.name
    assert abs(m.lat - HSV[0]) < 0.01 and "within 30 km" in m.query
    # the bounded search came first and every request asked for names and states
    assert requests[0].get("bounded") == "1"
    assert all(r["namedetails"] == "1" and r["addressdetails"] == "1" for r in requests)


def test_huntsville_decoys_alone_never_give_a_match(real_mode, nominatim):
    """Same photo without the AI model: only the decoys come back. None of
    them has both words in its name, so there is no match at all."""
    nominatim(lambda p: DECOYS)
    assert lookup_mod.lookup([_sign(source="")], _geo_dc()) == []


def test_ai_place_without_coordinates_uses_a_structured_city_search(real_mode, nominatim):
    def responder(p):
        if p.get("city") == "Huntsville" and p.get("state") == "Alabama":
            return [THE_PUB]
        return DECOYS

    requests = nominatim(responder)
    ai = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT"],
                   guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6, why="sign")])
    (m,) = lookup_mod.lookup([_sign()], _geo_dc(), ai)
    assert m.confidence == "strong" and "Huntsville" in m.name
    assert "amenity" in requests[0]  # before any nationwide search


def test_name_match_in_a_state_nothing_points_to_is_only_possible(real_mode, nominatim):
    """The models say D.C.; the region-hinted search returns the right name,
    but in Virginia. Possible match only."""
    far = rec("Poppy & Parliament", 37.54, -77.43, "Richmond", "Virginia", "US-VA")
    nominatim(lambda p: [far] if "Washington" in p.get("q", "") else [])
    (m,) = lookup_mod.lookup([_sign(source="")], _geo_dc())
    assert m.confidence == "weak" and "Virginia" in m.note


# ----------------------------------------------------------- garbled text
def test_garbled_reading_gives_no_strong_match(real_mode, nominatim):
    """OCR misread a sign as "TAPPY PARLMNT". Nominatim, asked for anything,
    returns a "Tappy's" on Parliament Road and a "Parliament" motel: neither
    name has both words, so nothing is claimed."""
    junk = [rec("Tappy's Tavern", 37.54, -77.45, "Richmond", "Virginia", "US-VA", road="Parliament Road"),
            rec("Parliament Motel", 38.9, -77.03, "Washington", "District of Columbia", "US-DC"),
            rec("Tappy Lane", 38.91, -77.02, "Washington", "District of Columbia", "US-DC", kind="residential")]
    nominatim(lambda p: junk)
    out = lookup_mod.lookup([_sign("TAPPY PARLMNT", source="")], _geo_dc())
    assert not [m for m in out if m.confidence == "strong"]


def test_ocr_slip_in_one_word_still_matches_the_name():
    assert lookup_mod.name_score("POPPY & PARLAMENT", "The Poppy and Parliament") >= config.NAME_MATCH_MIN


@pytest.mark.parametrize("sign,name,ok", [
    ("POPPY & PARLIAMENT", "The Poppy and Parliament", True),
    ("POPPY & PARLIAMENT", "Poppy & Parliament", True),
    ("THE POPPY AND PARLIAMENT", "Poppy & Parliament", True),
    ("POPPY & PARLIAMENT", "Poppy", False),            # the other word is only in the address
    ("POPPY & PARLIAMENT", "Parliament Pub", False),
    ("PARAMOUNT", "The Paramount Theater", True),
    ("POPPY", "Poppy's Bakery", True),
    ("illy", "Lilly's Cafe", False),                   # short words must match exactly
    ("MAIN ST", "Main Street", True),
    ("POPPY & PARLIAMENT PUB", "The Poppy & Parliament", True),  # one extra word when there are 3+
    ("TAPPY PARLMNT", "Tappy's Tavern", False),
])
def test_name_verification(sign, name, ok):
    assert (lookup_mod.name_score(sign, name) >= config.NAME_MATCH_MIN) is ok


def test_alt_names_count_and_unnamed_results_are_dropped():
    pub = lookup_mod._hit(rec("The British Pub", *HSV, "Huntsville", "Alabama", "US-AL",
                              alt="Poppy & Parliament"))
    house = lookup_mod._hit(DECOYS[2])
    assert pub[6] == "AL" and pub[7] == "us" and "Poppy & Parliament" in pub[5]
    assert house[5] == ()
    assert lookup_mod.verify("POPPY & PARLIAMENT", (pub, house)) == (pub,)


# ----------------------------------------------------------- phone numbers
def test_phone_numbers_resolve_offline_and_skip_toll_free():
    assert phone_places("Call (256) 533-1234") == [("Huntsville", "AL")]
    assert phone_places("256-555-0123") == [("", "AL")]
    assert phone_places("202.555.0100") == [("", "DC")]
    assert phone_places("1-800-555-1212") == []


def test_phone_area_code_backs_a_match_and_is_never_sent(real_mode, nominatim):
    """No AI model; StreetCLIP says D.C.; a phone number on the window has a
    Huntsville exchange. The pub found in Huntsville is strong, and the number
    never leaves the machine."""

    def responder(p):
        if p.get("city") == "Huntsville" or "Huntsville" in p.get("q", ""):
            return [THE_PUB]
        return []

    requests = nominatim(responder)
    phone = Clue(id="ph", kind="text", category="text:phone", label="CALL (256) 533-1234", score=0.9,
                 box=[0, 0, 5, 5], reason="r", weight=3.5)
    out = {m.clue_id: m for m in lookup_mod.lookup([_sign(source=""), phone], _geo_dc())}
    assert out["sign"].confidence == "strong" and "Huntsville" in out["sign"].name
    assert "ph" not in out
    sent = " ".join(" ".join(r.values()) for r in requests)
    assert "256" not in sent and "533" not in sent and "1234" not in sent


def test_prior_collects_every_kind_of_evidence():
    ai = AIReading(model="m", sign_texts=["Tel 205.555.0100"],
                   guesses=[AIGuess(place="Nashville, Tennessee, USA", confidence=0.5, why="w"),
                            AIGuess(place="Somewhere", confidence=0.9, why="w")])
    clues = [_sign("Welcome to Georgia", kind="place"), _sign("Huntsville, AL", kind="place")]
    prior, places = lookup_mod.build_prior(_geo_dc(), ai, clues)
    assert set(prior) == {"DC", "VA", "TN", "AL", "GA"}
    assert prior["DC"] == pytest.approx(0.82) and prior["GA"] == pytest.approx(config.PRIOR_STATE_TEXT)
    assert prior["AL"] == pytest.approx(config.PRIOR_PHONE + config.PRIOR_STATE_TEXT)


@pytest.mark.parametrize("text,codes", [
    ("Welcome to Alabama", ["AL"]),
    ("WEST VIRGINIA UNIVERSITY", ["WV"]),
    ("Washington, D.C.", ["DC"]),
    ("Huntsville, AL 35801", ["AL"]),
    ("EAT, IN OR OUT", []),
    ("POPPY & PARLIAMENT", []),
])
def test_states_in_text(text, codes):
    assert states_in(text) == codes


# ----------------------------------------------------------- queries
def test_variants_cover_ampersand_spellings():
    assert lookup_mod.variants("THE POPPY & PARLIAMENT") == [
        "POPPY & PARLIAMENT", "POPPY and PARLIAMENT", "POPPY PARLIAMENT"]
    assert lookup_mod.variants("PARAMOUNT") == ["PARAMOUNT"]
    assert not any(ch.isdigit() for v in lookup_mod.variants("CALL 256-533-1234") for ch in v)


def test_hinted_queries_come_before_nationwide_ones():
    geo = _geo_dc()
    plans = lookup_mod._queries("POPPY & PARLIAMENT", geo, ["Huntsville, Alabama, USA"],
                                [("Huntsville, Alabama, USA", *HSV)])
    kinds = [(k, hinted) for k, _, _, hinted in plans]
    first_unhinted = kinds.index(("text", False))
    assert kinds[:3] == [("bounded", True)] * 3 and ("structured", True) in kinds[:first_unhinted]
    assert all(h for _, h in kinds[:first_unhinted]) and not any(h for _, h in kinds[first_unhinted:])
    box = plans[0][1][1]
    x1, y1, x2, y2 = map(float, box.split(","))
    assert 0.5 < y1 - y2 < 0.6 and x1 < HSV[1] < x2  # about 30 km each way


def test_query_budget_counts_every_kind_of_search(real_mode, nominatim, monkeypatch):
    monkeypatch.setattr(config, "LOOKUP_MAX_QUERIES", 4)
    requests = nominatim(lambda p: [])
    ai = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT", "BUTCHER SHOPPE GROVE"],
                   guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6, why="w",
                                    lat=HSV[0], lon=HSV[1])])
    clues = [_sign(), _sign("BUTCHER SHOPPE GROVE").model_copy(update={"id": "b"})]
    assert lookup_mod.lookup(clues, _geo_dc(), ai) == []
    assert len(requests) == 4


def test_one_sign_cannot_use_the_whole_budget(real_mode, nominatim, monkeypatch):
    monkeypatch.setattr(config, "LOOKUP_MAX_PER_TEXT", 2)
    requests = nominatim(lambda p: [])
    lookup_mod.lookup([_sign(source="")], _geo_dc())
    assert len(requests) == 2


# ----------------------------------------------------------- reading agreement
def test_readings_only_one_reader_saw_are_not_searched(real_mode, nominatim):
    requests = nominatim(lambda p: [])
    ai = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT", "GROVE STREET MARKET"])
    clues = [
        _sign("POPPY & PARLIAMENT", source="ocr+ai"),
        _sign("Pa-pints Grill", source="").model_copy(update={"id": "ocr-only"}),   # OCR alone
        _sign("GROVE STREET MARKET", source="ai", box=None).model_copy(update={"id": "ai-only"}),
        _sign("GROVE STRET MARKET", source="", score=0.4).model_copy(update={"id": "ocr-2"}),
    ]
    picked = {c.id for c in lookup_mod.agreed(lookup_mod._candidates(clues), clues, ai)}
    # the AI-only reading counts once OCR read nearly the same thing elsewhere
    assert picked == {"sign", "ai-only", "ocr-2"}
    lookup_mod.lookup(clues, _geo_dc(), ai)
    assert requests and not any("pints" in r.get("q", "").lower() for r in requests)


def test_ai_reading_nobody_else_saw_is_not_searched():
    ai = AIReading(model="m", sign_texts=["ROYAL OAK TAVERN"])
    clues = [_sign("ROYAL OAK TAVERN", source="ai", box=None)]
    assert lookup_mod.agreed(lookup_mod._candidates(clues), clues, ai) == []


def test_without_the_ai_model_single_reader_text_is_still_searched():
    clues = [_sign("PARAMOUNT", source="")]
    assert lookup_mod.agreed(lookup_mod._candidates(clues), clues, None) == clues
    assert lookup_mod.agreed(lookup_mod._candidates(clues), clues, AIReading(model="m")) == clues
