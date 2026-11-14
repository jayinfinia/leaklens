"""#1 local AI model and #2 object proximity search, with Ollama and
Overpass faked (no network, no model)."""

import json

import pytest

from app import config
from app.pipeline import features, lookup as lookup_mod, vlm
from app.pipeline.risk import score
from app.schemas import AIGuess, AIReading, Clue, ExifReport, Geolocation, GuessLabel

HSV = (34.7304, -86.5861)


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "VLM_MODE", "auto")
    monkeypatch.setattr(lookup_mod, "_overpass", lambda points, regexes: [])
    monkeypatch.setattr(features, "_fetch", lambda q: [])
    vlm._status.update(checked=0.0)


def _text(cid, label, kind="name", score=0.5, box=(0, 0, 10, 10)):
    return Clue(id=cid, kind="text", category=f"text:{kind}", label=label, score=score,
                box=list(box) if box else None, reason="r", weight=2.5)


def _geo_dc():
    return Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                       region=[GuessLabel(name="Washington, D.C.", prob=0.82)])


# ----------------------------------------------------------- Ollama plumbing
def test_status_reports_missing_model_with_the_pull_command(real_mode, monkeypatch):
    monkeypatch.setattr(vlm, "_http", lambda path, payload=None, timeout=5: {"models": [{"name": "qwen3-vl:4b"}]})
    monkeypatch.setattr(config, "VLM_MODEL", "qwen3-vl:8b")
    st = vlm.status(refresh=True)
    assert not st["ok"] and "ollama pull qwen3-vl:8b" in st["reason"] and "qwen3-vl:4b" in st["reason"]


def test_status_when_ollama_is_not_running(real_mode, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(vlm, "_http", boom)
    st = vlm.status(refresh=True)
    assert not st["ok"] and "not reachable" in st["reason"]


def test_ask_sends_image_and_schema_and_parses_the_answer(real_mode, monkeypatch):
    from PIL import Image

    sent = {}
    answer = {
        "sign_texts": ["POPPY & PARLIAMENT", "x", "POPPY & PARLIAMENT"],
        "clues": [{"clue": "red British phone box", "suggests": "themed pub district"}],
        "features": ["telephone box", "butcher shop", "spaceship"],
        "guesses": [{"place": "Huntsville, Alabama, USA", "confidence": 1.7, "why": "sign + phone box"}],
    }

    def fake(path, payload=None, timeout=5):
        if path == "/api/tags":
            return {"models": [{"name": "qwen3-vl:8b"}]}
        sent.update(payload)
        return {"message": {"content": "Sure! " + json.dumps(answer)}}

    monkeypatch.setattr(vlm, "_http", fake)
    monkeypatch.setattr(config, "VLM_MODEL", "qwen3-vl:8b")
    r = vlm.ask(Image.new("RGB", (3000, 2000)))
    assert sent["format"]["required"] == ["sign_texts", "clues", "features", "guesses"]
    assert sent["messages"][0]["images"] and sent["options"]["temperature"] == 0
    assert r.sign_texts == ["POPPY & PARLIAMENT"]  # deduped, junk dropped
    assert r.features == ["butcher shop", "telephone box"]  # unknown feature dropped
    assert r.guesses[0].confidence == 1.0 and not r.error


def test_ask_reports_errors_instead_of_crashing(real_mode, monkeypatch):
    from PIL import Image

    def fake(path, payload=None, timeout=5):
        if path == "/api/tags":
            return {"models": [{"name": config.VLM_MODEL}]}
        raise TimeoutError("timed out")

    monkeypatch.setattr(vlm, "_http", fake)
    r = vlm.ask(Image.new("RGB", (100, 100)))
    assert r.error.startswith("model call failed")


# ----------------------------------------------------------- merging readings
def test_ai_reading_upgrades_the_ocr_box_and_adds_unseen_signs():
    reading = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT", "Butcher Shoppe Grove", "TELEPHONE"])
    out = vlm.merge_texts([_text("t0", "POPPY"), _text("t1", "CLASS OF 2024", "generic")], reading)
    by = {c.label: c for c in out}
    assert by["POPPY & PARLIAMENT"].box == [0, 0, 10, 10] and by["POPPY & PARLIAMENT"].source == "ocr+ai"
    assert by["Butcher Shoppe Grove"].box is None and by["Butcher Shoppe Grove"].source == "ai"
    assert "TELEPHONE" not in by  # generic text the model read is not a clue


def test_same_sign_matching():
    assert vlm.same_sign("POPPY", "POPPY & PARLIAMENT")
    assert vlm.same_sign("UVA MED SCHOOL CLASS OF 2024", "CONGRATULATIONS UVA MED SCHOOL")
    assert not vlm.same_sign("POPPY", "PARAMOUNT")


# ----------------------------------------------------------- AI + lookup
def test_ai_place_guess_narrows_the_search_despite_a_wrong_vision_model(real_mode, monkeypatch):
    """Huntsville: StreetCLIP says D.C.; the AI model reads the full sign and
    says Huntsville. The hinted search finds one place: strong."""
    calls = []

    def fake(q, countrycodes=""):
        calls.append(q)
        if q == "POPPY & PARLIAMENT, Huntsville, Alabama":
            return (("Poppy & Parliament, 2nd Street, Huntsville, Alabama", *HSV, "pub"),)
        return ()

    monkeypatch.setattr(lookup_mod, "_search", fake)
    ai = AIReading(model="m", guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.7, why="sign")])
    m = lookup_mod.lookup([_text("t0", "POPPY & PARLIAMENT")], _geo_dc(), ai)[0]
    assert calls[0] == "POPPY & PARLIAMENT, Huntsville, Alabama"
    assert m.confidence == "strong" and "Huntsville" in m.name


def test_ai_guess_location_corroborates_an_ambiguous_word(real_mode, monkeypatch):
    poppys = (("Poppy, Richmond, Virginia", 37.54, -77.45, "restaurant"),
              ("Poppy, Seattle, Washington", 47.61, -122.33, "restaurant"),
              ("Poppy, Huntsville, Alabama", *HSV, "restaurant"))
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": poppys if q == "POPPY" else ())
    ai = AIReading(model="m", guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6, why="w",
                                               lat=34.73, lon=-86.586)])
    m = lookup_mod.lookup([_text("t0", "POPPY")], _geo_dc(), ai)[0]
    assert m.confidence == "strong" and "Huntsville" in m.name and "AI model" in m.note


def test_geocoding_ai_guesses(real_mode, monkeypatch):
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": (("Huntsville", *HSV, "city"),))
    ai = AIReading(model="m", guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6, why="w")])
    g = lookup_mod.geocode_guesses(ai, True).guesses[0]
    assert (g.lat, g.lon) == HSV


def test_ai_guess_counts_toward_risk_but_capped():
    ai = AIReading(model="m", guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=1.0, why="w")])
    r = score(ExifReport(), [], Geolocation(), ai=ai)
    assert r.factors[0].source == "ai" and r.score == 45


# ----------------------------------------------------------- #2 object proximity
def test_features_from_detector_and_ai_rarest_first():
    clues = [Clue(id="o", kind="object", category="transit", label="bus stop sign", score=0.4,
                  box=[0, 0, 1, 1], reason="r", weight=2.5)]
    keys = features.features_in(clues, ["telephone box", "restaurant", "not-a-feature"])
    assert keys == ["telephone box", "bus stop", "restaurant"]


def test_overpass_query_covers_every_point_and_tag():
    q = features._query([(1.0, 2.0), (3.0, 4.0)], ["telephone box", "pub or bar"])
    assert q.count("around:150,1.000000,2.000000") == 3  # telephone + pub + bar
    assert '["amenity"="telephone"]' in q and '["amenity"="bar"]' in q


def test_score_points_counts_each_feature_once_per_place(monkeypatch):
    els = [
        {"lat": HSV[0] + 0.0003, "lon": HSV[1], "tags": {"amenity": "telephone"}},
        {"lat": HSV[0] - 0.0004, "lon": HSV[1], "tags": {"shop": "butcher"}},
        {"lat": HSV[0] - 0.0005, "lon": HSV[1], "tags": {"shop": "butcher"}},
        {"lat": 37.5401, "lon": -77.45, "tags": {"highway": "bus_stop"}},
    ]
    monkeypatch.setattr(features, "_fetch", lambda q: els)
    scores = features.score_points([HSV, (37.54, -77.45)], ["telephone box", "butcher shop", "bus stop"])
    assert scores[0] == (5.0, ["telephone box", "butcher shop"]) and scores[1] == (0.5, ["bus stop"])
    assert features.pick(scores) == 0


@pytest.mark.parametrize("scores,winner", [
    ([(5.0, []), (0.5, [])], 0),
    ([(3.0, []), (2.5, [])], None),   # too close to call
    ([(1.0, []), (0.0, [])], None),   # not enough evidence
    ([(0.0, []), (3.0, [])], 1),
])
def test_pick_needs_evidence_and_a_clear_margin(scores, winner):
    assert features.pick(scores) == winner


def test_objects_around_a_candidate_settle_an_ambiguous_word(real_mode, monkeypatch):
    poppys = (("Poppy, Richmond, Virginia", 37.54, -77.45, "restaurant"),
              ("Poppy, Huntsville, Alabama", *HSV, "restaurant"))
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": poppys if q == "POPPY" else ())
    monkeypatch.setattr(features, "_fetch", lambda q: [
        {"lat": HSV[0] + 0.0003, "lon": HSV[1], "tags": {"amenity": "telephone"}},
        {"lat": HSV[0], "lon": HSV[1] + 0.0004, "tags": {"shop": "butcher"}},
    ])
    m = lookup_mod.lookup([_text("t0", "POPPY")], _geo_dc(), None, ["telephone box", "butcher shop"])[0]
    assert m.confidence == "strong" and "Huntsville" in m.name and "phone box" in m.note


# ----------------------------------------------------------- end to end (mock)
def test_mock_pipeline_includes_ai_reading(monkeypatch):
    from tests.conftest import textured_image, to_jpeg
    from app.pipeline import analyze, load_image

    monkeypatch.setattr(config, "MOCK_MODE", True)
    raw, upright = load_image(to_jpeg(textured_image(), gps=False))
    a = analyze(raw, upright, lookup=True, use_ai=True)
    assert a.ai and a.ai.guesses[0].lat is not None
    assert any(c.source == "ai" and c.box is None for c in a.clues)
    assert any(f.source == "ai" for f in a.risk.factors)
    off = analyze(raw, upright, use_ai=False)
    assert off.ai is None and all(c.box is not None for c in off.clues)


def test_famous_place_beats_same_named_roads_in_the_hinted_region(real_mode, monkeypatch):
    hits = (("The Paramount Theater, East Main Street, Charlottesville", 38.0306, -78.4817, "theatre", 0.42),
            ("Paramount Drive, Scott County, Virginia", 36.7, -82.6, "residential", 0.10),
            ("Paramount Lane, Fairfax, Virginia", 38.85, -77.3, "residential", 0.10))
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": hits if "," in q else ())
    geo = Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                      region=[GuessLabel(name="Virginia", prob=0.41)],
                      points=[__import__("app.schemas", fromlist=["GuessPoint"]).GuessPoint(lat=36.7, lon=-82.5, prob=0.1)])
    m = lookup_mod.lookup([_text("t0", "PARAMOUNT")], geo)[0]
    assert m.confidence == "strong" and "Theater" in m.name  # not the road nearest the (wrong) pin


def test_no_dominant_result_stays_possible_and_ignores_the_pin(real_mode, monkeypatch):
    hits = (("Paramount Drive, Scott County, Virginia", 36.7, -82.6, "residential", 0.10),
            ("Paramount Lane, Fairfax, Virginia", 38.85, -77.3, "residential", 0.10))
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": hits if "," in q else ())
    geo = Geolocation(region=[GuessLabel(name="Virginia", prob=0.41)],
                      points=[__import__("app.schemas", fromlist=["GuessPoint"]).GuessPoint(lat=38.8, lon=-77.3, prob=0.1)])
    m = lookup_mod.lookup([_text("t0", "PARAMOUNT")], geo)[0]
    assert m.confidence == "weak" and "Scott County" in m.name  # OSM order, not nearest pin


def test_with_an_ai_reading_only_text_the_ai_also_read_is_searched(real_mode, monkeypatch):
    calls = []
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": calls.append(q) or ())
    ai = AIReading(model="m", sign_texts=["PARAMOUNT"])
    clues = [_text("a", "PARAMOUNT").model_copy(update={"source": "ocr+ai"}), _text("b", "Pa-pints")]
    lookup_mod.lookup(clues, _geo_dc(), ai)
    assert calls and all("PARAMOUNT" in q for q in calls)


def test_cville_names_a_city():
    from app.pipeline.text_intel import classify

    assert classify("DOWN TOWN CVILLE").kind == "place"


def test_truncated_answers_are_salvaged():
    cut = ('{"sign_texts": ["POPPY & PARLIAMENT", "TELEPHONE"], "clues": [{"clue": "red phone box", '
           '"suggests": "British theme"}], "features": ["telephone box"], "guesses": [{"place": '
           '"Huntsville, Alabama, USA", "confidence": 0.6, "why": "the sign and the phone b')
    data = vlm._parse(cut)
    assert data["sign_texts"] == ["POPPY & PARLIAMENT", "TELEPHONE"]
    assert data["features"] == ["telephone box"]
    r = vlm.to_reading(data, "m", 1)
    assert r.sign_texts and not r.error


def test_ask_requests_a_large_context(real_mode, monkeypatch):
    from PIL import Image

    sent = {}

    def fake(path, payload=None, timeout=5):
        if path == "/api/tags":
            return {"models": [{"name": config.VLM_MODEL}]}
        sent.update(payload)
        return {"message": {"content": '{"sign_texts": [], "clues": [], "features": [], "guesses": []}'}}

    monkeypatch.setattr(vlm, "_http", fake)
    r = vlm.ask(Image.new("RGB", (100, 100)))
    assert sent["options"]["num_ctx"] >= 8192
    assert "nothing usable" in r.error
