"""One final answer: gazetteer, GeoCLIP clusters and verification, the AI
chooser's parsing, and the fusion rules."""

import pytest

from app import config
from app.pipeline import fuse, gazetteer as gz, vlm
from app.pipeline.geolocate import cluster
from app.schemas import (AIChoice, AIGuess, AIReading, ExifReport, Geolocation, GuessLabel, GuessPoint,
                         PlaceCluster, TextMatch)

RICHMOND = (37.5538, -77.4603)
DENVER = (39.7392, -104.9847)


def us_geo(states=("Virginia", "North Carolina"), clusters=()):
    return Geolocation(
        country=[GuessLabel(name="United States", prob=0.9)],
        region=[GuessLabel(name=s, prob=p) for s, p in zip(states, (0.4, 0.2, 0.1))],
        clusters=list(clusters))


# ------------------------------------------------------------ gazetteer
def test_gazetteer_is_bundled_and_sane():
    cs = gz.cities()
    assert len(cs) > 2000 and all(c.population >= 20000 for c in cs)
    assert {c.state for c in cs} == set(gz.STATES)  # every state and D.C.


@pytest.mark.parametrize("place,label", [
    ("Charlottesville, Virginia, USA", "Charlottesville, VA"),
    ("charlottesville, va", "Charlottesville, VA"),
    ("Huntsville, Alabama, United States", "Huntsville, AL"),
    ("Huntsville, TX", "Huntsville, TX"),
    ("New York, NY", "New York City, NY"),
    ("Washington, D.C., USA", "Washington, DC"),
    ("St. Louis, Missouri", "St. Louis, MO"),
    ("Downtown, Richmond, Virginia", "Richmond, VA"),
])
def test_find_places(place, label):
    assert gz.find(place).label == label


@pytest.mark.parametrize("place", ["Toronto, Ontario, Canada", "Somewhere in the eastern US", "", "Paris, France"])
def test_find_rejects_unknown_or_foreign_places(place):
    assert gz.find(place) is None


def test_nearest_and_describe():
    c, d = gz.nearest(38.03, -78.48)
    assert c.label == "Charlottesville, VA" and d < 2
    assert gz.describe(38.03, -78.48)[0] == "Charlottesville, VA"
    assert gz.describe(38.2, -78.7)[0].startswith("near ")
    assert gz.describe(48.85, 2.35) == ("", None)  # Paris: not in a US list
    assert gz.state_at(*RICHMOND) == "VA" and gz.state_at(48.85, 2.35) == ""


# ------------------------------------------------------------ GeoCLIP clusters
def test_cluster_merges_pins_within_25_km_and_names_them():
    pts = [GuessPoint(lat=38.03, lon=-78.48, prob=0.2), GuessPoint(lat=38.1, lon=-78.4, prob=0.1),
           GuessPoint(lat=RICHMOND[0], lon=RICHMOND[1], prob=0.15), GuessPoint(lat=48.85, lon=2.35, prob=0.01)]
    cs = cluster(pts)
    assert [c.n for c in cs] == [2, 1, 1]
    assert cs[0].place == "Charlottesville, VA" and cs[0].prob == pytest.approx(0.3)
    assert cs[1].place == "Richmond, VA"
    assert cs[2].place == ""  # Paris
    assert 38.03 < cs[0].lat < 38.1  # probability-weighted center


# ------------------------------------------------------------ GeoCLIP verification
def test_geoscorer_ranks_points_and_scores_any_coordinate():
    torch = pytest.importorskip("torch")
    from PIL import Image

    from app.pipeline.geolocate import GeoScorer
    from tests.fake_geoclip import FakeGeoCLIP

    s = GeoScorer(Image.new("RGB", (64, 64)), model=FakeGeoCLIP(target=RICHMOND))
    pins = s.points(30)
    assert len(pins) == 30 and gz.km(pins[0].lat, pins[0].lon, *RICHMOND) < 40
    assert pins[0].prob >= pins[-1].prob
    near, mid, far = s.score([RICHMOND, (38.03, -78.48), DENVER])
    assert near == 1.0 and near > mid > far >= 0.0
    assert s.score([]) == []
    assert torch.is_tensor(s.sorted_logits)


def test_geolocate_uses_top_k_pins_and_clusters(monkeypatch):
    pytest.importorskip("torch")
    from PIL import Image

    import importlib

    from app import models
    from tests.fake_geoclip import FakeGeoCLIP

    geo_mod = importlib.import_module("app.pipeline.geolocate")

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "USE_GEOCLIP", True)
    monkeypatch.setattr(models, "geoclip", lambda: FakeGeoCLIP(target=RICHMOND))
    img = Image.new("RGB", (64, 64))
    s = geo_mod.scorer(img)
    pins = s.points(config.GEOCLIP_TOP_K)
    assert len(pins) == config.GEOCLIP_TOP_K
    cs = cluster(pins)
    assert cs[0].place.removeprefix("near ") == "Richmond, VA"
    assert geo_mod.score_points(img, [RICHMOND])[0] == 1.0


# ------------------------------------------------------------ chooser parsing
OPTS = ["Richmond, VA", "Charlottesville, VA", "Norfolk, VA"]


@pytest.mark.parametrize("answer,place,other", [
    ("Charlottesville, VA", "Charlottesville, VA", False),
    ("charlottesville, va", "Charlottesville, VA", False),
    ("Charlottesville", "Charlottesville, VA", False),
    ("Charlottesville, Virginia, USA", "Charlottesville, VA", False),
    ("2", "Charlottesville, VA", False),
    ("other: Lynchburg, Virginia", "Lynchburg, Virginia", True),
    ("Other - Roanoke, VA", "Roanoke, VA", True),
    ("Staunton, VA", "Staunton, VA", True),  # not on the list: counts as other
])
def test_parse_choice(answer, place, other):
    c = vlm.parse_choice({"choice": answer, "confidence": 0.7, "state": "Virginia", "evidence": "x"}, OPTS)
    assert (c.place, c.other) == (place, other)
    assert c.confidence == 0.7 and c.state == "Virginia" and c.options == OPTS and not c.error


def test_parse_choice_with_no_place_or_bad_confidence():
    c = vlm.parse_choice({"choice": "other", "confidence": "high"}, OPTS)
    assert c.error and not c.place and c.confidence == 0.0
    c = vlm.parse_choice({"choice": "Norfolk, VA", "confidence": 3}, OPTS)
    assert c.confidence == 1.0


def test_choose_prompt_lists_options_and_first_pass_findings():
    r = AIReading(model="m", sign_texts=["POPPY & PARLIAMENT"], state="Alabama",
                  guesses=[AIGuess(place="Huntsville, Alabama", confidence=0.6)])
    p = vlm.choose_prompt(r, OPTS, ["Virginia", "Maryland"])
    assert all(f"- {o}" in p for o in OPTS)
    assert "POPPY & PARLIAMENT" in p and "Huntsville, Alabama (60%)" in p and "Virginia, Maryland" in p
    assert '"other: City, State"' in p


def test_options_are_named_deduped_shuffled_and_capped():
    geo = us_geo(clusters=[
        PlaceCluster(lat=38.03, lon=-78.48, prob=0.3, place="Charlottesville, VA", km=1),
        PlaceCluster(lat=37.6, lon=-77.5, prob=0.2, place="near Richmond, VA", km=20),
        PlaceCluster(lat=48.85, lon=2.35, prob=0.1, place="")])
    ai = AIReading(model="m", guesses=[
        AIGuess(place="Charlottesville, Virginia, USA", confidence=0.5, lat=38.0293, lon=-78.4767),
        AIGuess(place="Somewhere in the eastern US", confidence=0.1)])
    weak = TextMatch(clue_id="t", text="CORNER", query="q", name="Corner, Norfolk, Virginia",
                     lat=36.85, lon=-76.29, confidence="weak")
    opts = fuse.options(geo, ai, [weak])
    names = [o.name for o in opts]
    # fewer than 4: the biggest city of each likely state is added
    assert sorted(names) == ["Charlotte, NC", "Charlottesville, VA", "Norfolk, VA", "Richmond, VA",
                             "Virginia Beach, VA"]
    assert names == [o.name for o in fuse.options(geo, ai, [weak])]  # deterministic
    assert len(fuse.options(geo, ai, [weak], limit=2)) == 2
    rich = next(o for o in opts if o.name == "Richmond, VA")
    assert (rich.lat, rich.lon) == (37.6, -77.5)  # "near": the cluster's own center


def test_locate_choice_from_list_and_other():
    opts = [fuse.Option("Richmond, VA", 37.6, -77.5, "geoclip")]
    c = fuse.locate_choice(AIChoice(place="Richmond, VA", confidence=0.6), opts)
    assert (c.lat, c.lon) == (37.6, -77.5)
    c = fuse.locate_choice(AIChoice(place="Lynchburg, Virginia", confidence=0.6, other=True), opts)
    assert c.lat == pytest.approx(37.41, abs=0.05)
    c = fuse.locate_choice(AIChoice(place="Tiny Hamlet, Virginia", confidence=0.6, other=True), opts)
    assert c.lat is None


# ------------------------------------------------------------ fusion
def test_strong_text_match_beats_a_confident_ai():
    ai = AIReading(model="m", guesses=[AIGuess(place="Denver, CO", confidence=0.95, lat=DENVER[0], lon=DENVER[1])],
                   choice=AIChoice(place="Denver, CO", confidence=0.9, lat=DENVER[0], lon=DENVER[1]))
    m = TextMatch(clue_id="t", text="POPPY & PARLIAMENT", query="q", lat=34.7306, lon=-86.5859,
                  name="Poppy & Parliament, 2nd Street, Huntsville, Alabama", confidence="strong")
    geo = us_geo(("Colorado",), [PlaceCluster(lat=DENVER[0], lon=DENVER[1], prob=0.5, place="Denver, CO")])
    b = fuse.fuse(geo, ai, [m])
    assert b.source == "text" and b.lat == 34.7306 and "Huntsville" in b.place
    assert b.confidence > 0.9 and "POPPY & PARLIAMENT" in b.why


def test_gps_metadata_wins_over_everything():
    exif = ExifReport(has_exif=True, gps=GuessPoint(lat=38.03, lon=-78.48, prob=1.0))
    b = fuse.fuse(us_geo(), None, [], exif=exif)
    assert b.source == "metadata" and b.confidence == 1.0 and b.place == "Charlottesville, VA"


def test_candidate_backed_by_several_sources_beats_one_confident_guess():
    ai = AIReading(model="m", state="Virginia", guesses=[
        AIGuess(place="Denver, Colorado", confidence=0.7, lat=DENVER[0], lon=DENVER[1]),
        AIGuess(place="Richmond, Virginia", confidence=0.3, lat=RICHMOND[0], lon=RICHMOND[1])])
    geo = us_geo(("Virginia",), [PlaceCluster(lat=37.6, lon=-77.4, prob=0.4, place="near Richmond, VA"),
                                 PlaceCluster(lat=33.0, lon=-97.0, prob=0.1, place="near Denton, TX")])
    b = fuse.fuse(geo, ai, [])
    assert b.place == "Richmond, VA" and b.source == "ai"
    assert b.supports == ["ai", "geoclip"]
    assert gz.km(b.lat, b.lon, *RICHMOND) < 1  # the named city, not the cluster center
    assert "GeoCLIP" in b.why and "Virginia" in b.why


def test_chooser_pick_backed_by_geoclip_wins():
    ai = AIReading(model="m", guesses=[AIGuess(place="Denver, Colorado", confidence=0.6,
                                               lat=DENVER[0], lon=DENVER[1])],
                   choice=AIChoice(place="Richmond, VA", confidence=0.5, lat=RICHMOND[0], lon=RICHMOND[1]))
    geo = us_geo(("Virginia",), [PlaceCluster(lat=37.6, lon=-77.4, prob=0.3, place="near Richmond, VA")])
    b = fuse.fuse(geo, ai, [])
    assert b.source == "chooser" and b.place == "Richmond, VA"
    assert b.supports == ["chooser", "geoclip"] and "picked it" in b.why


def test_geoclip_verification_breaks_a_tie():
    class Scorer:
        def score(self, coords):
            return [1.0 if gz.km(lat, lon, *DENVER) < 50 else 0.2 for lat, lon in coords]

    ai = AIReading(model="m", guesses=[
        AIGuess(place="Richmond, Virginia", confidence=0.5, lat=RICHMOND[0], lon=RICHMOND[1]),
        AIGuess(place="Denver, Colorado", confidence=0.5, lat=DENVER[0], lon=DENVER[1])])
    geo = Geolocation(country=[GuessLabel(name="Canada", prob=0.5)])
    assert fuse.fuse(geo, ai, []).place == "Richmond, VA"  # tie: first listed
    assert fuse.fuse(geo, ai, [], scorer=Scorer()).place == "Denver, CO"


def test_geoclip_alone_and_state_fallback():
    geo = us_geo(("Ohio",), [PlaceCluster(lat=40.4, lon=-82.9, prob=0.04, place="near Delaware, OH")])
    b = fuse.fuse(geo, None, [])
    assert b.source == "geoclip" and b.place == "near Delaware, OH" and b.supports == ["geoclip"]

    b = fuse.fuse(us_geo(("Ohio",)), None, [])
    assert b.source == "state" and b.place == "somewhere in Ohio" and b.confidence <= 0.3
    assert gz.state_at(b.lat, b.lon) == "OH"

    assert fuse.fuse(Geolocation(), None, []) is None


def test_place_guesses_fills_coordinates_offline():
    ai = AIReading(model="m", guesses=[AIGuess(place="Huntsville, Alabama, USA", confidence=0.6),
                                       AIGuess(place="Kyoto, Japan", confidence=0.1),
                                       AIGuess(place="Mobile, AL", confidence=0.2, lat=1.0, lon=2.0)])
    out = fuse.place_guesses(ai)
    assert out.guesses[0].lat == pytest.approx(34.7304) and out.guesses[1].lat is None
    assert out.guesses[2].lat == 1.0  # existing coordinates are kept
    assert fuse.place_guesses(None) is None


def test_mock_analysis_has_a_sensible_best_guess(photo_no_gps):
    from app.pipeline import analyze, load_image

    raw, upright = load_image(photo_no_gps)
    a = analyze(raw, upright, lookup=False, use_ai=True)
    assert a.best and a.best.place == "Charlottesville, VA" and a.best.source == "chooser"
    assert a.ai.choice and "Charlottesville, VA" in a.ai.choice.options and len(a.ai.choice.options) >= 2
    assert 0 < a.best.confidence <= config.FUSE_MAX_CONF

    a = analyze(raw, upright, lookup=True, use_ai=True)
    assert a.best.source == "text" and a.ai.choice is None  # strong match: no second AI call

    a = analyze(raw, upright, lookup=False, use_ai=False)
    assert a.best and a.best.source == "geoclip" and a.ai is None


def test_chooser_can_be_turned_off(photo_no_gps, monkeypatch):
    from app.pipeline import analyze, load_image

    monkeypatch.setattr(config, "VLM_CHOOSE", False)
    raw, upright = load_image(photo_no_gps)
    a = analyze(raw, upright, lookup=False, use_ai=True)
    assert a.ai.choice is None and a.best.source == "ai" and a.best.place == "Charlottesville, VA"
