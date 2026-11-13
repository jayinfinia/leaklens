import pytest

from app.pipeline import lookup as lookup_mod
from app.pipeline.ocr import _Item, merge_lines, stack_vertical_letters, to_clues
from app.pipeline.risk import combine, score
from app.pipeline.text_intel import classify
from app.schemas import Clue, ExifReport, Geolocation, GuessLabel, TextMatch


# ----------------------------------------------------------- text intel
@pytest.mark.parametrize("text,kind", [
    ("CONGRATULATIONS", "generic"),
    ("CLASS OF 2024", "generic"),
    ("2024", "generic"),
    ("UVA MED SCHOOL", "institution"),
    ("Main St", "street"),
    ("434-555-0182", "phone"),
    ("Charlottesville", "place"),
    ("Welcome to Alabama", "place"),
    ("www.poppyandparliament.com", "web"),
    ("POPPY & PARLIAMENT", "name"),
    ("PARAMOUNT", "name"),
    ("217", "number"),
])
def test_classify(text, kind):
    assert classify(text).kind == kind


def test_specific_text_outweighs_generic():
    assert classify("Main St").weight > classify("POPPY").weight > classify("CONGRATULATIONS").weight


# ----------------------------------------------------------- OCR stitching
def test_vertical_marquee_letters_become_one_word():
    letters = "PARAMOUNT"
    items = [_Item([100, 50 + i * 40, 130, 85 + i * 40], ch, 0.3) for i, ch in enumerate(letters)]
    items.append(_Item([400, 60, 410, 80], "x", 0.9))  # stray single letter elsewhere
    out = stack_vertical_letters(items)
    assert [i.text for i in out] == ["PARAMOUNT"]
    assert out[0].box == [100, 50, 130, 85 + 8 * 40]


def test_words_on_one_sign_merge_and_far_words_dont():
    items = [
        _Item([10, 10, 200, 40], "CONGRATULATIONS", 0.98),
        _Item([20, 48, 90, 78], "UVA", 0.8), _Item([100, 48, 160, 78], "MED", 0.8),
        _Item([170, 48, 260, 78], "SCHOOL", 0.8),
        _Item([900, 500, 1000, 530], "POPPY", 0.5),
    ]
    texts = sorted(i.text for i in merge_lines(items))
    assert texts == ["CONGRATULATIONS UVA MED SCHOOL", "POPPY"]


def test_merged_sign_is_weighted_by_its_most_revealing_part():
    clues = to_clues(merge_lines([
        _Item([10, 10, 200, 40], "CONGRATULATIONS", 0.98),
        _Item([20, 48, 160, 78], "UVA MED SCHOOL", 0.8),
    ]))
    assert len(clues) == 1 and clues[0].category == "text:institution"


# ----------------------------------------------------------- risk calibration
def _clue(i, cat="storefront", conf=0.3, weight=2.5):
    return Clue(id=f"c{i}", kind="object", category=cat, label=cat, score=conf,
                box=[0, 0, 10, 10], reason="r", weight=weight)


def test_many_weak_duplicates_do_not_saturate():
    geo = Geolocation(country=[GuessLabel(name="United States", prob=0.8)])
    r = score(ExifReport(), [_clue(i) for i in range(15)], geo)
    assert r.score < 60, r.score


def test_gps_alone_is_100():
    from app.schemas import GuessPoint

    r = score(ExifReport(has_exif=True, gps=GuessPoint(lat=1, lon=2, prob=1)), [], Geolocation())
    assert r.score == 100 and r.level == "high"


def test_text_match_drives_high_risk():
    clue = Clue(id="txt-0", kind="text", category="text:name", label="POPPY", score=0.5,
                box=[0, 0, 1, 1], reason="r", weight=2.5)
    m = TextMatch(clue_id="txt-0", text="POPPY", query="POPPY, Alabama", name="Poppy, Huntsville",
                  lat=34.7, lon=-86.6)
    assert score(ExifReport(), [clue], Geolocation(), matches=[m]).level == "high"


def test_combine_bounds():
    assert combine([]) == 0
    assert combine([0.5, 0.5]) == 75
    assert combine([1.0, 0.2]) == 100


# ----------------------------------------------------------- lookup
def test_lookup_uses_region_hint_then_falls_back(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    calls = []

    def fake_search(q):
        calls.append(q)
        return (("Paramount Theater, East Main Street, Charlottesville, Virginia", 38.03, -78.48, "theatre"),) \
            if q == "PARAMOUNT, Virginia" else ()

    monkeypatch.setattr(lookup_mod, "_search", fake_search)
    geo = Geolocation(country=[GuessLabel(name="United States", prob=0.8)],
                      region=[GuessLabel(name="Virginia", prob=0.41)])
    clues = [
        Clue(id="txt-1", kind="text", category="text:name", label="PARAMOUNT", score=0.6,
             box=[0, 0, 1, 1], reason="r", weight=2.5),
        Clue(id="txt-2", kind="text", category="text:generic", label="CLASS OF 2024", score=0.99,
             box=[0, 0, 1, 1], reason="r", weight=0.4),
    ]
    matches = lookup_mod.lookup(clues, geo)
    assert calls == ["PARAMOUNT, Virginia"]  # generic text is never searched
    assert matches[0].clue_id == "txt-1" and "Charlottesville" in matches[0].name


def test_lookup_respects_query_budget(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "LOOKUP_MAX_QUERIES", 3)
    calls = []
    monkeypatch.setattr(lookup_mod, "_search", lambda q: calls.append(q) or ())
    clues = [Clue(id=f"t{i}", kind="text", category="text:name", label=f"NAME{i}", score=0.9,
                  box=[0, 0, 1, 1], reason="r", weight=2.5) for i in range(5)]
    geo = Geolocation(region=[GuessLabel(name="Ohio", prob=0.5)])
    assert lookup_mod.lookup(clues, geo) == [] and len(calls) == 3
