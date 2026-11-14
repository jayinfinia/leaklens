import pytest

from app.pipeline import lookup as lookup_mod
from app.pipeline.ocr import _Item, merge_lines, stack_vertical_letters, to_clues
from app.pipeline.risk import combine, score
from app.pipeline.text_intel import classify
from app.schemas import Clue, ExifReport, Geolocation, GuessLabel, TextMatch


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(lookup_mod, "_overpass", lambda points, regexes: [])


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

    def fake_search(q, countrycodes=""):
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
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": calls.append(q) or ())
    clues = [Clue(id=f"t{i}", kind="text", category="text:name", label=f"NAME{i}", score=0.9,
                  box=[0, 0, 1, 1], reason="r", weight=2.5) for i in range(5)]
    geo = Geolocation(region=[GuessLabel(name="Ohio", prob=0.5)])
    assert lookup_mod.lookup(clues, geo) == [] and len(calls) == 3


def _geo_dc():
    from app.schemas import GuessPoint

    return Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                       region=[GuessLabel(name="Washington, D.C.", prob=0.82)],
                       points=[GuessPoint(lat=38.9, lon=-77.0, prob=0.1)])


def test_single_common_word_across_the_country_is_only_a_possible_match(monkeypatch):
    """Huntsville photo: OCR reads only "POPPY", the model wrongly says D.C.,
    and "POPPY" matches businesses nationwide. We must not claim Richmond."""
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    hits = (("Poppy, Huntsville, Alabama", 34.73, -86.59, "restaurant"),
            ("Poppy, Richmond, Virginia", 37.54, -77.45, "restaurant"),
            ("Poppy, Seattle, Washington", 47.61, -122.33, "restaurant"))
    monkeypatch.setattr(lookup_mod, "_search",
                        lambda q, countrycodes="": () if "," in q else hits)
    clue = Clue(id="t", kind="text", category="text:name", label="POPPY", score=0.51,
                box=[0, 0, 1, 1], reason="r", weight=2.5)
    m = lookup_mod.lookup([clue], _geo_dc())[0]
    assert m.confidence == "weak"
    assert m.name.startswith("Poppy, Huntsville")  # search ranking, not the (wrong) model pin
    assert "share this name" in m.note
    weak = score(ExifReport(), [clue], Geolocation(), matches=[m])
    strong = score(ExifReport(), [clue], Geolocation(), matches=[m.model_copy(update={"confidence": "strong"})])
    assert weak.score < strong.score


def test_one_clear_place_in_the_hinted_region_is_strong(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    hits = (("The Paramount Theater, 215 East Main Street, Charlottesville", 38.0306, -78.4817, "theatre"),
            ("Paramount Theater (entrance), Charlottesville", 38.0307, -78.4819, "entrance"))
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": hits)
    geo = Geolocation(country=[GuessLabel(name="United States", prob=0.9)],
                      region=[GuessLabel(name="Virginia", prob=0.41)])
    clue = Clue(id="t", kind="text", category="text:name", label="PARAMOUNT", score=0.86,
                box=[0, 0, 1, 1], reason="r", weight=2.5)
    m = lookup_mod.lookup([clue], geo)[0]
    assert m.confidence == "strong" and "Charlottesville" in m.name


def test_specific_multiword_name_is_strong_without_a_hint(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    hit = (("Poppy & Parliament, Huntsville, Alabama", 34.73, -86.59, "pub"),)
    monkeypatch.setattr(lookup_mod, "_search",
                        lambda q, countrycodes="": () if "," in q else hit)
    clue = Clue(id="t", kind="text", category="text:name", label="POPPY & PARLIAMENT", score=0.5,
                box=[0, 0, 1, 1], reason="r", weight=2.5)
    m = lookup_mod.lookup([clue], _geo_dc())[0]
    assert m.confidence == "strong" and "Huntsville" in m.name


def test_query_cleanup():
    from app.pipeline.text_intel import clean_query

    assert clean_query("CONGRATULATIONS UVA MED_SCHOOL CLASS OF 2024") == "UVA MED SCHOOL"


def test_vertical_sign_unstacks_into_a_row():
    pytest.importorskip("scipy")
    from PIL import Image, ImageDraw

    from app.pipeline.vertical import is_tall, letter_rows, unstack

    col = Image.new("RGB", (60, 9 * 40), (150, 30, 30))  # red sign face
    d = ImageDraw.Draw(col)
    for i in range(9):
        d.rectangle([20, i * 40 + 8, 40, i * 40 + 32], fill=(240, 230, 200))  # cream "letters"
        d.ellipse([2, i * 40 + 15, 8, i * 40 + 21], fill=(250, 200, 60))  # border bulbs
    assert is_tall([0, 0, 60, 360])
    assert len(letter_rows(col)) == 9
    row, n = unstack(col)
    assert n == 9 and row.width > row.height * 3


# ----------------------------------------------------------- co-location
HUNTSVILLE = (34.7304, -86.5861)
POPPYS = (("Poppy, Richmond, Virginia", 37.54, -77.45, "restaurant"),
          ("Poppy, Seattle, Washington", 47.61, -122.33, "restaurant"),
          ("Poppy, Huntsville, Alabama", HUNTSVILLE[0], HUNTSVILLE[1], "restaurant"))


def _text(cid, label, kind="name", score=0.6):
    return Clue(id=cid, kind="text", category=f"text:{kind}", label=label, score=score,
                box=[0, 0, 1, 1], reason="r", weight=2.5)


def test_two_signs_near_each_other_settle_an_ambiguous_word(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    butcher = (("Butcher Shoppe, Huntsville, Alabama", HUNTSVILLE[0] + 0.0008, HUNTSVILLE[1], "shop"),
               ("Butcher Shoppe, Portland, Oregon", 45.52, -122.68, "shop"))

    def fake(q, countrycodes=""):
        if "," in q:
            return ()
        return POPPYS if q == "POPPY" else butcher

    monkeypatch.setattr(lookup_mod, "_search", fake)
    out = {m.text: m for m in lookup_mod.lookup([_text("a", "POPPY"), _text("b", "BUTCHER SHOPPE")], _geo_dc())}
    assert out["POPPY"].confidence == "strong"
    assert "Huntsville" in out["POPPY"].name and "Butcher Shoppe" in out["POPPY"].note


def test_overpass_finds_another_signs_name_next_to_one_candidate(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(lookup_mod, "_search", lambda q, countrycodes="": () if "," in q or q != "POPPY" else POPPYS)
    seen = {}

    def fake_overpass(points, regexes):
        seen["regexes"] = regexes
        return [("Poppy & Parliament", HUNTSVILLE[0] + 0.0002, HUNTSVILLE[1]),
                ("Poppy", 37.54, -77.45)]  # a candidate's own name never counts as support

    monkeypatch.setattr(lookup_mod, "_overpass", fake_overpass)
    clues = [_text("a", "POPPY"), _text("b", "PARLIAMENT", score=0.3), _text("c", "CLASS OF 2024", "generic")]
    m = [x for x in lookup_mod.lookup(clues, _geo_dc()) if x.text == "POPPY"][0]
    assert m.confidence == "strong" and "Huntsville" in m.name and "Poppy & Parliament" in m.note
    assert len(seen["regexes"]) == 1  # generic text is never used as evidence


@pytest.mark.parametrize("sign,osm_name,ok", [
    ("MAIN ST", "Main Street", True),
    ("POPPY & PARLIAMENT", "Poppy and Parliament", True),
    ("POPPY & PARLIAMENT", "Poppy & Parliament", True),
    ("illy", "Lilly's Cafe", False),
    ("PARAMOUNT", "The Paramount Theater", True),
])
def test_term_regex(sign, osm_name, ok):
    import re

    pattern = lookup_mod.term_regex(sign)
    assert bool(re.search(pattern, osm_name, re.I)) is ok


# ----------------------------------------------------------- zoomed crops
class _FakeEngine:
    min_conf = 0.3

    def __init__(self, reads):
        self.reads = reads

    def readtext(self, image):
        return self.reads(image)


def test_crop_reading_maps_boxes_back_to_the_full_image():
    from PIL import Image

    from app.pipeline.ocr import read_crop

    img = Image.new("RGB", (1000, 800))
    eng = _FakeEngine(lambda crop: [([[10, 10], [110, 10], [110, 40], [10, 40]], "PARLIAMENT", 0.9)])
    (item,) = read_crop(eng, img, [400, 300, 500, 340])  # 100x40 box, upscaled ~8x capped to 3x
    x0, y0, x1, y1 = item.box
    assert 388 <= x0 <= 400 and 295 <= y0 <= 300 and x1 < 440


def test_dedupe_keeps_the_fuller_reading():
    from app.pipeline.ocr import _Item, dedupe

    out = dedupe([_Item([100, 100, 200, 140], "POPPY", 0.9),
                  _Item([95, 98, 420, 142], "POPPY & PARLIAMENT", 0.6),
                  _Item([600, 100, 700, 140], "illy", 0.5)])
    assert sorted(i.text for i in out) == ["POPPY & PARLIAMENT", "illy"]


def test_engine_falls_back_to_easyocr_off_mac(monkeypatch):
    import sys

    from app import config, models
    from app.pipeline import ocr_engines

    monkeypatch.setattr(config, "OCR_ENGINE", "auto")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(models, "ocr", lambda: object())
    ocr_engines.get_engine.cache_clear()
    try:
        assert ocr_engines.get_engine().name == "easyocr"
    finally:
        ocr_engines.get_engine.cache_clear()


def test_ocr_slips_of_generic_words_stay_generic():
    assert classify("TELEPHONM").kind == "generic"
    assert classify("TELEPHONE").kind == "generic"
    assert classify("PARLIAMENT").kind == "name"


def test_combined_engine_pools_readings_with_each_cutoff():
    from app.pipeline.ocr_engines import CombinedEngine

    class E:
        def __init__(self, name, cutoff, rows):
            self.name, self.min_conf, self.rows = name, cutoff, rows

        def readtext(self, image):
            return self.rows

    a = E("apple", 0.25, [([[0, 0], [9, 0], [9, 9], [0, 9]], "POPPY & PARLIAMENT", 0.3),
                          ([[0, 0], [9, 0], [9, 9], [0, 9]], "junk", 0.1)])
    b = E("easyocr", 0.4, [([[0, 0], [9, 0], [9, 9], [0, 9]], "POPPY", 0.49),
                           ([[0, 0], [9, 0], [9, 9], [0, 9]], "lidly", 0.2)])
    texts = [r[1] for r in CombinedEngine([a, b]).readtext(None)]
    assert texts == ["POPPY & PARLIAMENT", "POPPY"]
