"""LeakLens's real HTTP code against a fake Ollama that reproduces the
failure modes reported for qwen3-vl and other thinking models."""

import pytest
from PIL import Image

from app import config
from app.pipeline import vlm
from tests.fake_ollama import FakeOllama


@pytest.fixture
def ollama(monkeypatch):
    servers = []

    def start(mode):
        s = FakeOllama(mode)
        servers.append(s)
        monkeypatch.setattr(config, "MOCK_MODE", False)
        monkeypatch.setattr(config, "VLM_MODE", "auto")
        monkeypatch.setattr(config, "OLLAMA_URL", s.url)
        monkeypatch.setattr(config, "VLM_MODEL", "qwen3-vl:8b-instruct")
        vlm._status.update(checked=None)
        return s

    yield start
    for s in servers:
        s.close()
    vlm._status.update(checked=None)


@pytest.mark.parametrize("mode,attempts", [
    ("good", 1), ("thinking", 1), ("inline_think", 1), ("prose", 1), ("truncated", 1),
    ("schema_empty", 2), ("extra_data", 1),
    ("long_think", 2),  # one cut-off try, then the same request with room to finish
])
def test_every_reported_failure_mode_still_yields_a_reading(ollama, mode, attempts):
    s = ollama(mode)
    r = vlm.ask(Image.new("RGB", (2000, 1500), (90, 90, 90)))
    assert not r.error, r.error
    assert "POPPY & PARLIAMENT" in r.sign_texts
    assert r.guesses[0].place == "Huntsville, Alabama, USA"
    assert len(s.requests) == attempts
    first = s.requests[0]
    assert "think" not in first  # never send think=false (empty-answer bugs)
    assert first["options"]["num_ctx"] >= 8192 and first["messages"][0]["images"]


def test_fallback_order_is_schema_then_json_then_free(ollama):
    s = ollama("nothing")
    r = vlm.ask(Image.new("RGB", (64, 64)))
    assert [type(q.get("format")).__name__ for q in s.requests] == ["dict", "str", "NoneType"]
    assert s.requests[1]["format"] == "json"
    assert r.error.startswith("no usable answer after 3 attempts")


def test_full_pipeline_with_real_code_paths(ollama, monkeypatch):
    """Tiny real-architecture vision models + fake Ollama + faked map search:
    the Huntsville case end to end. OCR only sees POPPY, the vision model
    says D.C.; the AI reads the whole sign and guesses Huntsville."""
    pytest.importorskip("torch")
    import importlib

    from app.pipeline import analyze, features, load_image, lookup as lookup_mod
    from tests.conftest import textured_image, to_jpeg
    from tests import test_models_offline as offline

    ollama("thinking")  # the exact failure seen on the Mac
    import torch
    from transformers import CLIPConfig, CLIPModel, Owlv2Config, Owlv2ForObjectDetection

    from app import models
    from app.pipeline import ocr_engines

    torch.manual_seed(0)
    clip = CLIPModel(CLIPConfig(text_config=offline.TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=32, patch_size=16), projection_dim=16)).eval()
    owl = Owlv2ForObjectDetection(Owlv2Config(text_config=offline.TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=64, patch_size=16), projection_dim=32)).eval()

    class PoppyOCR:
        def readtext(self, arr):
            return [([[10, 10], [90, 10], [90, 40], [10, 40]], "POPPY", 0.51)]

        def recognize(self, arr, detail=1):
            return []

    monkeypatch.setattr(config, "USE_GEOCLIP", False)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    monkeypatch.setattr(config, "OCR_ENGINE", "easyocr")
    models.device.cache_clear()
    geo_mod = importlib.import_module("app.pipeline.geolocate")
    geo_mod._text_embeds.cache_clear()
    ocr_engines.get_engine.cache_clear()
    monkeypatch.setattr(models, "streetclip", lambda: (offline.TinyClipProcessor(), clip))
    monkeypatch.setattr(models, "owl", lambda: (offline.TinyOwlProcessor(), owl))
    monkeypatch.setattr(models, "ocr", lambda: PoppyOCR())

    searched = []

    def fake_search(q, countrycodes=""):
        searched.append(q)
        if q.startswith("Huntsville"):
            return (("Huntsville, Madison County, Alabama", 34.7304, -86.5861, "city", 0.7),)
        if q == "POPPY & PARLIAMENT, Huntsville, Alabama":
            return (("Poppy & Parliament, 2nd Street, Huntsville, Alabama", 34.7306, -86.5859, "pub", 0.2),)
        return ()

    monkeypatch.setattr(lookup_mod, "_search", fake_search)
    monkeypatch.setattr(lookup_mod, "_overpass", lambda points, regexes: [])
    monkeypatch.setattr(features, "_fetch", lambda q: [])
    try:
        raw, upright = load_image(to_jpeg(textured_image(400, 300), gps=False))
        a = analyze(raw, upright, lookup=True, use_ai=True)
    finally:
        geo_mod._text_embeds.cache_clear()
        ocr_engines.get_engine.cache_clear()
        models.device.cache_clear()

    assert a.ai and not a.ai.error
    poppy = next(c for c in a.clues if "POPPY" in c.label)
    assert poppy.label == "POPPY & PARLIAMENT" and poppy.source == "ocr+ai" and poppy.box
    assert any(c.label == "Butcher Shoppe" and c.box is None for c in a.clues)
    assert a.ai.guesses[0].lat == pytest.approx(34.7304)
    m = next(m for m in a.text_matches if m.text == "POPPY & PARLIAMENT")
    assert m.confidence == "strong" and "Huntsville" in m.name
    assert "POPPY & PARLIAMENT, Huntsville, Alabama" in searched
    assert a.risk.level == "high"


def test_an_installed_thinking_build_is_used_when_instruct_is_missing(ollama):
    s = ollama("good")  # fake serves "qwen3-vl:8b" only
    st = vlm.status(refresh=True)
    assert st["ok"] and st["model"] == "qwen3-vl:8b"


def test_thinking_build_hint_in_the_error(ollama, monkeypatch):
    ollama("nothing")
    r = vlm.ask(Image.new("RGB", (64, 64)))
    assert "ollama pull qwen3-vl:8b-instruct" in r.error


# ------------------------------------------------------------ second pass: the chooser
OPTIONS = ["Nashville, TN", "Huntsville, AL", "Birmingham, AL"]


@pytest.mark.parametrize("mode,attempts", [
    ("good", 1), ("thinking", 1), ("inline_think", 1), ("prose", 1), ("truncated", 1),
    ("schema_empty", 2), ("extra_data", 1), ("long_think", 2),
])
def test_chooser_survives_every_failure_mode(ollama, mode, attempts):
    s = ollama(mode)
    from tests.fake_ollama import ANSWER

    reading = vlm.to_reading(ANSWER, "m", 0)
    assert reading.state == "Alabama" and reading.state_evidence
    c = vlm.choose(Image.new("RGB", (800, 600)), reading, OPTIONS, ["Alabama", "Tennessee"])
    assert not c.error, c.error
    assert c.place == "Huntsville, AL" and not c.other and c.options == OPTIONS
    assert c.state == "Alabama"
    assert len(s.requests) == attempts
    req = s.requests[0]
    assert req["messages"][0]["images"] and "think" not in req
    prompt = req["messages"][0]["content"]
    assert "POPPY & PARLIAMENT" in prompt and "- Huntsville, AL" in prompt and "Alabama, Tennessee" in prompt
    if mode != "truncated":  # cut off inside the confidence value
        assert c.confidence == 0.7


def test_chooser_other_answer(ollama):
    s = ollama("good")
    s.choice = "other"
    c = vlm.choose(Image.new("RGB", (64, 64)), None, OPTIONS)
    assert c.other and c.place == "Decatur, Alabama"


def test_chooser_reports_failure(ollama):
    s = ollama("good")
    s.choice = "none"
    c = vlm.choose(Image.new("RGB", (64, 64)), None, OPTIONS)
    assert c.error.startswith("no usable answer") and not c.place
    assert [type(q.get("format")).__name__ for q in s.requests] == ["dict", "str", "NoneType"]


@pytest.mark.parametrize("choice", ["pick", "other"])
def test_end_to_end_best_guess_with_fake_ollama_and_geoclip(ollama, monkeypatch, choice):
    """Tiny real-architecture vision models, a fake GeoCLIP whose pins sit in
    Nashville, and the fake Ollama. No map search: the AI's Huntsville guess
    is placed offline, the chooser sees named GeoCLIP clusters plus that
    guess, and fusion returns one best guess."""
    pytest.importorskip("torch")
    import importlib

    import torch
    from transformers import CLIPConfig, CLIPModel, Owlv2Config, Owlv2ForObjectDetection

    from app import models
    from app.pipeline import analyze, load_image, ocr_engines
    from app.pipeline import gazetteer as gz
    from tests import test_models_offline as offline
    from tests.conftest import textured_image, to_jpeg
    from tests.fake_geoclip import FakeGeoCLIP

    s = ollama("good")
    s.choice = choice
    torch.manual_seed(0)
    clip = CLIPModel(CLIPConfig(text_config=offline.TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=32, patch_size=16), projection_dim=16)).eval()
    owl = Owlv2ForObjectDetection(Owlv2Config(text_config=offline.TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=64, patch_size=16), projection_dim=32)).eval()
    nashville = (36.1659, -86.7844)
    monkeypatch.setattr(config, "USE_GEOCLIP", True)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    monkeypatch.setattr(config, "OCR_ENGINE", "easyocr")
    monkeypatch.setattr(config, "ATTRIBUTION", False)
    models.device.cache_clear()
    geo_mod = importlib.import_module("app.pipeline.geolocate")
    geo_mod._text_embeds.cache_clear()
    ocr_engines.get_engine.cache_clear()
    monkeypatch.setattr(models, "streetclip", lambda: (offline.TinyClipProcessor(), clip))
    monkeypatch.setattr(models, "owl", lambda: (offline.TinyOwlProcessor(), owl))
    monkeypatch.setattr(models, "ocr", lambda: offline.FakeOCR())
    monkeypatch.setattr(models, "geoclip", lambda: FakeGeoCLIP(target=nashville))
    try:
        raw, upright = load_image(to_jpeg(textured_image(400, 300), gps=False))
        a = analyze(raw, upright, lookup=False, use_ai=True, explain=False)
    finally:
        geo_mod._text_embeds.cache_clear()
        ocr_engines.get_engine.cache_clear()
        models.device.cache_clear()

    assert len(s.requests) == 2  # blind first pass, then the chooser
    assert a.geolocation.clusters and a.geolocation.clusters[0].place.removeprefix("near ") == "Nashville, TN"
    assert len(a.geolocation.points) == config.GEOCLIP_PINS_SHOWN
    assert a.ai.guesses[0].lat == pytest.approx(34.7304)  # placed offline, no map search
    ch = a.ai.choice
    assert ch and not ch.error and "Huntsville, AL" in ch.options and len(ch.options) >= 2
    assert a.best is not None and 0 < a.best.confidence <= config.FUSE_MAX_CONF
    if choice == "pick":
        assert ch.place == "Huntsville, AL" and a.best.source == "chooser"
        assert gz.km(a.best.lat, a.best.lon, 34.7304, -86.5861) < 25
        assert a.best.supports[:2] == ["chooser", "ai"]
    else:  # "other: Decatur, Alabama" is placed offline; Huntsville (first pass) still leads
        assert ch.other and ch.lat == pytest.approx(34.6059, abs=0.01)
        assert a.best.place in ("Huntsville, AL", "Decatur, AL")
        assert gz.km(a.best.lat, a.best.lon, 34.7304, -86.5861) < 40
