"""Exercise the real (non-mock) model code paths with tiny, randomly
initialized models of the same architectures. No downloads: this checks
shapes, batching, coordinate handling and library API compatibility, not
accuracy. Skipped when torch/transformers aren't installed (e.g. CI).
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    BatchEncoding, BatchFeature, CLIPConfig, CLIPImageProcessor, CLIPModel,
    Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor,
)

from app import config, models  # noqa: E402
import importlib  # noqa: E402

from app.pipeline import analyze, load_image  # noqa: E402
from app.pipeline.safe import plan  # noqa: E402
from tests.conftest import textured_image, to_jpeg  # noqa: E402

geolocate = importlib.import_module("app.pipeline.geolocate")

TXT = dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
           vocab_size=100, eos_token_id=99, bos_token_id=98, pad_token_id=0)


def _ids(texts, n=8):
    rows = [[1 + b % 97 for b in hashlib.sha256(t.encode()).digest()[:n]] for t in texts]
    ids = torch.tensor(rows)
    return ids, torch.ones_like(ids)


class TinyTokenizer:
    def __call__(self, texts, padding=True, return_tensors="pt"):
        ids, mask = _ids(texts)
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})


class TinyClipProcessor:
    tokenizer = TinyTokenizer()
    image_processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})


class TinyOwlProcessor:
    image_processor = Owlv2ImageProcessor(size={"height": 64, "width": 64})

    def __call__(self, text, images, return_tensors="pt"):
        ids, mask = _ids(text[0])
        px = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return BatchFeature({"input_ids": ids, "attention_mask": mask, "pixel_values": px})


class FakeOCR:
    def readtext(self, arr):
        return [
            ([[10, 10], [90, 10], [90, 30], [10, 30]], "MAIN ST", 0.91),
            ([[100, 50], [110, 50], [110, 60], [100, 60]], "x", 0.99),  # too short
            ([[5, 100], [80, 100], [80, 120], [5, 120]], "BLURRY", 0.1),  # low confidence
        ]


@pytest.fixture
def real_paths(monkeypatch):
    torch.manual_seed(0)
    clip = CLIPModel(CLIPConfig(text_config=TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=32, patch_size=16), projection_dim=16)).eval()
    owl = Owlv2ForObjectDetection(Owlv2Config(text_config=TXT, vision_config=dict(
        hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
        image_size=64, patch_size=16), projection_dim=32)).eval()

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "USE_GEOCLIP", False)
    monkeypatch.setattr(config, "DETECT_THRESHOLD", 0.0)  # random weights: keep every box
    monkeypatch.setattr(config, "SCENE_MIN_PROB", 0.0)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    models.device.cache_clear()
    geolocate._text_embeds.cache_clear()
    monkeypatch.setattr(models, "streetclip", lambda: (TinyClipProcessor(), clip))
    monkeypatch.setattr(models, "owl", lambda: (TinyOwlProcessor(), owl))
    monkeypatch.setattr(models, "ocr", lambda: FakeOCR())
    monkeypatch.setattr(config, "OCR_ENGINE", "easyocr")
    from app.pipeline import ocr_engines

    ocr_engines.get_engine.cache_clear()
    yield
    geolocate._text_embeds.cache_clear()
    ocr_engines.get_engine.cache_clear()


def test_full_analysis_runs_on_real_architectures(real_paths):
    raw, upright = load_image(to_jpeg(textured_image(400, 300), gps=False))
    a = analyze(raw, upright)

    assert len(a.geolocation.country) == 5
    assert (a.geolocation.country[0].name == "United States") == bool(a.geolocation.region)
    assert abs(sum(p.prob for p in a.geolocation.country) - 1) < 1  # top-5 slice of a softmax

    texts = [c for c in a.clues if c.kind == "text"]
    assert [c.label for c in texts] == ["MAIN ST"]

    for c in a.clues:
        x0, y0, x1, y1 = c.box
        assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 300
    ranked = sorted(a.clues, key=lambda c: c.score, reverse=True)[: config.ATTRIBUTION_MAX_CLUES]
    assert all(c.impact is not None and c.impact >= 0 for c in ranked)

    assert len(a.scene) == len(config.SCENE_CUES)  # threshold 0 keeps every cue
    assert 0 <= a.risk.score <= 100


def test_batched_scores_match_single_scores(real_paths):
    imgs = [textured_image(200, 150), textured_image(320, 200).rotate(90, expand=True)]
    batch = geolocate.label_probs(imgs, geolocate.COUNTRIES)
    for i, im in enumerate(imgs):
        single = geolocate.label_probs([im], geolocate.COUNTRIES)[0]
        assert torch.allclose(batch[i], single, atol=1e-5)


def test_make_safe_plan_on_real_architectures(real_paths):
    raw, upright = load_image(to_jpeg(textured_image(400, 300), gps=False))
    a = analyze(raw, upright)
    ids, note = plan(upright, a)
    assert set(ids) <= {c.id for c in a.clues}
    assert note
