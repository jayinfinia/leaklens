import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import analyze, load_image
from app.pipeline.exif import read_exif
from app.pipeline.scrub import blur_regions

client = TestClient(app)


def post(path, data_bytes, name="photo.jpg", mime="image/jpeg", **form):
    return client.post(path, files={"file": (name, data_bytes, mime)}, data=form or None)


# ----------------------------------------------------------- metadata
def test_reads_gps_and_device(photo):
    raw, _ = load_image(photo)
    report = read_exif(raw)
    assert report.has_exif
    assert report.device == "TestCam Model X"
    assert abs(report.gps.lat - 38.03) < 0.001
    assert abs(report.gps.lon + 78.48) < 0.001


def test_gps_forces_high_risk(photo):
    raw, upright = load_image(photo)
    assert analyze(raw, upright).risk.level == "high"


def test_scrub_strips_all_metadata(photo):
    _, upright = load_image(photo)
    buf = io.BytesIO()
    blur_regions(upright, [[10, 10, 200, 120]]).save(buf, format="JPEG")
    assert not Image.open(io.BytesIO(buf.getvalue())).getexif()


# ----------------------------------------------------------- blurring
def test_blur_changes_only_the_region():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            img.putpixel((x, y), (0, 0, 0) if (x // 4 + y // 4) % 2 else (255, 255, 255))
    out = blur_regions(img, [[20, 20, 80, 80]])
    assert out.getpixel((150, 150)) == (255, 255, 255)
    assert out.crop((20, 20, 80, 80)).tobytes() != img.crop((20, 20, 80, 80)).tobytes()


def test_blur_ignores_degenerate_boxes():
    img = Image.new("RGB", (50, 50), (10, 20, 30))
    assert blur_regions(img, [[5, 5, 5, 5], [60, 60, 90, 90]]).tobytes() == img.tobytes()


# ----------------------------------------------------------- analysis
def test_analysis_has_clues_scene_and_attribution(photo_no_gps):
    raw, upright = load_image(photo_no_gps)
    a = analyze(raw, upright)
    assert {c.kind for c in a.clues} == {"object", "text"}
    assert all(c.impact is not None for c in a.clues)
    assert a.scene and a.scene[0].category == "power_lines"
    assert any(f.source == "scene" for f in a.risk.factors)


def test_explain_off_skips_attribution(photo_no_gps):
    raw, upright = load_image(photo_no_gps)
    assert all(c.impact is None for c in analyze(raw, upright, explain=False).clues)


def test_risk_factors_combine_as_noisy_or(photo_no_gps):
    import math

    raw, upright = load_image(photo_no_gps)
    r = analyze(raw, upright).risk
    expected = 100 * (1 - math.prod(1 - f.points / 100 for f in r.factors))
    assert abs(r.score - expected) <= 1
    assert r.factors == sorted(r.factors, key=lambda f: f.points, reverse=True)


def test_large_images_keep_boxes_in_original_coordinates():
    from tests.conftest import textured_image, to_jpeg

    raw, upright = load_image(to_jpeg(textured_image(2560, 1920), gps=False))
    a = analyze(raw, upright)
    assert (a.width, a.height) == (2560, 1920)
    # mock clue spans x up to 34% of the width: 870px here, 435px in the 1280px working copy
    assert abs(max(c.box[2] for c in a.clues) - 0.34 * 2560) < 2


# ----------------------------------------------------------- API
def test_api_analyze_then_scrub(photo):
    body = post("/api/analyze", photo).json()
    assert body["mock"] and body["clues"]

    boxes = json.dumps([c["box"] for c in body["clues"]])
    out = post("/api/scrub", photo, boxes=boxes).json()
    assert out["after"]["exif"]["has_exif"] is False
    assert out["after"]["clues"] == []  # blurred clues are gone on re-check
    assert out["after"]["risk"]["score"] < out["before"]["score"]


def test_api_make_safe_reaches_lower_risk(photo):
    out = post("/api/safe", photo).json()
    assert out["blurred"]
    assert out["note"]
    assert out["after"]["risk"]["score"] < out["before"]["score"]
    assert out["after"]["exif"]["gps"] is None


def test_api_rejects_bad_boxes(photo):
    assert post("/api/scrub", photo, boxes="[[1,2,3]]").status_code == 400
    assert post("/api/scrub", photo, boxes="not json").status_code == 400


def test_api_rejects_non_images():
    assert post("/api/analyze", b"hello", name="x.txt", mime="text/plain").status_code == 400


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200 and "LeakLens" in res.text


def test_heic_gets_a_browser_preview():
    pillow_heif = pytest.importorskip("pillow_heif")
    from tests.conftest import textured_image

    buf = io.BytesIO()
    try:
        textured_image(320, 240).save(buf, format="HEIF")
    except Exception as exc:  # encoder not built into this wheel
        pytest.skip(f"HEIF encoding unavailable: {exc}")
    body = post("/api/analyze", buf.getvalue(), name="IMG_0001.HEIC", mime="image/heic").json()
    assert body["preview_b64"]


def test_api_lookup_toggle(photo_no_gps):
    off = post("/api/analyze?lookup=false", photo_no_gps).json()
    on = post("/api/analyze?lookup=true", photo_no_gps).json()
    assert off["text_matches"] == [] and off["lookup_used"] is False
    assert on["text_matches"] and on["lookup_used"] is True
    assert on["risk"]["score"] >= off["risk"]["score"]

    safe = post("/api/safe?lookup=true", photo_no_gps).json()
    assert on["text_matches"][0]["clue_id"] in safe["blurred"]
    assert safe["after"]["text_matches"] == []
