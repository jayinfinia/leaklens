import io
import json
import os

os.environ["LEAKLENS_MOCK"] = "1"  # never download models in tests

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402
from app.pipeline import analyze, load_image  # noqa: E402
from app.pipeline.exif import read_exif  # noqa: E402
from app.pipeline.scrub import blur_regions  # noqa: E402


def make_jpeg(with_gps: bool = True) -> bytes:
    img = Image.new("RGB", (640, 480), (120, 160, 200))
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    exif[0x0110] = "Model X"  # Model
    if with_gps:
        gps = exif.get_ifd(0x8825)
        gps[1], gps[2] = "N", (38.0, 1.0, 48.0)
        gps[3], gps[4] = "W", (78.0, 28.0, 48.0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_reads_gps_and_device():
    raw, _ = load_image(make_jpeg())
    report = read_exif(raw)
    assert report.has_exif
    assert report.device == "TestCam Model X"
    assert abs(report.gps.lat - 38.03) < 0.001
    assert abs(report.gps.lon + 78.48) < 0.001


def test_gps_forces_high_risk():
    raw, upright = load_image(make_jpeg(with_gps=True))
    assert analyze(raw, upright).risk.level == "high"


def test_scrub_strips_all_metadata():
    _, upright = load_image(make_jpeg())
    clean = blur_regions(upright, [[10, 10, 200, 120]])
    buf = io.BytesIO()
    clean.save(buf, format="JPEG")
    reopened = Image.open(io.BytesIO(buf.getvalue()))
    assert not reopened.getexif()


def test_blur_changes_only_the_region():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            img.putpixel((x, y), (0, 0, 0) if (x // 4 + y // 4) % 2 else (255, 255, 255))
    out = blur_regions(img, [[20, 20, 80, 80]])
    assert out.getpixel((150, 150)) == (255, 255, 255)
    assert out.crop((20, 20, 80, 80)).tobytes() != img.crop((20, 20, 80, 80)).tobytes()


def test_api_round_trip():
    client = TestClient(app)
    files = {"file": ("photo.jpg", make_jpeg(), "image/jpeg")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["mock"] and body["clues"]

    boxes = json.dumps([c["box"] for c in body["clues"]])
    files = {"file": ("photo.jpg", make_jpeg(), "image/jpeg")}
    res = client.post("/api/scrub", files=files, data={"boxes": boxes})
    assert res.status_code == 200
    out = res.json()
    assert out["after"]["exif"]["has_exif"] is False
    assert out["after"]["risk"]["score"] < out["before"]["score"]


def test_rejects_non_images():
    client = TestClient(app)
    res = client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 400
