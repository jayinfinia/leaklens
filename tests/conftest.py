import io
import os
import random

os.environ["LEAKLENS_MOCK"] = "1"  # tests never download models

import pytest  # noqa: E402
from PIL import Image  # noqa: E402


def textured_image(w: int = 640, h: int = 480) -> Image.Image:
    """Busy pixels everywhere, so mock mode 'sees' clues until they're blurred."""
    rng = random.Random(0)
    img = Image.new("RGB", (w, h))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(w * h)])
    return img


def to_jpeg(img: Image.Image, gps: bool = True) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    exif[0x0110] = "Model X"  # Model
    if gps:
        g = exif.get_ifd(0x8825)
        g[1], g[2] = "N", (38.0, 1.0, 48.0)
        g[3], g[4] = "W", (78.0, 28.0, 48.0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif, quality=95)
    return buf.getvalue()


@pytest.fixture
def photo() -> bytes:
    return to_jpeg(textured_image())


@pytest.fixture
def photo_no_gps() -> bytes:
    return to_jpeg(textured_image(), gps=False)
