"""Orchestrates the four layers: metadata -> clues -> geolocation -> risk."""

import io

from PIL import Image, ImageOps

from .. import config
from ..schemas import Analysis, Clue
from .detect import detect_clues
from .exif import read_exif
from .geolocate import geolocate
from .ocr import find_text
from .risk import score


def load_image(data: bytes) -> tuple[Image.Image, Image.Image]:
    """Returns (raw image with metadata, upright RGB image)."""
    raw = Image.open(io.BytesIO(data))
    raw.load()
    upright = ImageOps.exif_transpose(raw).convert("RGB")
    return raw, upright


def _downscale(img: Image.Image) -> tuple[Image.Image, float]:
    side = max(img.size)
    if side <= config.MAX_ANALYSIS_SIDE:
        return img, 1.0
    factor = config.MAX_ANALYSIS_SIDE / side
    small = img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)
    return small, factor


def _rescale(clues: list[Clue], factor: float) -> list[Clue]:
    if factor == 1.0:
        return clues
    return [c.model_copy(update={"box": [v / factor for v in c.box]}) for c in clues]


def analyze(raw: Image.Image, upright: Image.Image) -> Analysis:
    exif = read_exif(raw)
    work, factor = _downscale(upright)
    clues = _rescale(detect_clues(work) + find_text(work), factor)
    geo = geolocate(work)
    return Analysis(
        width=upright.width, height=upright.height, exif=exif, clues=clues,
        geolocation=geo, risk=score(exif, clues, geo), mock=config.MOCK_MODE,
    )
