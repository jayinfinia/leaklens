"""Orchestrates the layers: metadata -> clues -> geolocation -> attribution -> risk."""

import base64
import io

from PIL import Image, ImageOps

from .. import config
from ..schemas import Analysis, Clue
from .attribution import attribute
from .detect import detect_clues
from .exif import read_exif
from .geolocate import geolocate
from .lookup import lookup as text_lookup
from .ocr import find_text
from .risk import score
from .scene import scene_clues

try:  # iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

BROWSER_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}


def load_image(data: bytes) -> tuple[Image.Image, Image.Image]:
    """Returns (raw image with metadata, upright RGB image)."""
    raw = Image.open(io.BytesIO(data))
    raw.load()
    upright = ImageOps.exif_transpose(raw).convert("RGB")
    return raw, upright


def jpeg_b64(img: Image.Image, max_side: int | None = None, quality: int = 90) -> str:
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def downscale(img: Image.Image, max_side: int | None = None) -> tuple[Image.Image, float]:
    max_side = max_side or config.MAX_ANALYSIS_SIDE
    side = max(img.size)
    if side <= max_side:
        return img, 1.0
    factor = max_side / side
    small = img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)
    return small, factor


def rescale(clues: list[Clue], factor: float) -> list[Clue]:
    if factor == 1.0:
        return clues
    return [c.model_copy(update={"box": [v / factor for v in c.box]}) for c in clues]


def analyze(raw: Image.Image, upright: Image.Image, explain: bool = True,
            lookup: bool | None = None) -> Analysis:
    exif = read_exif(raw)
    work, factor = downscale(upright)
    lookup = config.TEXT_LOOKUP if lookup is None else lookup

    embeds = None
    if not config.MOCK_MODE:  # embed once, share between geolocation and scene
        from .geolocate import _image_embeds

        embeds = _image_embeds([work])

    # OCR gets a sharper copy than the vision models: small sign text matters.
    ocr_img, ocr_factor = downscale(upright, config.OCR_MAX_SIDE)
    text = rescale(find_text(ocr_img), ocr_factor / factor)  # -> `work` coordinates

    clues = detect_clues(work) + text
    geo = geolocate(work, embeds)
    scene = scene_clues(work, embeds)
    if explain:
        clues = attribute(work, clues, geo)  # boxes still in `work` coordinates
    clues = rescale(clues, factor)
    matches = text_lookup(clues, geo) if lookup else []

    preview = None
    if (raw.format or "").upper() not in BROWSER_FORMATS:
        preview = jpeg_b64(upright, max_side=1600, quality=85)

    return Analysis(
        width=upright.width, height=upright.height, exif=exif, clues=clues, scene=scene,
        geolocation=geo, text_matches=matches, lookup_used=lookup,
        risk=score(exif, clues, geo, scene, matches), preview_b64=preview,
        mock=config.MOCK_MODE,
    )
