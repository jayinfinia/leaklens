"""Orchestrates the layers:

metadata -> clues (detector, OCR) -> geolocation (StreetCLIP labels, GeoCLIP
pins and clusters) -> scene -> attribution -> AI first pass (blind) -> sign-
text lookup -> AI chooser (picks among named candidates; skipped after a
strong text match or when GeoCLIP already backs the first guess) -> fusion
into one best guess -> risk.
"""

import base64
import io

from PIL import Image, ImageOps

from .. import config
from ..schemas import Analysis, Clue
from .attribution import attribute
from .detect import detect_with_candidates
from .exif import read_exif
from .geolocate import geolocate, scorer as geo_scorer
from . import fuse, vlm
from .features import features_in
from .lookup import geocode_guesses, lookup as text_lookup
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
    return [c if c.box is None else c.model_copy(update={"box": [v / factor for v in c.box]})
            for c in clues]


def analyze(raw: Image.Image, upright: Image.Image, explain: bool = True,
            lookup: bool | None = None, use_ai: bool | None = None) -> Analysis:
    exif = read_exif(raw)
    work, factor = downscale(upright)
    lookup = config.TEXT_LOOKUP if lookup is None else lookup
    use_ai = (config.VLM_MODE != "off") if use_ai is None else use_ai

    embeds = None
    if not config.MOCK_MODE:  # embed once, share between geolocation and scene
        from .geolocate import _image_embeds

        embeds = _image_embeds([work])

    objects, candidates = detect_with_candidates(work)
    # OCR gets a sharper copy than the vision models: small sign text matters.
    # Detector boxes get a second, zoomed OCR pass (tall ones as vertical signs).
    ocr_img, ocr_factor = downscale(upright, config.OCR_MAX_SIDE)
    to_ocr = ocr_factor / factor
    sign_boxes = [[v * to_ocr for v in b] for b in candidates]
    text = rescale(find_text(ocr_img, sign_boxes), to_ocr)  # -> `work` coordinates

    clues = objects + text
    scorer = geo_scorer(work)  # GeoCLIP embeds the photo once: pins now, verification later
    geo = geolocate(work, embeds, scorer)
    scene = scene_clues(work, embeds)
    if explain:
        clues = attribute(work, clues, geo)  # boxes still in `work` coordinates
    clues = rescale(clues, factor)

    # Local AI model: reads signs, lists objects, guesses places (#1).
    ai = vlm.ask(upright) if use_ai else None
    if ai and not ai.error:
        clues = vlm.merge_texts(clues, ai)
    feature_keys = features_in(clues, ai.features if ai and not ai.error else None)
    us = bool(geo.country) and geo.country[0].name == "United States"
    if lookup and ai and not ai.error:
        ai = geocode_guesses(ai, us)
    ai = fuse.place_guesses(ai)  # offline coordinates for guesses still without any
    matches = text_lookup(clues, geo, ai if ai and not ai.error else None, feature_keys) if lookup else []

    # Second AI pass: choose among named candidates. A verified sign already
    # settles it, and so does GeoCLIP agreeing with the first guess: skip the
    # extra call then.
    strong = any(m.confidence == "strong" for m in matches)
    if (ai and not ai.error and config.VLM_CHOOSE and not strong
            and not fuse.first_pass_confirmed(geo, ai)):
        opts = fuse.options(geo, ai, matches)
        if len(opts) >= config.CHOOSE_MIN_OPTIONS:
            states = geo.region[:3] if us else []  # names and probabilities
            choice = vlm.choose(upright, ai, [o.name for o in opts], states)
            ai = ai.model_copy(update={"choice": fuse.locate_choice(choice, opts, lookup, us)})
    best = fuse.fuse(geo, ai, matches, scorer, exif)

    preview = None
    if (raw.format or "").upper() not in BROWSER_FORMATS:
        preview = jpeg_b64(upright, max_side=1600, quality=85)

    return Analysis(
        width=upright.width, height=upright.height, exif=exif, clues=clues, scene=scene,
        geolocation=geo, text_matches=matches, lookup_used=lookup, ai=ai, best=best,
        risk=score(exif, clues, geo, scene, matches, ai), preview_b64=preview,
        mock=config.MOCK_MODE,
    )
