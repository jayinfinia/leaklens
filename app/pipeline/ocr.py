"""Layer 2b: readable text. EasyOCR finds names, addresses and numbers."""

import re

from .. import config
from ..schemas import Clue
from . import mock

_MEANINGFUL = re.compile(r"[A-Za-z0-9]{3,}")


def find_text(image) -> list[Clue]:
    if config.MOCK_MODE:
        return mock.text(image)

    import numpy as np

    from ..models import ocr

    results = ocr().readtext(np.array(image.convert("RGB")))
    clues = []
    for i, (points, text, conf) in enumerate(results):
        text = text.strip()
        if conf < config.OCR_MIN_CONF or not _MEANINGFUL.search(text):
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        clues.append(Clue(
            id=f"txt-{i}", kind="text", category="text", label=text[:80],
            score=round(float(conf), 3), box=[min(xs), min(ys), max(xs), max(ys)],
            reason=config.TEXT_CLUE_REASON, weight=config.TEXT_CLUE_WEIGHT,
        ))
    return clues

