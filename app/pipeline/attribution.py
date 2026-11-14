"""Which clue gave it away?

Occlusion test: blur each clue on its own, re-score the edited copy, and
measure how far confidence in the original top guess falls. A big drop
means the model was leaning on that clue. All copies go through the
model in one batch.
"""

from .. import config
from ..schemas import Clue, Geolocation
from . import mock
from .scrub import blur_regions


def attribute(image, clues: list[Clue], geo: Geolocation) -> list[Clue]:
    if not clues or not config.ATTRIBUTION or not geo.country:
        return clues
    if config.MOCK_MODE:
        return [c.model_copy(update={"impact": mock.impact(c)}) for c in clues]

    from .geolocate import label_probs, target

    ranked = sorted((c for c in clues if c.box is not None), key=lambda c: c.score,
                    reverse=True)[: config.ATTRIBUTION_MAX_CLUES]
    labels, top = target(geo)
    idx = labels.index(top)

    variants = [image] + [blur_regions(image, [c.box]) for c in ranked]
    probs = label_probs(variants, labels)[:, idx].tolist()
    base = probs[0]

    impacts = {c.id: round(max(0.0, base - p), 4) for c, p in zip(ranked, probs[1:])}
    return [c.model_copy(update={"impact": impacts.get(c.id)}) for c in clues]
