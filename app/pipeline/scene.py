"""Layer 2c: scene signals that can't be boxed (skyline, plants, road lines)."""

from .. import config
from ..schemas import SceneClue
from . import mock


def scene_clues(image, image_embeds=None) -> list[SceneClue]:
    if config.MOCK_MODE:
        return mock.scene(image)

    from .geolocate import clip_probs

    cats = list(config.SCENE_CUES)
    prompts = [config.SCENE_CUES[c]["prompt"] for c in cats] + config.NEUTRAL_SCENES
    probs = clip_probs([image], prompts, image_embeds)[0].tolist()

    found = []
    for cat, p in zip(cats, probs):
        if p >= config.SCENE_MIN_PROB:
            spec = config.SCENE_CUES[cat]
            found.append(SceneClue(category=cat, label=spec["prompt"], prob=round(p, 3),
                                   reason=spec["reason"], weight=spec["weight"]))
    return sorted(found, key=lambda s: s.prob, reverse=True)
