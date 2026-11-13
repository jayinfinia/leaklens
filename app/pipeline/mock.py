"""Canned results for LEAKLENS_MOCK=1.

Mock mode reacts to blurring: every fake clue lives in one fixed region,
and if that region has been flattened (low pixel variance) the clues
disappear and the location guess gets less confident. That keeps the
analyze -> scrub -> re-check flow meaningful without any models.
"""

from PIL import ImageStat

from .. import config
from ..schemas import Clue, Geolocation, GuessLabel, GuessPoint, SceneClue, TextMatch
from .text_intel import classify

REGION = (0.08, 0.12, 0.34, 0.26)  # x0, y0, x1, y1 as fractions
DETAIL_THRESHOLD = 12.0


def region_box(image) -> list[float]:
    w, h = image.size
    return [w * REGION[0], h * REGION[1], w * REGION[2], h * REGION[3]]


def has_detail(image) -> bool:
    crop = image.convert("L").crop(tuple(int(v) for v in region_box(image)))
    return ImageStat.Stat(crop).stddev[0] >= DETAIL_THRESHOLD


def clues(image) -> list[Clue]:
    if not has_detail(image):
        return []
    spec = config.CLUE_CATEGORIES["street_sign"]
    return [Clue(
        id="obj-0", kind="object", category="street_sign", label="a street sign",
        score=0.71, box=region_box(image), reason=spec["reason"], weight=spec["weight"],
    )]


def text(image) -> list[Clue]:
    if not has_detail(image):
        return []
    x0, y0, x1, y1 = region_box(image)
    intel = classify("MAIN ST")
    return [Clue(
        id="txt-0", kind="text", category=f"text:{intel.kind}", label="MAIN ST",
        score=0.88, box=[x0 + (x1 - x0) * 0.1, y0 + (y1 - y0) * 0.25, x1 - (x1 - x0) * 0.1, y1 - (y1 - y0) * 0.3],
        reason=intel.reason, weight=intel.weight,
    )]


def geolocation(image) -> Geolocation:
    if has_detail(image):
        country = [GuessLabel(name="United States", prob=0.81), GuessLabel(name="Canada", prob=0.07)]
        region = [GuessLabel(name="Virginia", prob=0.44), GuessLabel(name="Maryland", prob=0.18)]
        points = [GuessPoint(lat=38.9, lon=-77.04, prob=0.31)]
        summary = "Likely Virginia, USA (44% within the US)"
    else:
        country = [GuessLabel(name="United States", prob=0.38), GuessLabel(name="Canada", prob=0.21)]
        region = [GuessLabel(name="Ohio", prob=0.09), GuessLabel(name="Virginia", prob=0.08)]
        points = [GuessPoint(lat=40.4, lon=-82.9, prob=0.04)]
        summary = "Likely United States (38%)"
    return Geolocation(country=country, region=region, points=points, summary=summary)


def scene(image) -> list[SceneClue]:
    spec = config.SCENE_CUES["power_lines"]
    return [SceneClue(category="power_lines", label=spec["prompt"], prob=0.22,
                      reason=spec["reason"], weight=spec["weight"])]


def impact(clue: Clue) -> float:
    return round(0.4 * clue.score, 3)
