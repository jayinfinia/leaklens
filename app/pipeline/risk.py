"""Layer 4a: turn everything found into one explainable 0-100 score.

The low/medium/high split is loosely adapted from the three-level visual
privacy risk framework in "Doxing via the Lens" (Luo et al., ICLR 2026).
Every point added is listed as a factor so the UI can explain the score.
"""

from .. import config
from ..schemas import Clue, ExifReport, Geolocation, Risk, RiskFactor, SceneClue

SCALE = 8.0  # raw points -> 0-100
GPS_POINTS = 12.5  # exact GPS alone maxes the score (12.5 * 8 = 100)
IMPACT_POINTS = 4.0  # extra points for a clue the model visibly leans on


def level_for(value: int) -> str:
    if value >= config.RISK_HIGH:
        return "high"
    return "medium" if value >= config.RISK_MEDIUM else "low"


def score(exif: ExifReport, clues: list[Clue], geo: Geolocation,
          scene: list[SceneClue] | None = None) -> Risk:
    factors: list[RiskFactor] = []

    if exif.gps:
        factors.append(RiskFactor(source="metadata", points=GPS_POINTS,
                                  reason="The file contains exact GPS coordinates."))
    if exif.taken_at:
        factors.append(RiskFactor(source="metadata", points=0.5,
                                  reason=f"Timestamp in metadata ({exif.taken_at})."))
    if exif.device:
        factors.append(RiskFactor(source="metadata", points=0.3,
                                  reason=f"Device model in metadata ({exif.device})."))

    for c in clues:
        name = f"“{c.label}”" if c.kind == "text" else c.category.replace("_", " ")
        points = c.weight * c.score
        reason = f"{name}: {c.reason}"
        if c.impact:
            points += IMPACT_POINTS * c.impact
            reason += f" Hiding it alone cuts the model's confidence by {c.impact * 100:.0f} pts."
        factors.append(RiskFactor(source=c.id, points=round(points, 2), reason=reason))

    for s in scene or []:
        factors.append(RiskFactor(source="scene", points=round(s.weight * s.prob, 2),
                                  reason=f"Scene: {s.label}. {s.reason}"))

    # How sure the geolocation models are counts too, even with no clues.
    if geo.country:
        top = geo.country[0]
        factors.append(RiskFactor(source="geolocation", points=round(2.0 * top.prob, 2),
                                  reason=f"Models place it in {top.name} ({top.prob:.0%})."))
    if geo.region:
        top = geo.region[0]
        factors.append(RiskFactor(source="geolocation", points=round(3.0 * top.prob, 2),
                                  reason=f"Models narrow it to {top.name} ({top.prob:.0%})."))

    raw = sum(f.points for f in factors)
    value = int(min(100, round(raw * SCALE)))
    factors.sort(key=lambda f: f.points, reverse=True)
    return Risk(score=value, level=level_for(value), factors=factors)
