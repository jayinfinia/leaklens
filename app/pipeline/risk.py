"""Layer 4a: turn everything found into one explainable 0-100 score.

The low/medium/high split is loosely adapted from the three-level visual
privacy risk framework in "Doxing via the Lens" (Luo et al., ICLR 2026).
Every point added is listed as a factor so the UI can explain the score.
"""

from .. import config
from ..schemas import Clue, ExifReport, Geolocation, Risk, RiskFactor

SCALE = 8.0  # raw points -> 0-100
GPS_POINTS = 13.0  # alone pushes the score past the "high" cutoff


def score(exif: ExifReport, clues: list[Clue], geo: Geolocation) -> Risk:
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
        name = c.label if c.kind == "text" else c.category.replace("_", " ")
        factors.append(RiskFactor(source=c.id, points=round(c.weight * c.score, 2),
                                  reason=f"{name}: {c.reason}"))

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
    level = "high" if value >= config.RISK_HIGH else "medium" if value >= config.RISK_MEDIUM else "low"
    factors.sort(key=lambda f: f.points, reverse=True)
    return Risk(score=value, level=level, factors=factors)
