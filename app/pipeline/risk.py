"""Layer 5a: turn everything found into one explainable 0-100 score.

Each factor gets a "leak strength" between 0 and 1: roughly, the chance
that this signal alone lets someone narrow down where the photo was taken.
They combine like independent chances (noisy-OR):

    score = 100 * (1 - product(1 - leak_i))

so ten weak duplicate boxes can't add up to "certain", while one strong
signal (exact GPS, a matched address) dominates. Within a category, each
extra clue counts half as much as the one before it.

The low/medium/high split is loosely adapted from the three-level visual
privacy risk framework in "Doxing via the Lens" (Luo et al., ICLR 2026).
"""

from collections import defaultdict

from .. import config
from ..schemas import Clue, ExifReport, Geolocation, Risk, RiskFactor, SceneClue, TextMatch

MAX_LEAK = 0.95
CLUE_SCALE = 0.5 / 3.0  # weight 3 at full confidence -> 0.5
IMPACT_SCALE = 1.5  # a clue that cuts the top guess by 10 pts adds 0.15
GPS_LEAK = 1.0
MATCH_LEAK = 0.85


def level_for(value: int) -> str:
    if value >= config.RISK_HIGH:
        return "high"
    return "medium" if value >= config.RISK_MEDIUM else "low"


def combine(leaks: list[float]) -> int:
    remaining = 1.0
    for v in leaks:
        remaining *= 1.0 - min(max(v, 0.0), 1.0)
    return int(round(100 * (1 - remaining)))


def _clue_leak(c: Clue) -> float:
    v = c.weight * c.score * CLUE_SCALE + IMPACT_SCALE * (c.impact or 0.0)
    return min(MAX_LEAK, v)


def score(exif: ExifReport, clues: list[Clue], geo: Geolocation,
          scene: list[SceneClue] | None = None,
          matches: list[TextMatch] | None = None) -> Risk:
    factors: list[tuple[float, RiskFactor]] = []

    def add(leak: float, source: str, reason: str):
        factors.append((leak, RiskFactor(source=source, points=round(100 * leak, 1), reason=reason)))

    if exif.gps:
        add(GPS_LEAK, "metadata", "The file contains exact GPS coordinates.")
    if exif.taken_at:
        add(0.03, "metadata", f"Timestamp in metadata ({exif.taken_at}).")
    if exif.device:
        add(0.02, "metadata", f"Device model in metadata ({exif.device}).")

    for m in matches or []:
        add(MATCH_LEAK, m.clue_id, f"Sign text “{m.text}” matches a real place: {m.name}.")

    by_cat: dict[str, list[Clue]] = defaultdict(list)
    for c in clues:
        by_cat[c.category].append(c)
    for cat_clues in by_cat.values():
        for rank, c in enumerate(sorted(cat_clues, key=_clue_leak, reverse=True)):
            name = f"“{c.label}”" if c.kind == "text" else c.category.replace("_", " ")
            reason = f"{name}: {c.reason}"
            if c.impact and c.impact >= 0.01:
                reason += f" Hiding it alone cuts the model's confidence by {c.impact * 100:.0f} pts."
            add(_clue_leak(c) * 0.5 ** rank, c.id, reason)

    for s in scene or []:
        add(0.3 * s.weight * s.prob, "scene", f"Scene: {s.label}. {s.reason}")

    # How sure the geolocation models are counts too, even with no clues.
    if geo.country:
        top = geo.country[0]
        add(0.15 * top.prob, "geolocation", f"Models place it in {top.name} ({top.prob:.0%}).")
    if geo.region:
        top = geo.region[0]
        add(0.35 * top.prob, "geolocation", f"Models narrow it to {top.name} ({top.prob:.0%}).")

    value = combine([leak for leak, _ in factors])
    ordered = [f for _, f in sorted(factors, key=lambda x: x[0], reverse=True)]
    return Risk(score=value, level=level_for(value), factors=ordered)
