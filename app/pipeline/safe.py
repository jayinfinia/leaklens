""""Make it safe": blur the fewest clues needed to reach low risk.

Clues are tried in order of how much the model leans on them (attribution
impact, then detector weight). Each cumulative step is re-scored in one
batch, and the first step that reaches "low" wins. If none does, every
clue is blurred and the note explains what is left (usually scene-level
signals that need cropping rather than blurring).
"""

from ..schemas import Analysis, ExifReport
from .geolocate import geolocate_many
from .risk import score
from .scrub import blur_regions


def plan(upright, analysis: Analysis) -> tuple[list[str], str]:
    matched = {m.clue_id for m in analysis.text_matches}
    # Text that matched a real place goes first, then what the model leans on most.
    clues = sorted(analysis.clues,
                   key=lambda c: (c.id in matched, c.impact or 0, c.weight * c.score), reverse=True)
    if not clues:
        return [], "No visual clues to blur; metadata will still be stripped."

    steps = [clues[: i + 1] for i in range(len(clues))]
    variants = [blur_regions(upright, [c.box for c in step]) for step in steps]
    geos = geolocate_many(variants)

    no_exif = ExifReport()
    for step, geo in zip(steps, geos):
        hidden = {c.id for c in step}
        remaining = [c for c in analysis.clues if c.id not in hidden]
        matches = [m for m in analysis.text_matches if m.clue_id not in hidden]
        if score(no_exif, remaining, geo, analysis.scene, matches).level == "low":
            return [c.id for c in step], f"Blurred {len(step)} of {len(clues)} clues to reach low risk."

    note = "Blurred every clue, but the scene itself still points to a region."
    if analysis.scene:
        note += f" Consider cropping out: {', '.join(s.category.replace('_', ' ') for s in analysis.scene[:3])}."
    return [c.id for c in clues], note
