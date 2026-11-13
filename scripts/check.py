"""Run the full pipeline on one photo from the command line.

    python scripts/check.py path/to/photo.jpg

Handy for tuning prompts and thresholds without the UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import analyze, load_image  # noqa: E402


def main(path: str) -> None:
    raw, upright = load_image(Path(path).read_bytes())
    a = analyze(raw, upright)
    print(f"Risk: {a.risk.score}/100 ({a.risk.level})")
    print(f"Geo:  {a.geolocation.summary}")
    if a.exif.gps:
        print(f"GPS:  {a.exif.gps.lat}, {a.exif.gps.lon}")
    for c in sorted(a.clues, key=lambda c: c.impact or 0, reverse=True):
        impact = f"  impact -{c.impact * 100:.0f} pts" if c.impact is not None else ""
        print(f"  [{c.score:.2f}] {c.category:<16} {c.label}{impact}")
    for s in a.scene:
        print(f"  scene {s.category:<16} {s.prob:.0%}")
    print("Factors:")
    for f in a.risk.factors:
        print(f"  +{f.points:<5} {f.reason}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
