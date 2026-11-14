"""Print exactly what each OCR engine reads in a photo.

    python scripts/ocr_debug.py path/to/photo.jpg

Shows raw readings (text, confidence, box) per engine on the full image,
then the final merged sign clues LeakLens would use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.pipeline import downscale, load_image  # noqa: E402
from app.pipeline import ocr_engines  # noqa: E402
from app.pipeline.ocr import find_text  # noqa: E402


def show(engine, image):
    print(f"\n== {engine.name} (cutoff {engine.min_conf}) ==")
    try:
        rows = engine.readtext(image)
    except Exception as exc:
        print(f"  failed: {exc}")
        return
    for points, text, conf in sorted(rows, key=lambda r: (r[0][0][1], r[0][0][0])):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        flag = "" if conf >= engine.min_conf else "   (below cutoff)"
        print(f"  {conf:4.2f}  {text!r:40} box=({min(xs):.0f},{min(ys):.0f})-({max(xs):.0f},{max(ys):.0f}){flag}")


def main(path: str) -> None:
    _, upright = load_image(Path(path).read_bytes())
    image, _ = downscale(upright, config.OCR_MAX_SIDE)
    print(f"image {upright.size} -> OCR copy {image.size}")
    engines = []
    apple = ocr_engines._apple_or_none(required=False)
    if apple:
        engines.append(apple)
    engines.append(ocr_engines.EasyOCREngine())
    for e in engines:
        show(e, image)
    print(f"\n== final clues (engine: {ocr_engines.get_engine().name}, no detector crops) ==")
    for c in find_text(image):
        print(f"  {c.score:4.2f}  {c.category:18} {c.label}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
