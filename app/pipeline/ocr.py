"""Layer 2b: readable text.

EasyOCR returns single words and even single letters. Investigators read
whole signs, so we:
  1. stitch vertically stacked letters (P-A-R-A-M-O-U-N-T) into words,
  2. merge nearby words on the same sign into one clue,
  3. unstack tall vertical signs and read them as a row (see vertical),
  4. weight each sign by what its text reveals (see text_intel).
"""

import re
import statistics

from .. import config
from ..schemas import Clue
from . import mock
from .text_intel import classify

_MEANINGFUL = re.compile(r"[A-Za-z0-9]{3,}")


class _Item:
    __slots__ = ("box", "text", "conf")

    def __init__(self, box, text, conf):
        self.box, self.text, self.conf = box, text, conf

    @property
    def h(self):
        return self.box[3] - self.box[1]

    @property
    def w(self):
        return self.box[2] - self.box[0]

    @property
    def cx(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self):
        return (self.box[1] + self.box[3]) / 2


def _union(items) -> list[float]:
    return [min(i.box[0] for i in items), min(i.box[1] for i in items),
            max(i.box[2] for i in items), max(i.box[3] for i in items)]


def _is_single(i: _Item) -> bool:
    t = i.text.strip()
    return len(t) == 1 and t.isalnum()


def stack_vertical_letters(items: list[_Item]) -> list[_Item]:
    """Join single letters stacked in a column (vertical marquee signs)."""
    singles = sorted((i for i in items if _is_single(i)), key=lambda i: i.cy)
    rest = [i for i in items if not _is_single(i)]
    used, columns = set(), []
    for i, a in enumerate(singles):
        if i in used:
            continue
        col = [a]
        used.add(i)
        for j in range(i + 1, len(singles)):
            if j in used:
                continue
            b, last = singles[j], col[-1]
            width = max(last.w, b.w, 1)
            gap = b.box[1] - last.box[3]
            if abs(b.cx - last.cx) <= 0.7 * width and gap <= 1.5 * max(last.h, b.h):
                col.append(b)
                used.add(j)
        columns.append(col)
    out = list(rest)
    for col in columns:
        if len(col) >= 3:
            out.append(_Item(_union(col), "".join(c.text.strip() for c in col),
                             statistics.mean(c.conf for c in col)))
    return out


def merge_lines(items: list[_Item]) -> list[_Item]:
    """Merge words that sit close together into one sign (union-find)."""
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            ha, hb = max(a.h, 1), max(b.h, 1)
            if max(ha, hb) / min(ha, hb) > 1.8:
                continue  # very different text sizes: different signs
            pad = 0.6 * min(ha, hb)
            if (a.box[0] - pad <= b.box[2] and b.box[0] - pad <= a.box[2] and
                    a.box[1] - pad <= b.box[3] and b.box[1] - pad <= a.box[3]):
                parent[find(i)] = find(j)

    groups: dict[int, list[_Item]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])

    merged = []
    for g in groups.values():
        row_h = statistics.median(max(i.h, 1) for i in g)
        ordered = sorted(g, key=lambda i: (round(i.cy / (0.7 * row_h)), i.box[0]))
        merged.append(_Item(_union(g), " ".join(i.text.strip() for i in ordered),
                            statistics.mean(i.conf for i in g)))
    return merged


def normalize(text: str) -> str:
    """OCR often reads spaces as underscores ("MED_SCHOOL")."""
    return " ".join(text.replace("_", " ").replace("|", " ").split())


def to_clues(items: list[_Item]) -> list[Clue]:
    clues = []
    for k, it in enumerate(sorted(items, key=lambda i: (i.box[1], i.box[0]))):
        text = normalize(it.text)
        if not _MEANINGFUL.search(text):
            continue
        intel = classify(text)
        clues.append(Clue(
            id=f"txt-{k}", kind="text", category=f"text:{intel.kind}", label=text[:80],
            score=round(float(it.conf), 3), box=[float(v) for v in it.box],
            reason=intel.reason, weight=intel.weight,
        ))
    return clues


def _overlaps(a, b, thresh=0.5) -> bool:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return area > 0 and inter / area >= thresh


def _alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _to_items(raw, min_conf: float, offset=(0.0, 0.0), scale: float = 1.0) -> list[_Item]:
    items = []
    for points, text, conf in raw:
        single = len(text.strip()) == 1
        # stacked marquee letters read with low confidence one at a time;
        # they only survive if 3+ line up into a word (stack_vertical_letters)
        if not text.strip() or conf < (0.15 if single else min_conf):
            continue
        xs = [float(p[0]) / scale + offset[0] for p in points]
        ys = [float(p[1]) / scale + offset[1] for p in points]
        items.append(_Item([min(xs), min(ys), max(xs), max(ys)], text, float(conf)))
    return items


def read_crop(engine, image, box, target: int = 1024, pad: float = 0.12) -> list[_Item]:
    """Re-read one detected sign as a zoomed crop; small text reads better big."""
    w, h = box[2] - box[0], box[3] - box[1]
    x0 = max(0, int(box[0] - pad * w))
    y0 = max(0, int(box[1] - pad * h))
    x1 = min(image.width, int(box[2] + pad * w))
    y1 = min(image.height, int(box[3] + pad * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    crop = image.crop((x0, y0, x1, y1))
    scale = min(3.0, target / max(crop.size))
    if scale > 1.1:
        from PIL import Image

        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    else:
        scale = 1.0
    return _to_items(engine.readtext(crop), engine.min_conf, offset=(x0, y0), scale=scale)


def dedupe(items: list[_Item]) -> list[_Item]:
    """Full-image and crop passes often read the same sign; keep the fuller
    reading ("POPPY & PARLIAMENT" beats "POPPY")."""
    kept: list[_Item] = []
    for it in sorted(items, key=lambda i: (_alnum(i.text), i.conf), reverse=True):
        if not any(_overlaps(it.box, k.box, 0.6) or (_overlaps(k.box, it.box, 0.6) and
                                                     _alnum(k.text) >= _alnum(it.text))
                   for k in kept):
            kept.append(it)
    return kept


def find_text(image, extra_boxes: list[list[float]] | None = None) -> list[Clue]:
    """`extra_boxes`: detector boxes (same coordinates as `image`). Tall ones are
    tried as vertical signs; the rest are re-read as zoomed crops."""
    if config.MOCK_MODE:
        return mock.text(image)

    from .ocr_engines import get_engine
    from .vertical import is_tall, read_vertical

    engine = get_engine()
    raw = engine.readtext(image)

    # Tall boxes are usually stacked letters read as nonsense: re-read them.
    vertical: list[_Item] = []
    tall_boxes = []
    for points, _, _ in raw:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        box = [min(xs), min(ys), max(xs), max(ys)]
        if is_tall(box):
            tall_boxes.append(box)
    for box in extra_boxes or []:
        if is_tall(box) and not any(_overlaps(box, t) or _overlaps(t, box) for t in tall_boxes):
            tall_boxes.append(box)
    for box in tall_boxes:
        try:
            hit = read_vertical(engine, image, box)
        except Exception as exc:  # never let one odd sign break the analysis
            print(f"[leaklens] vertical read failed: {exc}")
            hit = None
        if hit and hit[1] >= 0.4:
            vertical.append(_Item(box, hit[0], hit[1]))

    items = [i for i in _to_items(raw, engine.min_conf)
             if not any(_overlaps(i.box, v.box) for v in vertical)]

    # Zoomed second look at each detected sign (largest-confidence first).
    area = image.width * image.height
    crops = [b for b in (extra_boxes or []) if not is_tall(b)
             and (b[2] - b[0]) * (b[3] - b[1]) < config.OCR_CROP_MAX_AREA * area]
    for box in crops[: config.OCR_MAX_CROPS]:
        try:
            items += [i for i in read_crop(engine.crop_engine, image, box)
                      if not any(_overlaps(i.box, v.box) for v in vertical)]
        except Exception as exc:
            print(f"[leaklens] crop read failed: {exc}")

    # vertical signs stay separate: merging would glue them to nearby text
    rest = merge_lines(stack_vertical_letters(dedupe(items)))
    return to_clues(vertical + rest)
