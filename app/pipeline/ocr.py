"""Layer 2b: readable text.

EasyOCR returns single words and even single letters. Investigators read
whole signs, so we:
  1. stitch vertically stacked letters (P-A-R-A-M-O-U-N-T) into words,
  2. merge nearby words on the same sign into one clue,
  3. weight each sign by what its text reveals (see text_intel).
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
            if max(ha, hb) / min(ha, hb) > 2.5:
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


def to_clues(items: list[_Item]) -> list[Clue]:
    clues = []
    for k, it in enumerate(sorted(items, key=lambda i: (i.box[1], i.box[0]))):
        text = " ".join(it.text.split())
        if not _MEANINGFUL.search(text):
            continue
        intel = classify(text)
        clues.append(Clue(
            id=f"txt-{k}", kind="text", category=f"text:{intel.kind}", label=text[:80],
            score=round(float(it.conf), 3), box=[float(v) for v in it.box],
            reason=intel.reason, weight=intel.weight,
        ))
    return clues


def find_text(image) -> list[Clue]:
    if config.MOCK_MODE:
        return mock.text(image)

    import numpy as np

    from ..models import ocr

    raw = ocr().readtext(np.array(image.convert("RGB")))
    items = []
    for points, text, conf in raw:
        single = len(text.strip()) == 1
        # stacked marquee letters read with low confidence one at a time;
        # they only survive if 3+ line up into a word (stack_vertical_letters)
        if not text.strip() or conf < (0.15 if single else config.OCR_MIN_CONF):
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        items.append(_Item([min(xs), min(ys), max(xs), max(ys)], text, float(conf)))
    return to_clues(merge_lines(stack_vertical_letters(items)))
