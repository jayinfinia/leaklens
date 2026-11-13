"""Read vertical signs whose letters are stacked top to bottom (P/A/R/A/M/O/U/N/T).

OCR models read left to right, so a stacked marquee comes back as one tall
box with garbage text. We "unstack" it:

  1. find the letter rows: pixels that differ from the sign's background
     colour, profiled row by row; gaps between letters are empty rows
  2. binarize to black-on-white using the letter colour, which also
     drops the sign face and most of the bulbs around the edge
  3. in each letter cell keep only the blobs that reach the middle
     (removes the leftover border bulbs)
  4. lay the cells side by side and let the OCR recognizer read the row
"""

import numpy as np
from PIL import Image


def _otsu(x: np.ndarray) -> float:
    hist, edges = np.histogram(x, bins=64)
    w = hist.cumsum()
    mu = (hist * edges[:-1]).cumsum()
    tot, mt = w[-1], mu[-1]
    best, t = -1.0, float(edges[32])
    for i in range(1, 63):
        w0, w1 = w[i], tot - w[i]
        if w0 == 0 or w1 == 0:
            continue
        v = w0 * w1 * (mu[i] / w0 - (mt - mu[i]) / w1) ** 2
        if v > best:
            best, t = v, float(edges[i])
    return t


def _center_band(a: np.ndarray, lo=0.2, hi=0.8) -> np.ndarray:
    w = a.shape[1]
    return a[:, int(w * lo):max(int(w * hi), int(w * lo) + 1)]


def letter_rows(col: Image.Image) -> list[tuple[int, int]]:
    a = np.asarray(col.convert("RGB"), dtype=float)
    band = _center_band(a)
    bg = np.median(band.reshape(-1, 3), axis=0)
    d = np.sqrt(((band - bg) ** 2).sum(axis=2))
    prof = (d > _otsu(d)).mean(axis=1)
    if prof.max() == 0:
        return []
    on = prof > max(0.04, 0.12 * prof.max())
    segs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            segs.append((start, y))
            start = None
    if start is not None:
        segs.append((start, len(on)))
    if not segs:
        return []
    med = float(np.median([b - a for a, b in segs]))
    return [s for s in segs if s[1] - s[0] >= 0.35 * med]


def binarize(col: Image.Image) -> Image.Image:
    a = np.asarray(col.convert("RGB"), dtype=float)
    band = _center_band(a, 0.25, 0.75).reshape(-1, 3)
    bg = np.median(band, axis=0)
    d_bg = np.sqrt(((band - bg) ** 2).sum(axis=1))
    ink = band[d_bg > _otsu(d_bg)]
    if not len(ink):
        return col.convert("L")
    ink_color = np.median(ink, axis=0)
    sep = float(np.sqrt(((ink_color - bg) ** 2).sum()))
    d_ink = np.sqrt(((a - ink_color) ** 2).sum(axis=2))
    return Image.fromarray(np.where(d_ink < 0.45 * sep, 0, 255).astype(np.uint8))


def clean_cell(cell: Image.Image) -> Image.Image:
    from scipy import ndimage

    a = np.asarray(cell) < 128
    lab, n = ndimage.label(a)
    if n == 0:
        return cell
    w = a.shape[1]
    sizes = ndimage.sum(a, lab, range(1, n + 1))
    big = float(np.max(sizes))
    keep = np.zeros_like(a)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        x0, x1 = sl[1].start, sl[1].stop
        if x1 > 0.3 * w and x0 < 0.7 * w and sizes[k - 1] >= 0.15 * big:
            keep |= lab == k
    cols = np.where(keep.any(axis=0))[0]
    out = Image.fromarray(np.where(keep, 0, 255).astype(np.uint8))
    if len(cols):
        out = out.crop((max(0, int(cols[0]) - 2), 0, min(w, int(cols[-1]) + 3), a.shape[0]))
    return out


def unstack(col: Image.Image, height: int = 48, pad: int = 14):
    """Returns (row image, letter count), or None if it doesn't look stacked."""
    rows = letter_rows(col)
    if len(rows) < 3:
        return None
    bw = binarize(col)
    cells = [clean_cell(bw.crop((0, max(0, a - 2), bw.width, min(bw.height, b + 2)))) for a, b in rows]
    cells = [c.resize((max(1, int(c.width * height / max(c.height, 1))), height)) for c in cells]
    row = Image.new("L", (sum(c.width for c in cells) + pad * (len(cells) + 1), height + 2 * pad), 255)
    x = pad
    for c in cells:
        row.paste(c, (x, pad))
        x += c.width + pad
    return row, len(rows)


def is_tall(box, min_ratio: float = 2.5, min_height: float = 60) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return h >= min_height and h >= min_ratio * max(w, 1)


def read_vertical(reader, image: Image.Image, box) -> tuple[str, float] | None:
    """OCR a tall region as stacked letters. Returns (text, confidence)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    col = image.crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
    out = unstack(col)
    if out is None:
        return None
    row, _ = out
    res = reader.recognize(np.array(row), detail=1)
    if not res:
        return None
    _, text, conf = res[0]
    text = "".join(ch for ch in text if ch.isalnum() or ch in " &'-").strip()
    return (text, float(conf)) if sum(ch.isalnum() for ch in text) >= 3 else None
