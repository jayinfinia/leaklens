"""Layer 2a: visual clues. OWLv2 draws boxes around location-revealing objects."""

from .. import config
from ..schemas import Clue
from . import mock


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def _containment(inner, outer) -> float:
    """Share of `inner` that lies inside `outer`."""
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def _nms(clues: list[Clue], iou_thresh: float = 0.5, contain_thresh: float = 0.85) -> list[Clue]:
    """Keep the strongest box when several prompts hit the same spot.

    Also drops a weaker box that sits almost entirely inside a stronger one
    (a sign inside a storefront), across every category.
    """
    kept: list[Clue] = []
    for c in sorted(clues, key=lambda c: c.score, reverse=True):
        if all(_iou(c.box, k.box) < iou_thresh and _containment(c.box, k.box) < contain_thresh
               for k in kept):
            kept.append(c)
    return kept


def detect_clues(image) -> list[Clue]:
    return detect_with_candidates(image)[0]


def detect_with_candidates(image) -> tuple[list[Clue], list[list[float]]]:
    """Returns (clues, OCR candidate boxes).

    Clues are boxes the detector is reasonably sure of (DETECT_THRESHOLD).
    Candidates also include weaker boxes (CROP_THRESHOLD): too unsure to show
    as clues, but worth a zoomed OCR read, since a faint "sign" box often
    holds the most revealing text in the photo.
    """
    if config.MOCK_MODE:
        clues = mock.clues(image)
        return clues, [c.box for c in clues]

    import torch

    from ..models import device, owl

    processor, model = owl()
    prompts, categories = [], []
    for cat, spec in config.CLUE_CATEGORIES.items():
        for p in spec["prompts"]:
            prompts.append(p)
            categories.append(cat)

    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device())
    with torch.no_grad():
        outputs = model(**inputs)

    # OWLv2 pads the image to a square, so boxes come back in padded space.
    side = max(image.size)
    target = torch.tensor([[side, side]], device=device())
    post = getattr(processor, "post_process_object_detection", None) or \
        processor.image_processor.post_process_object_detection
    low = min(config.DETECT_THRESHOLD, config.CROP_THRESHOLD)
    result = post(outputs=outputs, threshold=low, target_sizes=target)[0]

    w, h = image.size
    clues = []
    for i, (score, label, box) in enumerate(
        zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist())
    ):
        x0, y0, x1, y1 = box
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        cat = categories[label]
        spec = config.CLUE_CATEGORIES[cat]
        clues.append(Clue(
            id=f"obj-{i}", kind="object", category=cat, label=prompts[label],
            score=round(score, 3), box=[x0, y0, x1, y1],
            reason=spec["reason"], weight=spec["weight"],
        ))
    kept = _nms(clues)
    strong = [c for c in kept if c.score >= config.DETECT_THRESHOLD][: config.MAX_OBJECT_CLUES]
    candidates = [c.box for c in kept][: config.MAX_OBJECT_CLUES + config.OCR_MAX_CROPS]
    return strong, candidates

