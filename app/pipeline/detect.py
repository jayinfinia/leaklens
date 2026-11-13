"""Layer 2a: visual clues. OWLv2 draws boxes around location-revealing objects."""

from .. import config
from ..schemas import Clue


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def _nms(clues: list[Clue], iou_thresh: float = 0.5) -> list[Clue]:
    """Keep the strongest box when several prompts hit the same spot."""
    kept: list[Clue] = []
    for c in sorted(clues, key=lambda c: c.score, reverse=True):
        if all(_iou(c.box, k.box) < iou_thresh for k in kept):
            kept.append(c)
    return kept


def detect_clues(image) -> list[Clue]:
    if config.MOCK_MODE:
        return _mock(image)

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
    result = post(outputs=outputs, threshold=config.DETECT_THRESHOLD, target_sizes=target)[0]

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
    return _nms(clues)


def _mock(image) -> list[Clue]:
    w, h = image.size
    spec = config.CLUE_CATEGORIES["street_sign"]
    return [Clue(
        id="obj-0", kind="object", category="street_sign", label="a street sign",
        score=0.71, box=[w * 0.08, h * 0.12, w * 0.34, h * 0.26],
        reason=spec["reason"], weight=spec["weight"],
    )]
