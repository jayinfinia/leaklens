"""Layer 3: the "attack". Where do public models think this photo was taken?

StreetCLIP gives readable labels (country, then US state).
GeoCLIP gives coordinates for the map pin.

Text embeddings are computed once and cached, and images are embedded in
batches, so re-scoring many edited copies of a photo (clue attribution,
"make it safe") costs one image forward pass each.
"""

import os
import tempfile
from functools import lru_cache

from .. import config
from ..schemas import Geolocation, GuessLabel, GuessPoint
from . import mock

COUNTRIES = [
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Colombia",
    "Peru", "United Kingdom", "Ireland", "France", "Spain", "Portugal", "Italy",
    "Germany", "Netherlands", "Belgium", "Switzerland", "Austria", "Poland",
    "Czech Republic", "Sweden", "Norway", "Finland", "Denmark", "Greece", "Turkey",
    "Russia", "Ukraine", "Romania", "Israel", "United Arab Emirates", "Saudi Arabia",
    "Egypt", "Morocco", "Nigeria", "Ghana", "Kenya", "South Africa", "India",
    "Pakistan", "Bangladesh", "Thailand", "Vietnam", "Malaysia", "Singapore",
    "Indonesia", "Philippines", "China", "Japan", "South Korea", "Taiwan",
    "Australia", "New Zealand", "Jamaica", "Barbados", "Trinidad and Tobago",
    "Dominican Republic", "Puerto Rico", "Iceland",
]

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "Washington, D.C.",
]

TEMPLATE = "A Street View photo in {}."
BATCH = 8


# ------------------------------------------------------------ embeddings
def _as_tensor(out):
    """transformers v4 returns a tensor here, v5 returns an output object."""
    import torch

    return out if torch.is_tensor(out) else out.pooler_output


@lru_cache(maxsize=32)
def _text_embeds(prompts: tuple[str, ...]):
    import torch

    from ..models import device, streetclip

    processor, model = streetclip()
    tok = processor.tokenizer(list(prompts), padding=True, return_tensors="pt").to(device())
    with torch.no_grad():
        t = _as_tensor(model.get_text_features(**tok))
    return t / t.norm(dim=-1, keepdim=True)


def _image_embeds(images: list):
    import torch

    from ..models import device, streetclip

    processor, model = streetclip()
    chunks = []
    for i in range(0, len(images), BATCH):
        px = processor.image_processor(images=[im.convert("RGB") for im in images[i:i + BATCH]],
                                       return_tensors="pt")["pixel_values"].to(device())
        with torch.no_grad():
            chunks.append(_as_tensor(model.get_image_features(pixel_values=px)))
    e = torch.cat(chunks)
    return e / e.norm(dim=-1, keepdim=True)


def clip_probs(images: list, prompts: list[str], image_embeds=None):
    """Softmax over prompts for each image -> tensor [n_images, n_prompts]."""
    from ..models import streetclip

    _, model = streetclip()
    img = _image_embeds(images) if image_embeds is None else image_embeds
    logits = model.logit_scale.exp() * img @ _text_embeds(tuple(prompts)).T
    return logits.softmax(dim=-1).cpu()


def label_probs(images: list, labels: list[str], image_embeds=None):
    return clip_probs(images, [TEMPLATE.format(x) for x in labels], image_embeds)


def _top(row, labels, k=5) -> list[GuessLabel]:
    top = row.topk(min(k, len(labels)))
    return [GuessLabel(name=labels[i], prob=round(p, 4))
            for p, i in zip(top.values.tolist(), top.indices.tolist())]


# ------------------------------------------------------------ public API
def _summarize(country, region) -> str:
    if not country:
        return "No confident guess."
    best = country[0]
    if region and best.name == "United States" and region[0].prob >= 0.25:
        return f"Likely {region[0].name}, USA ({region[0].prob:.0%} within the US)"
    return f"Likely {best.name} ({best.prob:.0%})"


def geolocate_many(images: list, image_embeds=None) -> list[Geolocation]:
    """Labels only (no map pins). Cheap enough to call on many variants."""
    if config.MOCK_MODE:
        return [mock.geolocation(im) for im in images]

    embeds = _image_embeds(images) if image_embeds is None else image_embeds
    cp = label_probs(images, COUNTRIES, embeds)
    sp = label_probs(images, US_STATES, embeds)
    out = []
    for i in range(len(images)):
        country = _top(cp[i], COUNTRIES)
        region = _top(sp[i], US_STATES) if country[0].name == "United States" else []
        out.append(Geolocation(country=country, region=region, summary=_summarize(country, region)))
    return out


def _points(image, top_k: int = 5) -> list[GuessPoint]:
    import torch

    from ..models import geoclip

    model = geoclip()
    fd, path = tempfile.mkstemp(suffix=".jpg")  # GeoCLIP.predict wants a path
    os.close(fd)
    try:
        image.convert("RGB").save(path, quality=92)
        with torch.no_grad():
            gps, probs = model.predict(path, top_k=top_k)
    finally:
        os.remove(path)
    return [GuessPoint(lat=round(float(g[0]), 5), lon=round(float(g[1]), 5), prob=round(float(p), 4))
            for g, p in zip(gps.tolist(), probs.tolist())]


def geolocate(image, image_embeds=None) -> Geolocation:
    if config.MOCK_MODE:
        return mock.geolocation(image)

    geo = geolocate_many([image], image_embeds)[0]
    if config.USE_GEOCLIP:
        try:
            geo.points = _points(image)
        except Exception as exc:  # keep the demo alive if GeoCLIP misbehaves
            print(f"[leaklens] GeoCLIP failed: {exc}")
    return geo


def target(geo: Geolocation) -> tuple[list[str], str]:
    """The label set and label that best describe the current top guess.

    Attribution measures how much each clue props up this one guess.
    """
    if geo.region:
        return US_STATES, geo.region[0].name
    return COUNTRIES, geo.country[0].name


def confidence(geo: Geolocation) -> float:
    if geo.region:
        return geo.region[0].prob
    return geo.country[0].prob if geo.country else 0.0
