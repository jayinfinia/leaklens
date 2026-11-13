"""Layer 3: the "attack". Where do public models think this photo was taken?

StreetCLIP gives readable labels (country, then US state).
GeoCLIP gives coordinates for the map pin.
"""

import os
import tempfile

from .. import config
from ..schemas import Geolocation, GuessLabel, GuessPoint

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


def _rank(image, labels: list[str], top_k: int = 5) -> list[GuessLabel]:
    import torch

    from ..models import device, streetclip

    processor, model = streetclip()
    texts = [f"A Street View photo in {name}." for name in labels]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device())
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=-1)[0]
    top = probs.topk(min(top_k, len(labels)))
    return [GuessLabel(name=labels[i], prob=round(p, 4))
            for p, i in zip(top.values.tolist(), top.indices.tolist())]


def _points(image, top_k: int = 5) -> list[GuessPoint]:
    from ..models import geoclip

    model = geoclip()
    fd, path = tempfile.mkstemp(suffix=".jpg")  # GeoCLIP.predict wants a path
    os.close(fd)
    try:
        image.convert("RGB").save(path, quality=92)
        gps, probs = model.predict(path, top_k=top_k)
    finally:
        os.remove(path)
    return [GuessPoint(lat=round(float(g[0]), 5), lon=round(float(g[1]), 5), prob=round(float(p), 4))
            for g, p in zip(gps.tolist(), probs.tolist())]


def _summarize(country, region) -> str:
    if not country:
        return "No confident guess."
    best = country[0]
    if region and best.name == "United States" and region[0].prob >= 0.25:
        return f"Likely {region[0].name}, USA ({region[0].prob:.0%} within the US)"
    return f"Likely {best.name} ({best.prob:.0%})"


def geolocate(image) -> Geolocation:
    if config.MOCK_MODE:
        return Geolocation(
            country=[GuessLabel(name="United States", prob=0.81), GuessLabel(name="Canada", prob=0.07)],
            region=[GuessLabel(name="Virginia", prob=0.44), GuessLabel(name="Maryland", prob=0.18)],
            points=[GuessPoint(lat=38.9, lon=-77.04, prob=0.31)],
            summary="Likely Virginia, USA (44% within the US)",
        )

    country = _rank(image, COUNTRIES)
    region = _rank(image, US_STATES) if country and country[0].name == "United States" else []
    points = []
    if config.USE_GEOCLIP:
        try:
            points = _points(image)
        except Exception as exc:  # keep the demo alive if GeoCLIP misbehaves
            print(f"[leaklens] GeoCLIP failed: {exc}")
    return Geolocation(country=country, region=region, points=points,
                       summary=_summarize(country, region))
