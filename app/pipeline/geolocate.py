"""Layer 3: the "attack". Where do public models think this photo was taken?

StreetCLIP gives readable labels (country, then US state).
GeoCLIP gives coordinates: its top 40 pins are grouped into ~25 km clusters
and named offline (gazetteer.py), and GeoScorer rates any coordinate
against the photo, which fuse.py uses to check candidate places.

Text embeddings are computed once and cached, and images are embedded in
batches, so re-scoring many edited copies of a photo (clue attribution,
"make it safe") costs one image forward pass each.
"""

from functools import lru_cache

from .. import config
from ..schemas import Geolocation, GuessLabel, GuessPoint, PlaceCluster
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


class GeoScorer:
    """GeoCLIP on one image: the image is embedded once, then any coordinates
    can be scored against it (GeoCLIP's image <-> GPS similarity, the
    "geo-verification" step of G3, NeurIPS 2024).

    GeoCLIP logits aren't probabilities, so a spot's score is its rank among
    GeoCLIP's 100K-point world gallery on a log scale: 1.0 = better than every
    gallery point, 0.8 = top 10, 0.6 = top 100, 0.4 = top 1,000, 0 = bottom.
    """

    CHUNK = 8192

    def __init__(self, image, model=None):
        import torch

        from ..models import geoclip

        self.model = model if model is not None else geoclip()
        m = self.model
        self.device = m.logit_scale.device
        with torch.no_grad():
            px = m.image_encoder.preprocess_image(image.convert("RGB")).to(self.device)
            f = m.image_encoder(px)
            self.image_features = f / f.norm(dim=-1, keepdim=True)
            self.scale = m.logit_scale.exp()
            self.gallery = m.gps_gallery
            self.gallery_logits = self._logits(self.gallery)  # [N]
        self.sorted_logits = self.gallery_logits.sort().values.cpu()

    def _logits(self, coords):
        import torch

        out = []
        for i in range(0, len(coords), self.CHUNK):
            loc = self.model.location_encoder(coords[i:i + self.CHUNK].to(self.device).float())
            loc = loc / loc.norm(dim=-1, keepdim=True)
            out.append((self.scale * self.image_features @ loc.T)[0])
        return torch.cat(out) if out else torch.zeros(0)

    def points(self, top_k: int) -> list[GuessPoint]:
        """Top-k gallery points, like GeoCLIP.predict."""
        probs = self.gallery_logits.softmax(dim=-1)
        top = probs.topk(min(top_k, len(probs)))
        gps = self.gallery.index_select(0, top.indices).cpu().tolist()
        return [GuessPoint(lat=round(float(g[0]), 5), lon=round(float(g[1]), 5), prob=round(float(p), 4))
                for g, p in zip(gps, top.values.cpu().tolist())]

    def score(self, coords: list[tuple[float, float]]) -> list[float]:
        import math

        import torch

        if not coords:
            return []
        with torch.no_grad():
            logits = self._logits(torch.tensor(coords, dtype=torch.float32)).cpu()
        n = len(self.sorted_logits)
        above = n - torch.searchsorted(self.sorted_logits, logits, right=True)  # gallery points beating it
        return [round(max(0.0, 1.0 - math.log10(1 + int(a)) / math.log10(n)), 4) for a in above.tolist()]


class MockScorer:
    """LEAKLENS_MOCK=1: scores fall off with distance from the mock pin."""

    def __init__(self, image):
        self.geo = mock.geolocation(image)

    def points(self, top_k: int) -> list[GuessPoint]:
        return self.geo.points[:top_k]

    def score(self, coords: list[tuple[float, float]]) -> list[float]:
        import math

        from .gazetteer import km

        p = self.geo.points[0]
        return [round(math.exp(-km(lat, lon, p.lat, p.lon) / 400), 4) for lat, lon in coords]


def scorer(image):
    """A GeoCLIP scorer for this image, or None when GeoCLIP is off or fails."""
    if config.MOCK_MODE:
        return MockScorer(image)
    if not config.USE_GEOCLIP:
        return None
    try:
        return GeoScorer(image)
    except Exception as exc:  # keep the demo alive if GeoCLIP misbehaves
        print(f"[leaklens] GeoCLIP failed: {exc}")
        return None


def score_points(image, coords: list[tuple[float, float]]) -> list[float]:
    """GeoCLIP verification score (0-1) of each (lat, lon) for this image."""
    s = scorer(image)
    return s.score(coords) if s else [0.0] * len(coords)


def cluster(points: list[GuessPoint], km_radius: float | None = None,
            limit: int | None = None) -> list[PlaceCluster]:
    """Greedy clustering, most probable pin first: a pin within `km_radius` of
    a cluster's first pin joins it. Centers are probability-weighted; each
    cluster is named after the nearest city in the offline gazetteer."""
    from .gazetteer import describe, km

    radius = config.CLUSTER_KM if km_radius is None else km_radius
    groups: list[list[GuessPoint]] = []
    for p in sorted(points, key=lambda p: p.prob, reverse=True):
        g = next((g for g in groups if km(g[0].lat, g[0].lon, p.lat, p.lon) <= radius), None)
        if g is None:
            groups.append([p])
        else:
            g.append(p)
    out = []
    for g in groups:
        w = sum(p.prob for p in g) or 1e-9
        lat = sum(p.lat * p.prob for p in g) / w
        lon = sum(p.lon * p.prob for p in g) / w
        place, dist = describe(lat, lon)
        out.append(PlaceCluster(lat=round(lat, 5), lon=round(lon, 5), prob=round(sum(p.prob for p in g), 4),
                                n=len(g), place=place, km=dist))
    out.sort(key=lambda c: c.prob, reverse=True)
    return out[: (config.MAX_CLUSTERS if limit is None else limit)]


def geolocate(image, image_embeds=None, geo_scorer=None) -> Geolocation:
    """Labels from StreetCLIP; pins and named pin clusters from GeoCLIP.
    Pass a scorer (see `scorer`) to reuse its image embedding later."""
    if config.MOCK_MODE:
        geo = mock.geolocation(image)
        geo.clusters = cluster(geo.points)
        return geo

    geo = geolocate_many([image], image_embeds)[0]
    s = geo_scorer if geo_scorer is not None else scorer(image)
    if s is not None:
        try:
            pins = s.points(config.GEOCLIP_TOP_K)
            geo.points = pins[: config.GEOCLIP_PINS_SHOWN]
            geo.clusters = cluster(pins)
        except Exception as exc:
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
