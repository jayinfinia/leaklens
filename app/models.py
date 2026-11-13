"""Lazy, cached model loaders.

Heavy imports (torch, transformers, easyocr, geoclip) happen inside the
loaders so the API, the tests and mock mode start instantly.
"""

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def device() -> str:
    if config.DEVICE:
        return config.DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def owl():
    """Open-vocabulary detector: draws boxes for text prompts."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(config.OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(config.OWL_MODEL).to(device()).eval()
    return processor, model


@lru_cache(maxsize=1)
def streetclip():
    """CLIP fine-tuned for geolocation (CC-BY-NC-4.0, non-commercial)."""
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(config.STREETCLIP_MODEL)
    model = CLIPModel.from_pretrained(config.STREETCLIP_MODEL).to(device()).eval()
    return processor, model


@lru_cache(maxsize=1)
def geoclip():
    """Image-to-GPS model; returns coordinates rather than labels."""
    from geoclip import GeoCLIP

    model = GeoCLIP()
    try:
        model = model.to(device())
    except Exception:
        pass  # GeoCLIP manages its own device on some versions
    model.eval()
    return model


@lru_cache(maxsize=1)
def ocr():
    import easyocr

    return easyocr.Reader(config.OCR_LANGS, gpu=device() == "cuda")


def warm_up() -> None:
    """Load every model once so the first request isn't slow."""
    if config.MOCK_MODE:
        return
    owl()
    streetclip()
    from .pipeline.ocr_engines import get_engine

    get_engine()
    if config.USE_GEOCLIP:
        geoclip()
