"""LeakLens API.

Run:  uvicorn app.main:app --reload
Mock: LEAKLENS_MOCK=1 uvicorn app.main:app --reload   (no model downloads)
"""

import json
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from . import config
from .models import warm_up
from .pipeline import analyze, jpeg_b64, load_image
from .pipeline.safe import plan
from .pipeline.scrub import blur_regions
from .schemas import Analysis, ScrubResult

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD = 25 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_up()
    yield


app = FastAPI(title="LeakLens", lifespan=lifespan)


async def _read(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 25 MB.")
    try:
        return load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image LeakLens can read.")


def _parse_boxes(boxes: str) -> list[list[float]]:
    try:
        box_list = json.loads(boxes)
        if not all(isinstance(b, list) and len(b) == 4 for b in box_list):
            raise ValueError
        return [[float(v) for v in b] for b in box_list]
    except (ValueError, TypeError):
        raise HTTPException(400, "boxes must be a JSON list of [x0, y0, x1, y1].")


def _result(upright, boxes, before, blurred, note="", lookup=None, ai=None) -> ScrubResult:
    clean = blur_regions(upright, boxes)
    return ScrubResult(
        image_b64=jpeg_b64(clean, quality=92),
        before=before,
        after=analyze(clean, clean, explain=False, lookup=lookup, use_ai=ai),
        blurred=blurred,
        note=note,
    )


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    engine = "mock"
    if not config.MOCK_MODE:
        try:
            from .pipeline.ocr_engines import get_engine

            engine = get_engine().name
        except Exception as exc:
            engine = f"unavailable ({exc})"
    from .pipeline import vlm

    ai = vlm.status()
    return {"ok": True, "mock": config.MOCK_MODE, "lookup_default": config.TEXT_LOOKUP,
            "ocr_engine": engine, "ai": {"available": bool(ai.get("ok")), "model": ai.get("model", config.VLM_MODEL),
                                          "reason": ai.get("reason", "")}}


# `lookup` (search sign text on OpenStreetMap) defaults to the server
# setting; the UI passes it explicitly from its checkbox.

@app.post("/api/analyze", response_model=Analysis)
async def analyze_image(file: UploadFile = File(...), explain: bool = True,
                        lookup: Optional[bool] = None, ai: Optional[bool] = None):
    raw, upright = await _read(file)
    return analyze(raw, upright, explain=explain, lookup=lookup, use_ai=ai)


@app.post("/api/scrub", response_model=ScrubResult)
async def scrub_image(file: UploadFile = File(...), boxes: str = Form("[]"),
                      lookup: Optional[bool] = None, ai: Optional[bool] = None):
    """Blur the given boxes, strip metadata, then re-run the analysis."""
    box_list = _parse_boxes(boxes)
    raw, upright = await _read(file)
    before = analyze(raw, upright, explain=False, lookup=lookup, use_ai=ai).risk
    return _result(upright, box_list, before, blurred=[], lookup=lookup, ai=ai)


@app.post("/api/safe", response_model=ScrubResult)
async def make_safe(file: UploadFile = File(...), lookup: Optional[bool] = None,
                    ai: Optional[bool] = None):
    """Blur the fewest clues needed to reach low risk."""
    raw, upright = await _read(file)
    analysis = analyze(raw, upright, explain=True, lookup=lookup, use_ai=ai)
    ids, note = plan(upright, analysis)
    boxes = [c.box for c in analysis.clues if c.id in ids and c.box is not None]
    return _result(upright, boxes, analysis.risk, blurred=ids, note=note, lookup=lookup, ai=ai)
