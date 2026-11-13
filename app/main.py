"""LeakLens API.

Run:  uvicorn app.main:app --reload
Mock: LEAKLENS_MOCK=1 uvicorn app.main:app --reload   (no model downloads)
"""

import json
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


def _result(upright, boxes, before, blurred, note="") -> ScrubResult:
    clean = blur_regions(upright, boxes)
    return ScrubResult(
        image_b64=jpeg_b64(clean, quality=92),
        before=before,
        after=analyze(clean, clean, explain=False),
        blurred=blurred,
        note=note,
    )


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "mock": config.MOCK_MODE}


@app.post("/api/analyze", response_model=Analysis)
async def analyze_image(file: UploadFile = File(...), explain: bool = True):
    raw, upright = await _read(file)
    return analyze(raw, upright, explain=explain)


@app.post("/api/scrub", response_model=ScrubResult)
async def scrub_image(file: UploadFile = File(...), boxes: str = Form("[]")):
    """Blur the given boxes, strip metadata, then re-run the analysis."""
    box_list = _parse_boxes(boxes)
    raw, upright = await _read(file)
    before = analyze(raw, upright, explain=False).risk
    return _result(upright, box_list, before, blurred=[])


@app.post("/api/safe", response_model=ScrubResult)
async def make_safe(file: UploadFile = File(...)):
    """Blur the fewest clues needed to reach low risk."""
    raw, upright = await _read(file)
    analysis = analyze(raw, upright, explain=True)
    ids, note = plan(upright, analysis)
    boxes = [c.box for c in analysis.clues if c.id in ids]
    return _result(upright, boxes, analysis.risk, blurred=ids, note=note)
