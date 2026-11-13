"""LeakLens API.

Run:  uvicorn app.main:app --reload
Mock: LEAKLENS_MOCK=1 uvicorn app.main:app --reload   (no model downloads)
"""

import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from . import config
from .models import warm_up
from .pipeline import analyze, load_image
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


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "mock": config.MOCK_MODE}


@app.post("/api/analyze", response_model=Analysis)
async def analyze_image(file: UploadFile = File(...)):
    raw, upright = await _read(file)
    return analyze(raw, upright)


@app.post("/api/scrub", response_model=ScrubResult)
async def scrub_image(file: UploadFile = File(...), boxes: str = Form("[]")):
    """Blur the given boxes, strip metadata, then re-run the full analysis."""
    try:
        box_list = json.loads(boxes)
        assert all(isinstance(b, list) and len(b) == 4 for b in box_list)
    except (ValueError, AssertionError):
        raise HTTPException(400, "boxes must be a JSON list of [x0, y0, x1, y1].")

    raw, upright = await _read(file)
    before = analyze(raw, upright).risk
    clean = blur_regions(upright, box_list)
    after = analyze(clean, clean)

    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=92)
    return ScrubResult(
        image_b64=base64.b64encode(buf.getvalue()).decode(),
        before=before,
        after=after,
    )
