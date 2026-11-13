"""Response shapes shared by the pipeline and the API."""

from typing import Optional

from pydantic import BaseModel, Field


class Clue(BaseModel):
    id: str
    kind: str  # "object" (OWLv2) or "text" (OCR)
    category: str
    label: str  # what was matched: the prompt, or the OCR text
    score: float  # detector confidence, 0-1
    box: list[float]  # [x0, y0, x1, y1] in original-image pixels
    reason: str
    weight: float


class GuessLabel(BaseModel):
    name: str
    prob: float


class GuessPoint(BaseModel):
    lat: float
    lon: float
    prob: float


class Geolocation(BaseModel):
    country: list[GuessLabel] = Field(default_factory=list)
    region: list[GuessLabel] = Field(default_factory=list)
    points: list[GuessPoint] = Field(default_factory=list)
    summary: str = ""


class ExifReport(BaseModel):
    has_exif: bool = False
    gps: Optional[GuessPoint] = None
    taken_at: Optional[str] = None
    device: Optional[str] = None
    fields: dict[str, str] = Field(default_factory=dict)


class RiskFactor(BaseModel):
    source: str
    points: float
    reason: str


class Risk(BaseModel):
    score: int  # 0-100
    level: str  # "low" | "medium" | "high"
    factors: list[RiskFactor]


class Analysis(BaseModel):
    width: int
    height: int
    exif: ExifReport
    clues: list[Clue]
    geolocation: Geolocation
    risk: Risk
    mock: bool = False


class ScrubResult(BaseModel):
    image_b64: str  # JPEG, no metadata
    before: Risk
    after: Analysis
