"""Response shapes shared by the pipeline and the API."""

from typing import Optional

from pydantic import BaseModel, Field


class Clue(BaseModel):
    id: str
    kind: str  # "object" (OWLv2) or "text" (OCR)
    category: str
    label: str  # what was matched: the prompt, or the OCR text
    score: float  # detector confidence, 0-1
    # [x0, y0, x1, y1] in original-image pixels; None for text only the AI
    # model read (we know it's in the photo, not where)
    box: Optional[list[float]] = None
    reason: str
    weight: float
    # How much blurring this clue alone lowers confidence in the top
    # location guess (0-1). None when attribution is off.
    impact: Optional[float] = None
    source: str = ""  # "", "ocr+ai" (OCR box, AI reading) or "ai"


class SceneClue(BaseModel):
    """A whole-image signal (skyline, vegetation...) with no box to blur."""
    category: str
    label: str
    prob: float
    reason: str
    weight: float


class TextMatch(BaseModel):
    """A real place found by searching a sign's text (opt-in lookup)."""
    clue_id: str
    text: str
    query: str
    name: str
    lat: float
    lon: float
    kind: str = ""
    confidence: str = "strong"  # "strong" (one clear place) or "weak" (possible match)
    n_results: int = 1
    note: str = ""


class AIGuess(BaseModel):
    place: str
    confidence: float
    why: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


class AIReading(BaseModel):
    """What the local AI vision model extracted and concluded."""
    model: str
    ms: int = 0
    sign_texts: list[str] = Field(default_factory=list)
    clues: list[dict] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    guesses: list[AIGuess] = Field(default_factory=list)
    error: str = ""


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
    points: float  # leak strength of this factor alone, 0-100
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
    scene: list[SceneClue] = Field(default_factory=list)
    geolocation: Geolocation
    text_matches: list[TextMatch] = Field(default_factory=list)
    lookup_used: bool = False
    ai: Optional[AIReading] = None
    risk: Risk
    # JPEG preview for formats browsers can't show (e.g. HEIC from iPhones).
    preview_b64: Optional[str] = None
    mock: bool = False


class ScrubResult(BaseModel):
    image_b64: str  # JPEG, no metadata
    before: Risk
    after: Analysis
    blurred: list[str] = Field(default_factory=list)  # clue ids
    note: str = ""
