"""#1: a local AI vision model that reasons about clues, like a real attacker.

"Doxing via the Lens" (ICLR 2026) found the strongest geolocators win by
clue-based reasoning, and their GeoMiner attack splits the job into clue
extraction, then reasoning. We ask a small open model the same two
questions in one structured call:

  1. extract: every readable sign (verbatim), notable clues, and which
     objects from a fixed feature list appear (see features.py)
  2. reason: up to three candidate places with confidence and why

It runs locally through Ollama (default: Qwen3-VL 8B), so the photo never
leaves the machine. Readings feed the text layer, guesses become search
hints and corroboration for the OpenStreetMap lookup.

LEAKLENS_VLM = auto | off      auto: used when Ollama is running with the model
"""

import base64
import io
import json
import time
import urllib.request

from .. import config
from ..schemas import AIGuess, AIReading
from .features import FEATURES

PROMPT = f"""You are the "attacker" half of a privacy checker. The user wants to know what this photo
reveals about where it was taken, so be thorough and concrete.

Step 1, clue extraction:
- sign_texts: copy every readable sign, storefront name, street name, plate, banner or label EXACTLY as
  written, one entry per sign (keep words of one sign together, e.g. "POPPY & PARLIAMENT").
  Skip text you cannot actually read. Do not invent text.
- clues: notable location clues (architecture, vegetation, road markings, vehicles, climate, objects)
  and what each suggests.
- features: which of these objects are clearly visible (use only these exact names):
  {", ".join(FEATURES)}

Step 2, reasoning:
- guesses: up to 3 candidate places, most likely first, as "City, State/Region, Country", each with a
  confidence from 0 to 1 and a one-sentence reason that cites the clues. Use lower confidence when unsure.

Answer with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sign_texts": {"type": "array", "items": {"type": "string"}},
        "clues": {"type": "array", "items": {
            "type": "object",
            "properties": {"clue": {"type": "string"}, "suggests": {"type": "string"}},
            "required": ["clue", "suggests"]}},
        "features": {"type": "array", "items": {"type": "string", "enum": list(FEATURES)}},
        "guesses": {"type": "array", "items": {
            "type": "object",
            "properties": {"place": {"type": "string"}, "confidence": {"type": "number"},
                           "why": {"type": "string"}},
            "required": ["place", "confidence", "why"]}},
    },
    "required": ["sign_texts", "clues", "features", "guesses"],
}

_status = {"checked": 0.0, "ok": False, "reason": "not checked"}


def _http(path: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    url = config.OLLAMA_URL.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def status(refresh: bool = False) -> dict:
    """Is Ollama running with the configured model? Cached for 30 s."""
    if config.VLM_MODE == "off":
        return {"ok": False, "reason": "turned off (LEAKLENS_VLM=off)"}
    if config.MOCK_MODE:
        return {"ok": True, "reason": "mock", "model": "mock-vlm"}
    if not refresh and time.monotonic() - _status["checked"] < 30:
        return dict(_status)
    _status["checked"] = time.monotonic()
    try:
        names = [m.get("name", "") for m in _http("/api/tags").get("models", [])]
    except Exception:
        _status.update(ok=False, reason=f"Ollama not reachable at {config.OLLAMA_URL}")
        return dict(_status)
    want = config.VLM_MODEL
    base = want.split(":")[0]
    match = next((n for n in names if n == want or n == f"{want}:latest"), None)
    if match is None:
        _status.update(ok=False, reason=f"model {want} not pulled (run: ollama pull {want})"
                       + (f"; found {', '.join(n for n in names if n.startswith(base))}" if any(
                           n.startswith(base) for n in names) else ""))
        return dict(_status)
    _status.update(ok=True, reason="ready", model=match)
    return dict(_status)


def _encode(image, max_side: int) -> str:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _parse(content: str) -> dict:
    content = content.strip()
    if not content.startswith("{"):
        start, end = content.find("{"), content.rfind("}")
        content = content[start:end + 1] if start >= 0 and end > start else "{}"
    return json.loads(content)


def to_reading(data: dict, model: str, ms: int) -> AIReading:
    guesses = []
    for g in (data.get("guesses") or [])[:3]:
        try:
            conf = max(0.0, min(1.0, float(g.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
        place = str(g.get("place", "")).strip()
        if place:
            guesses.append(AIGuess(place=place[:120], confidence=round(conf, 2), why=str(g.get("why", ""))[:300]))
    texts = []
    for t in data.get("sign_texts") or []:
        t = " ".join(str(t).split())
        if t and sum(ch.isalnum() for ch in t) >= 3 and t not in texts:
            texts.append(t[:120])
    clues = [{"clue": str(c.get("clue", ""))[:160], "suggests": str(c.get("suggests", ""))[:200]}
             for c in (data.get("clues") or [])[:10] if isinstance(c, dict) and c.get("clue")]
    feats = [f for f in (data.get("features") or []) if f in FEATURES]
    return AIReading(model=model, ms=ms, sign_texts=texts[:15], clues=clues,
                     features=sorted(set(feats)), guesses=guesses)


def ask(image) -> AIReading | None:
    if config.MOCK_MODE:
        from . import mock

        if not mock.has_detail(image):  # blur-aware, like the other mocks
            return AIReading(model="mock-vlm", ms=0, guesses=[
                AIGuess(place="Somewhere in the eastern US", confidence=0.1, why="generic street")])
        return AIReading(model="mock-vlm", ms=0, sign_texts=["MAIN ST", "CORNER JUICE"],
                         clues=[{"clue": "brick storefronts", "suggests": "older US downtown"}],
                         features=["bus stop"],
                         guesses=[AIGuess(place="Charlottesville, Virginia, USA", confidence=0.6,
                                          why="brick downtown mall and Virginia-style signage")])
    st = status()
    if not st.get("ok"):
        return AIReading(model=config.VLM_MODEL, ms=0, error=st.get("reason", "unavailable"))
    t0 = time.monotonic()
    payload = {
        "model": st.get("model", config.VLM_MODEL),
        "messages": [{"role": "user", "content": PROMPT, "images": [_encode(image, config.VLM_MAX_SIDE)]}],
        "format": SCHEMA,
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": 0},
    }
    try:
        resp = _http("/api/chat", payload, timeout=config.VLM_TIMEOUT)
        data = _parse(resp.get("message", {}).get("content", ""))
    except Exception as exc:
        return AIReading(model=payload["model"], ms=int(1000 * (time.monotonic() - t0)),
                         error=f"model call failed: {exc}")
    return to_reading(data, payload["model"], int(1000 * (time.monotonic() - t0)))


# ------------------------------------------------------------ merge readings
def _norm(t: str) -> str:
    return "".join(ch.lower() for ch in t if ch.isalnum())


def _tokens(t: str) -> set:
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in t).split() if len(w) >= 3}


def same_sign(ocr_text: str, ai_text: str) -> bool:
    a, b = _norm(ocr_text), _norm(ai_text)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ta, tb = _tokens(ocr_text), _tokens(ai_text)
    return bool(ta and tb) and len(ta & tb) / min(len(ta), len(tb)) >= 0.5


def merge_texts(clues: list, reading: AIReading | None) -> list:
    """Upgrade OCR readings with the AI model's fuller reading of the same sign
    ("POPPY" -> "POPPY & PARLIAMENT"); add signs only the model read, without a
    box (we know it's in the photo, not where)."""
    from ..schemas import Clue
    from .text_intel import classify

    if not reading or not reading.sign_texts:
        return clues
    out = list(clues)
    used = set()
    for k, text in enumerate(reading.sign_texts):
        hit = next((i for i, c in enumerate(out) if c.kind == "text" and c.box is not None
                    and i not in used and same_sign(c.label, text)), None)
        intel = classify(text)
        if hit is not None:
            used.add(hit)
            c = out[hit]
            if sum(ch.isalnum() for ch in text) > sum(ch.isalnum() for ch in c.label):
                out[hit] = c.model_copy(update={
                    "label": text, "category": f"text:{intel.kind}", "weight": intel.weight,
                    "reason": intel.reason, "source": "ocr+ai"})
            else:
                out[hit] = c.model_copy(update={"source": "ocr+ai"})
        elif intel.kind not in ("generic", "garbled"):
            out.append(Clue(id=f"ai-{k}", kind="text", category=f"text:{intel.kind}", label=text,
                            score=0.7, box=None, weight=intel.weight, source="ai",
                            reason=intel.reason + " Read by the AI model; not located in the photo."))
    return out
