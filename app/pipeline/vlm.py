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

Keep it short: at most 8 sign_texts, 6 clues and 3 guesses, one sentence each. Answer with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sign_texts": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
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
    if match is None:  # use another installed build of the same family, instruct first
        family = sorted((n for n in names if n.startswith(base)), key=lambda n: ("instruct" not in n, n))
        match = family[0] if family else None
    if match is None:
        _status.update(ok=False, reason=f"model {want} not pulled (run: ollama pull {want})"
                       + (f"; installed: {', '.join(names[:6])}" if names else ""))
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


def _repair(text: str) -> str:
    """Close a JSON answer that was cut off mid-way: drop the dangling partial
    value, then close any open string, arrays and objects."""
    stack, in_str, esc, last_ok = [], False, False, 0
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            last_ok = i + 1
        elif ch == ",":
            last_ok = i
    # cut back to the last complete element, then close what is still open
    cut = text[:last_ok].rstrip().rstrip(",")
    stack, in_str, esc = [], False, False
    for ch in cut:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return cut + "".join(reversed(stack))


KEYS = ("sign_texts", "clues", "features", "guesses")


def _parse(content: str) -> dict:
    """First JSON object in the text that looks like our answer. Handles
    prose around it, several objects in a row ("Extra data"), and an answer
    cut off at the end. Thinking traces often quote partial JSON, so the
    LAST complete answer-like object wins."""
    content = (content or "").strip()
    dec = json.JSONDecoder()
    found = []
    i = content.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(content, i)
            if isinstance(obj, dict) and any(k in obj for k in KEYS):
                found.append(obj)
            i = content.find("{", end)
        except json.JSONDecodeError:
            i = content.find("{", i + 1)
    if found:
        return found[-1]
    start = content.find("{")
    if start < 0:
        return {}
    try:
        obj = json.loads(_repair(content[start:]))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


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


def _answer_text(resp: dict) -> str:
    """Thinking models on Ollama (qwen3-vl among them) sometimes leave
    message.content empty and put the whole answer in message.thinking."""
    msg = resp.get("message") or {}
    content = (msg.get("content") or "").strip()
    if "<think>" in content:  # some builds inline the trace
        content = content.split("</think>")[-1].strip()
    if content and content not in ("{}", "[]"):
        return content
    return (msg.get("thinking") or content or "").strip()


def _useful(data: dict) -> bool:
    return bool(data.get("sign_texts") or data.get("guesses") or data.get("clues"))


# Attempts, most to least constrained. Public reports show each can fail on
# some Ollama/model versions: schema output landing in `thinking`, `format`
# ignored when thinking is off, empty content with think=false. We never send
# think=false; we read both fields and fall back step by step.
ATTEMPTS = [
    ("schema", lambda: SCHEMA),
    ("json", lambda: "json"),
    ("free", lambda: None),
]


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
    model = st.get("model", config.VLM_MODEL)
    image_b64 = _encode(image, config.VLM_MAX_SIDE)
    t0 = time.monotonic()
    problems = []
    budget = 2048
    for name, fmt in ATTEMPTS:
        prompt = PROMPT if name != "free" else PROMPT + "\nReturn a single JSON object with keys " \
            '"sign_texts", "clues", "features", "guesses". No other text.'
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "keep_alive": "15m",
            "options": {"temperature": 0, "num_ctx": config.VLM_NUM_CTX, "num_predict": budget},
        }
        if fmt() is not None:
            payload["format"] = fmt()
        try:
            resp = _http("/api/chat", payload, timeout=config.VLM_TIMEOUT)
            msg = resp.get("message") or {}
            if (resp.get("done_reason") == "length" and not (msg.get("content") or "").strip()
                    and msg.get("thinking") and budget < config.VLM_THINK_BUDGET):
                # A thinking build used the whole budget on its reasoning. Let it finish once.
                print(f"[leaklens] {model} is a thinking build and ran out of room; retrying with "
                      f"{config.VLM_THINK_BUDGET} tokens (an '-instruct' build is much faster)")
                budget = config.VLM_THINK_BUDGET
                payload["options"]["num_predict"] = budget
                resp = _http("/api/chat", payload, timeout=config.VLM_TIMEOUT * 2)
        except Exception as exc:
            problems.append(f"{name}: {exc}")
            print(f"[leaklens] AI attempt '{name}' failed: {exc}")
            if "timed out" in str(exc).lower():
                break  # a timeout will just repeat; don't triple the wait
            continue
        text = _answer_text(resp)
        try:
            data = _parse(text)
        except Exception as exc:
            data = {}
            problems.append(f"{name}: unparseable ({exc})")
        if _useful(data):
            reading = to_reading(data, model, int(1000 * (time.monotonic() - t0)))
            print(f"[leaklens] AI model ({name} mode): {len(reading.sign_texts)} signs, "
                  f"{len(reading.guesses)} guesses, {reading.ms / 1000:.1f}s; top guess: "
                  f"{reading.guesses[0].place if reading.guesses else 'none'}")
            return reading
        where = "thinking" if not (resp.get("message") or {}).get("content", "").strip() else "content"
        problems.append(f"{name}: empty answer (looked in {where}; done_reason={resp.get('done_reason')})")
        print(f"[leaklens] AI attempt '{name}' gave nothing usable; trying the next mode")
    hint = (" This looks like a thinking build; try: ollama pull qwen3-vl:8b-instruct"
            if "instruct" not in model else "")
    return AIReading(model=model, ms=int(1000 * (time.monotonic() - t0)),
                     error="no usable answer after 3 attempts: " + "; ".join(problems)[:400] + hint)


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
