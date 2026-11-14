"""Check the local AI model end to end on one photo.

    python scripts/ai_check.py path/to/photo.jpg

Prints Ollama's status, then for each attempt (schema, json, free) where
the answer landed (content or thinking) and what LeakLens parsed from it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.pipeline import load_image, vlm  # noqa: E402


def main(path: str) -> None:
    st = vlm.status(refresh=True)
    print(f"Ollama at {config.OLLAMA_URL}, model {config.VLM_MODEL}: {st.get('reason')}")
    if not st.get("ok"):
        sys.exit(1)
    _, upright = load_image(Path(path).read_bytes())
    image_b64 = vlm._encode(upright, config.VLM_MAX_SIDE)
    for name, fmt in vlm.ATTEMPTS:
        payload = {"model": st["model"], "stream": False, "keep_alive": "15m",
                   "messages": [{"role": "user", "content": vlm.PROMPT, "images": [image_b64]}],
                   "options": {"temperature": 0, "num_ctx": config.VLM_NUM_CTX, "num_predict": 2048}}
        if fmt() is not None:
            payload["format"] = fmt()
        t0 = time.monotonic()
        try:
            resp = vlm._http("/api/chat", payload, timeout=config.VLM_TIMEOUT)
        except Exception as exc:
            print(f"\n[{name}] request failed: {exc}")
            continue
        msg = resp.get("message") or {}
        print(f"\n[{name}] {time.monotonic() - t0:.1f}s, done_reason={resp.get('done_reason')}")
        print(f"  content  ({len(msg.get('content') or '')} chars): {(msg.get('content') or '')[:160]!r}")
        print(f"  thinking ({len(msg.get('thinking') or '')} chars): {(msg.get('thinking') or '')[:160]!r}")
        try:
            data = vlm._parse(vlm._answer_text(resp))
        except Exception as exc:
            print(f"  parse failed: {exc}")
            continue
        r = vlm.to_reading(data, st["model"], 0)
        print(f"  parsed: signs={r.sign_texts} guesses={[(g.place, g.confidence) for g in r.guesses]}")
        if vlm._useful(data):
            print(f"\nOK: LeakLens would use the '{name}' answer.")
            return
    print("\nNo attempt produced a usable answer. Paste this output to debug.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
