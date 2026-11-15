"""A tiny stand-in for Ollama's HTTP API that reproduces failure modes
reported against qwen3-vl and other thinking models:

  good          answer in message.content
  thinking      content empty, answer in message.thinking (marker #998)
  schema_empty  JSON-schema request returns "{}", plain "json" works
  inline_think  "<think>...</think>" trace before the JSON in content
  prose         format ignored: prose plus a fenced JSON block
  truncated     answer cut off mid-string (small default context)
  nothing       every attempt comes back empty
  long_think    thinking build: with a small num_predict it only thinks
                (done_reason=length, content empty); with room it answers
  extra_data    two JSON objects back to back (json.loads: "Extra data")

The chooser (second pass, vlm.choose) is recognized by its prompt. Its
answer is set with `choice`:

  pick          picks the listed option that mentions Huntsville (default)
  other         "other: Decatur, Alabama" (a place not on the list)
  none          the chooser comes back empty every time
The failure `mode` applies to chooser answers too.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = {
    "sign_texts": ["POPPY & PARLIAMENT", "TELEPHONE", "Butcher Shoppe"],
    "clues": [{"clue": "red British-style phone box", "suggests": "a themed pub district in a US city"}],
    "features": ["telephone box", "butcher shop", "pub or bar"],
    "state_evidence": ["no plates visible; 256 phone number on the butcher's sign"],
    "state": "Alabama",
    "guesses": [{"place": "Huntsville, Alabama, USA", "confidence": 0.62,
                 "why": "POPPY & PARLIAMENT is a known pub in downtown Huntsville"}],
}


CHOOSER_MARK = "Similar photos were taken near these places"


def options_in(prompt: str) -> list[str]:
    """The candidate list from a chooser prompt, in the order shown."""
    tail = prompt.split(CHOOSER_MARK, 1)[1].split("\n\n", 1)[0]
    return [line[2:] for line in tail.splitlines() if line.startswith("- ")]


class FakeOllama:
    def __init__(self, mode="good", model="qwen3-vl:8b", choice="pick"):
        self.mode, self.model, self.choice, self.requests = mode, model, choice, []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"models": [{"name": fake.model}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(req)
                self._send(fake.reply(req))

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def is_chooser(req) -> bool:
        return CHOOSER_MARK in req["messages"][0]["content"]

    def chooser_answer(self, req) -> dict:
        opts = options_in(req["messages"][0]["content"])
        if self.choice == "other":
            pick = "other: Decatur, Alabama"
        else:
            pick = next((o for o in opts if "Huntsville" in o), opts[0] if opts else "")
        return {"state": "Alabama", "evidence": "POPPY & PARLIAMENT is a pub in downtown Huntsville",
                "choice": pick, "confidence": 0.7}

    def reply(self, req):
        if self.is_chooser(req):
            if self.choice == "none":
                return {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}
            full = json.dumps(self.chooser_answer(req))
            return self._shape(req, full, cut=full[: full.index('"confidence"') + 14])
        full = json.dumps(ANSWER)
        return self._shape(req, full, cut=full[: full.index("known pub") + 5])

    def _shape(self, req, full, cut):
        fmt = req.get("format")
        msg = {"role": "assistant", "content": full}
        if self.mode == "thinking":
            msg = {"role": "assistant", "content": "", "thinking": full}
        elif self.mode == "schema_empty":
            msg["content"] = "{}" if isinstance(fmt, dict) else full
        elif self.mode == "inline_think":
            msg["content"] = "<think>The sign says POPPY & PARLIAMENT...</think>\n" + full
        elif self.mode == "prose":
            msg["content"] = "Here is my analysis:\n```json\n" + full + "\n```\nHope this helps."
        elif self.mode == "truncated":
            msg["content"] = cut
            return {"message": msg, "done": True, "done_reason": "length"}
        elif self.mode == "long_think":
            if (req.get("options") or {}).get("num_predict", 0) < 4000:
                msg = {"role": "assistant", "content": "",
                       "thinking": 'Let me look... the sign reads {"partial": "POPP' + " hmm" * 400}
                return {"message": msg, "done": True, "done_reason": "length"}
            msg = {"role": "assistant", "content": full, "thinking": "long reasoning..."}
        elif self.mode == "extra_data":
            msg["content"] = '{"note": "draft"} ' + full
        elif self.mode == "nothing":
            msg = {"role": "assistant", "content": ""}
        return {"message": msg, "done": True, "done_reason": "stop"}

    def close(self):
        self.server.shutdown()
