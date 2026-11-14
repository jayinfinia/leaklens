"""A tiny stand-in for Ollama's HTTP API that reproduces failure modes
reported against qwen3-vl and other thinking models:

  good          answer in message.content
  thinking      content empty, answer in message.thinking (marker #998)
  schema_empty  JSON-schema request returns "{}", plain "json" works
  inline_think  "<think>...</think>" trace before the JSON in content
  prose         format ignored: prose plus a fenced JSON block
  truncated     answer cut off mid-string (small default context)
  nothing       every attempt comes back empty
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = {
    "sign_texts": ["POPPY & PARLIAMENT", "TELEPHONE", "Butcher Shoppe"],
    "clues": [{"clue": "red British-style phone box", "suggests": "a themed pub district in a US city"}],
    "features": ["telephone box", "butcher shop", "pub or bar"],
    "guesses": [{"place": "Huntsville, Alabama, USA", "confidence": 0.62,
                 "why": "POPPY & PARLIAMENT is a known pub in downtown Huntsville"}],
}


class FakeOllama:
    def __init__(self, mode="good", model="qwen3-vl:8b"):
        self.mode, self.model, self.requests = mode, model, []
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

    def reply(self, req):
        full = json.dumps(ANSWER)
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
            msg["content"] = full[: full.index("known pub") + 5]
            return {"message": msg, "done": True, "done_reason": "length"}
        elif self.mode == "nothing":
            msg = {"role": "assistant", "content": ""}
        return {"message": msg, "done": True, "done_reason": "stop"}

    def close(self):
        self.server.shutdown()
