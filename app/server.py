"""Stdlib HTTP server for the interactive UI. No web framework, no new dependencies.

SSE over a POST rather than an `EventSource`, because a message has to be sent and
its events streamed back on one connection. That means the browser reads the body
with `fetch` + a stream reader instead of `EventSource`, which is why the frontend
parses SSE frames by hand.

Run:  uv run python -m app.server        (then open http://127.0.0.1:8765)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agents.providers.base import ProviderConfig
from app.session import PROVENANCE, InteractiveSession, catalog

STATIC = Path(__file__).parent / "static"
GROQ_URL = "https://api.groq.com/openai/v1"

#: Verified against the live endpoint: gpt-oss-120b emits clean tool calls.
#: qwen/qwen3.8-27b is listed but needs max_tokens <= 1000 on the free tier (OTPM).
MODELS: dict[str, dict[str, Any]] = {
    "gpt-oss-120b": {
        "label": "openai/gpt-oss-120b (Groq, reasoning)",
        "cfg": ProviderConfig(
            kind="openai_compat", model="openai/gpt-oss-120b", context_window=131_072,
            base_url=GROQ_URL, api_key_env="GROQ_API_KEY", temperature=0.0, seed=7,
            max_tokens=2048, reasoning_effort="low",
        ),
    },
    "qwen3-8b": {
        "label": "qwen3:8b (Ollama, local, no key)",
        "cfg": ProviderConfig(
            kind="ollama", model="qwen3:8b", context_window=32_768,
            base_url="http://localhost:11434", temperature=0.0, seed=7,
            max_tokens=2048, think=False,
        ),
    },
}

_sessions: dict[str, InteractiveSession] = {}
_lock = threading.Lock()


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, default=str)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "simmart-ui"

    # ------------------------------------------------------------------ plumbing

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        if "GET /api/session/" in fmt % args or "POST" in fmt % args:
            print(f"  {fmt % args}")

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if not path.is_file() or STATIC.resolve() not in path.parents:
            self._json({"error": "not found"}, 404)
            return
        ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css"}.get(
            path.suffix.lstrip("."), "application/octet-stream"
        )
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _session(self, sid: str) -> InteractiveSession | None:
        with _lock:
            return _sessions.get(sid)

    # --------------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):])
        elif path == "/api/catalog":
            self._json({
                "catalog": catalog(),
                "models": {k: v["label"] for k, v in MODELS.items()},
                "provenance": PROVENANCE,
                "has_groq_key": bool(os.environ.get("GROQ_API_KEY", "").strip()),
            })
        elif path.startswith("/api/session/"):
            sid = path.rsplit("/", 1)[-1]
            s = self._session(sid)
            if s is None:
                self._json({"error": "no such session"}, 404)
            else:
                self._json({"summary": s.summary(), "log": s.log_rows()})
        else:
            self._json({"error": "not found"}, 404)

    # -------------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            if path == "/api/session":
                self._create()
            elif path.startswith("/api/session/") and path.endswith("/message"):
                self._message(path.split("/")[3])
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # surface the traceback in the UI, not just a 500
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _create(self) -> None:
        req = self._body()
        model_key = req.get("model", "gpt-oss-120b")
        if model_key not in MODELS:
            self._json({"error": f"unknown model {model_key}"}, 400)
            return
        sid = uuid.uuid4().hex[:12]
        s = InteractiveSession(
            session_id=sid,
            role=req.get("role", "support"),
            provider_cfg=MODELS[model_key]["cfg"],
            control=req.get("control", "C1"),
            drift=req.get("drift", "D0"),
            task_type=req.get("task_type") or None,
            seed=int(req.get("seed", 7)),
            max_turns=int(req.get("max_turns", 12)),
        )
        with _lock:
            _sessions[sid] = s
        self._json({"session_id": sid, "summary": s.summary()})

    def _message(self, sid: str) -> None:
        s = self._session(sid)
        if s is None:
            self._json({"error": "no such session"}, 404)
            return
        text = (self._body().get("text") or "").strip()
        if not text:
            self._json({"error": "empty message"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def frames() -> Iterator[bytes]:
            try:
                for ev in s.send(text):
                    yield _sse(ev)
            except Exception as e:
                traceback.print_exc()
                yield _sse({"type": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                yield _sse({"type": "summary", "summary": s.summary()})
                yield b"data: [DONE]\n\n"

        for chunk in frames():
            self.wfile.write(chunk)
            self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="SimMart interactive agent UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    print(f"SimMart UI on http://{args.host}:{args.port}")
    print(f"  GROQ_API_KEY : {'set' if os.environ.get('GROQ_API_KEY') else 'NOT SET'}")
    print(f"  models       : {', '.join(MODELS)}")
    print("  sessions are interactive demos, not experimental data (provenance tagged)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
