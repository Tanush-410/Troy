"""Check that an Ollama model really gets the context window we ask for.

Sends prompts of increasing size with num_ctx set and checks that the server's
prompt_eval_count grows with them (Ollama silently truncates an overflowing
prompt to about half the window). Local and free:
    uv run python -m experiments.context_probe qwen3:8b 32768
"""

from __future__ import annotations

import sys

from agents.providers.ollama_provider import DEFAULT_URL, _post


def model_context_length(model: str) -> int | None:
    info = _post(f"{DEFAULT_URL}/api/show", {"model": model}).get("model_info", {})
    return next((v for k, v in info.items() if k.endswith("context_length")), None)


def probe(model: str, num_ctx: int) -> bool:
    native = model_context_length(model)
    print(f"{model}: native context_length={native}, requested num_ctx={num_ctx}")
    ok = native is None or num_ctx <= native
    if not ok:
        print("  FAIL: num_ctx exceeds the model's native window")
    last = 0
    for frac in (0.25, 0.5, 0.75, 0.9):
        n_items = int(num_ctx * frac / 4.5)  # "itemNNNN " is ~4-5 tokens
        filler = " ".join(f"item{i}" for i in range(n_items))
        r = _post(f"{DEFAULT_URL}/api/chat", {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": filler + "\nReply with OK."}],
            "options": {"num_ctx": num_ctx, "num_predict": 2}, "think": False,
        })
        got = r.get("prompt_eval_count", 0)
        grew = got > last
        print(f"  ~{frac:.0%} of window: prompt_eval_count={got} {'ok' if grew else 'TRUNCATED'}")
        ok = ok and grew
        last = got
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    model, num_ctx = sys.argv[1], int(sys.argv[2])
    sys.exit(0 if probe(model, num_ctx) else 1)
