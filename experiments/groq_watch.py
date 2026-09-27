"""Check Groq's rate limits periodically with a tiny request; exit when they exceed the free tier.

    uv run python -m experiments.groq_watch [--every 3600] [--min-tpm 16000]

Each check costs a few tokens and is appended to logs/groq_limits.log.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

LOG = Path("logs/groq_limits.log")


def check() -> dict:
    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                           max_retries=0)
    raw = client.chat.completions.with_raw_response.create(
        model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "OK"}], max_tokens=4)
    return {"requests_per_day": int(raw.headers.get("x-ratelimit-limit-requests", "0")),
            "tokens_per_minute": int(raw.headers.get("x-ratelimit-limit-tokens", "0"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3600)
    ap.add_argument("--min-tpm", type=int, default=16_000)
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            limits = check()
            entry = {"checked": now, **limits}
        except Exception as e:  # a failed check is logged and retried next hour, never fatal
            limits, entry = None, {"checked": now, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if limits and limits["tokens_per_minute"] >= args.min_tpm:
            print(f"PAID LIMITS: {json.dumps(entry)}")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
