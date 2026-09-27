"""Check Groq's rate limits periodically with a tiny request; exit when they exceed the free tier.

    uv run python -m experiments.groq_watch [--every 3600] [--min-tpm 16000] [--once]

Each check costs a few tokens and is appended to logs/groq_limits.log. Exit code 0
means the paid limit is available, 1 means a check failed (retry next hour).

Two failure modes this fixes, both seen while watching the free tier:
- Clock drift: `time.sleep(every)` after each check made the gap between checks
  `every + check duration`, so the cadence crept by seconds per hour and a
  long watch drifted by hours. It now sleeps to an absolute deadline.
- Idle connections: the client was rebuilt on every check, and an hourly
  keep-alive-free socket to Groq is dropped by intermediaries, so checks started
  failing with connection resets. One client is now reused, and a check that
  fails on a transport error rebuilds it before retrying.
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
MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"


def new_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=BASE_URL, api_key=os.environ["GROQ_API_KEY"], max_retries=0,
                         timeout=30.0, max_connections=1)


def check(client: openai.OpenAI) -> dict:
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL, messages=[{"role": "user", "content": "OK"}], max_tokens=4)
    return {"requests_per_day": int(raw.headers.get("x-ratelimit-limit-requests", "0")),
            "tokens_per_minute": int(raw.headers.get("x-ratelimit-limit-tokens", "0"))}


def check_once(client: openai.OpenAI) -> dict:
    """One check, rebuilding the client once if the connection had gone stale."""
    try:
        return check(client)
    except (openai.APIConnectionError, openai.APITimeoutError):
        return check(new_client())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3600)
    ap.add_argument("--min-tpm", type=int, default=16_000)
    ap.add_argument("--once", action="store_true", help="one check, then exit")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    client = new_client()
    next_at = time.monotonic()
    while True:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            limits = check_once(client)
            entry = {"checked": now, **limits}
        except Exception as e:  # a failed check is logged and retried next hour, never fatal
            client = new_client()  # the connection may be the reason
            limits, entry = None, {"checked": now, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if limits and limits["tokens_per_minute"] >= args.min_tpm:
            print(f"PAID LIMITS: {json.dumps(entry)}")
            return 0
        print(f"{now}: {json.dumps(entry)}", flush=True)
        if args.once:
            return 1
        # Sleep to an absolute deadline, so the cadence does not drift by the check duration.
        next_at += args.every
        now_m = time.monotonic()
        if next_at <= now_m:  # a check took longer than the interval: restart the cadence
            next_at = now_m
        time.sleep(next_at - now_m)
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
