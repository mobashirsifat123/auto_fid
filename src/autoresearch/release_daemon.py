"""Dedicated local daemon entry point used by the product launcher."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="autoresearch-daemon")
    parser.add_argument("--host", default=os.getenv("AUTORESEARCH_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AUTORESEARCH_API_PORT", "8000")))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("AutoResearch local daemon must bind to a loopback host.")
    uvicorn.run("autoresearch.api.app:create_app", factory=True, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
