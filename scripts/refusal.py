# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Small authenticated CLI for the loopback refusal administration endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


TOKEN_ENV = "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE"
DEFAULT_TOKEN = "/srv/secrets/refusal-admin-token"


def request_state(url: str, token_file: Path, value: float | None = None) -> dict:
    token = token_file.read_text(encoding="utf-8").rstrip("\r\n")
    if not token:
        raise ValueError("admin token file is empty")
    payload = None if value is None else json.dumps({"lambda": value}).encode("utf-8")
    request = Request(
        url.rstrip("/") + "/admin/refusal_lambda",
        data=payload,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="GET" if value is None else "POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("admin endpoint returned a non-object response")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=("0", "1"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, default=Path(os.environ.get(TOKEN_ENV, DEFAULT_TOKEN)))
    args = parser.parse_args(argv)
    try:
        value = 0.0 if args.state == "0" else 2.5
        result = request_state(args.base_url, args.token_file, value)
    except (OSError, ValueError, URLError, HTTPError, json.JSONDecodeError) as exc:
        print(f"refusal admin request failed: {exc}", file=sys.stderr)
        return 1
    if value is not None and result.get("lambda") != value:
        print("refusal admin endpoint did not confirm the requested value", file=sys.stderr)
        return 1
    state = args.state if value is not None else ("0" if result.get("lambda") == 0 else "1")
    print(json.dumps({"state": state, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
