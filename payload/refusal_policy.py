# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Admission-time refusal strength and prefix-cache namespace policy."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

ENV_DEFAULT = "VLLM_REFUSAL_LAMBDA_INIT"
ENV_DIRECTIONS = "VLLM_REFUSAL_DIRS"
PREFIX = "staticduo-refusal-v2:"
VERSION = 2
LAMBDA_MIN = 0.0
LAMBDA_MAX = 4.0
ARTIFACT_SHA256 = "d9820fff33a289ed9b0ce8a08669512250bae35d13fba8598acb521a25a2daaa"
SCOPE = "API-process default; stamps future admissions only; admitted requests retain their stamp"


def _validate_lambda(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("refusal lambda must be a finite number in [0,4]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("refusal lambda must be a finite number in [0,4]") from exc
    if not math.isfinite(result) or not LAMBDA_MIN <= result <= LAMBDA_MAX:
        raise ValueError("refusal lambda must be a finite number in [0,4]")
    return 0.0 if result == 0.0 else result


def _initial_lambda() -> float:
    return _validate_lambda(os.environ.get(ENV_DEFAULT, "1"))


_lock = threading.RLock()
_default_lambda = _initial_lambda()
_epoch = 0


@lru_cache(maxsize=8)
def _hash_artifact(path_string: str) -> str:
    path = Path(path_string)
    if not path.is_file():
        raise ValueError("refusal directions artifact is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_enabled() -> bool:
    return bool(os.environ.get(ENV_DIRECTIONS))


def artifact_sha256() -> str | None:
    raw_path = os.environ.get(ENV_DIRECTIONS)
    if not raw_path:
        return None
    digest = _hash_artifact(str(Path(raw_path).expanduser().resolve()))
    if digest != ARTIFACT_SHA256:
        raise ValueError("refusal directions artifact checksum does not match this build")
    return digest


def get_default_state() -> tuple[float, int]:
    with _lock:
        return _default_lambda, _epoch


def set_default_lambda(value: Any) -> tuple[float, int]:
    global _default_lambda, _epoch
    validated = _validate_lambda(value)
    with _lock:
        _default_lambda = validated
        _epoch += 1
        return _default_lambda, _epoch


def _request_override(cache_salt: str | None) -> tuple[float | None, str | None]:
    if cache_salt is None:
        return None, None
    if not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string or null")
    if not cache_salt.startswith("refusal:"):
        return None, cache_salt
    raw = cache_salt[len("refusal:"):].strip()
    if not raw:
        raise ValueError("refusal cache_salt requires a numeric value")
    return _validate_lambda(raw), None


def _decode_envelope(value: str, expected_artifact: str | None = None) -> dict[str, Any]:
    if not value.startswith(PREFIX):
        raise ValueError("not a refusal policy envelope")
    encoded = value[len(PREFIX):]
    try:
        envelope = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid refusal policy envelope") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"version", "lambda", "artifact_sha256", "tenant_salt"}:
        raise ValueError("invalid refusal policy envelope fields")
    if envelope["version"] != VERSION:
        raise ValueError("unsupported refusal policy envelope version")
    if not isinstance(envelope["lambda"], str):
        raise ValueError("invalid refusal policy lambda")
    try:
        value_lambda = _validate_lambda(float.fromhex(envelope["lambda"]))
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid refusal policy lambda") from exc
    if value_lambda.hex() != envelope["lambda"]:
        raise ValueError("noncanonical refusal policy lambda")
    artifact = envelope["artifact_sha256"]
    if artifact is not None and (not isinstance(artifact, str) or len(artifact) != 64 or
                                 any(char not in "0123456789abcdef" for char in artifact)):
        raise ValueError("invalid refusal policy artifact checksum")
    if expected_artifact is not None and artifact != expected_artifact:
        raise ValueError("refusal policy envelope artifact does not match this server")
    tenant_salt = envelope["tenant_salt"]
    if tenant_salt is not None and not isinstance(tenant_salt, str):
        raise ValueError("invalid refusal policy tenant salt")
    if json.dumps(envelope, sort_keys=True, separators=(",", ":")) != encoded:
        raise ValueError("noncanonical refusal policy envelope")
    return envelope


def stamp_cache_salt(cache_salt: str | None) -> str | None:
    """Freeze the effective value and artifact identity at request admission."""
    if not feature_enabled():
        return cache_salt
    artifact = artifact_sha256()
    if isinstance(cache_salt, str) and cache_salt.startswith(PREFIX):
        _decode_envelope(cache_salt, expected_artifact=artifact)
        return cache_salt
    override, tenant_salt = _request_override(cache_salt)
    with _lock:
        effective = _default_lambda if override is None else override
    envelope = {
        "version": VERSION,
        "lambda": effective.hex(),
        "artifact_sha256": artifact,
        "tenant_salt": tenant_salt,
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    return PREFIX + encoded


def parse_effective_lambda(cache_salt: str | None) -> float:
    """Parse an immutable admission stamp without consulting process globals."""
    if not feature_enabled():
        return 0.0
    if cache_salt is None:
        raise ValueError("refusal admission is missing its policy envelope")
    if not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string or null")
    if not cache_salt.startswith(PREFIX):
        raise ValueError("refusal admission is missing its policy envelope")
    return float.fromhex(_decode_envelope(cache_salt, expected_artifact=ARTIFACT_SHA256)["lambda"])


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
