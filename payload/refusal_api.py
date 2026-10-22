# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Loopback-only admin API for the refusal strength used by future admissions."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from vllm import refusal_policy as policy

TOKEN_FILE_ENV = "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE"
ROUTE = "/admin/refusal_lambda"


class LambdaBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lambda_value: float = Field(alias="lambda", ge=0.0, le=4.0)


def _admin_token() -> str:
    path = os.environ.get(TOKEN_FILE_ENV)
    if not path:
        raise HTTPException(status_code=503, detail="admin credential is not configured")
    try:
        token = Path(path).read_text().rstrip("\r\n")
    except OSError:
        raise HTTPException(status_code=503, detail="admin credential is unavailable") from None
    if not token:
        raise HTTPException(status_code=503, detail="admin credential is unavailable")
    return token


def _authorize(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    client = request.client
    if client is None or not policy.is_loopback_host(client.host):
        raise HTTPException(status_code=403, detail="loopback access required")
    token = _admin_token()
    supplied = authorization or ""
    expected = f"Bearer {token}"
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="admin authentication required")


def _state() -> dict:
    value, epoch = policy.get_default_state()
    return {
        "lambda": value,
        "epoch": epoch,
        "artifact_sha256": policy.artifact_sha256(),
        "scope": policy.SCOPE,
        "effect": "future request admissions only; already stamped requests are unchanged",
    }


def create_router() -> APIRouter:
    router = APIRouter()
    if not policy.feature_enabled():
        return router

    @router.get(ROUTE)
    def get_refusal_lambda(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict:
        _authorize(request, authorization)
        return _state()

    @router.post(ROUTE)
    def set_refusal_lambda(
        body: LambdaBody,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict:
        _authorize(request, authorization)
        value, epoch = policy.set_default_lambda(body.lambda_value)
        state = _state()
        state["lambda"] = value
        state["epoch"] = epoch
        return state

    return router


router = create_router()


def attach_router(app) -> None:
    """Mount refusal administration only when the refusal artifact is configured."""
    if policy.feature_enabled():
        app.include_router(create_router())
