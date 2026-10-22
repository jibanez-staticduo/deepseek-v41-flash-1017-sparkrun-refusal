# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Per-request refusal projection for DeepSeek V4.1 Flash 1017 WO-B layers."""

import hashlib
import json
import math
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    import triton as _triton
    import triton.language as _tl
except ImportError:
    _triton = None
    _tl = None


EXPECTED_SHA256 = "d9820fff33a289ed9b0ce8a08669512250bae35d13fba8598acb521a25a2daaa"
EXPECTED_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
EXPECTED_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
EXPECTED_SCOPE = "backbone attention WO-B layers2..39; MTP/DSpark untouched"
HIDDEN_SIZE = 5120
FIRST_LAYER = 2
LAST_LAYER = 39
SCHEMA = "deepseek-refusal-projection-v2"
MAX_HEADER_BYTES = 1024 * 1024
_EXPECTED_KEYS = tuple(f"layers.{layer}.attn.wo_b" for layer in range(FIRST_LAYER, LAST_LAYER + 1))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensors(path: Path) -> tuple[dict[str, Any], int, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated refusal directions file: {path}")
        (header_size,) = struct.unpack("<Q", prefix)
        if not 0 < header_size <= MAX_HEADER_BYTES or header_size > size - 8:
            raise ValueError(f"invalid safetensors header length in {path}")
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header in {path}") from exc
    if not isinstance(header, dict):
        raise ValueError("refusal directions safetensors header must be an object")
    return header, 8 + header_size, size


def _validate_header(header: dict[str, Any], data_size: int) -> None:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError("refusal directions file is missing __metadata__")
    if metadata.get("model") != EXPECTED_MODEL:
        raise ValueError("refusal directions model metadata does not match DeepSeek V4.1 Flash 1017")
    if metadata.get("revision") != EXPECTED_REVISION:
        raise ValueError("refusal directions model revision does not match the official 1017 snapshot")
    if metadata.get("scope") != EXPECTED_SCOPE:
        raise ValueError("refusal directions scope must be exactly WO-B layers 2..39")
    tensor_keys = tuple(key for key in header if key != "__metadata__")
    if set(tensor_keys) != set(_EXPECTED_KEYS) or len(tensor_keys) != len(_EXPECTED_KEYS):
        raise ValueError("refusal directions must contain exactly WO-B layers 2..39")

    spans: list[tuple[int, int, str]] = []
    for key in _EXPECTED_KEYS:
        meta = header[key]
        if not isinstance(meta, dict) or meta.get("dtype") != "F32" or meta.get("shape") != [HIDDEN_SIZE]:
            raise ValueError(f"{key} must be F32[{HIDDEN_SIZE}]")
        offsets = meta.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(offset) is not int for offset in offsets)):
            raise ValueError(f"invalid data offsets for {key}")
        start, end = offsets
        if start < 0 or end < start or end > data_size or end - start != HIDDEN_SIZE * 4:
            raise ValueError(f"invalid data bounds for {key}")
        spans.append((start, end, key))
    spans.sort()
    if not spans or spans[0][0] != 0 or spans[-1][1] != data_size:
        raise ValueError("refusal directions payload has gaps or trailing bytes")
    for previous, current in zip(spans, spans[1:]):
        if previous[1] != current[0]:
            raise ValueError(f"refusal directions payload overlaps or has a gap near {current[2]}")


@lru_cache(maxsize=4)
def _load_directions(path_string: str, expected_sha256: str) -> tuple[np.ndarray, str]:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"refusal directions file does not exist: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"refusal directions SHA256 mismatch: expected {expected_sha256}, got {actual_sha}")
    header, data_start, file_size = _read_safetensors(path)
    _validate_header(header, file_size - data_start)

    vectors = np.empty((len(_EXPECTED_KEYS), HIDDEN_SIZE), dtype=np.float32)
    with path.open("rb") as handle:
        for row, key in enumerate(_EXPECTED_KEYS):
            start, end = header[key]["data_offsets"]
            handle.seek(data_start + start)
            raw = handle.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"short read for refusal direction {key}")
            vector = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(vector).all():
                raise ValueError(f"refusal direction {key} contains non-finite values")
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                raise ValueError(f"refusal direction {key} must be unit length; norm={norm:.9g}")
            vectors[row] = vector
    vectors.setflags(write=False)
    return vectors, actual_sha


@lru_cache(maxsize=1)
def _configured_artifact() -> tuple[str, str] | None:
    raw_path = os.environ.get("VLLM_REFUSAL_DIRS")
    if not raw_path:
        return None
    path = str(Path(raw_path).expanduser().resolve())
    _, digest = _load_directions(path, EXPECTED_SHA256)
    return path, digest


def artifact_identity() -> str | None:
    """Return the immutable model/directions identity, independent of request lambda."""
    configured = _configured_artifact()
    if configured is None:
        return None
    _, digest = configured
    return f"{SCHEMA}:{EXPECTED_MODEL}@{EXPECTED_REVISION}:sha256={digest}"


def _rank_order_parts_type() -> type | tuple[()]:
    try:
        from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts
    except ImportError:
        return ()
    return RankOrderParts


_RANK_ORDER_PARTS_TYPE = _rank_order_parts_type()


if _triton is not None:
    @_triton.jit
    def project_rows_kernel(x_ptr, r_ptr, lambda_ptr, out_ptr,
                            x_stride_row, x_stride_col, lambda_stride,
                            HIDDEN: _tl.constexpr, BLOCK: _tl.constexpr):
        row = _tl.program_id(0)
        cols = _tl.arange(0, BLOCK)
        mask = cols < HIDDEN
        row_lambda = _tl.load(lambda_ptr + row * lambda_stride).to(_tl.float32)
        direction = _tl.load(r_ptr + cols, mask=mask, other=0).to(_tl.float32)
        input_bf16 = _tl.load(x_ptr + row * x_stride_row + cols * x_stride_col,
                              mask=mask, other=0)
        input_fp32 = input_bf16.to(_tl.float32)
        dot = _tl.sum(input_fp32 * direction, axis=0)
        projected = (input_fp32 - row_lambda * dot * direction).to(_tl.bfloat16)
        # Select the original BF16 bits directly for zero rows, including NaNs and signed zero.
        output = _tl.where(row_lambda == 0.0, input_bf16, projected)
        _tl.store(out_ptr + row * HIDDEN + cols, output, mask=mask)
else:
    project_rows_kernel = None


@torch.library.custom_op("vllm_refusal_v2::project_rows", mutates_args=())
def _project_rows(value: torch.Tensor, direction: torch.Tensor,
                  lambdas: torch.Tensor) -> torch.Tensor:
    _validate_projection_inputs(value, direction, lambdas)
    if value.requires_grad or direction.requires_grad or lambdas.requires_grad:
        raise RuntimeError("refusal projection is inference-only and does not support autograd")
    if value.is_cuda:
        if project_rows_kernel is None or _triton is None:
            raise RuntimeError("Triton is required for CUDA refusal projection")
        output = torch.empty(value.shape, dtype=value.dtype, device=value.device)
        if value.shape[0]:
            project_rows_kernel[(value.shape[0],)](
                value,
                direction,
                lambdas,
                output,
                value.stride(0),
                value.stride(1),
                lambdas.stride(0),
                HIDDEN_SIZE,
                8192,
                num_warps=4,
            )
        return output
    return _project_rows_cpu(value, direction, lambdas)


def _validate_projection_inputs(value: torch.Tensor, direction: torch.Tensor,
                                lambdas: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"projection input must have shape [rows,{HIDDEN_SIZE}]")
    if value.dtype != torch.bfloat16:
        raise TypeError(f"projection input must be BF16, got {value.dtype}")
    if direction.dtype != torch.float32 or direction.ndim != 1 or direction.numel() != HIDDEN_SIZE:
        raise ValueError(f"projection direction must be F32[{HIDDEN_SIZE}]")
    if lambdas.dtype != torch.float32 or lambdas.ndim != 1 or lambdas.shape[0] != value.shape[0]:
        raise ValueError("row lambdas must be an FP32 vector with one value per input row")
    if direction.device != value.device or lambdas.device != value.device:
        raise ValueError("input, direction, and row lambdas must be on the same device")
    if not value.is_cuda:
        if not torch.isfinite(lambdas).all():
            raise ValueError("row lambdas must be finite")
        if torch.any((lambdas < 0.0) | (lambdas > 4.0)):
            raise ValueError("row lambdas must be in [0,4]")


def _project_rows_cpu(value: torch.Tensor, direction: torch.Tensor,
                      lambdas: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(value)
    direction_fp32 = direction.float()
    for start in range(0, value.shape[0], 128):
        stop = min(start + 128, value.shape[0])
        rows = value[start:stop].float()
        row_lambdas = lambdas[start:stop, None]
        dot = torch.sum(rows * direction_fp32, dim=-1, keepdim=True)
        projected = (rows - row_lambdas * dot * direction_fp32).to(torch.bfloat16)
        original = value[start:stop]
        output[start:stop].copy_(torch.where(row_lambdas == 0.0, original, projected))
    return output


@_project_rows.register_fake
def _project_rows_fake(value: torch.Tensor, direction: torch.Tensor,
                       lambdas: torch.Tensor) -> torch.Tensor:
    return torch.empty(value.shape, dtype=value.dtype, device=value.device)


class RefusalProjection(nn.Module):
    """Apply per-row refusal lambdas using one direction for a target layer."""

    def __init__(self, layer: int, hidden_size: int):
        super().__init__()
        if type(layer) is not int or not FIRST_LAYER <= layer <= LAST_LAYER:
            raise ValueError(f"RefusalProjection only supports target layers {FIRST_LAYER}..{LAST_LAYER}")
        if type(hidden_size) is not int or hidden_size != HIDDEN_SIZE:
            raise ValueError(f"DeepSeek V4.1 Flash 1017 refusal requires hidden_size={HIDDEN_SIZE}")
        configured = _configured_artifact()
        if configured is None:
            raise ValueError("VLLM_REFUSAL_DIRS is required to construct target projections")
        path, _ = configured
        vectors, _ = _load_directions(path, EXPECTED_SHA256)
        direction = torch.tensor(vectors[layer - FIRST_LAYER].copy(), dtype=torch.float32)
        self.register_buffer("direction", direction, persistent=False)
        self.layer = layer
        self.hidden_size = hidden_size

    def forward(self, value: Any, lambdas: torch.Tensor) -> Any:
        if isinstance(value, _RANK_ORDER_PARTS_TYPE):
            value = value.sum()
        if not isinstance(value, torch.Tensor):
            raise TypeError("refusal projection expects a BF16 tensor or RankOrderParts")
        if self.direction.device != value.device:
            raise RuntimeError("refusal direction must be materialized on the model device before forward/capture")
        return _project_rows(value, self.direction, lambdas)


def make_projection(layer: int, hidden_size: int) -> RefusalProjection | None:
    """Build target-layer modules whenever the directions artifact is configured."""
    if type(layer) is not int or not FIRST_LAYER <= layer <= LAST_LAYER:
        return None
    if type(hidden_size) is not int or hidden_size != HIDDEN_SIZE:
        raise ValueError(f"DeepSeek V4.1 Flash 1017 refusal requires hidden_size={HIDDEN_SIZE}")
    if _configured_artifact() is None:
        return None
    return RefusalProjection(layer, hidden_size)


__all__ = ["RefusalProjection", "artifact_identity", "make_projection"]
