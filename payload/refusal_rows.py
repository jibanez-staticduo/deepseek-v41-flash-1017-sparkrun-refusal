# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Persistent target-token lambda rows, derived from final GPU query boundaries."""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _fill_rows(slots, starts, mapping, padding, output,
                   padded_rows: tl.constexpr, capacity: tl.constexpr,
                   BLOCK: tl.constexpr):
        req = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        first = tl.load(starts + req)
        end = tl.load(starts + req + 1)
        slot = tl.load(mapping + req)
        strength = tl.load(slots + slot)
        row = first + offsets
        valid = (row < end) & (row < padded_rows) & (row < capacity)
        is_padding = tl.load(padding + row, mask=valid, other=1)
        tl.store(output + row, tl.where(is_padding, 0.0, strength), mask=valid)


class RefusalRows:
    """One worker's request slots and stable padded target-row buffer."""

    def __init__(self, max_reqs, max_tokens, device):
        if max_reqs <= 0 or max_tokens <= 0:
            raise ValueError("positive refusal row capacities are required")
        self.max_reqs = int(max_reqs)
        self.max_tokens = int(max_tokens)
        self.request_values = torch.zeros(self.max_reqs, dtype=torch.float32, device=device)
        self.rows = torch.zeros(self.max_tokens, dtype=torch.float32, device=device)
        self._indices = {}
        self._values = {}

    def set_request(self, index, value, req_id):
        value = float(value)
        if not 0 <= index < self.max_reqs:
            raise ValueError("refusal request slot outside capacity")
        if not math.isfinite(value) or not 0 <= value <= 4:
            raise ValueError("request refusal lambda must be finite and in [0,4]")
        old = self._indices.get(req_id)
        if req_id in self._values and self._values[req_id] != value:
            raise ValueError("a live request cannot change its refusal lambda")
        if old is not None and old != index:
            self.request_values[old].zero_()
        self.request_values[index].fill_(value)
        self._indices[req_id] = index
        self._values[req_id] = value

    def check_update(self, req_id, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 4:
            raise ValueError("request refusal lambda must be finite and in [0,4]")
        if req_id not in self._values:
            raise ValueError("streaming refusal request has no existing lambda")
        if self._values[req_id] != value:
            raise ValueError("a live request cannot change its refusal lambda")

    def remove_request(self, req_id):
        index = self._indices.pop(req_id, None)
        self._values.pop(req_id, None)
        if index is not None:
            self.request_values[index].zero_()

    def dummy(self, num_tokens):
        if not 0 <= num_tokens <= self.max_tokens:
            raise ValueError("dummy refusal rows outside capacity")
        self.rows.zero_()
        return self.rows[:num_tokens]

    def fill(self, input_batch):
        count = input_batch.num_reqs
        padded = input_batch.num_tokens_after_padding
        if not 0 <= count <= self.max_reqs or not 0 <= padded <= self.max_tokens:
            raise ValueError("refusal batch outside allocated capacity")
        starts, mapping = input_batch.query_start_loc, input_batch.idx_mapping
        padding = input_batch.is_padding
        if starts.numel() < count + 1 or mapping.numel() < count or padding.numel() < padded:
            raise ValueError("refusal batch metadata is truncated")
        self.rows.zero_()
        if self.rows.is_cuda:
            if triton is None:
                raise RuntimeError("Triton is required for CUDA refusal rows")
            if count:
                _fill_rows[(count, triton.cdiv(self.max_tokens, 256))](
                    self.request_values, starts, mapping, padding, self.rows,
                    padded, self.max_tokens, 256,
                )
        else:
            # CPU reference uses the same final query boundaries, never their CPU upper bounds.
            for i in range(count):
                first, end = int(starts[i]), int(starts[i + 1])
                slot = int(mapping[i])
                if not 0 <= first <= end <= padded or not 0 <= slot < self.max_reqs:
                    raise ValueError("invalid refusal query boundary or slot")
                self.rows[first:end].copy_(torch.where(
                    padding[first:end], 0.0, self.request_values[slot]
                ))
        return self.rows[:padded]
