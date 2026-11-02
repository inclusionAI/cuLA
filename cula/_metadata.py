# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only packed-sequence and contiguous-shard metadata helpers."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index

import torch


def as_boundaries(values: torch.Tensor | Sequence[int], *, name: str) -> tuple[int, ...]:
    """Validate a strictly increasing, zero-based packed prefix on the host."""

    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor; copying CUDA metadata would synchronize the device")
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if values.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must have int32 or int64 dtype")
        result = tuple(int(value) for value in values.tolist())
    else:
        try:
            result = tuple(index(value) for value in values)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be an integer sequence") from error
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least a start and an end")
    if result[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def balanced_shards(total_tokens: int, world_size: int) -> tuple[int, ...]:
    if total_tokens < world_size:
        raise ValueError("packed context parallelism requires at least one token on every rank")
    return tuple(shard * total_tokens // world_size for shard in range(world_size + 1))


def validate_shards(shard_offsets: Sequence[int], *, total_tokens: int, world_size: int) -> tuple[int, ...]:
    try:
        offsets = tuple(index(value) for value in shard_offsets)
    except (TypeError, ValueError) as error:
        raise TypeError("shard_offsets must be an integer sequence") from error
    if len(offsets) != world_size + 1:
        raise ValueError(f"shard_offsets must contain world_size + 1 ({world_size + 1}) entries")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("shard_offsets must cover the complete packed token range")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("shard_offsets must be strictly increasing; empty CP shards are unsupported")
    return offsets


__all__ = ["as_boundaries", "balanced_shards", "validate_shards"]
