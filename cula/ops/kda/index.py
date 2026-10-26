# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Small tensor-index helpers shared by the standalone KDA kernels.

These functions intentionally use only PyTorch.  Keeping this module local to
cuLA prevents low-level CuTeDSL kernels from acquiring a runtime dependency on
a higher-level linear-attention framework.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch


def single_entry_tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Cache the most recent call by object identity.

    Metadata helpers are commonly invoked several times with the same
    ``cu_seqlens`` tensor during one operator dispatch.  An identity cache is
    sufficient for that case and, unlike a value cache, never synchronizes a
    CUDA tensor merely to build a key.
    """

    last_args: tuple[Any, ...] | None = None
    last_kwargs: dict[str, Any] | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            same_args = len(args) == len(last_args) and all(a is b for a, b in zip(args, last_args, strict=True))
            same_kwargs = len(kwargs) == len(last_kwargs) and all(
                key in last_kwargs and value is last_kwargs[key] for key, value in kwargs.items()
            )
            if same_args and same_kwargs:
                return last_result

        last_result = fn(*args, **kwargs)
        last_args = args
        last_kwargs = kwargs
        return last_result

    return wrapper


@single_entry_tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Return per-sequence lengths for a cumulative-length tensor."""

    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be one-dimensional, got shape {tuple(cu_seqlens.shape)}")
    return torch.diff(cu_seqlens)


@single_entry_tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map each logical chunk to ``(sequence_id, local_chunk_id)``.

    The output is an integer tensor of shape ``[num_chunks, 2]`` on the same
    device and with the same dtype as ``cu_seqlens``.  ``cu_seqlens_cpu`` may
    be supplied to avoid a device-to-host synchronization while determining
    how many chunks belong to each sequence.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    source = cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu
    if source.ndim != 1:
        raise ValueError(f"cu_seqlens must be one-dimensional, got shape {tuple(source.shape)}")
    if source.numel() == 0:
        raise ValueError("cu_seqlens must contain at least one element")

    counts = torch.div(prepare_lens(source) + chunk_size - 1, chunk_size, rounding_mode="floor").tolist()
    sequence_ids: list[int] = []
    local_chunk_ids: list[int] = []
    for sequence_id, count in enumerate(counts):
        count = int(count)
        sequence_ids.extend([sequence_id] * count)
        local_chunk_ids.extend(range(count))

    if not sequence_ids:
        return torch.empty((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.tensor(
        list(zip(sequence_ids, local_chunk_ids, strict=True)),
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )


__all__ = ["prepare_chunk_indices", "prepare_lens", "single_entry_tensor_cache"]
