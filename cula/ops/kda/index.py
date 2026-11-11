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

    entry: tuple[tuple[Any, ...], dict[str, Any], tuple[int | None, ...], Any] | None = None

    def tensor_versions(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int | None, ...]:
        versions = []
        for value in (*args, *kwargs.values()):
            if not isinstance(value, torch.Tensor):
                versions.append(None)
                continue
            try:
                versions.append(value._version)
            except RuntimeError:
                # Inference tensors do not expose a version counter. Treat
                # them as uncacheable rather than risking stale metadata.
                versions.append(None)
        return tuple(versions)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal entry
        cached = entry
        if cached is not None:
            cached_args, cached_kwargs, cached_versions, cached_result = cached
            same_args = len(args) == len(cached_args) and all(
                arg is cached_arg for arg, cached_arg in zip(args, cached_args, strict=True)
            )
            same_kwargs = len(kwargs) == len(cached_kwargs) and all(
                key in cached_kwargs and value is cached_kwargs[key] for key, value in kwargs.items()
            )
            versions = tensor_versions(args, kwargs)
            versions_available = all(
                version is not None
                for value, version in zip((*args, *kwargs.values()), versions, strict=True)
                if isinstance(value, torch.Tensor)
            )
            if same_args and same_kwargs and versions_available and versions == cached_versions:
                return cached_result

        result = fn(*args, **kwargs)
        entry = (args, kwargs, tensor_versions(args, kwargs), result)
        return result

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

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    source = cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu
    if source.ndim != 1:
        raise ValueError(f"cu_seqlens must be one-dimensional, got shape {tuple(source.shape)}")
    if source.numel() == 0:
        raise ValueError("cu_seqlens must contain at least one element")

    counts = torch.div(prepare_lens(source) + chunk_size - 1, chunk_size, rounding_mode="floor")
    repeat_counts = counts.to(torch.int64)
    sequence_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=source.dtype, device=source.device), repeat_counts
    )
    sequence_starts = torch.cumsum(counts, dim=0) - counts
    local_chunk_ids = torch.arange(sequence_ids.numel(), dtype=source.dtype, device=source.device) - (
        torch.repeat_interleave(sequence_starts, repeat_counts)
    )
    return torch.stack((sequence_ids, local_chunk_ids), dim=-1).to(dtype=cu_seqlens.dtype, device=cu_seqlens.device)


__all__ = ["prepare_chunk_indices", "prepare_lens", "single_entry_tensor_cache"]
