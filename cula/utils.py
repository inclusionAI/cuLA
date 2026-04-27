# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for cuLA debugging and development.
"""

import functools
import os
from collections.abc import Callable

import cutlass
import torch
from cutlass import cute

# ---------------------------------------------------------------------------
# Fast-math flag (read once at import time)
# ---------------------------------------------------------------------------
USE_FAST_MATH: bool = os.getenv("CULA_USE_FAST_MATH", "1") == "1"

# ---------------------------------------------------------------------------
# Device architecture helpers
# ---------------------------------------------------------------------------


@functools.cache
def get_device_sm_version(device: torch.device | str | int | None = None) -> tuple[int, int]:
    """Return the CUDA compute capability (major, minor) for *device*.

    Args:
        device: Any value accepted by ``torch.device``.  When ``None`` the
                currently active CUDA device is used.

    Returns:
        ``(major, minor)`` tuple, e.g. ``(9, 0)`` for sm90a or ``(10, 0)``
        for sm100a.

    Example::

        major, minor = get_device_sm_version()
        if major == 10:   # Blackwell
            ...
        elif major == 9:  # Hopper
            ...
    """
    if device is None:
        device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    return prop.major, prop.minor


def is_blackwell(device: torch.device | str | int | None = None) -> bool:
    """Check whether *device* is a Blackwell-architecture GPU (SM100/SM103).

    Returns ``True`` for SM100 (e.g. B200) and SM103 (e.g. B300).
    """
    major, minor = get_device_sm_version(device)
    return major == 10 and minor in (0, 3)


def assert_blackwell(device: torch.device | str | int | None = None) -> None:
    """Assert that *device* is a Blackwell-architecture GPU (SM100/SM103).

    Raises:
        RuntimeError: If the device is not Blackwell.
    """
    major, minor = get_device_sm_version(device)
    if not (major == 10 and minor in (0, 3)):
        raise RuntimeError(f"Only Blackwell GPUs (SM100/SM103) are supported, got compute capability sm_{major}{minor}.")


def assert_hopper(device: torch.device | str | int | None = None) -> None:
    """Assert that *device* is a Hopper-architecture GPU (SM90).

    Raises:
        RuntimeError: If the device is not Hopper.
    """
    major, minor = get_device_sm_version(device)
    if not (major == 9 and minor == 0):
        raise RuntimeError(f"Only Hopper GPUs (SM90) are supported, got compute capability sm_{major}{minor}.")


def get_kda_fused_fwd(device: torch.device | str | int | None = None) -> Callable:
    """Return the appropriate ``kda_prefill`` implementation for *device*.

    - sm100/sm103 (Blackwell) → cula.kda.kda_prefill_blackwell (not yet available)
    - sm90  (Hopper)          → cula.kda.kda_prefill_hopper

    Args:
        device: CUDA device to query.  Defaults to the currently active device.

    Raises:
        RuntimeError: If the device architecture is not supported.
    """
    major, minor = get_device_sm_version(device)
    if major == 10 and minor in (0, 3):
        # TODO
        raise NotImplementedError(
            "The Blackwell implementation of fused prefill is not yet available. "
            "Please use a sm90a (Hopper) device or wait for future updates."
        )
    elif major == 9 and minor == 0:
        from cula.kda import kda_prefill_hopper

        return kda_prefill_hopper
    else:
        raise RuntimeError(
            f"Unsupported CUDA compute capability sm_{major}{minor}. "
            f"Only sm90a (Hopper) and Blackwell (SM100/SM103) are supported."
        )


@cute.jit
def print_tensor_2d(tensor: cute.Tensor):
    """
    Print a 2D tensor with 6 decimal places.

    Args:
        tensor: A 2D cute.Tensor to print
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])

    cute.printf("---------- tensor [%d x %d] ----------\n", rows, cols)

    for i in cutlass.range_constexpr(rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        cute.printf("]\n")

    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor(tensor: cute.Tensor):
    """
    Print a 2D tensor with 6 decimal places.
    For higher dimension tensors, flattens and prints as 2D.

    Args:
        tensor: A cute.Tensor to print (assumes 2D indexing)
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])

    cute.printf("---------- tensor [%d x %d] ----------\n", rows, cols)

    for i in cutlass.range_constexpr(rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        cute.printf("]\n")

    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor_flat(tensor: cute.Tensor):
    """
    Print all elements of a tensor in flat order with 6 decimal places.

    Args:
        tensor: A cute.Tensor to print
    """
    total_size = cute.size(tensor)

    cute.printf("---------- tensor [flat, size=%d] ----------\n", total_size)

    for i in cutlass.range_constexpr(total_size):
        cute.printf("[%d] = %10.6f\n", i, tensor.flat_ref(i).to(cutlass.Float32))

    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor_partial(tensor: cute.Tensor, max_rows: int, max_cols: int):
    """
    Print a partial view of a 2D tensor (first max_rows x max_cols elements).
    Useful for large tensors where printing everything is impractical.

    Args:
        tensor: A 2D cute.Tensor to print
        max_rows: Maximum number of rows to print
        max_cols: Maximum number of columns to print
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])

    print_rows = rows if rows < max_rows else max_rows
    print_cols = cols if cols < max_cols else max_cols

    cute.printf("---------- tensor [%d x %d] (showing %d x %d) ----------\n", rows, cols, print_rows, print_cols)

    for i in cutlass.range_constexpr(print_rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(print_cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        if print_cols < cols:
            cute.printf(", ...")
        cute.printf("]\n")

    if print_rows < rows:
        cute.printf("... (%d more rows)\n", rows - print_rows)

    cute.printf("----------------------------------\n")


@functools.cache
def prepare_uniform_cu_seqlens(batch_size: int, seqlen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, device=device, dtype=dtype)
    return cu_seqlens


@functools.cache
def get_device_sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_cache_buf = {}


def _get_cache_buf(name: str, nbytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None or buf.size(0) < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Tensor cache
# Adapted from: https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Original copyright: 2024 The FLA Authors (same Apache 2.0 license)
# ---------------------------------------------------------------------------
_CULA_DISABLE_TENSOR_CACHE: bool = os.getenv("CULA_DISABLE_TENSOR_CACHE", "0") == "1"


def tensor_cache(fn):
    """Single-entry cache for functions with tensor inputs (identity-based)."""
    last_args = None
    last_kwargs = None
    last_result = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if _CULA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Sequence-length helpers
# Adapted from: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# Original copyright: 2024 The FLA Authors (same Apache 2.0 license)
# ---------------------------------------------------------------------------
@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    import triton  # already available as a transitive dep of cutlass-dsl

    if cu_seqlens_cpu is not None:
        indices = torch.cat(
            [torch.arange(n, device=cu_seqlens.device) for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()]
        )
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
