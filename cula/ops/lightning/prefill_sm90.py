# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Production Torch wrapper for the SM90a Lightning Attention prefill kernel."""

from __future__ import annotations

import functools
import math
from importlib.metadata import version

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cula.ops.lightning.sm90.prefill_kernel import (
    EXPECTED_CUTLASS_DSL_VERSION,
    HEAD_DIM,
    VALUE_DIM,
    LightningSm90PrefillKernel,
)
from cula.ops.lightning.sm90.schedule import TARGET_ARCH
from cula.utils import _get_cache_buf, get_device_sm_count

CHUNK_SIZE = 64
FIXED_BACKEND_IDENTITY = "cula.lightning.sm90a.cutedsl.prefill.fixed"
VARLEN_NONPERSISTENT_BACKEND_IDENTITY = "cula.lightning.sm90a.cutedsl.prefill.varlen.nonpersistent"
VARLEN_PERSISTENT_BACKEND_IDENTITY = "cula.lightning.sm90a.cutedsl.prefill.varlen.persistent_static"

_compiled_fixed_variants: dict[tuple[int, int, int, int, int, bool, bool], object] = {}
_compiled_varlen_variants: dict[tuple[int, int, int, int, int, int, bool, int | None], object] = {}


def _require_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"SM90 Lightning prefill requires chunk_size={CHUNK_SIZE}")


@functools.cache
def _require_sm90_environment(device: torch.device) -> None:
    """Validate process and architecture metadata once per CUDA device."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SM90 Lightning prefill requires CUDA tensors")
    installed_version = version("nvidia-cutlass-dsl")
    if installed_version != EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            "SM90 Lightning prefill is pinned to "
            f"nvidia-cutlass-dsl=={EXPECTED_CUTLASS_DSL_VERSION}, found {installed_version}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 Lightning prefill requires SM90, found {capability}")


def _validate_fixed_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_s: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 BTHD tensors")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes")
    B, T, H, D = q.shape
    if v.shape[:2] != (B, T) or v.shape[-1] != D:
        raise ValueError("v must share q/k batch, sequence, and head dimension")
    HV = v.shape[2]
    if D != HEAD_DIM:
        raise ValueError(f"SM90 prefill requires head dimension {HEAD_DIM}, found {D}")
    if HV < H or HV % H:
        raise ValueError("SM90 prefill requires HV >= H and HV % H == 0")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must use torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if decay_s.dtype != torch.float32 or decay_s.shape not in {(H,), (HV,)}:
        raise ValueError(f"decay_s must have FP32 shape ({H},) or ({HV},)")
    if not decay_s.is_cuda or not decay_s.is_contiguous():
        raise ValueError("decay_s must be a contiguous CUDA tensor")
    if any(tensor.device != q.device for tensor in (k, v, decay_s)):
        raise ValueError("all inputs must be on the same CUDA device")
    if initial_state is not None:
        if initial_state.shape != (B, HV, VALUE_DIM, HEAD_DIM):
            raise ValueError(f"initial_state must have public BHVK shape ({B},{HV},{VALUE_DIM},{HEAD_DIM})")
        if initial_state.dtype != torch.float32:
            raise ValueError("initial_state must use torch.float32")
        if not initial_state.is_cuda or not initial_state.is_contiguous() or initial_state.device != q.device:
            raise ValueError("initial_state must be contiguous on the input CUDA device")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a finite Python number")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    return B, T, H, HV, decay_s.numel()


def _validate_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_s: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_pool: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None,
    scale: float,
) -> tuple[int, int, int, int, int, int]:
    """Validate host-visible packed metadata without reading device values."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("packed q, k, and v must be rank-4 tensors")
    if q.shape != k.shape:
        raise ValueError("packed q and k must have identical shapes")
    B, T, H, D = q.shape
    if B != 1 or T < 1 or D != HEAD_DIM:
        raise ValueError(f"packed q/k require shape [1,T,H,{HEAD_DIM}] with T>0")
    if v.shape[:2] != (1, T) or v.shape[-1] != HEAD_DIM:
        raise ValueError(f"packed v must have shape [1,T,HV,{HEAD_DIM}]")
    HV = v.shape[2]
    if HV < H or HV % H:
        raise ValueError("packed SM90 prefill requires HV >= H and HV % H == 0")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must use torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if decay_s.dtype != torch.float32 or decay_s.shape not in {(H,), (HV,)}:
        raise ValueError(f"decay_s must have FP32 shape ({H},) or ({HV},)")
    if not decay_s.is_cuda or not decay_s.is_contiguous():
        raise ValueError("decay_s must be a contiguous CUDA tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        raise ValueError("cu_seqlens must have shape [N+1] with N>=1")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous CUDA int32 tensor")
    N = cu_seqlens.shape[0] - 1
    if initial_state_indices is not None:
        if initial_state_indices.shape != (N,) or initial_state_indices.dtype != torch.int32:
            raise ValueError(f"initial_state_indices must have int32 shape ({N},)")
        if not initial_state_indices.is_cuda or not initial_state_indices.is_contiguous():
            raise ValueError("initial_state_indices must be a contiguous CUDA tensor")
    if state_pool is not None:
        if state_pool.ndim != 4 or state_pool.shape[1:] != (HV, VALUE_DIM, HEAD_DIM):
            raise ValueError(f"state_pool must have shape [pool,{HV},{VALUE_DIM},{HEAD_DIM}]")
        if state_pool.shape[0] < 1 or state_pool.dtype != torch.float32:
            raise ValueError("state_pool must contain at least one FP32 slot")
        if not state_pool.is_cuda or not state_pool.is_contiguous():
            raise ValueError("state_pool must be a contiguous CUDA tensor")
        if initial_state_indices is None and state_pool.shape[0] < N:
            raise ValueError("state_pool must contain at least N slots when initial_state_indices is omitted")
    tensors = [k, v, decay_s, cu_seqlens]
    if state_pool is not None:
        tensors.append(state_pool)
    if initial_state_indices is not None:
        tensors.append(initial_state_indices)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all packed inputs must be on the same CUDA device")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a finite Python number")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    pool_size = N if state_pool is None else state_pool.shape[0]
    return T, H, HV, N, pool_size, decay_s.numel()


def _fake_compact(dtype, shape: tuple[int, ...]):
    kwargs = {"assumed_align": 16}
    if len(shape) > 1:
        kwargs["stride_order"] = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, shape, **kwargs)


def _compile_fixed_variant(
    batch_size: int,
    sequence_length: int,
    qk_heads: int,
    value_heads: int,
    decay_heads: int,
    has_initial_state: bool,
    output_final_state: bool,
):
    schedule = LightningSm90PrefillKernel(
        batch_size=batch_size,
        sequence_length=sequence_length,
        qk_heads=qk_heads,
        value_heads=value_heads,
        decay_heads=decay_heads,
        needs_initial_state=has_initial_state,
        needs_final_state=output_final_state,
    )
    q_fake = _fake_compact(
        cutlass.BFloat16,
        (batch_size, sequence_length, qk_heads, HEAD_DIM),
    )
    k_fake = _fake_compact(
        cutlass.BFloat16,
        (batch_size, sequence_length, qk_heads, HEAD_DIM),
    )
    v_fake = _fake_compact(
        cutlass.BFloat16,
        (batch_size, sequence_length, value_heads, VALUE_DIM),
    )
    output_fake = _fake_compact(
        cutlass.BFloat16,
        (batch_size, sequence_length, value_heads, VALUE_DIM),
    )
    decay_fake = _fake_compact(cutlass.Float32, (decay_heads,))
    state_fake = _fake_compact(
        cutlass.Float32,
        (batch_size, value_heads, VALUE_DIM, HEAD_DIM),
    )
    initial_fake = state_fake if has_initial_state else decay_fake
    final_fake = state_fake if output_final_state else decay_fake
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[(cute.GPUArch(TARGET_ARCH), cute.EnableTVMFFI(True))](
        schedule,
        q_fake,
        k_fake,
        v_fake,
        decay_fake,
        initial_fake,
        final_fake,
        decay_fake,
        decay_fake,
        decay_fake,
        cutlass.Float32(1.0),
        cutlass.Int32(sequence_length),
        output_fake,
        stream_fake,
    )


def _get_compiled_fixed_variant(
    batch_size: int,
    sequence_length: int,
    qk_heads: int,
    value_heads: int,
    decay_heads: int,
    has_initial_state: bool,
    output_final_state: bool,
):
    key = (
        batch_size,
        sequence_length,
        qk_heads,
        value_heads,
        decay_heads,
        has_initial_state,
        output_final_state,
    )
    compiled = _compiled_fixed_variants.get(key)
    if compiled is None:
        compiled = _compile_fixed_variant(*key)
        _compiled_fixed_variants[key] = compiled
    return compiled


def _compile_varlen_variant(
    total_length: int,
    qk_heads: int,
    value_heads: int,
    decay_heads: int,
    num_sequences: int,
    state_pool_size: int,
    persistent: bool,
    persistent_ctas: int | None,
):
    schedule = LightningSm90PrefillKernel(
        batch_size=1,
        sequence_length=total_length,
        qk_heads=qk_heads,
        value_heads=value_heads,
        decay_heads=decay_heads,
        needs_initial_state=True,
        needs_final_state=True,
        is_varlen=True,
        num_sequences=num_sequences,
        state_pool_size=state_pool_size,
        persistent=persistent,
        persistent_ctas=persistent_ctas,
    )
    q_fake = _fake_compact(
        cutlass.BFloat16,
        (1, total_length, qk_heads, HEAD_DIM),
    )
    k_fake = _fake_compact(
        cutlass.BFloat16,
        (1, total_length, qk_heads, HEAD_DIM),
    )
    v_fake = _fake_compact(
        cutlass.BFloat16,
        (1, total_length, value_heads, VALUE_DIM),
    )
    output_fake = _fake_compact(
        cutlass.BFloat16,
        (1, total_length, value_heads, VALUE_DIM),
    )
    decay_fake = _fake_compact(cutlass.Float32, (decay_heads,))
    state_fake = _fake_compact(
        cutlass.Float32,
        (state_pool_size, value_heads, VALUE_DIM, HEAD_DIM),
    )
    cu_seqlens_fake = _fake_compact(cutlass.Int32, (num_sequences + 1,))
    indices_fake = _fake_compact(cutlass.Int32, (num_sequences,))
    tensormaps_fake = make_fake_compact_tensor(
        cutlass.Uint8,
        (cute.sym_int(),),
        assumed_align=128,
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[(cute.GPUArch(TARGET_ARCH), cute.EnableTVMFFI(True))](
        schedule,
        q_fake,
        k_fake,
        v_fake,
        decay_fake,
        state_fake,
        state_fake,
        cu_seqlens_fake,
        indices_fake,
        tensormaps_fake,
        cutlass.Float32(1.0),
        cutlass.Int32(total_length),
        output_fake,
        stream_fake,
    )


def _get_compiled_varlen_variant(
    total_length: int,
    qk_heads: int,
    value_heads: int,
    decay_heads: int,
    num_sequences: int,
    state_pool_size: int,
    persistent: bool,
    persistent_ctas: int | None,
):
    key = (
        total_length,
        qk_heads,
        value_heads,
        decay_heads,
        num_sequences,
        state_pool_size,
        persistent,
        persistent_ctas,
    )
    compiled = _compiled_varlen_variants.get(key)
    if compiled is None:
        compiled = _compile_varlen_variant(*key)
        _compiled_varlen_variants[key] = compiled
    return compiled


def lightning_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fixed-length Lightning Attention on SM90a."""

    _require_chunk_size(chunk_size)
    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be boolean")
    B, T, H, HV, decay_heads = _validate_fixed_inputs(
        Q,
        K,
        V,
        decay,
        initial_state,
        scale,
    )
    if T < 1:
        raise ValueError("SM90 Lightning prefill requires a positive sequence length")
    _require_sm90_environment(Q.device)
    output = torch.empty_like(V)
    final_state = (
        torch.empty((B, HV, VALUE_DIM, HEAD_DIM), dtype=torch.float32, device=Q.device) if output_final_state else None
    )
    compiled = _get_compiled_fixed_variant(
        B,
        T,
        H,
        HV,
        decay_heads,
        initial_state is not None,
        output_final_state,
    )
    initial_arg = initial_state if initial_state is not None else decay
    final_arg = final_state if final_state is not None else decay
    compiled(
        Q,
        K,
        V,
        decay,
        initial_arg,
        final_arg,
        decay,
        decay,
        decay,
        cutlass.Float32(float(scale)),
        cutlass.Int32(T),
        output,
    )
    return output, final_state


def lightning_attn_fwd_varlen(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    state_pool: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
    persistent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed-varlen Lightning Attention on SM90a."""

    _require_chunk_size(chunk_size)
    if not isinstance(persistent, bool):
        raise TypeError("persistent must be boolean")
    T, H, HV, N, pool_size, decay_heads = _validate_varlen_inputs(
        Q,
        K,
        V,
        decay,
        cu_seqlens,
        state_pool,
        initial_state_indices,
        scale,
    )
    _require_sm90_environment(Q.device)
    if state_pool is None:
        state_pool = torch.zeros(
            (N, HV, VALUE_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=Q.device,
        )
    if initial_state_indices is None:
        initial_state_indices = torch.arange(N, dtype=torch.int32, device=Q.device)
    output = torch.empty_like(V)
    sm_count = get_device_sm_count(Q.device)
    persistent_ctas = min(N * HV, sm_count) if persistent else None
    tensormaps = _get_cache_buf(
        "lightning_sm90_prefill_tensormaps",
        sm_count * 128,
        Q.device,
    )
    compiled = _get_compiled_varlen_variant(
        T,
        H,
        HV,
        decay_heads,
        N,
        pool_size,
        persistent,
        persistent_ctas,
    )
    compiled(
        Q,
        K,
        V,
        decay,
        state_pool,
        state_pool,
        cu_seqlens,
        initial_state_indices,
        tensormaps,
        cutlass.Float32(float(scale)),
        cutlass.Int32(T),
        output,
    )
    return output, state_pool


def get_sm90_lightning_attn_prefill_backend_identity(
    *,
    varlen: bool = False,
    persistent: bool = True,
) -> str:
    """Return the exact SM90 implementation identity without launching it."""

    if not isinstance(varlen, bool) or not isinstance(persistent, bool):
        raise TypeError("varlen and persistent must be boolean")
    if not varlen:
        return FIXED_BACKEND_IDENTITY
    return VARLEN_PERSISTENT_BACKEND_IDENTITY if persistent else VARLEN_NONPERSISTENT_BACKEND_IDENTITY


__all__ = [
    "FIXED_BACKEND_IDENTITY",
    "VARLEN_NONPERSISTENT_BACKEND_IDENTITY",
    "VARLEN_PERSISTENT_BACKEND_IDENTITY",
    "get_sm90_lightning_attn_prefill_backend_identity",
    "lightning_attn_fwd",
    "lightning_attn_fwd_varlen",
]
