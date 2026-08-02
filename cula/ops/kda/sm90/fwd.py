# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 MoonshotAI
# Licensed under the MIT License.
# Based on MoonshotAI/FlashKDA (https://github.com/MoonshotAI/FlashKDA)

"""
FlashKDA Prefill

two-kernel (K1 Prepare + K2 Recurrence), CHUNK=16, D=128.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch

from cula.ops.kda.sm90._common import _stream_key

CHUNK: int = 16
D: int = 128  # only 128 supported

_VARLEN_LAYOUT_CACHE_MAXSIZE = 64


def _compute_total_tiles(seq_lens: list[int] | tuple[int, ...]) -> int:
    return sum((sl + CHUNK - 1) // CHUNK for sl in seq_lens)


@dataclass
class _VarlenMetadata:
    cu_values: tuple[int, ...]
    seq_lens: tuple[int, ...]
    total_tiles: int
    needs_padding: bool
    total_aligned: int
    cu_tiles: torch.Tensor | None
    tile_starts: torch.Tensor
    tile_actual_lens: torch.Tensor


@dataclass
class _PrefillProblem:
    B: int
    T: int
    H: int
    total_tiles: int
    is_varlen: bool
    varlen_meta: _VarlenMetadata | None = None


def _seq_tiles_from_problem(problem: _PrefillProblem) -> list[int]:
    if problem.varlen_meta is not None:
        return [(sl + CHUNK - 1) // CHUNK for sl in problem.varlen_meta.seq_lens]
    return [(problem.T + CHUNK - 1) // CHUNK] * problem.B


def _validate_inputs(
    q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens, cu_seqlens_cpu=None
) -> _PrefillProblem:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, D], got {tuple(q.shape)}")
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise TypeError(f"q must be a CUDA bfloat16 tensor, got dtype={q.dtype}, device={q.device}")
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not tensor.is_cuda or tensor.device != q.device or tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be a CUDA bfloat16 tensor, got dtype={tensor.dtype}, device={tensor.device}")
    if any(not tensor.is_contiguous() for tensor in (q, k, v, g, beta)):
        raise ValueError("q, k, v, g and beta must be contiguous")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(f"q/k/g shapes must match, got q={tuple(q.shape)}, k={tuple(k.shape)}, g={tuple(g.shape)}")
    if v.shape != q.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match q shape {tuple(q.shape)}")

    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        raise ValueError(f"B, T and H must be positive, got B={B}, T={T}, H={H}")
    if K != D:
        raise ValueError(f"only K=V={D} supported, got K={K} V={v.shape[-1]}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta shape mismatch: {tuple(beta.shape)} vs ({B},{T},{H})")
    if (
        A_log is None
        or A_log.device != q.device
        or not A_log.is_contiguous()
        or A_log.shape != (H,)
        or A_log.dtype != torch.float32
    ):
        raise ValueError(
            f"A_log must be float32 with shape ({H},), got {None if A_log is None else (A_log.dtype, tuple(A_log.shape))}"
        )
    if (
        dt_bias is None
        or dt_bias.device != q.device
        or not dt_bias.is_contiguous()
        or dt_bias.shape != (H, K)
        or dt_bias.dtype != torch.float32
    ):
        raise ValueError(
            f"dt_bias must be float32 with shape ({H}, {K}), "
            f"got {None if dt_bias is None else (dt_bias.dtype, tuple(dt_bias.shape))}"
        )

    is_varlen = cu_seqlens is not None
    if is_varlen:
        if B != 1:
            raise ValueError(f"varlen requires B=1, got B={B}")
        if cu_seqlens.device != q.device or cu_seqlens.ndim != 1 or not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be a contiguous 1D tensor on the q device")
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least two entries")
        if cu_seqlens_cpu is not None and (
            cu_seqlens_cpu.device.type != "cpu"
            or cu_seqlens_cpu.dtype != torch.int32
            or cu_seqlens_cpu.ndim != 1
            or cu_seqlens_cpu.numel() != cu_seqlens.numel()
        ):
            raise ValueError(
                "cu_seqlens_cpu must be a 1D CPU tensor with the same numel as "
                f"cu_seqlens ({cu_seqlens.numel()}), got device={cu_seqlens_cpu.device}, "
                f"shape={tuple(cu_seqlens_cpu.shape)}"
            )
        varlen_meta = _get_or_build_varlen_metadata(cu_seqlens, cu_seqlens_cpu)
        N = len(varlen_meta.seq_lens)
        if varlen_meta.cu_values[0] != 0:
            raise ValueError("cu_seqlens must start at 0")
        if varlen_meta.cu_values[-1] != T:
            raise ValueError(f"cu_seqlens[-1] must equal packed T={T}, got {varlen_meta.cu_values[-1]}")
        seq_lens = varlen_meta.seq_lens
        if any(sl <= 0 for sl in seq_lens):
            raise ValueError(f"all variable-length sequences must be non-empty, got seq_lens={seq_lens}")
        total_tiles = varlen_meta.total_tiles
    else:
        N = B
        total_tiles = B * ((T + CHUNK - 1) // CHUNK)
        varlen_meta = None

    if initial_state is not None:
        if initial_state.shape != (N, H, D, D):
            raise ValueError(f"initial_state shape must be ({N}, {H}, {D}, {D}), got {tuple(initial_state.shape)}")
        if initial_state.device != q.device or initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
            raise TypeError("initial_state must be a contiguous float32 tensor on the q device")
    if final_state is not None:
        if final_state.shape != (N, H, D, D):
            raise ValueError(f"final_state shape must be ({N}, {H}, {D}, {D}), got {tuple(final_state.shape)}")
        if final_state.device != q.device or final_state.dtype != torch.float32 or not final_state.is_contiguous():
            raise TypeError("final_state must be a contiguous float32 tensor on the q device")

    return _PrefillProblem(
        B=B,
        T=T,
        H=H,
        total_tiles=total_tiles,
        is_varlen=is_varlen,
        varlen_meta=varlen_meta,
    )


def _validate_launch_options(q, out, lower_bound, use_gate_in_kernel) -> None:
    if out.shape != q.shape or out.device != q.device or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(
            f"out must be contiguous bfloat16 with shape {tuple(q.shape)} on {q.device}, "
            f"got dtype={out.dtype}, device={out.device}, shape={tuple(out.shape)}"
        )
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "CuTeDSL FlashKDA prefill only supports use_gate_in_kernel=True. "
            "Pre-gated inputs would require the torch reference, which is test-only."
        )
    if lower_bound is None:
        raise ValueError("lower_bound must be specified.")
    if not (-5 <= lower_bound < 0):
        raise ValueError(f"lower_bound must be in the safe range [-5, 0), got {lower_bound}.")


_VARLEN_LAYOUT_CACHE: dict = {}
_VARLEN_METADATA_CACHE: dict[tuple, tuple[weakref.ReferenceType[torch.Tensor], tuple, _VarlenMetadata]] = {}
_K1_SYMBOLS = None
_K2_LAUNCHER = None


_WS_ARENA_ALIGN = 256
_WS_ARENA: dict = {}  # (device, stream_ptr) -> [arena uint8 tensor, {sizes_key: views}]
_WS_ARENA_MAXSIZE = 8
_WS_VIEWS_MAXSIZE = 32


def _get_or_alloc_workspaces(n_qk: int, n_cc: int, n_gt: int, n_beta: int, device, dtype):
    """Carve K1/K2 scratch (ws_qd/kd/kr/gt/inv/mqk, ws_beta) out of a grow-only
    per-(device, stream) arena instead of allocating per call.

    Reusing the arena is safe because every producer/consumer runs on the
    keyed stream: the next call's K1 cannot overwrite a workspace before this
    call's K2 finished reading it.
    """
    arena_key = _stream_key(device)
    sizes_key = (n_qk, n_cc, n_gt, n_beta, dtype)
    entry = _WS_ARENA.get(arena_key)
    if entry is not None:
        views = entry[1].get(sizes_key)
        if views is not None:
            return views

    nbytes_list = (
        n_qk * 2,  # ws_qd bf16
        n_qk * 2,  # ws_kd
        n_qk * 2,  # ws_kr
        n_gt * 4,  # ws_gt fp32
        n_cc * 2,  # ws_inv bf16
        n_cc * 2,  # ws_mqk
        n_beta * dtype.itemsize,  # ws_beta
    )
    offsets = []
    total = 0
    for nbytes in nbytes_list:
        offsets.append(total)
        total += -(-nbytes // _WS_ARENA_ALIGN) * _WS_ARENA_ALIGN

    if entry is None or entry[0].numel() < total:
        if entry is None and len(_WS_ARENA) >= _WS_ARENA_MAXSIZE:
            _WS_ARENA.pop(next(iter(_WS_ARENA)))
        # Growing replaces the arena; stale views die with the old entry.
        entry = [torch.empty(total, dtype=torch.uint8, device=device), {}]
        _WS_ARENA[arena_key] = entry
    arena = entry[0]

    def carve(idx: int, numel: int, view_dtype: torch.dtype):
        return arena.narrow(0, offsets[idx], numel * view_dtype.itemsize).view(view_dtype)

    views = (
        carve(0, n_qk, torch.bfloat16),
        carve(1, n_qk, torch.bfloat16),
        carve(2, n_qk, torch.bfloat16),
        carve(3, n_gt, torch.float32),
        carve(4, n_cc, torch.bfloat16),
        carve(5, n_cc, torch.bfloat16),
        carve(6, n_beta, dtype),
    )
    if len(entry[1]) >= _WS_VIEWS_MAXSIZE:
        entry[1].pop(next(iter(entry[1])))
    entry[1][sizes_key] = views
    return views


def _get_or_build_varlen_layout(seq_lens: tuple[int, ...], device, cu_dtype):
    """CHUNK-aligned cumulative token offsets and tile counts for non-aligned varlen."""
    key = (seq_lens, _stream_key(device), cu_dtype)
    cached = _VARLEN_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached

    out_offsets = [0]
    for sl in seq_lens:
        aligned = ((sl + CHUNK - 1) // CHUNK) * CHUNK
        out_offsets.append(out_offsets[-1] + aligned)

    cu_pad = torch.tensor(out_offsets, dtype=cu_dtype, device=device)
    cu_tiles = torch.tensor([off // CHUNK for off in out_offsets], dtype=torch.int32, device=device)
    cached = (cu_pad, cu_tiles)
    if len(_VARLEN_LAYOUT_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        _VARLEN_LAYOUT_CACHE.pop(next(iter(_VARLEN_LAYOUT_CACHE)))
    _VARLEN_LAYOUT_CACHE[key] = cached
    return cached


def _get_or_build_varlen_metadata(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None) -> _VarlenMetadata:
    """Cache varlen metadata (seq_lens, tile offsets, padding flags) for cu_seqlens."""
    cache_key = (id(cu_seqlens), _stream_key(cu_seqlens.device))
    version = 0 if torch.is_inference(cu_seqlens) else int(cu_seqlens._version)
    attrs = (
        cu_seqlens.data_ptr(),
        tuple(cu_seqlens.shape),
        str(cu_seqlens.device),
        cu_seqlens.dtype,
        version,
    )
    cached = _VARLEN_METADATA_CACHE.get(cache_key)
    if cached is not None:
        tensor_ref, cached_attrs, meta = cached
        if tensor_ref() is cu_seqlens and cached_attrs == attrs:
            return meta
        _VARLEN_METADATA_CACHE.pop(cache_key, None)

    src_cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.detach().to("cpu")
    cu_values = tuple(int(v) for v in src_cpu.tolist())
    seq_lens = tuple(cu_values[i + 1] - cu_values[i] for i in range(len(cu_values) - 1))
    total_tiles = _compute_total_tiles(seq_lens)
    needs_padding = any((sl % CHUNK) != 0 for sl in seq_lens)
    aligned_lens = tuple(((sl + CHUNK - 1) // CHUNK) * CHUNK for sl in seq_lens)
    total_aligned = sum(aligned_lens)
    tile_starts_list: list[int] = []
    tile_actual_lens_list: list[int] = []
    for bos, sl in zip(cu_values[:-1], seq_lens):
        for offset in range(0, sl, CHUNK):
            tile_starts_list.append(bos + offset)
            tile_actual_lens_list.append(min(CHUNK, sl - offset))
    tile_starts = torch.tensor(tile_starts_list, dtype=torch.int32, device=cu_seqlens.device)
    tile_actual_lens = torch.tensor(tile_actual_lens_list, dtype=torch.int32, device=cu_seqlens.device)
    cu_tiles = None
    if not needs_padding:
        cu_tiles = torch.tensor(
            [v // CHUNK for v in cu_values],
            dtype=torch.int32,
            device=cu_seqlens.device,
        )

    meta = _VarlenMetadata(
        cu_values=cu_values,
        seq_lens=seq_lens,
        total_tiles=total_tiles,
        needs_padding=needs_padding,
        total_aligned=total_aligned,
        cu_tiles=cu_tiles,
        tile_starts=tile_starts,
        tile_actual_lens=tile_actual_lens,
    )
    if len(_VARLEN_METADATA_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        for k, (ref, _a, _m) in list(_VARLEN_METADATA_CACHE.items()):
            if ref() is None:
                _VARLEN_METADATA_CACHE.pop(k, None)
    if len(_VARLEN_METADATA_CACHE) >= _VARLEN_LAYOUT_CACHE_MAXSIZE:
        _VARLEN_METADATA_CACHE.pop(next(iter(_VARLEN_METADATA_CACHE)))
    _VARLEN_METADATA_CACHE[cache_key] = (weakref.ref(cu_seqlens), attrs, meta)
    return meta


def _get_k1_symbols():
    global _K1_SYMBOLS
    if _K1_SYMBOLS is None:
        from cula.ops.kda.sm90.k1 import CHUNK as k1_chunk
        from cula.ops.kda.sm90.k1 import D as k1_d
        from cula.ops.kda.sm90.k1 import launch_k1 as k1_launch

        _K1_SYMBOLS = (k1_chunk, k1_d, k1_launch)
    return _K1_SYMBOLS


def _get_k2_launcher():
    global _K2_LAUNCHER
    if _K2_LAUNCHER is not None:
        return _K2_LAUNCHER
    from cula.ops.kda.sm90.k2 import launch_k2

    _K2_LAUNCHER = launch_k2
    return launch_k2


def flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    state_transposed: bool = False,
    use_gate_in_kernel: bool = True,
    _problem: _PrefillProblem | None = None,
) -> None:
    """FlashKDA fwd. ``out`` and ``final_state`` are written in-place.

    Args:
        q, k, v, g: [B, T, H, D] bf16.
        beta: [B, T, H] bf16 (pre-sigmoid).
        scale: attention scale.
        out: [B, T, H, D] bf16 output (written in-place).
        A_log: [H] fp32.
        dt_bias: [H, D] fp32.
        lower_bound: gate floor (negative).
        initial_state: [N, H, D, D] bf16/fp32 or None.
        final_state: [N, H, D, D] bf16/fp32 or None (written in-place).
        cu_seqlens: [N+1] int32/int64 for variable-length, or None.
        cu_seqlens_cpu: optional CPU copy of cu_seqlens (same values) to skip the
            GPU->host sync when first building varlen metadata.
        state_transposed: False -> [N,H,V,K] (default), True -> [N,H,K,V].
    """
    if _problem is None:
        problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens, cu_seqlens_cpu)
        _validate_launch_options(q, out, lower_bound, use_gate_in_kernel)
    else:
        problem = _problem

    _dispatch_cute(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        final_state,
        cu_seqlens,
        problem,
        state_transposed=state_transposed,
    )


def _dispatch_cute(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    final_state,
    cu_seqlens,
    problem: _PrefillProblem,
    *,
    state_transposed: bool = False,
):
    """Launch K1 + K2."""
    K1_CHUNK, K1_D, launch_k1 = _get_k1_symbols()

    T_orig = problem.T
    need_t_pad = (not problem.is_varlen) and (T_orig % K1_CHUNK != 0)
    if need_t_pad:
        T_pad = ((T_orig + K1_CHUNK - 1) // K1_CHUNK) * K1_CHUNK
        B, H = problem.B, problem.H
        pad_len = T_pad - T_orig
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad_len), value=-1e6)
        beta = torch.nn.functional.pad(beta, (0, 0, 0, pad_len), value=-80.0)
        out_orig = out
        out = torch.empty_like(q)
        problem = _PrefillProblem(
            B=B,
            T=T_pad,
            H=H,
            total_tiles=B * (T_pad // K1_CHUNK),
            is_varlen=False,
        )

    k1_q, k1_k, k1_g, k1_beta = q, k, g, beta
    k1_total_tiles = problem.total_tiles
    k1_tile_starts = None
    k1_tile_actual_lens = None
    k1_is_varlen = False

    # Varlen: K1/K2 read original q/k/g/v; beta remains padded for the
    # existing compact workspace layout.
    k2_cu_seqlens_tiles_cached = None
    k2_v_tile_starts = None
    k2_v_tile_actual_lens = None
    if problem.is_varlen:
        varlen_meta = problem.varlen_meta
        seq_lens_list = varlen_meta.seq_lens
        if varlen_meta.needs_padding:
            total_aligned = varlen_meta.total_aligned

            k1_total_tiles = varlen_meta.total_tiles
            k1_tile_starts = varlen_meta.tile_starts
            k1_tile_actual_lens = varlen_meta.tile_actual_lens
            k1_is_varlen = True
            k2_v_tile_starts = varlen_meta.tile_starts
            k2_v_tile_actual_lens = varlen_meta.tile_actual_lens

            # Padded tile boundaries for K2's per-sequence recurrence. K1 emits ws_beta
            # directly, so varlen needs no host-side beta padding/gather.
            cu_pad, k2_cu_seqlens_tiles_cached = _get_or_build_varlen_layout(
                tuple(seq_lens_list),
                q.device,
                cu_seqlens.dtype,
            )

            problem_pad = _PrefillProblem(
                B=1,
                T=total_aligned,
                H=problem.H,
                total_tiles=total_aligned // K1_CHUNK,
                is_varlen=True,
            )
            cu_seqlens, problem = cu_pad, problem_pad
        else:
            k2_cu_seqlens_tiles_cached = varlen_meta.cu_tiles

    _launch_k2 = _get_k2_launcher()

    B, T, H = problem.B, problem.T, problem.H

    if problem.is_varlen:
        T_total = T
        k2_cu_seqlens_tiles = k2_cu_seqlens_tiles_cached
    else:
        T_total = B * T
        k2_cu_seqlens_tiles = None

    total_tiles = T_total // K1_CHUNK

    n_qk = total_tiles * H * K1_CHUNK * K1_D
    n_cc = total_tiles * H * K1_CHUNK * K1_CHUNK
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta = _get_or_alloc_workspaces(
        n_qk, n_cc, total_tiles * H * K1_D, T_total * H, q.device, beta.dtype
    )

    # K1 reads beta from its original packed [T, H] layout and emits raw beta
    # into ws_beta (tail rows = -80); K2 reads ws_beta directly, so no
    # host-side transpose/padding/gather of beta is needed.
    launch_k1(
        k1_q,
        k1_k,
        k1_g,
        A_log,
        dt_bias,
        k1_beta.reshape(-1),
        scale,
        lower_bound,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        ws_beta,
        tile_starts=k1_tile_starts,
        tile_actual_lens=k1_tile_actual_lens,
        total_tiles=k1_total_tiles,
        is_varlen=k1_is_varlen,
    )
    _launch_k2(
        v,
        ws_beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out,
        k2_cu_seqlens_tiles,
        initial_state=initial_state,
        final_state=final_state,
        state_transposed=state_transposed,
        v_tile_starts=k2_v_tile_starts,
        v_tile_actual_lens=k2_v_tile_actual_lens,
    )

    if need_t_pad:
        out_orig.copy_(out[:, :T_orig])
