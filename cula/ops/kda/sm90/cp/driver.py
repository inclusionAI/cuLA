# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP prefill executor: K1 once → pre_scan → merge → segment-K2.

Only *runs* a given CPPlan; planning lives in cp.plan. The wrapper calls
_run_cp with a plan from plan_prefill; intracard_prefill is the plan-and-run
entry for direct callers.
"""

from __future__ import annotations

import torch

from cula.ops.kda.cp_mode import NotSplittableError
from cula.ops.kda.sm90._common import _stream_key
from cula.ops.kda.sm90.cp.merge import launch_merge
from cula.ops.kda.sm90.cp.plan import CPPlan, plan_auto, plan_manual, split_balanced
from cula.ops.kda.sm90.cp.pre_scan import launch_pre_scan
from cula.ops.kda.sm90.fwd import (
    _get_or_alloc_workspaces,
    _get_or_build_varlen_metadata,
    _PrefillProblem,
    _seq_tiles_from_problem,
    _validate_inputs,
    _validate_launch_options,
    flash_kda_fwd,
)
from cula.ops.kda.sm90.k1 import launch_k1
from cula.ops.kda.sm90.k2 import CHUNK, D, launch_k2
from cula.utils import get_device_sm_count

_SCRATCH_CACHE: dict = {}
_PLAN_TENSOR_CACHE: dict = {}
_SCRATCH_CACHE_MAXSIZE = 8
_PLAN_TENSOR_CACHE_MAXSIZE = 64


def _get_plan_tensor(values: tuple, dtype, device: torch.device) -> torch.Tensor:
    key = (values, dtype, _stream_key(device))
    cached = _PLAN_TENSOR_CACHE.get(key)
    if cached is None:
        if len(_PLAN_TENSOR_CACHE) >= _PLAN_TENSOR_CACHE_MAXSIZE:
            _PLAN_TENSOR_CACHE.pop(next(iter(_PLAN_TENSOR_CACHE)))
        cached = torch.tensor(values, dtype=dtype, device=device)
        _PLAN_TENSOR_CACHE[key] = cached
    return cached


def _get_scratch(key_name: str, shape: tuple, dtype, device) -> torch.Tensor:
    key = (key_name, shape, dtype, _stream_key(device))
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        if len(_SCRATCH_CACHE) >= _SCRATCH_CACHE_MAXSIZE:
            _SCRATCH_CACHE.pop(next(iter(_SCRATCH_CACHE)))
        cached = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH_CACHE[key] = cached
    return cached


def _run_cp(
    plan: CPPlan,
    q,
    k,
    v,
    g,
    beta,
    *,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    _problem: _PrefillProblem,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    state_transposed=False,
) -> None:
    """Execute a non-trivial CPPlan verbatim -- no re-planning, no fallback."""
    if not _problem.is_varlen and _problem.T % CHUNK != 0:
        _run_padded_dense(
            plan,
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
            state_transposed,
        )
    else:
        _run_pipeline(
            plan,
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
            state_transposed,
        )


def intracard_prefill(
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
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    state_transposed=False,
    s_split=None,
    allow_fallback=True,
) -> None:
    """Plan-and-run for direct callers. s_split forces a manual split;
    allow_fallback=False means forced CP (structural split only, raise on a
    trivial plan); otherwise a trivial plan falls back to the serial kernel."""
    problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens)
    _validate_launch_options(q, out, lower_bound, True)
    seq_tiles = _seq_tiles_from_problem(problem)
    H = q.shape[2]
    if s_split is not None:
        plan = plan_manual(seq_tiles, s_split)
    elif allow_fallback:
        plan = plan_auto(seq_tiles, H, get_device_sm_count(q.device))
    else:
        plan = split_balanced(seq_tiles, H, get_device_sm_count(q.device))
    if plan.trivial:
        if not allow_fallback:
            raise NotSplittableError("SM90 intracard CP cannot split this shape.")
        flash_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            out=out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            state_transposed=state_transposed,
            _problem=problem,
        )
        return
    _run_cp(
        plan,
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        _problem=problem,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        state_transposed=state_transposed,
    )


# Dense partial-tile support — pad to CHUNK multiple, run aligned CP, strip back.
# (Varlen partial-tile is handled natively via ceil tiles + tile_starts mask.)
# The plan is built on ceil tile counts, so it is valid for the padded tensors.
def _pad_cp_inputs(pad, q, k, v, g, beta):
    """No-op sentinels: q/k/v=0, g=-1e6 (decay~0), beta=-80 (sigmoid~0)."""
    return pad(q, 0.0), pad(k, 0.0), pad(v, 0.0), pad(g, -1e6), pad(beta, -80.0)


def _run_padded_dense(
    plan,
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
    state_transposed,
) -> None:
    B, T, H, _ = q.shape
    pad_t = ((T + CHUNK - 1) // CHUNK) * CHUNK - T

    def _pad(x: torch.Tensor, fill: float) -> torch.Tensor:
        spec = (0, 0, 0, pad_t) if x.ndim == 3 else (0, 0, 0, 0, 0, pad_t)
        return torch.nn.functional.pad(x, spec, value=fill)

    pq, pk, pv, pg, pbeta = _pad_cp_inputs(_pad, q, k, v, g, beta)
    pout = out.new_empty((B, T + pad_t, H, D))
    _run_pipeline(
        plan,
        pq,
        pk,
        pv,
        pg,
        pbeta,
        scale,
        pout,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        final_state,
        None,
        state_transposed,
    )
    out.copy_(pout[:, :T])


def _run_pipeline(
    plan: CPPlan,
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
    state_transposed,
) -> None:
    B, T, H, K = q.shape
    device = q.device

    if cu_seqlens is None:
        T_total = B * T
        tile_starts = tile_actual_lens = None
        is_varlen_padded = False
    else:
        varlen_meta = _get_or_build_varlen_metadata(cu_seqlens)
        T_total = T
        is_varlen_padded = varlen_meta.needs_padding
        tile_starts = varlen_meta.tile_starts if is_varlen_padded else None
        tile_actual_lens = varlen_meta.tile_actual_lens if is_varlen_padded else None

    n_seg = plan.n_seg_total
    seg_cu_tiles = _get_plan_tensor(plan.seg_cu, torch.int32, device)

    n_qk = plan.total_tiles * H * CHUNK * D
    n_cc = plan.total_tiles * H * CHUNK * CHUNK
    # ws_beta uses tile layout (total_tiles*CHUNK*H), not packed token layout (T_total*H).
    ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta = _get_or_alloc_workspaces(
        n_qk, n_cc, plan.total_tiles * H * D, plan.total_tiles * CHUNK * H, device, beta.dtype
    )
    launch_k1(
        q,
        k,
        g,
        A_log,
        dt_bias,
        beta.reshape(-1),
        scale,
        lower_bound,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        ws_beta,
        tile_starts=tile_starts,
        tile_actual_lens=tile_actual_lens,
        total_tiles=plan.total_tiles if is_varlen_padded else None,
        is_varlen=is_varlen_padded,
    )

    b_seg = _get_scratch("b_seg", (n_seg, H, D, D), torch.float32, device)
    m_seg = _get_scratch("m_seg", (n_seg, H, D, D), torch.float32, device)
    v_flat = v.view(1, T_total, H, D) if B > 1 else v
    # Longest-first launch order: stable sort -> identity for uniform splits.
    seg_tiles = plan.seg_tiles
    seg_order = _get_plan_tensor(tuple(sorted(range(n_seg), key=lambda i: -seg_tiles[i])), torch.int32, device)
    # pre_scan skips each sequence's last segment.
    last_segs = {first + n - 1 for first, n in plan.per_seq}
    prescan_order = _get_plan_tensor(
        tuple(sorted((i for i in range(n_seg) if i not in last_segs), key=lambda i: -seg_tiles[i])),
        torch.int32,
        device,
    )
    launch_pre_scan(
        v_flat,
        ws_beta,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        b_seg,
        m_seg,
        seg_cu_tiles,
        v_tile_starts=tile_starts,
        v_tile_actual_lens=tile_actual_lens,
        total_tiles=plan.total_tiles,
        seg_order=prescan_order,
    )

    init_bhvk = None
    if initial_state is not None:
        init_bhvk = initial_state
        if state_transposed:
            init_bhvk = init_bhvk.transpose(-1, -2).contiguous()
    carries = _get_scratch("carries", (n_seg, H, D, D), torch.float32, device)
    launch_merge(carries, m_seg, b_seg, plan.per_seq, init_bhvk)

    out_flat = out.view(1, T_total, H, D) if B > 1 else out
    seg_final = None
    if final_state is not None:
        seg_final = _get_scratch("seg_final", (n_seg, H, D, D), torch.float32, device)
    launch_k2(
        v_flat,
        ws_beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out_flat,
        seg_cu_tiles,
        initial_state=carries,
        final_state=seg_final,
        state_transposed=False,
        v_tile_starts=tile_starts,
        v_tile_actual_lens=tile_actual_lens,
        seq_order=seg_order,
    )

    if final_state is not None:
        last_idx = _get_plan_tensor(tuple(first + n - 1 for first, n in plan.per_seq), torch.long, device)
        if not state_transposed:
            torch.index_select(seg_final, 0, last_idx, out=final_state)
        else:
            final_state.copy_(seg_final.index_select(0, last_idx).transpose(-1, -2))
