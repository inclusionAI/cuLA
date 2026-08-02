# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fused l2-norm for paired (q, k) tensors — one Triton kernel handles both.

cuLA's baseline `cula_kda_prefill` calls `l2norm_fwd(q)` and `l2norm_fwd(k)`
as two separate Triton kernel launches. Each launch costs ~50-80 μs of CPU
overhead (Python wrapper + torch.empty + CUDA driver dispatch), even though
the actual GPU work is tiny for D=128.

The two operations are mathematically identical and operate on disjoint
inputs/outputs. By writing one kernel whose grid covers both q and k
(distinguishing them via `tl.program_id(1)`), we cut the Python-driver
overhead in half — saving one launch (~50 μs) per fwd at any T.

Combined with skipping the gate-stream optimization (which didn't pay off
due to torch.cuda.stream context overhead — see hopper_fused_fwd_opt.py),
this is the cleanest small-T speedup we found.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _l2norm_fwd_qk_kernel(
    q_ptr,
    k_ptr,  # input pointers (T*H, D) each
    yq_ptr,
    yk_ptr,  # output pointers (T*H, D) each
    rstd_q_ptr,
    rstd_k_ptr,  # output rstd (T*H,) each
    eps,
    D,
    BD: tl.constexpr,
):
    i_row = tl.program_id(0)  # 0..T*H-1
    i_qk = tl.program_id(1)  # 0=q, 1=k

    cols = tl.arange(0, BD)
    mask = cols < D

    # is_q is uniform across all threads in the block (driven by
    # tl.program_id(1)), so the if/else compiles to a single conditional
    # branch — no warp divergence, and only one tensor is actually loaded.
    is_q = i_qk == 0
    base_off = i_row * D
    if is_q:
        b_x = tl.load(q_ptr + base_off + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        b_x = tl.load(k_ptr + base_off + cols, mask=mask, other=0.0).to(tl.float32)

    b_rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd

    # Symmetric stores — same uniform-branch logic as the load above.
    if is_q:
        tl.store(yq_ptr + base_off + cols, b_y, mask=mask)
        tl.store(rstd_q_ptr + i_row, b_rstd)
    else:
        tl.store(yk_ptr + base_off + cols, b_y, mask=mask)
        tl.store(rstd_k_ptr + i_row, b_rstd)


def l2norm_fwd_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
):
    """L2-normalize q and k along the last dim, in a single fused kernel.

    Args:
        q, k: shape (..., D). Last dim is normalised; preceding dims are
              treated as a flat "row" index. q and k must have identical shape.
        eps: numerical safety eps.

    Returns:
        (y_q, y_k, rstd_q, rstd_k)
            y_q, y_k: normalized outputs, same shape as q, k.
            rstd_q, rstd_k: 1/sqrt(sum(x^2)+eps), shape q.shape[:-1].
    """
    assert q.shape == k.shape, f"q.shape {q.shape} != k.shape {k.shape}"
    assert q.dtype == k.dtype
    assert q.device == k.device
    assert q.is_contiguous() and k.is_contiguous(), "q, k must be contiguous"

    D = q.shape[-1]
    T = q.numel() // D

    y_q = torch.empty_like(q)
    y_k = torch.empty_like(k)
    rstd_q = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    rstd_k = torch.empty(k.shape[:-1], dtype=torch.float32, device=k.device)

    BD = triton.next_power_of_2(D)
    if D > BD or 65536 // q.element_size() < BD:
        raise RuntimeError(f"D={D} too large for fused l2norm_fwd_qk")

    # Grid: (T*H, 2) — program_id(1) picks q (0) or k (1).
    # Heuristic on num_warps: small D (e.g. 128) needs only 1 warp; up to 4.
    num_warps = 1 if BD <= 256 else (2 if BD <= 1024 else 4)
    _l2norm_fwd_qk_kernel[(T, 2)](
        q_ptr=q,
        k_ptr=k,
        yq_ptr=y_q,
        yk_ptr=y_k,
        rstd_q_ptr=rstd_q,
        rstd_k_ptr=rstd_k,
        eps=eps,
        D=D,
        BD=BD,
        num_warps=num_warps,
    )
    return y_q, y_k, rstd_q, rstd_k
