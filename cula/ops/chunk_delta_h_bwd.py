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

"""
SM90 CuTe DSL prototype for chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.

This is intentionally a small, non-persistent kernel for the first Hopper path:
  - fixed chunk size BT=64
  - K in {64, 128, 256}
  - non-varlen tensors [B, T, H, D]
  - non-transposed state layout [B, NT, H, K, V]
  - optional gk final-state decay

It mirrors the Triton bwd_dhu recurrence in FLA's common/chunk_delta_h.py.
The implementation favors clarity and testability over throughput; later
iterations can replace the shared-memory matrix products with WGMMA/TMA tiles.
"""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.utils import USE_FAST_MATH, assert_hopper

BT = 64
BV = 32
NUM_THREADS = 256


class ChunkDeltaRuleBwdDHUSm90:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        use_gk: bool,
        use_dht: bool,
        use_dh0: bool,
        use_exp2: bool,
        scale: float,
        use_fast_math: bool = True,
    ):
        assert head_dim_k in (64, 128, 256), f"prototype only supports K in {{64, 128, 256}}, got K={head_dim_k}"
        self.B = batch_size
        self.T = seq_len
        self.H = num_heads
        self.K = head_dim_k
        self.V = head_dim_v
        self.use_gk = use_gk
        self.use_dht = use_dht
        self.use_dh0 = use_dh0
        self.use_exp2 = use_exp2
        self.scale = scale
        self.use_fast_math = use_fast_math
        self.BT = BT
        self.BK = head_dim_k
        self.BV = BV
        self.num_threads = NUM_THREADS

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        gk_in: cute.Tensor,
        dht_in: cute.Tensor,
        dh0_in: cute.Tensor,
        do_in: cute.Tensor,
        dh_in: cute.Tensor,
        dv_in: cute.Tensor,
        dv2_in: cute.Tensor,
        stream: cuda.CUstream,
    ):
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        w_ptr = w_in.iterator
        gk_ptr = gk_in.iterator
        dht_ptr = dht_in.iterator
        dh0_ptr = dh0_in.iterator
        do_ptr = do_in.iterator
        dh_ptr = dh_in.iterator
        dv_ptr = dv_in.iterator
        dv2_ptr = dv2_in.iterator

        NT = (self.T + self.BT - 1) // self.BT

        q_layout = cute.make_layout(
            (self.B, self.T, self.H, self.BK),
            stride=(self.T * self.H * self.BK, self.H * self.BK, self.BK, 1),
        )
        q = cute.make_tensor(q_ptr, q_layout)
        k = cute.make_tensor(k_ptr, q_layout)
        w = cute.make_tensor(w_ptr, q_layout)

        v_layout = cute.make_layout(
            (self.B, self.T, self.H, self.V),
            stride=(self.T * self.H * self.V, self.H * self.V, self.V, 1),
        )
        do = cute.make_tensor(do_ptr, v_layout)
        dv = cute.make_tensor(dv_ptr, v_layout)
        dv2 = cute.make_tensor(dv2_ptr, v_layout)

        gk_layout = cute.make_layout(
            (self.B, self.T, self.H, self.BK),
            stride=(self.T * self.H * self.BK, self.H * self.BK, self.BK, 1),
        )
        gk = cute.make_tensor(gk_ptr, gk_layout)

        state_layout = cute.make_layout(
            (self.B, NT, self.H, self.BK, self.V),
            stride=(
                NT * self.H * self.BK * self.V,
                self.H * self.BK * self.V,
                self.BK * self.V,
                self.V,
                1,
            ),
        )
        dh = cute.make_tensor(dh_ptr, state_layout)

        final_layout = cute.make_layout(
            (self.B, self.H, self.BK, self.V),
            stride=(self.H * self.BK * self.V, self.BK * self.V, self.V, 1),
        )
        dht = cute.make_tensor(dht_ptr, final_layout)
        dh0 = cute.make_tensor(dh0_ptr, final_layout)

        self.kernel(q, k, w, gk, dht, dh0, do, dh, dv, dv2).launch(
            grid=[cute.ceil_div(self.V, self.BV), self.B * self.H, 1],
            block=[self.num_threads, 1, 1],
            smem=(self.BK * self.BV + self.BT * self.BV) * 4 + 512,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        w: cute.Tensor,
        gk: cute.Tensor,
        dht: cute.Tensor,
        dh0: cute.Tensor,
        do: cute.Tensor,
        dh: cute.Tensor,
        dv: cute.Tensor,
        dv2: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i_v_tile, i_bh, _ = cute.arch.block_idx()
        i_b = i_bh // self.H
        i_h = i_bh - i_b * self.H
        v_base = i_v_tile * self.BV
        NT = (self.T + self.BT - 1) // self.BT

        smem = cutlass.utils.SmemAllocator()
        s_dh = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            16,
        )
        s_dv = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            16,
        )

        linear = tidx
        while linear < self.BK * self.BV:
            k_idx = linear // self.BV
            v_rel = linear - k_idx * self.BV
            v_idx = v_base + v_rel
            init = cutlass.Float32(0.0)
            if cutlass.const_expr(self.use_dht):
                if v_idx < self.V:
                    init = cutlass.Float32(dht[i_b, i_h, k_idx, v_idx])
            s_dh[k_idx, v_rel] = init
            linear += self.num_threads
        cute.arch.barrier()

        for chunk_rev in cutlass.range_constexpr(NT):
            i_t = NT - 1 - chunk_rev
            chunk_start = i_t * self.BT
            chunk_end = cutlass.min(chunk_start + self.BT, self.T)
            last_idx = chunk_end - 1

            linear = tidx
            while linear < self.BK * self.BV:
                k_idx = linear // self.BV
                v_rel = linear - k_idx * self.BV
                v_idx = v_base + v_rel
                if v_idx < self.V:
                    dh[i_b, i_t, i_h, k_idx, v_idx] = s_dh[k_idx, v_rel].to(dh.element_type)
                linear += self.num_threads
            cute.arch.barrier()

            linear = tidx
            while linear < self.BT * self.BV:
                t_rel = linear // self.BV
                v_rel = linear - t_rel * self.BV
                t_idx = chunk_start + t_rel
                v_idx = v_base + v_rel
                acc = cutlass.Float32(0.0)
                if t_idx < self.T and v_idx < self.V:
                    acc = cutlass.Float32(dv[i_b, t_idx, i_h, v_idx])
                    for k_idx in cutlass.range(self.BK, unroll_full=True):
                        acc += cutlass.Float32(k[i_b, t_idx, i_h, k_idx]) * s_dh[k_idx, v_rel]
                    dv2[i_b, t_idx, i_h, v_idx] = acc.to(dv2.element_type)
                s_dv[t_rel, v_rel] = acc
                linear += self.num_threads
            cute.arch.barrier()

            linear = tidx
            while linear < self.BK * self.BV:
                k_idx = linear // self.BV
                v_rel = linear - k_idx * self.BV
                v_idx = v_base + v_rel
                acc = s_dh[k_idx, v_rel]
                if v_idx < self.V:
                    if cutlass.const_expr(self.use_gk):
                        gk_last = cutlass.Float32(gk[i_b, last_idx, i_h, k_idx])
                        if cutlass.const_expr(self.use_exp2):
                            acc *= cute.exp2(gk_last, fastmath=self.use_fast_math)
                        else:
                            acc *= cute.exp(gk_last, fastmath=self.use_fast_math)
                    for t_rel in cutlass.range(self.BT, unroll_full=True):
                        t_idx = chunk_start + t_rel
                        if t_idx < self.T:
                            q_term = cutlass.Float32(q[i_b, t_idx, i_h, k_idx])
                            do_term = cutlass.Float32(do[i_b, t_idx, i_h, v_idx])
                            w_term = cutlass.Float32(w[i_b, t_idx, i_h, k_idx])
                            acc += q_term * do_term * self.scale - w_term * s_dv[t_rel, v_rel]
                    s_dh[k_idx, v_rel] = acc
                linear += self.num_threads
            cute.arch.barrier()

        if cutlass.const_expr(self.use_dh0):
            linear = tidx
            while linear < self.BK * self.BV:
                k_idx = linear // self.BV
                v_rel = linear - k_idx * self.BV
                v_idx = v_base + v_rel
                if v_idx < self.V:
                    dh0[i_b, i_h, k_idx, v_idx] = s_dh[k_idx, v_rel]
                linear += self.num_threads


def _as_cute(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16)


@functools.lru_cache(maxsize=64)
def _compile_bwd_dhu_sm90(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    use_gk: bool,
    use_dht: bool,
    use_dh0: bool,
    use_exp2: bool,
    scale: float,
):
    kernel = ChunkDeltaRuleBwdDHUSm90(
        batch_size=B,
        seq_len=T,
        num_heads=H,
        head_dim_k=K,
        head_dim_v=V,
        use_gk=use_gk,
        use_dht=use_dht,
        use_dh0=use_dh0,
        use_exp2=use_exp2,
        scale=scale,
        use_fast_math=USE_FAST_MATH,
    )

    q_fake = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k_fake = torch.empty_like(q_fake)
    w_fake = torch.empty_like(q_fake)
    do_fake = torch.empty(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    dv_fake = torch.empty_like(do_fake)
    dv2_fake = torch.empty_like(do_fake)
    gk_fake = torch.empty(B, T, H, K, device="cuda", dtype=torch.float32)
    dht_fake = torch.empty(B, H, K, V, device="cuda", dtype=torch.float32)
    dh0_fake = torch.empty_like(dht_fake)
    dh_fake = torch.empty(B, math.ceil(T / BT), H, K, V, device="cuda", dtype=torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    return cute.compile(
        kernel,
        _as_cute(q_fake),
        _as_cute(k_fake),
        _as_cute(w_fake),
        _as_cute(gk_fake),
        _as_cute(dht_fake),
        _as_cute(dh0_fake),
        _as_cute(do_fake),
        _as_cute(dh_fake),
        _as_cute(dv_fake),
        _as_cute(dv2_fake),
        stream=stream,
        options="--enable-tvm-ffi",
    )


def chunk_gated_delta_rule_bwd_dhu_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = BT,
    chunk_indices: torch.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """FLA-compatible wrapper for the current SM90 bwd_dhu prototype."""
    del chunk_indices
    assert_hopper(q.device)
    if cu_seqlens is not None:
        raise NotImplementedError("SM90 bwd_dhu prototype only supports non-varlen tensors.")
    if transpose_state_layout:
        raise NotImplementedError("SM90 bwd_dhu prototype only supports [B, NT, H, K, V] state layout.")
    if g is not None:
        raise NotImplementedError("SM90 bwd_dhu prototype supports gk gating, not scalar g gating yet.")
    if chunk_size != BT:
        raise NotImplementedError(f"SM90 bwd_dhu prototype only supports chunk_size={BT}.")

    B, T, H, K = q.shape
    V = do.shape[-1]
    if K not in (64, 128, 256):
        raise NotImplementedError(f"SM90 bwd_dhu prototype only supports K in {{64, 128, 256}}, got K={K}.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("q, k, and w must be bfloat16 for the SM90 bwd_dhu prototype.")
    if do.dtype != torch.bfloat16 or dv.dtype != torch.bfloat16:
        raise TypeError("do and dv must be bfloat16 for the SM90 bwd_dhu prototype.")
    if not q.is_contiguous() or not k.is_contiguous() or not w.is_contiguous():
        raise ValueError("q, k, and w must be contiguous.")
    if not do.is_contiguous() or not dv.is_contiguous():
        raise ValueError("do and dv must be contiguous.")

    NT = math.ceil(T / BT)
    scale_value = 1.0 if scale is None else float(scale)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    gk_arg = gk if gk is not None else torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dht_arg = dht if dht is not None else torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    dh0_arg = dh0 if dh0 is not None else torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    if gk is not None and (gk.dtype != torch.float32 or not gk.is_contiguous()):
        raise ValueError("gk must be contiguous float32.")
    if dht is not None and (dht.dtype != torch.float32 or not dht.is_contiguous()):
        raise ValueError("dht must be contiguous float32.")

    compiled = _compile_bwd_dhu_sm90(
        B,
        T,
        H,
        K,
        V,
        gk is not None,
        dht is not None,
        h0 is not None,
        use_exp2,
        scale_value,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(q, k, w, gk_arg, dht_arg, dh0_arg, do, dh, dv, dv2, stream)
    return dh, dh0, dv2


# Shorter alias for users who import this module directly.
chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu_sm90
