#!/usr/bin/env python3
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
Benchmark: la_decode_mtp (CuTe DSL) vs alternatives on Lightning Attention MTP.

Compares three implementations of T > 1 Lightning Attention decode:
  1. cula `linear_attention_decode_mtp`   (this work — fused single-launch)
  2. fla  `fused_recurrent_fwd`            (Triton, T-aware)
  3. cula `linear_attention_decode` × T    (cula self-comparison; T sequential calls)

Two timing modes (mirroring bench_la_decode_vs_fla.py):
  - kernel-only: pre-allocated buffers, pre-compiled kernel handle, pre-built stream
  - wrapper:     full Python entry point per call (cache lookup, CUstream, ...)

Bandwidth analysis (SOL% against B200 HBM3e peak ~8 TB/s) printed alongside.

Usage:
    python benchmarks/bench_la_decode_mtp.py
    python benchmarks/bench_la_decode_mtp.py --heads 64 --head-dim 128 --T 4
    python benchmarks/bench_la_decode_mtp.py --batch-sizes 1 4 16 64 --T 2
"""

import argparse
import os
import sys

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import cuda.bindings.driver as cuda_drv
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fla.ops.common.fused_recurrent import fused_recurrent_fwd

    HAS_FLA = True
except ImportError:
    HAS_FLA = False

from benchmarks.utils import benchmark_cuda_fn, relative_rms_error
from cula.lightning.la_decode_mtp import get_compiled_la_mtp_handle, linear_attention_decode_mtp
from cula.ops.lightning.decode import linear_attention_decode
from cula.utils import USE_FAST_MATH


# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth model — see spec §9.3
# ─────────────────────────────────────────────────────────────────────────────
def la_mtp_bytes(B, T, H, HV, K, V, cache_intermediate_states, disable_state_update):
    fp32 = 4
    qkv = B * T * H * K * fp32 * 2 + B * T * HV * V * fp32  # q, k, v reads
    out_w = B * T * HV * V * fp32  # o writes
    h0_r = B * HV * V * K * fp32  # h0 reads
    h0_w = 0 if disable_state_update else B * HV * V * K * fp32  # h0 writes
    inter = B * T * HV * V * K * fp32 if cache_intermediate_states else 0
    return qkv + out_w + h0_r + h0_w + inter


def sol_pct(byte_count: int, kernel_ms: float, peak_bps: float) -> float:
    """Speed-of-light percent of HBM peak."""
    return (byte_count / (kernel_ms * 1e-3)) / peak_bps * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark for one (B, T) configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(
    B, T, H, HV, K, V, layer_idx, num_layers, peak_bps, cache_intermediate_states=False, disable_state_update=False
):
    device = "cuda"
    dtype = torch.float32
    scale = K**-0.5
    pool_size = B

    # Per-head log decay (Lightning Attention formula)
    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, device=device, dtype=torch.float32)
    decay_scales = -g_gamma  # la_decode_mtp convention: exp(-decay_scales)

    # =========================================================================
    # Layer 1 — Inputs & buffers
    # =========================================================================
    torch.manual_seed(42)
    q_4d = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k_4d = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v_4d = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    state_init = torch.randn(B, HV, K, V, device=device, dtype=torch.float32) * 0.01  # K-major

    s_offsets = torch.arange(B, device=device, dtype=torch.int32)
    cu_seqlens_dummy = torch.empty(1, device=device, dtype=torch.int32)
    inter = (
        torch.zeros(B * T * HV, V, K, device=device, dtype=torch.float32)
        if cache_intermediate_states
        else torch.empty(1, 1, 1, device=device, dtype=torch.float32)
    )

    # =========================================================================
    # Layer 2 — Correctness + compile warmup (wrapper, same config as benchmark)
    # =========================================================================
    # fla reference
    o_fla = None
    if HAS_FLA:
        state_fla = state_init.clone()
        with torch.no_grad():
            o_fla_fp32, _ht_fla = fused_recurrent_fwd(
                q_4d,
                k_4d,
                v_4d,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=state_fla,
                output_final_state=True,
            )
        o_fla = o_fla_fp32.to(dtype)

    # cuLA MTP — also populates the compile cache for kernel-only timing below
    s_cute = state_init.clone().permute(0, 1, 3, 2).contiguous()  # [B, HV, V, K]
    out_cute = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    with torch.no_grad():
        linear_attention_decode_mtp(
            q_4d,
            k_4d,
            v_4d,
            s_cute,
            inter,
            out_cute,
            decay_scales=decay_scales,
            s_offsets=s_offsets,
            cu_seqlens=cu_seqlens_dummy,
            softmax_scale=scale,
            T=T,
            cache_intermediate_states=cache_intermediate_states,
            disable_state_update=disable_state_update,
            is_varlen=False,
        )

    rmse = rel_maxdiff = float("nan")
    if o_fla is not None and HV == H:
        rmse = relative_rms_error(o_fla.float(), out_cute.float())
        ref_cmp = o_fla.float()
        out_cmp = out_cute.float()
        max_ref = torch.abs(ref_cmp).max().item()
        rel_maxdiff = torch.abs(out_cmp - ref_cmp).max().item() / (max_ref + 1e-8)

    # =========================================================================
    # Layer 3a — Kernel-only timing (compiled handle + pre-built stream)
    # =========================================================================
    compiled_cute = get_compiled_la_mtp_handle(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        scale,
        q_4d.device,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
        is_varlen=False,
    )
    stream_handle = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    state_kk = state_init.clone().permute(0, 1, 3, 2).contiguous().view(pool_size * HV, V, K)
    out_kk = torch.empty(B, T, HV, V, device=device, dtype=dtype)
    inter_kk = inter

    def kernel_cute_mtp():
        compiled_cute(
            state_kk,
            inter_kk,
            decay_scales,
            q_4d,
            k_4d,
            v_4d,
            out_kk,
            s_offsets,
            cu_seqlens_dummy,
            stream_handle,
        )

    # cula self-baseline: T sequential la_decode (T=1) wrapper calls
    state_seq = state_init.clone().permute(0, 1, 3, 2).contiguous().view(B * HV, V, K)
    out_seq_buf = torch.empty(B, HV, V, device=device, dtype=dtype)
    q_slices = [q_4d[:, t].contiguous() for t in range(T)]
    k_slices = [k_4d[:, t].contiguous() for t in range(T)]
    v_slices = [v_4d[:, t].contiguous() for t in range(T)]

    def kernel_cute_seq():
        for t in range(T):
            linear_attention_decode(
                q_slices[t],
                k_slices[t],
                v_slices[t],
                state_seq,
                out_seq_buf,
                softmax_scale=scale,
                stride_q=0,
                stride_k=0,
                stride_v=0,
                stride_s=0,
                stride_o=0,
                s_offsets=s_offsets,
                decay_scales=decay_scales,
                HEAD_DIM=K,
                K_SPLIT_DIM=K,
                V_SPLIT_DIM=V,
            )

    with torch.no_grad():
        cute_mtp_ms = benchmark_cuda_fn(kernel_cute_mtp)
        cute_seq_ms = benchmark_cuda_fn(kernel_cute_seq)

    # =========================================================================
    # Layer 3b — Wrapper timing (full Python entry path per call)
    # =========================================================================
    s_wrap = state_init.clone().permute(0, 1, 3, 2).contiguous()
    out_wrap = torch.empty(B, T, HV, V, device=device, dtype=dtype)
    inter_wrap = (
        torch.zeros(B * T * HV, V, K, device=device, dtype=torch.float32)
        if cache_intermediate_states
        else torch.empty(1, 1, 1, device=device, dtype=torch.float32)
    )

    def wrapper_cute_mtp():
        linear_attention_decode_mtp(
            q_4d,
            k_4d,
            v_4d,
            s_wrap,
            inter_wrap,
            out_wrap,
            decay_scales=decay_scales,
            s_offsets=s_offsets,
            cu_seqlens=cu_seqlens_dummy,
            softmax_scale=scale,
            T=T,
            cache_intermediate_states=cache_intermediate_states,
            disable_state_update=disable_state_update,
            is_varlen=False,
        )

    with torch.no_grad():
        wrap_cute_ms = benchmark_cuda_fn(wrapper_cute_mtp)

    fla_ms = float("nan")
    if HAS_FLA:
        state_fla_bench = state_init.clone()

        def wrapper_fla():
            fused_recurrent_fwd(
                q_4d,
                k_4d,
                v_4d,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=state_fla_bench,
                output_final_state=True,
            )

        with torch.no_grad():
            fla_ms = benchmark_cuda_fn(wrapper_fla)

    # =========================================================================
    # Layer 4 — Roofline & summary
    # =========================================================================
    bytes_moved = la_mtp_bytes(
        B,
        T,
        H,
        HV,
        K,
        V,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
    )
    sol = sol_pct(bytes_moved, cute_mtp_ms, peak_bps)

    return {
        "B": B,
        "T": T,
        "cute_mtp_ms": cute_mtp_ms,
        "cute_seq_ms": cute_seq_ms,
        "fla_ms": fla_ms,
        "wrap_cute_ms": wrap_cute_ms,
        "speedup_seq": cute_seq_ms / cute_mtp_ms,
        "speedup_fla": fla_ms / cute_mtp_ms if HAS_FLA else float("nan"),
        "rmse": rmse,
        "rel_maxdiff": rel_maxdiff,
        "sol_pct": sol,
        "bytes_GB": bytes_moved / 1e9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark la_decode_mtp")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--T", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--num-v-heads", type=int, default=None, help="HV (defaults to --heads for MHA)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--peak-bps", type=float, default=8e12, help="HBM peak bytes/sec for SOL%% (B200 HBM3e ≈ 8e12)")
    parser.add_argument("--cache-intermediate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-state-update", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    H = args.heads
    HV = args.num_v_heads if args.num_v_heads is not None else H
    K = V = args.head_dim

    print("Lightning Attention MTP Decode Benchmark")
    print(f"  H={H}, HV={HV}, K={K}, V={V}, layer={args.layer_idx}/{args.num_layers}")
    print(f"  dtype=fp32, state=fp32, peak={args.peak_bps:.2e} B/s")
    print(f"  cache_intermediate_states={args.cache_intermediate}, disable_state_update={args.disable_state_update}")
    print(f"  USE_FAST_MATH={USE_FAST_MATH}, fla available={HAS_FLA}")

    fla_avail = HAS_FLA and HV == H
    if HAS_FLA and HV != H:
        print(f"  [warning] GQA HV={HV} != H={H}; fla baseline disabled (fla assumes HV==H)")

    cols = (
        f"{'B':>4} | {'T':>3} | {'cute_mtp(ms)':>12} | {'cute×T(ms)':>10} | "
        f"{'fla(ms)':>9} | {'spd_seq':>7} | {'spd_fla':>7} | "
        f"{'wrap(ms)':>9} | {'SOL%':>5} | {'GB':>6} | {'RMSE':>9}"
    )
    print(f"\n{cols}")
    print("─" * len(cols))

    for T in args.T:
        for B in args.batch_sizes:
            r = run_config(
                B,
                T,
                H,
                HV,
                K,
                V,
                args.layer_idx,
                args.num_layers,
                args.peak_bps,
                cache_intermediate_states=args.cache_intermediate,
                disable_state_update=args.disable_state_update,
            )
            print(
                f"{r['B']:>4} | {r['T']:>3} | {r['cute_mtp_ms']:>12.4f} | "
                f"{r['cute_seq_ms']:>10.4f} | "
                f"{(r['fla_ms'] if fla_avail else float('nan')):>9.4f} | "
                f"{r['speedup_seq']:>6.2f}x | "
                f"{(r['speedup_fla'] if fla_avail else float('nan')):>6.2f}x | "
                f"{r['wrap_cute_ms']:>9.4f} | {r['sol_pct']:>5.1f} | "
                f"{r['bytes_GB']:>6.3f} | {r['rmse']:>9.6f}"
            )
        print()

    print("Notes:")
    print("  cute_mtp  : linear_attention_decode_mtp (fused single launch, T tokens)")
    print("  cute×T    : T sequential linear_attention_decode (T=1) calls — cula self-baseline")
    print("  fla       : fused_recurrent_fwd (Triton); kernel still re-launched per T internally")
    print("  spd_seq   : cute×T / cute_mtp  (fusion benefit within cula)")
    print("  spd_fla   : fla / cute_mtp     (vs industry reference)")
    print("  wrap(ms)  : cute_mtp full Python entry (cache lookup + CUstream + kernel)")
    print(f"  SOL%      : (bytes / kernel_ms) / peak_bps × 100  (peak = {args.peak_bps:.2e} B/s)")


if __name__ == "__main__":
    main()
