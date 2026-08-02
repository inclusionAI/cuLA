#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Benchmark: cuLA LA KVBuffer verify + state-update kernels.

Times the KVBuffer path (verify writes k/v to a pool buffer; state-update advances
the pooled state from it) and validates against a PyTorch reference.

An optional SGLang baseline (seg_la_mtp_kernel + fused_mamba_state_scatter_with_mask)
is compared when available. Set LA_SGLANG_PYTHON=/path/to/sglang/python for a custom
SGLang checkout.

Timing follows bench_la_decode_vs_fla.py:
  - Layer 2: wrapper call for correctness + compile warmup (same config as benchmark)
  - Layer 3: kernel-only via pre-compiled handles + pre-built stream

Usage:
    python benchmarks/bench_la_kvbuffer.py
    python benchmarks/bench_la_kvbuffer.py --batch-sizes 1 4 16 64 --T 2 4 8
    LA_SGLANG_PYTHON=~/sglang/python python benchmarks/bench_la_kvbuffer.py --T 4
"""

import argparse
import os
import sys

import torch

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import cuda.bindings.driver as cuda_drv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Optional SGLang baseline ─────────────────────────────────────────────────
_HAVE_SGLANG, _SGLANG_ERR = True, ""
SegLaMeta = seg_la_mtp_kernel = seg_la_sum_kernel = None
fused_mamba_state_scatter_with_mask = None
try:
    _sg_path = os.environ.get("LA_SGLANG_PYTHON", "")
    if _sg_path and os.path.isdir(_sg_path):
        sys.path.insert(0, _sg_path)
    from sglang.srt.layers.attention.linear.seg_la import (
        SegLaMeta,
        seg_la_mtp_kernel,
        seg_la_sum_kernel,
    )
    from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )
except Exception as e:  # noqa: BLE001
    _HAVE_SGLANG, _SGLANG_ERR = False, repr(e)

from benchmarks.utils import benchmark_cuda_fn, relative_rms_error  # noqa: E402
from cula.lightning.la_state_update_kvbuffer import (  # noqa: E402
    get_compiled_state_update_kvbuffer_handle,
    linear_attention_state_update_kvbuffer,
)
from cula.lightning.la_verify_kvbuffer import (  # noqa: E402
    get_compiled_verify_kvbuffer_handle,
    linear_attention_verify_kvbuffer,
)
from cula.utils import USE_FAST_MATH  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Reference & SGLang helpers
# ─────────────────────────────────────────────────────────────────────────────
def torch_la_mtp_ref(q, k, v, state, decay_scales, softmax_scale):
    """Pure PyTorch reference for MTP decode (output only)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    state = state.clone().float()
    out = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)
    decay = torch.exp(-decay_scales).float()

    for t in range(T):
        qt = q[:, t].float() * softmax_scale
        kt = k[:, t].float()
        vt = v[:, t].float()
        state = state * decay[None, :, None, None] + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", qt, state)

    return out


def run_sglang_mtp(
    q_3d,
    k_3d,
    v_3d,
    s_sglang,
    caches_sglang,
    s_offsets,
    cache_indices,
    decay_scales,
    meta,
    softmax_scale,
    HEAD_DIM,
    step,
    K_SPLIT_DIM=32,
    V_SPLIT_DIM=64,
):
    """Invoke seg_la_mtp_kernel the same way seg_la_fwd does for the MTP path."""
    length = q_3d.shape[0]
    qo_heads = q_3d.shape[1]
    bs = meta.batch_size

    k_dim_block = HEAD_DIM // K_SPLIT_DIM
    v_dim_block = HEAD_DIM // V_SPLIT_DIM
    tmp = torch.empty((k_dim_block, length, qo_heads, HEAD_DIM), device=q_3d.device, dtype=q_3d.dtype)
    grid = (bs, qo_heads, k_dim_block * v_dim_block)

    seg_la_mtp_kernel[grid](
        q_3d,
        k_3d,
        v_3d,
        s_sglang,
        caches_sglang,
        tmp,
        softmax_scale,
        q_3d.stride(0),
        k_3d.stride(0),
        v_3d.stride(0),
        s_sglang.stride(0),
        caches_sglang.stride(0),
        tmp.stride(0),
        s_offsets,
        cache_indices,
        decay_scales,
        step,
        HEAD_DIM=HEAD_DIM,
        K_SPLIT_DIM=K_SPLIT_DIM,
        V_SPLIT_DIM=V_SPLIT_DIM,
        num_warps=2,
        num_stages=3,
    )

    if k_dim_block > 1:
        if length < 2048:
            o = tmp.sum(0)
        else:
            o = torch.empty((length, qo_heads, HEAD_DIM), device=q_3d.device, dtype=q_3d.dtype)
            seg_la_sum_kernel[(length,)](
                tmp,
                o,
                DIM=qo_heads * HEAD_DIM,
                NUM_BLOCK=k_dim_block,
                num_warps=2,
                num_stages=3,
            )
    else:
        o = tmp[0]
    return o


def run_sglang_commit(s_sglang, caches_sglang, s_offsets, step_indices, B, H, K, V, T):
    """Invoke fused_mamba_state_scatter_with_mask (SGLang commit step)."""
    elem_per_entry = H * K * V
    dst = s_sglang.reshape(1, -1, elem_per_entry)
    src = caches_sglang.reshape(1, B, T, elem_per_entry)
    fused_mamba_state_scatter_with_mask(dst, src, s_offsets, step_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark for one (B, T) configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(B, T, H, K, V, layer_idx, num_layers):
    device = "cuda"
    input_dtype = torch.float32
    out_dtype = torch.float32
    scale = K**-0.5
    HV = H  # SGLang seg_la does not support GQA
    pool_size = B

    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, device=device, dtype=torch.float32)
    decay_scales = -g_gamma

    # =========================================================================
    # Layer 1 — Inputs & buffers (benchmark config: verify writes k/v, commit reads k/v buffers)
    # =========================================================================
    torch.manual_seed(42)
    q_4d = torch.randn(B, T, H, K, device=device, dtype=input_dtype)
    k_4d = torch.randn(B, T, H, K, device=device, dtype=input_dtype)
    v_4d = torch.randn(B, T, HV, V, device=device, dtype=input_dtype)
    state_init = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01  # K-major

    # cuLA state pool [pool_size, HV, V, K]
    s_kvbuf = state_init.permute(0, 1, 3, 2).contiguous()
    s_kk_view = s_kvbuf.view(pool_size * HV, V, K)

    out_kvbuf = torch.zeros(B, T, HV, V, device=device, dtype=out_dtype)
    out_kk = torch.empty(B, T, HV, V, device=device, dtype=out_dtype)

    h0_indices = torch.arange(B, device=device, dtype=torch.int32)
    accepted_len = torch.full((B,), T, device=device, dtype=torch.int32)

    k_buf = torch.zeros(pool_size, T, H, K, device=device, dtype=torch.float32)
    v_buf = torch.zeros(pool_size, T, HV, V, device=device, dtype=torch.float32)

    # SGLang 3D views (length = B*T)
    q_3d = q_4d.reshape(B * T, H, K).contiguous()
    k_3d = k_4d.reshape(B * T, H, K).contiguous()
    v_3d = v_4d.reshape(B * T, HV, V).contiguous()

    # =========================================================================
    # Layer 2 — Correctness + compile warmup (wrapper, same config as benchmark)
    # =========================================================================
    with torch.no_grad():
        o_ref = torch_la_mtp_ref(q_4d, k_4d, v_4d, state_init, decay_scales, scale)

        linear_attention_verify_kvbuffer(
            q_4d,
            k_4d,
            v_4d,
            s_kvbuf,
            out_kvbuf,
            decay_scales,
            h0_indices,
            scale,
            T,
            k_buf=k_buf,
            v_buf=v_buf,
        )
        linear_attention_state_update_kvbuffer(
            k_buf,
            v_buf,
            s_kvbuf,
            decay_scales,
            h0_indices,
            accepted_len,
            T,
        )

    rmse_kv = relative_rms_error(o_ref, out_kvbuf.float())

    # SGLang baseline (optional): correctness call also JIT-compiles Triton kernels
    rmse_sg = float("nan")
    s_sglang = caches_sglang = s_offsets_sg = cache_indices_sg = meta = None
    K_SPLIT_DIM = 32
    V_SPLIT_DIM = 32 if B <= 2 else 64
    if _HAVE_SGLANG:
        s_sglang = state_init.reshape(pool_size, H, K, V).contiguous()
        caches_sglang = torch.zeros(pool_size * T, H, K, V, device=device, dtype=torch.float32)
        s_offsets_sg = torch.arange(B, device=device, dtype=torch.int64)
        cache_indices_sg = torch.arange(B, device=device, dtype=torch.int64) * T
        meta = SegLaMeta(
            batch_size=B,
            max_q_length=T,
            q_offsets=torch.arange(B + 1, device=device, dtype=torch.int64) * T,
            s_offsets=s_offsets_sg,
            q_lengths=torch.full((B,), T, device=device, dtype=torch.int64),
            s_scales=torch.ones(B, device=device, dtype=torch.int64),
        )
        with torch.no_grad():
            o_sg = run_sglang_mtp(
                q_3d,
                k_3d,
                v_3d,
                s_sglang.clone(),
                caches_sglang.clone(),
                s_offsets_sg,
                cache_indices_sg,
                decay_scales,
                meta,
                scale,
                K,
                T,
                K_SPLIT_DIM,
                V_SPLIT_DIM,
            )
        rmse_sg = relative_rms_error(o_ref, o_sg.reshape(B, T, HV, V).float())

    # =========================================================================
    # Layer 3 — Kernel-only timing (compiled handles + pre-built stream)
    # =========================================================================
    stream_handle = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_verify = get_compiled_verify_kvbuffer_handle(
        B, T, H, HV, K, V, pool_size, scale, write_kv=True, device=q_4d.device
    )
    compiled_update = get_compiled_state_update_kvbuffer_handle(B, T, H, HV, K, V, pool_size, device=q_4d.device)

    def kernel_kvbuf_verify():
        compiled_verify(
            s_kk_view,
            decay_scales,
            q_4d,
            k_4d,
            v_4d,
            out_kk,
            h0_indices,
            k_buf,
            v_buf,
            stream_handle,
        )

    def kernel_kvbuf_update():
        compiled_update(
            s_kk_view,
            decay_scales,
            h0_indices,
            accepted_len,
            k_buf,
            v_buf,
            stream_handle,
        )

    step_indices_sg = torch.full((B,), T - 1, device=device, dtype=torch.int32)

    def kernel_sglang_verify():
        run_sglang_mtp(
            q_3d,
            k_3d,
            v_3d,
            s_sglang,
            caches_sglang,
            s_offsets_sg,
            cache_indices_sg,
            decay_scales,
            meta,
            scale,
            K,
            T,
            K_SPLIT_DIM,
            V_SPLIT_DIM,
        )

    def kernel_sglang_commit():
        run_sglang_commit(
            s_sglang,
            caches_sglang,
            s_offsets_sg.int(),
            step_indices_sg,
            B,
            H,
            K,
            V,
            T,
        )

    with torch.no_grad():
        cu_vfy_ms = benchmark_cuda_fn(kernel_kvbuf_verify)
        cu_cmt_ms = benchmark_cuda_fn(kernel_kvbuf_update)
        if _HAVE_SGLANG:
            sg_vfy_ms = benchmark_cuda_fn(kernel_sglang_verify)
            sg_cmt_ms = benchmark_cuda_fn(kernel_sglang_commit)
        else:
            sg_vfy_ms = sg_cmt_ms = float("nan")

    # =========================================================================
    # Layer 4 — Summary
    # =========================================================================
    sg_total_ms = sg_vfy_ms + sg_cmt_ms
    cu_total_ms = cu_vfy_ms + cu_cmt_ms

    return {
        "B": B,
        "T": T,
        "sg_vfy_ms": sg_vfy_ms,
        "sg_cmt_ms": sg_cmt_ms,
        "sg_total_ms": sg_total_ms,
        "cu_vfy_ms": cu_vfy_ms,
        "cu_cmt_ms": cu_cmt_ms,
        "cu_total_ms": cu_total_ms,
        "speedup": (sg_total_ms / cu_total_ms) if _HAVE_SGLANG else float("nan"),
        "rmse_sg": rmse_sg,
        "rmse_kv": rmse_kv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark LA KVBuffer verify + state-update")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--T", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    H = args.heads
    K = V = args.head_dim

    print("LA KVBuffer verify + state-update benchmark (cuLA, optional SGLang baseline)")
    print(f"  H={H}, K={K}, V={V}, layer={args.layer_idx}/{args.num_layers}")
    print("  q/k/v=fp32, out=fp32, state=fp32, kv_buffer=fp32")
    print(f"  USE_FAST_MATH={USE_FAST_MATH}")
    print("  Timing: kernel-only (wrapper for compile warmup; compiled handle for measure)")
    if _HAVE_SGLANG:
        print("  SGLang baseline: AVAILABLE (sg_* columns active)")
    else:
        print(f"  SGLang baseline: UNAVAILABLE — sg_* columns show nan. ({_SGLANG_ERR})")
        print("    set LA_SGLANG_PYTHON=/path/to/sglang/python to enable the comparison.")

    hdr = (
        f"{'B':>4} | {'T':>3} | "
        f"{'sg_vfy(ms)':>10} | {'sg_cmt(ms)':>10} | {'sg_total':>9} | "
        f"{'cu_vfy(ms)':>10} | {'cu_cmt(ms)':>10} | {'cu_total':>9} | "
        f"{'speedup':>7} | {'rmse_sg':>9} | {'rmse_kv':>9}"
    )
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for T_val in args.T:
        for B in args.batch_sizes:
            r = run_config(B, T_val, H, K, V, args.layer_idx, args.num_layers)
            print(
                f"{r['B']:>4} | {r['T']:>3} | "
                f"{r['sg_vfy_ms']:>10.4f} | {r['sg_cmt_ms']:>10.4f} | {r['sg_total_ms']:>9.4f} | "
                f"{r['cu_vfy_ms']:>10.4f} | {r['cu_cmt_ms']:>10.4f} | {r['cu_total_ms']:>9.4f} | "
                f"{r['speedup']:>6.2f}x | "
                f"{r['rmse_sg']:>9.6f} | {r['rmse_kv']:>9.6f}"
            )
        print()

    sg_mem = B * T_val * H * K * V * 4
    cu_mem = B * T_val * (H * K + H * V) * 4
    print(f"Memory per-pool (B={args.batch_sizes[-1]}, T={args.T[-1]}):")
    print(f"  SGLang intermediate caches: {sg_mem / 1e6:.1f} MB")
    print(f"  cuLA KV buffer:             {cu_mem / 1e6:.1f} MB")
    print(f"  Ratio:                      {sg_mem / cu_mem:.0f}×")

    print("\nColumns:")
    print("  sg_vfy     : seg_la_mtp_kernel (Triton, SGLang upstream)")
    print("  sg_cmt     : fused_mamba_state_scatter_with_mask (Triton, SGLang)")
    print("  cu_vfy     : verify_kvbuffer with KV buffer write (CuTe DSL)")
    print("  cu_cmt     : state_update_kvbuffer reading from buffer (CuTe DSL)")
    print("  speedup    : sg_total / cu_total")
    print("  rmse_*     : relative RMS error vs PyTorch reference")


if __name__ == "__main__":
    main()
