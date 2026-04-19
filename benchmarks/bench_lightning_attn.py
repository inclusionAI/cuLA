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
Unified Benchmark: Lightning Attention with decay — CuteDSL vs Triton (FLA).

Modes:
  no_state  — no initial/final state, standard prefill
  h0_ht     — provide random h0 and output ht
  varlen    — variable-length packed sequences (persistent vs non-persistent vs FLA)

Usage:
  # Standard prefill benchmarks
  python benchmarks/bench_lightning_attn.py --modes no_state h0_ht

  # Varlen only
  python benchmarks/bench_lightning_attn.py --modes varlen

  # All modes with report and plot
  python benchmarks/bench_lightning_attn.py --modes no_state h0_ht varlen --report --plot

  # Custom varlen workloads
  python benchmarks/bench_lightning_attn.py --modes varlen --num-heads 32 64 --iterations 50
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fla.ops.simple_gla.chunk import chunk_simple_gla_fwd

from cula.ops.lightning_attn import lightning_attn_fwd, lightning_attn_fwd_varlen

# =============================================================================
# Constants
# =============================================================================
D_DEFAULT = 128
C = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def reset_cuda_error():
    """Reset CUDA error state after an error."""
    try:
        torch.cuda.synchronize()
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaGetLastError()
        torch.cuda.empty_cache()
    except Exception:
        pass


def compute_decay(H, layer_idx=12, num_layers=24):
    """Compute per-head decay: decay_s[h] = (8/H)*(1-layer_idx/num_layers)*h."""
    return (8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def torch_naive_lightning_attn(Q, K, V, decay, scale=1.0, initial_state=None, output_final_state=False):
    """Recurrent FP32 reference for lightning attention (simple_gla).

    O(B*T*H*D^2) — exact ground truth, all computation in FP32.
    """
    B, T, H, D = Q.shape
    q, k, v = Q.float(), K.float(), V.float()
    decay_factor = torch.exp(-decay.float())  # [H]

    S = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, D, D, dtype=torch.float32, device=Q.device)
    )
    O = torch.zeros(B, T, H, D, dtype=torch.float32, device=Q.device)

    for t in range(T):
        S = S * decay_factor[None, :, None, None]
        S = S + torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])
        O[:, t] = scale * torch.einsum("bhd,bhde->bhe", q[:, t], S)

    ht = S if output_final_state else None
    return O, ht


# =============================================================================
# Sequence length generators (for varlen)
# =============================================================================
def gen_uniform(N, T):
    """All sequences have equal length."""
    per = T // N
    return [per] * N


def gen_skewed(N, T):
    """One long sequence + many short ones."""
    if N == 1:
        return [T]
    short = max(1, T // (2 * (N - 1)))
    long_len = T - short * (N - 1)
    return [long_len] + [short] * (N - 1)


def gen_random(N, T, seed=42):
    """Random sequence lengths summing to ~T."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(N))
    lens = np.maximum(1, np.round(raw * T).astype(int))
    diff = T - lens.sum()
    lens[0] += diff
    lens = np.maximum(1, lens)
    return lens.tolist()


# =============================================================================
# Timing helper
# =============================================================================
def time_fn(fn, warmup, iters):
    """Time a CUDA function using events. Returns ms/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# =============================================================================
# Runners
# =============================================================================
def run_fla(Q, K, V, decay, initial_state, output_final_state, warmup, iters):
    """Run FLA chunk_simple_gla_fwd (standard, non-varlen)."""
    g_gamma = -decay
    scale = 1.0

    def fn():
        return chunk_simple_gla_fwd(
            q=Q,
            k=K,
            v=V,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=C,
        )

    fn()  # compile
    ms = time_fn(fn, warmup, iters)
    o, ht = fn()
    return o, ht, ms


def run_cutedsl(Q, K, V, decay, h0, output_final_state, warmup, iters):
    """Run CuteDSL kernel (standard, non-varlen)."""
    scale = 1.0

    def fn():
        return lightning_attn_fwd(
            Q,
            K,
            V,
            decay,
            scale=scale,
            initial_state=h0,
            output_final_state=output_final_state,
            chunk_size=C,
        )

    t0 = time.time()
    fn()
    compile_ms = (time.time() - t0) * 1000

    ms = time_fn(fn, warmup, iters)
    O, ht = fn()
    return O, ht, ms, compile_ms


def run_cutedsl_varlen(Q, K, V, decay, cu_seqlens, persistent, warmup, iters):
    """Run CuteDSL varlen kernel (persistent or non-persistent)."""

    def fn():
        return lightning_attn_fwd_varlen(
            Q,
            K,
            V,
            decay,
            cu_seqlens,
            scale=1.0,
            chunk_size=C,
            persistent=persistent,
        )

    t0 = time.time()
    fn()
    compile_ms = (time.time() - t0) * 1000

    ms = time_fn(fn, warmup, iters)
    O, sp = fn()
    return ms, O, sp, compile_ms


def run_fla_varlen(Q, K, V, decay, cu_seqlens, warmup, iters):
    """Run FLA native varlen (single launch via cu_seqlens). FAIR baseline."""
    g_gamma = -decay
    cu_long = cu_seqlens.to(torch.long)

    def fn():
        return chunk_simple_gla_fwd(
            q=Q,
            k=K,
            v=V,
            g_gamma=g_gamma,
            scale=1.0,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_long,
            chunk_size=C,
        )

    fn()  # compile
    ms = time_fn(fn, warmup, iters)
    return ms


# =============================================================================
# Standard (non-varlen) benchmark
# =============================================================================
def benchmark_standard_config(B, T, H, D, layer_idx, num_layers, mode, warmup, iters):
    """Benchmark a single standard (non-varlen) config.

    mode: "no_state" — no initial/final state
          "h0_ht"   — provide random h0 and output ht
    """
    torch.manual_seed(42)
    Q = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    K = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    V = torch.randn(B, T, H, D, dtype=DTYPE, device=DEVICE)
    decay = compute_decay(H, layer_idx, num_layers)

    has_h0 = mode == "h0_ht"
    output_ht = mode == "h0_ht"
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=DEVICE) * 0.01 if has_h0 else None
    h0_fla = h0.clone() if h0 is not None else None
    h0_cute = h0.transpose(-1, -2).contiguous() if h0 is not None else None  # BHVK for CuTe

    result = {"B": B, "T": T, "H": H, "D": D, "mode": mode}
    ht_fla = None
    ht_cute = None

    # --- FLA ---
    try:
        o_fla, ht_fla, fla_ms = run_fla(Q, K, V, decay, h0_fla, output_ht, warmup, iters)
        result["fla_ms"] = fla_ms
    except Exception as e:
        o_fla = None
        result["fla_ms"] = float("nan")
        result["fla_err"] = str(e)
        reset_cuda_error()

    # --- CuteDSL ---
    try:
        o_cute, ht_cute, cute_ms, compile_ms = run_cutedsl(Q, K, V, decay, h0_cute, output_ht, warmup, iters)
        result["cutedsl_ms"] = cute_ms
        result["compile_ms"] = compile_ms
    except Exception as e:
        o_cute = None
        result["cutedsl_ms"] = float("nan")
        result["compile_ms"] = float("nan")
        result["cutedsl_err"] = str(e)
        reset_cuda_error()

    # --- Naive FP32 reference ---
    o_naive, ht_naive = torch_naive_lightning_attn(
        Q,
        K,
        V,
        decay,
        scale=1.0,
        initial_state=h0,
        output_final_state=output_ht,
    )

    # --- Accuracy vs naive ---
    for label, o_test, ht_test in [("fla", o_fla, ht_fla), ("cute", o_cute, ht_cute)]:
        if o_test is not None:
            diff = o_naive - o_test.float()
            rms = o_naive.pow(2).mean().sqrt().item()
            rmse = diff.pow(2).mean().sqrt().item()
            result[f"{label}_o_rmse_ratio"] = rmse / (rms + 1e-8)
            result[f"{label}_o_maxdiff"] = diff.abs().max().item()
            if output_ht and ht_naive is not None and ht_test is not None:
                # CuTe kernel outputs BHVK state; transpose to BHKV for comparison
                ht_cmp = ht_test.transpose(-1, -2).float() if label == "cute" else ht_test.float()
                diff_ht = ht_naive - ht_cmp
                ht_rms = ht_naive.pow(2).mean().sqrt().item()
                ht_rmse = diff_ht.pow(2).mean().sqrt().item()
                result[f"{label}_ht_rmse_ratio"] = ht_rmse / (ht_rms + 1e-8)
            else:
                result[f"{label}_ht_rmse_ratio"] = float("nan")
        else:
            result[f"{label}_o_rmse_ratio"] = float("nan")
            result[f"{label}_o_maxdiff"] = float("nan")
            result[f"{label}_ht_rmse_ratio"] = float("nan")

    # --- Speedup ---
    fla_ok = _valid(result["fla_ms"])
    cute_ok = _valid(result["cutedsl_ms"])
    result["speedup"] = result["fla_ms"] / result["cutedsl_ms"] if fla_ok and cute_ok else float("nan")
    return result


# =============================================================================
# Varlen benchmark
# =============================================================================
def benchmark_varlen_config(N, seq_lens, H, D, warmup, iters, dist=""):
    """Benchmark a varlen config: persistent vs non-persistent vs FLA varlen."""
    T = sum(seq_lens)
    torch.manual_seed(42)
    Q = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    K = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    V = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    cu = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=DEVICE)
    decay = compute_decay(H)

    result = {
        "B": N,
        "T": T,
        "H": H,
        "D": D,
        "mode": "varlen",
        "seq_lens": seq_lens,
        "dist": dist,
        "min_seq": min(seq_lens),
        "max_seq": max(seq_lens),
    }

    # --- Persistent ---
    try:
        ms_p, O_p, sp_p, compile_ms = run_cutedsl_varlen(Q, K, V, decay, cu, True, warmup, iters)
        result["persistent_ms"] = ms_p
        result["compile_ms"] = compile_ms
    except Exception as e:
        result["persistent_ms"] = float("nan")
        result["persistent_err"] = str(e)
        O_p = None
        reset_cuda_error()

    # --- Non-persistent ---
    try:
        ms_np, O_np, sp_np, _ = run_cutedsl_varlen(Q, K, V, decay, cu, False, warmup, iters)
        result["nonpersistent_ms"] = ms_np
    except Exception as e:
        result["nonpersistent_ms"] = float("nan")
        result["nonpersistent_err"] = str(e)
        O_np = None
        reset_cuda_error()

    # --- FLA native varlen (fair: single launch) ---
    try:
        fla_vl_ms = run_fla_varlen(Q, K, V, decay, cu, warmup, iters)
        result["fla_varlen_ms"] = fla_vl_ms
    except Exception as e:
        result["fla_varlen_ms"] = float("nan")
        result["fla_varlen_err"] = str(e)
        reset_cuda_error()

    # --- Accuracy: persistent vs non-persistent ---
    if O_p is not None and O_np is not None:
        diff_o = O_p.float() - O_np.float()
        result["p_vs_np_O_diff"] = diff_o.abs().max().item()
        o_rmse = diff_o.pow(2).mean().sqrt().item()
        o_rms = O_p.float().pow(2).mean().sqrt().item()
        result["p_vs_np_O_rmse"] = o_rmse
        result["p_vs_np_O_rmse_ratio"] = o_rmse / (o_rms + 1e-8)
        diff_ht = sp_p.float() - sp_np.float()
        result["p_vs_np_ht_diff"] = diff_ht.abs().max().item()
        ht_rmse = diff_ht.pow(2).mean().sqrt().item()
        ht_rms = sp_p.float().pow(2).mean().sqrt().item()
        result["p_vs_np_ht_rmse"] = ht_rmse
        result["p_vs_np_ht_rmse_ratio"] = ht_rmse / (ht_rms + 1e-8)
    else:
        result["p_vs_np_O_diff"] = float("nan")
        result["p_vs_np_ht_diff"] = float("nan")
        result["p_vs_np_O_rmse"] = float("nan")
        result["p_vs_np_O_rmse_ratio"] = float("nan")
        result["p_vs_np_ht_rmse"] = float("nan")
        result["p_vs_np_ht_rmse_ratio"] = float("nan")

    # --- Speedups ---
    p_ms = result["persistent_ms"]
    np_ms = result["nonpersistent_ms"]
    fla_vl_ms = result.get("fla_varlen_ms", float("nan"))
    result["p_vs_np_speedup"] = np_ms / p_ms if _valid(p_ms) and _valid(np_ms) else float("nan")
    result["p_vs_fla_vl_speedup"] = fla_vl_ms / p_ms if _valid(p_ms) and _valid(fla_vl_ms) else float("nan")
    # For unified summary
    result["cutedsl_ms"] = p_ms
    result["speedup"] = result["p_vs_fla_vl_speedup"]  # use fair comparison

    return result


def _valid(x):
    """Check if a value is a valid positive number (not NaN)."""
    return not np.isnan(x) and x > 0


# =============================================================================
# Print helpers
# =============================================================================
def print_standard_header():
    hdr = (
        f"{'Config':<28} {'Mode':<10} "
        f"{'FLA(ms)':>9} {'CuteDSL(ms)':>12} {'Speedup':>8} "
        f"{'FLA_O_RMSE%':>12} {'Cute_O_RMSE%':>13} "
        f"{'FLA_Ht_RMSE%':>13} {'Cute_Ht_RMSE%':>14}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_standard_result(r):
    cfg = f"B={r['B']},T={r['T']},H={r['H']}"

    fla = f"{r['fla_ms']:.3f}" if _valid(r.get("fla_ms", float("nan"))) else "ERR"
    dsl = f"{r['cutedsl_ms']:.3f}" if _valid(r.get("cutedsl_ms", float("nan"))) else "ERR"
    sp = f"{r['speedup']:.2f}x" if _valid(r.get("speedup", float("nan"))) else "-"
    fla_o = f"{r['fla_o_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("fla_o_rmse_ratio", float("nan"))) else "-"
    cute_o = f"{r['cute_o_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("cute_o_rmse_ratio", float("nan"))) else "-"
    fla_ht = f"{r['fla_ht_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("fla_ht_rmse_ratio", float("nan"))) else "-"
    cute_ht = f"{r['cute_ht_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("cute_ht_rmse_ratio", float("nan"))) else "-"

    print(f"{cfg:<28} {r['mode']:<10} {fla:>9} {dsl:>12} {sp:>8} {fla_o:>12} {cute_o:>13} {fla_ht:>13} {cute_ht:>14}")
    if r.get("fla_err"):
        print(f"  >> FLA error: {r['fla_err']}")
    if r.get("cutedsl_err"):
        print(f"  >> CuteDSL error: {r['cutedsl_err']}")


def print_varlen_header():
    hdr = (
        f"{'Config':<24} {'Dist':<8} "
        f"{'Persist(ms)':>12} {'NonPer(ms)':>11} {'FLA_vl(ms)':>11} "
        f"{'P/NP':>6} {'P/FLAvl':>8} "
        f"{'O diff':>10} {'O_RMSE%':>9} {'ht diff':>10} {'ht_RMSE%':>10}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_varlen_result(r):
    cfg = f"N={r['B']},T={r['T']},H={r['H']}"
    dist = r.get("dist", "")

    p_ms = f"{r['persistent_ms']:.3f}" if _valid(r.get("persistent_ms", float("nan"))) else "ERR"
    np_ms = f"{r['nonpersistent_ms']:.3f}" if _valid(r.get("nonpersistent_ms", float("nan"))) else "ERR"
    fla_vl = f"{r['fla_varlen_ms']:.3f}" if _valid(r.get("fla_varlen_ms", float("nan"))) else "ERR"

    pvnp = f"{r['p_vs_np_speedup']:.2f}x" if _valid(r.get("p_vs_np_speedup", float("nan"))) else "-"
    pvfla_vl = f"{r['p_vs_fla_vl_speedup']:.2f}x" if _valid(r.get("p_vs_fla_vl_speedup", float("nan"))) else "-"

    od = f"{r['p_vs_np_O_diff']:.1e}" if not np.isnan(r.get("p_vs_np_O_diff", float("nan"))) else "-"
    ormse = f"{r['p_vs_np_O_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("p_vs_np_O_rmse_ratio", float("nan"))) else "-"
    hd = f"{r['p_vs_np_ht_diff']:.1e}" if not np.isnan(r.get("p_vs_np_ht_diff", float("nan"))) else "-"
    htrmse = f"{r['p_vs_np_ht_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("p_vs_np_ht_rmse_ratio", float("nan"))) else "-"

    print(
        f"{cfg:<24} {dist:<8} {p_ms:>12} {np_ms:>11} {fla_vl:>11} {pvnp:>6} {pvfla_vl:>8} {od:>10} {ormse:>9} {hd:>10} {htrmse:>10}"
    )
    if r.get("persistent_err"):
        print(f"  >> Persistent error: {r['persistent_err']}")
    if r.get("nonpersistent_err"):
        print(f"  >> Non-persistent error: {r['nonpersistent_err']}")
    if r.get("fla_varlen_err"):
        print(f"  >> FLA varlen error: {r['fla_varlen_err']}")


# =============================================================================
# Main suite
# =============================================================================
def run_benchmark_suite(args):
    """Run benchmarks across all requested modes."""
    D = args.head_dim
    layer_idx = args.layer_idx
    num_layers = args.num_layers
    warmup = args.warmup
    iters = args.iterations
    modes = args.modes

    print("\n" + "=" * 100)
    print("Lightning Attention Benchmark: CuteDSL vs FLA")
    print("=" * 100)
    print(f"  Modes:         {modes}")
    print(f"  Batch sizes:   {args.batch_sizes}")
    print(f"  Seq lengths:   {args.seq_lens}")
    print(f"  Num heads:     {args.num_heads}")
    print(f"  Head dim:      {D}")
    print(f"  Layer:         {layer_idx}/{num_layers}")
    print(f"  Warmup/Iters:  {warmup}/{iters}")
    print("=" * 100 + "\n")

    all_results = []

    # ===================== Standard modes (no_state, h0_ht) =====================
    standard_modes = [m for m in modes if m in ("no_state", "h0_ht")]
    if standard_modes:
        print_standard_header()
        for mode in standard_modes:
            for B in args.batch_sizes:
                for T in args.seq_lens:
                    for H in args.num_heads:
                        total = B * T * H * D
                        if total > 2_147_483_648:
                            continue
                        if T > 4096 and B > 2:
                            continue
                        r = benchmark_standard_config(B, T, H, D, layer_idx, num_layers, mode, warmup, iters)
                        all_results.append(r)
                        print_standard_result(r)

    # ===================== Varlen mode =====================
    if "varlen" in modes:
        print(f"\n{'=' * 100}")
        print(" Varlen Mode: Persistent vs Non-Persistent vs FLA varlen")
        print(f"{'=' * 100}")

        # Build varlen workloads
        workloads = []

        # Focus on N=5..25 range (realistic serving batch sizes)
        for N in [5, 8, 10, 12, 16, 20, 25]:
            for T_total in [1024, 2048, 4096, 8192, 16384, 32768]:
                if T_total // N < 1:
                    continue
                workloads.append((N, T_total, "uniform"))
                workloads.append((N, T_total, "skewed"))
                workloads.append((N, T_total, "random"))

        # Deduplicate
        seen = set()
        unique = []
        for w in workloads:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        workloads = unique

        for H in args.num_heads:
            print(f"\n  --- H={H}, D={D} ---")
            print_varlen_header()

            for N, T_total, dist in workloads:
                if dist == "uniform":
                    seq_lens = gen_uniform(N, T_total)
                elif dist == "skewed":
                    seq_lens = gen_skewed(N, T_total)
                elif dist == "random":
                    seq_lens = gen_random(N, T_total)
                else:
                    raise ValueError(f"Unknown dist: {dist}")

                r = benchmark_varlen_config(N, seq_lens, H, D, warmup, iters, dist=dist)
                all_results.append(r)
                print_varlen_result(r)

    # ===================== Summary =====================
    print(f"\n{'=' * 100}")
    print("SUMMARY BY MODE")
    print(f"{'=' * 100}")

    for mode in modes:
        mode_r = [r for r in all_results if r["mode"] == mode and _valid(r.get("speedup", float("nan")))]
        if not mode_r:
            print(f"\n  [{mode}]  No successful results.")
            continue

        if mode == "varlen":
            print(f"\n  [varlen]  ({len(mode_r)} configs)")
            # persistent vs non-persistent
            pvnp = [r["p_vs_np_speedup"] for r in mode_r if _valid(r.get("p_vs_np_speedup", float("nan")))]
            if pvnp:
                print(
                    f"    Persist vs NonPer:      avg={np.mean(pvnp):.2f}x  min={np.min(pvnp):.2f}x  max={np.max(pvnp):.2f}x"
                )
            pvfla_vl = [r["p_vs_fla_vl_speedup"] for r in mode_r if _valid(r.get("p_vs_fla_vl_speedup", float("nan")))]
            if pvfla_vl:
                print(
                    f"    Persist vs FLA varlen:   avg={np.mean(pvfla_vl):.2f}x  min={np.min(pvfla_vl):.2f}x  max={np.max(pvfla_vl):.2f}x  (FAIR)"
                )
            # accuracy
            od = [r["p_vs_np_O_diff"] for r in mode_r if not np.isnan(r.get("p_vs_np_O_diff", float("nan")))]
            if od:
                print(f"    P vs NP O diff:         max={max(od):.2e}  (bit-exact={all(x == 0 for x in od)})")
            ormse = [
                r["p_vs_np_O_rmse_ratio"] * 100 for r in mode_r if not np.isnan(r.get("p_vs_np_O_rmse_ratio", float("nan")))
            ]
            if ormse:
                print(f"    P vs NP O RMSE ratio:   avg={np.mean(ormse):.4f}%  max={np.max(ormse):.4f}%")
        else:
            speedups = [r["speedup"] for r in mode_r]
            print(f"\n  [{mode}]  ({len(mode_r)} configs)")
            print(
                f"    Speedup (CuteDSL/FLA):  avg={np.mean(speedups):.2f}x  min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
            )
            for label, name in [("fla", "FLA"), ("cute", "CuteDSL")]:
                o_rmses = [
                    r[f"{label}_o_rmse_ratio"] * 100
                    for r in mode_r
                    if not np.isnan(r.get(f"{label}_o_rmse_ratio", float("nan")))
                ]
                if o_rmses:
                    print(f"    {name} O RMSE% (vs naive):  avg={np.mean(o_rmses):.4f}  max={np.max(o_rmses):.4f}")
                ht_rmses = [
                    r[f"{label}_ht_rmse_ratio"] * 100
                    for r in mode_r
                    if not np.isnan(r.get(f"{label}_ht_rmse_ratio", float("nan")))
                ]
                if ht_rmses:
                    print(f"    {name} Ht RMSE% (vs naive): avg={np.mean(ht_rmses):.4f}  max={np.max(ht_rmses):.4f}")

    # --- Plot ---
    if args.plot:
        plot_results(all_results, modes)

    # --- Report ---
    if args.report:
        generate_report(all_results, modes, args)

    print()
    return all_results


# =============================================================================
# Plotting
# =============================================================================
def plot_results(all_results, modes):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(9 * n_modes, 7), squeeze=False)
    fig.suptitle("Lightning Attention: CuteDSL vs FLA", fontsize=14, fontweight="bold")

    for col, mode in enumerate(modes):
        ax = axes[0, col]
        mr = [r for r in all_results if r["mode"] == mode]

        if mode == "varlen":
            hr = [
                r
                for r in mr
                if _valid(r.get("persistent_ms", float("nan"))) and _valid(r.get("nonpersistent_ms", float("nan")))
            ]
            if not hr:
                ax.set_title("varlen (no data)")
                continue
            labels = [f"N{r['B']}T{r['T']}\n{r.get('dist', '')[:3]}" for r in hr]
            p_ms = [r["persistent_ms"] for r in hr]
            np_ms = [r["nonpersistent_ms"] for r in hr]
            fla_ms = [r["fla_varlen_ms"] if _valid(r.get("fla_varlen_ms", float("nan"))) else 0 for r in hr]
            x = np.arange(len(labels))
            w = 0.25
            ax.bar(x - w, p_ms, w, label="Persistent", color="tab:green")
            ax.bar(x, np_ms, w, label="Non-Persistent", color="tab:orange")
            ax.bar(x + w, fla_ms, w, label="FLA Triton", color="tab:blue")
        else:
            hr = [r for r in mr if _valid(r.get("speedup", float("nan")))]
            if not hr:
                ax.set_title(f"{mode} (no data)")
                continue
            labels = [f"B{r['B']}T{r['T']}H{r['H']}" for r in hr]
            fla = [r["fla_ms"] for r in hr]
            dsl = [r["cutedsl_ms"] for r in hr]
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, fla, w, label="FLA Triton", color="steelblue")
            ax.bar(x + w / 2, dsl, w, label="CuteDSL", color="orange")

        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Mode: {mode}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_results.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


# =============================================================================
# Markdown report
# =============================================================================
def generate_report(all_results, modes, args):
    from datetime import datetime

    path = os.path.join(os.path.dirname(__file__), "benchmark_report.md")
    with open(path, "w") as f:
        f.write("# Lightning Attention Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Modes: {modes}\n")
        f.write(f"- Batch sizes: {args.batch_sizes}\n")
        f.write(f"- Seq lengths: {args.seq_lens}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Layer: {args.layer_idx}/{args.num_layers}\n")
        f.write(f"- Warmup/Iters: {args.warmup}/{args.iterations}\n\n")

        for mode in modes:
            mr = [r for r in all_results if r["mode"] == mode]
            if not mr:
                continue
            f.write(f"## Mode: {mode}\n\n")

            if mode == "varlen":
                f.write("| N | T | Dist | Persist(ms) | NonPer(ms) | FLA_vl(ms) | P/NP | P/FLAvl | O diff | ht diff |\n")
                f.write("|---|---|------|-------------|------------|------------|------|---------|--------|--------|\n")
                for r in mr:
                    p = f"{r['persistent_ms']:.3f}" if _valid(r.get("persistent_ms", float("nan"))) else "-"
                    np_ = f"{r['nonpersistent_ms']:.3f}" if _valid(r.get("nonpersistent_ms", float("nan"))) else "-"
                    fla_vl = f"{r['fla_varlen_ms']:.3f}" if _valid(r.get("fla_varlen_ms", float("nan"))) else "-"
                    pvnp = f"{r['p_vs_np_speedup']:.2f}x" if _valid(r.get("p_vs_np_speedup", float("nan"))) else "-"
                    pvfla_vl = (
                        f"{r['p_vs_fla_vl_speedup']:.2f}x" if _valid(r.get("p_vs_fla_vl_speedup", float("nan"))) else "-"
                    )
                    od = f"{r['p_vs_np_O_diff']:.1e}" if not np.isnan(r.get("p_vs_np_O_diff", float("nan"))) else "-"
                    hd = f"{r['p_vs_np_ht_diff']:.1e}" if not np.isnan(r.get("p_vs_np_ht_diff", float("nan"))) else "-"
                    f.write(
                        f"| {r['B']} | {r['T']} | {r.get('dist', '')} | {p} | {np_} | {fla_vl} | {pvnp} | {pvfla_vl} | {od} | {hd} |\n"
                    )
            else:
                has_ht = mode == "h0_ht"
                if has_ht:
                    f.write(
                        "| Config | FLA(ms) | CuteDSL(ms) | Speedup | FLA_O_RMSE% | Cute_O_RMSE% | FLA_Ht_RMSE% | Cute_Ht_RMSE% |\n"
                    )
                    f.write(
                        "|--------|---------|-------------|---------|-------------|--------------|--------------|---------------|\n"
                    )
                else:
                    f.write("| Config | FLA(ms) | CuteDSL(ms) | Speedup | FLA_O_RMSE% | Cute_O_RMSE% |\n")
                    f.write("|--------|---------|-------------|---------|-------------|---------------|\n")
                for r in mr:
                    cfg = f"B={r['B']},T={r['T']},H={r['H']}"
                    sp = f"{r['speedup']:.2f}x" if _valid(r.get("speedup", float("nan"))) else "-"
                    fla = f"{r['fla_ms']:.3f}" if _valid(r.get("fla_ms", float("nan"))) else "-"
                    dsl = f"{r['cutedsl_ms']:.3f}" if _valid(r.get("cutedsl_ms", float("nan"))) else "-"
                    fla_o = (
                        f"{r['fla_o_rmse_ratio'] * 100:.3f}%" if not np.isnan(r.get("fla_o_rmse_ratio", float("nan"))) else "-"
                    )
                    cute_o = (
                        f"{r['cute_o_rmse_ratio'] * 100:.3f}%"
                        if not np.isnan(r.get("cute_o_rmse_ratio", float("nan")))
                        else "-"
                    )
                    if has_ht:
                        fla_ht = (
                            f"{r['fla_ht_rmse_ratio'] * 100:.3f}%"
                            if not np.isnan(r.get("fla_ht_rmse_ratio", float("nan")))
                            else "-"
                        )
                        cute_ht = (
                            f"{r['cute_ht_rmse_ratio'] * 100:.3f}%"
                            if not np.isnan(r.get("cute_ht_rmse_ratio", float("nan")))
                            else "-"
                        )
                        f.write(f"| {cfg} | {fla} | {dsl} | {sp} | {fla_o} | {cute_o} | {fla_ht} | {cute_ht} |\n")
                    else:
                        f.write(f"| {cfg} | {fla} | {dsl} | {sp} | {fla_o} | {cute_o} |\n")
            f.write("\n")

        # Summary
        f.write("## Summary\n\n")
        for mode in modes:
            mr = [r for r in all_results if r["mode"] == mode and _valid(r.get("speedup", float("nan")))]
            if not mr:
                continue
            if mode == "varlen":
                pvfla_vl = [r["p_vs_fla_vl_speedup"] for r in mr if _valid(r.get("p_vs_fla_vl_speedup", float("nan")))]
                if pvfla_vl:
                    f.write(
                        f"- **varlen Persist vs FLA varlen (fair)**: avg {np.mean(pvfla_vl):.2f}x, "
                        f"min {np.min(pvfla_vl):.2f}x, max {np.max(pvfla_vl):.2f}x ({len(pvfla_vl)} configs)\n"
                    )
            else:
                speedups = [r["speedup"] for r in mr]
                f.write(
                    f"- **{mode}**: avg {np.mean(speedups):.2f}x, "
                    f"min {np.min(speedups):.2f}x, max {np.max(speedups):.2f}x ({len(mr)} configs)\n"
                )
                for label, name in [("fla", "FLA"), ("cute", "CuteDSL")]:
                    o_rmses = [
                        r[f"{label}_o_rmse_ratio"] * 100
                        for r in mr
                        if not np.isnan(r.get(f"{label}_o_rmse_ratio", float("nan")))
                    ]
                    if o_rmses:
                        f.write(f"  - {name} O RMSE% (vs naive): avg {np.mean(o_rmses):.4f}, max {np.max(o_rmses):.4f}\n")
        f.write("\n---\n*Generated by bench_lightning_attn.py*\n")

    print(f"\nReport saved to {path}")


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified benchmark: Lightning Attention (CuteDSL vs FLA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["no_state", "h0_ht", "varlen"],
        choices=["no_state", "h0_ht", "varlen"],
        help="Modes to benchmark (default: all three)",
    )
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8], help="Batch sizes for standard (non-varlen) modes")
    p.add_argument(
        "--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 32768], help="Sequence lengths for standard modes"
    )
    p.add_argument("--num-heads", type=int, nargs="+", default=[64], help="Number of heads to test")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layer-idx", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--plot", action="store_true", help="Save bar chart PNG")
    p.add_argument("--report", action="store_true", help="Generate markdown report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_suite(args)
