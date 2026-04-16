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
bench_kda_fwd_bwd_e2e.py — Benchmark: cuLA CuTe DSL vs FLA Triton baseline
                            for chunk_kda forward + backward (end-to-end)

Compares:
  - Accuracy: err_ratio, relative max diff between cuLA and FLA outputs & gradients
  - Performance: kernel execution time (ms) with CUDA events

Modes:
  - Fixed-length: B=1, B=2 with various T
  - Varlen: ~20 seqs with 2-3x length variation

Phases:
  - forward: forward pass only
  - e2e: forward + backward (end-to-end)

Usage:
  python bench_kda_fwd_bwd_e2e.py [--mode fixed|varlen|both] [--phase forward|e2e] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_fwd_bwd_e2e.py --mode varlen --ncu
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    exclusive_cumsum,
    generate_random_seq_lens,
    prepare_safe_gate_inputs,
    set_seed,
)
from cula.kda import chunk_kda as cula_chunk_kda

# ============================================================
# Constants
# ============================================================
H, D = 64, 128
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False
DISABLE_RECOMPUTE = False
PHASE = "e2e"  # "forward" or "e2e"


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    if n_iters is None:
        n_iters = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / n_iters


def accuracy_stats(ref, out):
    """Compute err_ratio, relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    err = diff.flatten().pow(2).mean().sqrt().item()
    base = ref_f.flatten().pow(2).mean().sqrt().item()
    err_ratio = err / (base + 1e-8)
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return err_ratio, rel_max, mean_diff


def run_kda_e2e(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound, do, dht, fn):
    """Run KDA forward (+ backward if PHASE == 'e2e').

    Clears gradients, runs forward, optionally backward.
    """
    q.grad = None
    k.grad = None
    v.grad = None
    g.grad = None
    beta.grad = None
    init_state.grad = None

    out, ht = fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        disable_recompute=DISABLE_RECOMPUTE,
    )
    if PHASE == "e2e":
        out.backward(do)
    return out, ht


def run_kda_e2e_with_grads(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound, do, dht, fn):
    """Run KDA forward + backward and return outputs + gradients for accuracy check."""
    q_c = q.detach().clone().requires_grad_(True)
    k_c = k.detach().clone().requires_grad_(True)
    v_c = v.detach().clone().requires_grad_(True)
    g_c = g.detach().clone().requires_grad_(True)
    b_c = beta.detach().clone().requires_grad_(True)
    h_c = init_state.detach().clone().requires_grad_(True)

    out, ht = fn(
        q=q_c,
        k=k_c,
        v=v_c,
        g=g_c,
        beta=b_c,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h_c,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        disable_recompute=DISABLE_RECOMPUTE,
    )
    loss = (out * do).sum() + (ht * dht).sum()
    loss.backward()

    return dict(
        o=out,
        ht=ht,
        dq=q_c.grad,
        dk=k_c.grad,
        dv=v_c.grad,
        dg=g_c.grad,
        dbeta=b_c.grad,
        dh0=h_c.grad,
    )


# ============================================================
# Determinism check
# ============================================================
def check_determinism(num_seqs=5, T=512, iters=20):
    """Verify that cuLA chunk_kda produces identical outputs across repeated runs."""
    device = torch.device("cuda")
    set_seed(SEED)

    seq_lens = generate_random_seq_lens(num_seqs, T, 63, seed=SEED)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=True)
    q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
    A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
    scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

    set_seed(SEED + 1)
    do = torch.randn_like(v)
    dht = torch.randn_like(init_state)

    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        init_state=init_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        do=do,
        dht=dht,
    )

    ref = run_kda_e2e_with_grads(**common, fn=cula_chunk_kda)
    for i in range(iters):
        out = run_kda_e2e_with_grads(**common, fn=cula_chunk_kda)
        for name in ("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0"):
            assert torch.equal(out[name], ref[name]), f"[determinism] cuLA {name} mismatch at iter {i}"
    return True


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 120)
    print(f" Fixed-Length E2E Benchmark: cuLA vs FLA  phase={PHASE}  disable_recompute={DISABLE_RECOMPUTE}")
    print("=" * 120)
    results = []

    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=True)
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        # Generate do, dht for backward
        set_seed(SEED + 1)
        do = torch.randn_like(v)
        dht = torch.randn_like(init_state)

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            do=do,
            dht=dht,
        )

        # Accuracy: compare outputs and gradients
        acc = {}
        if PHASE == "e2e":
            fla_results = run_kda_e2e_with_grads(**common, fn=fla_chunk_kda)
            cula_results = run_kda_e2e_with_grads(**common, fn=cula_chunk_kda)
            torch.cuda.synchronize()

            for name in ("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0"):
                err_ratio, rel_max, mean_diff = accuracy_stats(fla_results[name], cula_results[name])
                acc[name] = {"err_ratio": err_ratio, "rel_max": rel_max, "mean_diff": mean_diff}
        else:
            # forward-only accuracy
            o_fla, ht_fla = run_kda_e2e(**common, fn=fla_chunk_kda)
            o_cula, ht_cula = run_kda_e2e(**common, fn=cula_chunk_kda)
            torch.cuda.synchronize()
            for name, ref, out in [("o", o_fla, o_cula), ("ht", ht_fla, ht_cula)]:
                err_ratio, rel_max, mean_diff = accuracy_stats(ref, out)
                acc[name] = {"err_ratio": err_ratio, "rel_max": rel_max, "mean_diff": mean_diff}

        # For timing, use leaf tensors with requires_grad
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        g_t = g.detach().clone().requires_grad_(True)
        beta_t = beta.detach().clone().requires_grad_(True)
        h0_t = init_state.detach().clone().requires_grad_(True)

        timing_common = dict(
            q=q_t,
            k=k_t,
            v=v_t,
            g=g_t,
            beta=beta_t,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=h0_t,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            do=do,
            dht=dht,
        )

        def fn_fla(**kw):
            return lambda: run_kda_e2e(**kw, fn=fla_chunk_kda)

        def fn_cula(**kw):
            return lambda: run_kda_e2e(**kw, fn=cula_chunk_kda)

        ms_fla = time_kernel(fn_fla(**timing_common))
        ms_cula = time_kernel(fn_cula(**timing_common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        r = {
            "B": B,
            "T": T,
            "accuracy": acc,
            "ms_fla": ms_fla,
            "ms_cula": ms_cula,
            "speedup": speedup,
        }
        results.append(r)

        del q, k, v, g, beta, A_log, dt_bias, inputs, do, dht
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 120)
    print(f" Varlen E2E Benchmark: cuLA vs FLA  phase={PHASE}  disable_recompute={DISABLE_RECOMPUTE}")
    print("=" * 120)
    results = []

    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=True)
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        # Generate do, dht for backward
        set_seed(SEED + 1)
        do = torch.randn_like(v)
        dht = torch.randn_like(init_state)

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            do=do,
            dht=dht,
        )

        # Accuracy: compare outputs and gradients
        acc = {}
        if PHASE == "e2e":
            fla_results = run_kda_e2e_with_grads(**common, fn=fla_chunk_kda)
            cula_results = run_kda_e2e_with_grads(**common, fn=cula_chunk_kda)
            torch.cuda.synchronize()

            for name in ("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0"):
                err_ratio, rel_max, mean_diff = accuracy_stats(fla_results[name], cula_results[name])
                acc[name] = {"err_ratio": err_ratio, "rel_max": rel_max, "mean_diff": mean_diff}
        else:
            o_fla, ht_fla = run_kda_e2e(**common, fn=fla_chunk_kda)
            o_cula, ht_cula = run_kda_e2e(**common, fn=cula_chunk_kda)
            torch.cuda.synchronize()
            for name, ref, out in [("o", o_fla, o_cula), ("ht", ht_fla, ht_cula)]:
                err_ratio, rel_max, mean_diff = accuracy_stats(ref, out)
                acc[name] = {"err_ratio": err_ratio, "rel_max": rel_max, "mean_diff": mean_diff}

        # For timing, use leaf tensors with requires_grad
        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        v_t = v.detach().clone().requires_grad_(True)
        g_t = g.detach().clone().requires_grad_(True)
        beta_t = beta.detach().clone().requires_grad_(True)
        h0_t = init_state.detach().clone().requires_grad_(True)

        timing_common = dict(
            q=q_t,
            k=k_t,
            v=v_t,
            g=g_t,
            beta=beta_t,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=h0_t,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
            do=do,
            dht=dht,
        )

        def fn_fla(**kw):
            return lambda: run_kda_e2e(**kw, fn=fla_chunk_kda)

        def fn_cula(**kw):
            return lambda: run_kda_e2e(**kw, fn=cula_chunk_kda)

        ms_fla = time_kernel(fn_fla(**timing_common))
        ms_cula = time_kernel(fn_cula(**timing_common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        r = {
            "tag": tag,
            "dist": dist,
            "T_total": T,
            "n_seqs": n_seqs,
            "accuracy": acc,
            "ms_fla": ms_fla,
            "ms_cula": ms_cula,
            "speedup": speedup,
        }
        results.append(r)

        del q, k, v, g, beta, A_log, dt_bias, inputs, do, dht
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    sep = "=" * 130
    print(f"\n\n{sep}")
    print("                       BENCHMARK REPORT: chunk_kda forward+backward (E2E)")
    print("                       cuLA CuTe DSL vs FLA Triton")
    print(
        f"                       H={H}  D={D}  dtype=bf16  safe_gate=True  phase={PHASE}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                       Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    # Determine which accuracy keys to show
    if PHASE == "e2e":
        acc_keys = ["o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0"]
    else:
        acc_keys = ["o", "ht"]

    acc_header = "  ".join(f"{k:>10s}" for k in acc_keys)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 125}")

        # Header
        print(f"  {'B':>3s}  {'T':>5s}  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>11s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 125}")

        for r in fixed_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in acc_keys)
            err_ratio_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('err_ratio', 0.0):10.6f}" for k in acc_keys)
            # Line 1: timing + rel_max
            print(
                f"  {r['B']:3d}  {r['T']:5d}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:11.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            # Line 2: err_ratio (no timing columns)
            print(f"  {'':3s}  {'':5s}  │  {'':9s}  {'':11s}  {'':8s}  │  {'err_ratio:':>10s}{err_ratio_vals}")
        print(f"  {'─' * 125}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 140}")

        print(f"  {'Config':>45s}  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>11s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 140}")

        for r in varlen_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in acc_keys)
            err_ratio_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('err_ratio', 0.0):10.6f}" for k in acc_keys)
            # Line 1: timing + rel_max
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:11.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            # Line 2: err_ratio (no config/timing columns)
            print(f"  {'':>45s}  │  {'':9s}  {'':11s}  {'':8s}  │  {'err_ratio:':>10s}{err_ratio_vals}")
        print(f"  {'─' * 140}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_kda_fwd_bwd_e2e: cuLA vs FLA (forward + backward)")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="e2e",
        choices=["forward", "e2e"],
        help="Benchmark phase: forward only or end-to-end (default: e2e)",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="Sanitizer mode: warmup=1, iters=1 (avoid Triton memory leak under compute-sanitizer)",
    )
    parser.add_argument(
        "--disable_recompute",
        action="store_true",
        help="Disable recompute in both FLA and cuLA (pre-compute QG)",
    )
    parser.add_argument(
        "--check_determinism",
        action="store_true",
        help="Run determinism check: verify cuLA produces identical outputs across repeated runs",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE, DISABLE_RECOMPUTE, PHASE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")
    if args.disable_recompute:
        DISABLE_RECOMPUTE = True
        print("[Disable recompute] pre-compute QG in forward")
    PHASE = args.phase

    if args.check_determinism:
        det_configs = [(5, 1024), (10, 4096), (10, 8192), (10, 16384)]
        print("\n[Determinism Check] cuLA chunk_kda E2E ...")
        for num_seqs, T in det_configs:
            result = check_determinism(num_seqs=num_seqs, T=T, iters=20)
            print(f"  num_seqs={num_seqs}  T={T:5d}  iters=20  {'PASS' if result else 'FAIL'}")
        print("[Determinism Check] All passed.\n")
        return

    fixed_configs = [
        # (B, T)
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (2, 512),
        (2, 1024),
        (2, 4096),
        (2, 8192),
        (2, 16384),
    ]

    varlen_configs = build_varlen_configs(
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )

    fixed_res, varlen_res = [], []

    if args.mode in ("fixed", "both"):
        fixed_res = bench_fixed(fixed_configs)

    if args.mode in ("varlen", "both"):
        varlen_res = bench_varlen(varlen_configs)

    print_report(fixed_res, varlen_res)
    return fixed_res, varlen_res


if __name__ == "__main__":
    main()
