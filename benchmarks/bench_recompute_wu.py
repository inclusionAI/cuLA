#!/usr/bin/env python3
"""
bench_recompute_wu.py — Benchmark: CuTe DSL kernel vs FLA Triton baseline
                         for recompute_w_u_fwd (KDA forward)

Compares:
  - Accuracy: max_diff, mean_diff between CuTe DSL and FLA outputs
  - Performance: kernel execution time (ms) with CUDA events

K=128, V=128, BT=64, dtype=bf16.

Usage:
  python benchmarks/bench_recompute_wu.py [--ncu]
"""

import argparse
import os

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import importlib

import torch

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_wu_mod = importlib.import_module("cula.ops.recompute_wu")
recompute_w_u_fwd = _wu_mod.recompute_w_u_fwd
recompute_w_u_fwd_ref = _wu_mod.recompute_w_u_fwd_ref

# ─── FLA baseline imports ───
from fla.ops.kda.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd  # noqa: E402

# ============================================================
# Constants
# ============================================================
K, V, BT = 128, 128, 64
dtype = torch.bfloat16
device = "cuda"

WARMUP = 10
N_ITERS = 100
NCU_MODE = False


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if n_iters is None:
        n_iters = 1 if NCU_MODE else N_ITERS
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
    diff = (ref.float() - out.float()).abs()
    return diff.max().item(), diff.mean().item()


def make_inputs(B, T, H, seed=42):
    """Create test inputs for recompute_w_u_fwd."""
    NT = T // BT
    torch.manual_seed(seed)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    gk_raw = -torch.abs(torch.randn(B, T, H, K, device=device, dtype=torch.float32)) * 0.1
    gk = gk_raw.cumsum(dim=1)
    A = torch.tril(
        torch.randn(B, NT, H, BT, BT, device=device, dtype=dtype) * 0.1
    ).reshape(B, T, H, BT)
    return k, v, beta, A, gk


# ============================================================
# Benchmark
# ============================================================
def bench(configs):
    print("\n" + "=" * 80)
    print(" Benchmark: CuTe DSL (SM100a) vs FLA Triton — recompute_w_u_fwd")
    print("=" * 80)
    results = []

    for B, T, H in configs:
        torch.cuda.empty_cache()
        k, v, beta, A, gk = make_inputs(B, T, H)

        # ---- FLA baseline (accuracy reference) ----
        w_fla, u_fla, _, kg_fla = fla_recompute_w_u_fwd(k, v, beta, A, gk=gk)
        torch.cuda.synchronize()

        # ---- CuTe DSL ----
        w_cute, u_cute, _, kg_cute = recompute_w_u_fwd(k, v, beta, A, gk)
        torch.cuda.synchronize()

        # ---- Accuracy ----
        w_max, w_mean = accuracy_stats(w_fla, w_cute)
        u_max, u_mean = accuracy_stats(u_fla, u_cute)
        kg_max, kg_mean = accuracy_stats(kg_fla, kg_cute)

        # ---- Performance timing ----
        def run_fla(k=k, v=v, beta=beta, A=A, gk=gk):
            fla_recompute_w_u_fwd(k, v, beta, A, gk=gk)

        def run_cute(k=k, v=v, beta=beta, A=A, gk=gk):
            recompute_w_u_fwd(k, v, beta, A, gk)

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float('inf')

        r = {
            'B': B, 'T': T, 'H': H,
            'w_max': w_max, 'u_max': u_max, 'kg_max': kg_max,
            'ms_fla': ms_fla, 'ms_cute': ms_cute, 'speedup': speedup,
        }
        results.append(r)
        print(f"  B={B:2d} T={T:5d} H={H:2d} | "
              f"w_diff={w_max:.4f} u_diff={u_max:.4f} kg_diff={kg_max:.6f} | "
              f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
              f"speedup={speedup:.2f}x")

    return results


# ============================================================
# Reference correctness check
# ============================================================
def check_correctness():
    """Quick correctness check against PyTorch reference."""
    print("\n" + "=" * 80)
    print(" Correctness Check: CuTe DSL vs PyTorch Reference")
    print("=" * 80)

    for B, T, H in [(1, 128, 1), (2, 256, 4), (4, 512, 16)]:
        k, v, beta, A, gk = make_inputs(B, T, H, seed=123)
        w_ref, u_ref, _, kg_ref = recompute_w_u_fwd_ref(k, v, beta, A, gk)
        w_cute, u_cute, _, kg_cute = recompute_w_u_fwd(k, v, beta, A, gk)
        torch.cuda.synchronize()

        w_max, _ = accuracy_stats(w_ref, w_cute)
        u_max, _ = accuracy_stats(u_ref, u_cute)
        kg_max, _ = accuracy_stats(kg_ref, kg_cute)
        ok = w_max < 1.0 and u_max < 1.0 and kg_max < 1.0
        status = "PASS" if ok else "FAIL"
        print(f"  B={B:2d} T={T:4d} H={H:2d} | "
              f"w={w_max:.6f} u={u_max:.6f} kg={kg_max:.6f} | {status}")


# ============================================================
# Main
# ============================================================
def main():
    global NCU_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode (warmup=1, iters=1)")
    args = parser.parse_args()
    NCU_MODE = args.ncu

    # Correctness check
    check_correctness()

    # Benchmark configs: (B, T, H)
    configs = [
        (2, 4096, 64),
        (2, 8192, 64),
        (4, 4096, 64),
        (4, 8192, 64),
        (2, 16384, 64),
        (4, 16384, 64),
    ]
    bench(configs)


if __name__ == "__main__":
    main()
