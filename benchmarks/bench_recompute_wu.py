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
bench_recompute_wu.py — Benchmark: CuTe DSL kernel vs FLA Triton baseline
                         for recompute_w_u_fwd (KDA forward)

Compares:
  - Accuracy: max_diff, mean_diff between CuTe DSL and FLA outputs
  - Performance: kernel execution time (ms) with CUDA events

K=128, V=128, BT=64, dtype=bf16.

Usage:
  python benchmarks/bench_recompute_wu.py [--ncu] [--mode fixed|varlen|both]
"""

import argparse
import os

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

import importlib

import torch

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_wu_mod = importlib.import_module("cula.ops.recompute_wu")
recompute_w_u_fwd = _wu_mod.recompute_w_u_fwd
recompute_w_u_fwd_ref = _wu_mod.recompute_w_u_fwd_ref
_make_varlen_test_data = _wu_mod._make_varlen_test_data
build_chunk_indices_wu = _wu_mod.build_chunk_indices_wu

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
    ref_f = ref.double()
    out_f = out.double()
    diff = (ref_f - out_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    return max_diff, mean_diff, rmse, rel_max


def make_inputs(B, T, H, seed=42):
    """Create test inputs for recompute_w_u_fwd."""
    NT = T // BT
    torch.manual_seed(seed)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    gk_raw = -torch.abs(torch.randn(B, T, H, K, device=device, dtype=torch.float32)) * 0.1
    gk = gk_raw.cumsum(dim=1)
    A = torch.tril(torch.randn(B, NT, H, BT, BT, device=device, dtype=dtype) * 0.1).reshape(B, T, H, BT)
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
        w_max, _, w_rmse, w_rel = accuracy_stats(w_fla, w_cute)
        u_max, _, u_rmse, u_rel = accuracy_stats(u_fla, u_cute)
        kg_max, _, kg_rmse, kg_rel = accuracy_stats(kg_fla, kg_cute)

        ok = w_rel < 0.01 and u_rel < 0.01 and kg_rel < 0.01
        status = "PASS" if ok else "FAIL"

        # ---- Performance timing ----
        def run_fla(k=k, v=v, beta=beta, A=A, gk=gk):
            fla_recompute_w_u_fwd(k, v, beta, A, gk=gk)

        def run_cute(k=k, v=v, beta=beta, A=A, gk=gk):
            recompute_w_u_fwd(k, v, beta, A, gk)

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        r = {
            "B": B,
            "T": T,
            "H": H,
            "w_max": w_max,
            "u_max": u_max,
            "kg_max": kg_max,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
        }
        results.append(r)
        print(
            f"  B={B:2d} T={T:5d} H={H:2d} | "
            f"w: rmse={w_rmse:.2e} rel={w_rel:.2e}  "
            f"u: rmse={u_rmse:.2e} rel={u_rel:.2e}  "
            f"kg: rmse={kg_rmse:.2e} rel={kg_rel:.2e} | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
            f"speedup={speedup:.2f}x | {status}"
        )

    return results


# ============================================================
# Reference correctness check
# ============================================================
def check_correctness():
    """Quick correctness check against PyTorch reference."""
    print("\n" + "=" * 80)
    print(" Correctness Check: CuTe DSL vs PyTorch Reference")
    print("=" * 80)

    for B, T, H in [(1, 128, 1), (2, 256, 4), (4, 512, 16), (2, 16384, 64)]:
        k, v, beta, A, gk = make_inputs(B, T, H, seed=123)
        w_ref, u_ref, _, kg_ref = recompute_w_u_fwd_ref(k, v, beta, A, gk)
        w_cute, u_cute, _, kg_cute = recompute_w_u_fwd(k, v, beta, A, gk)
        torch.cuda.synchronize()

        w_max, _, _, _ = accuracy_stats(w_ref, w_cute)
        u_max, _, _, _ = accuracy_stats(u_ref, u_cute)
        kg_max, _, _, _ = accuracy_stats(kg_ref, kg_cute)
        ok = w_max < 1.0 and u_max < 1.0 and kg_max < 1.0
        status = "PASS" if ok else "FAIL"
        print(f"  B={B:2d} T={T:5d} H={H:2d} | w={w_max:.6f} u={u_max:.6f} kg={kg_max:.6f} | {status}")


# ============================================================
# Varlen Benchmark
# ============================================================
def make_varlen_inputs(seq_lens, H, seed=42):
    """Create varlen test inputs (THD format) for recompute_wu."""
    torch.manual_seed(seed)
    k, v, beta, A, gk = _make_varlen_test_data(seq_lens, H, K, V, BT, device=device)
    cu_list = [0]
    for sl in seq_lens:
        cu_list.append(cu_list[-1] + sl)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)
    return k, v, beta, A, gk, cu_seqlens


def bench_varlen():
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fla.ops.utils import prepare_chunk_indices
    from utils import build_varlen_configs

    print("\n" + "=" * 80)
    print(" Varlen Benchmark: CuTe DSL vs FLA Triton — recompute_w_u_fwd (varlen)")
    print("=" * 80)

    H = 64
    configs = build_varlen_configs(
        num_seqs_list=(1, 5, 10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )

    for seq_lens, total_len, dist in configs:
        torch.cuda.empty_cache()
        k, v, beta, A, gk, cu_seqlens = make_varlen_inputs(seq_lens, H)
        N = len(seq_lens)

        # ---- FLA baseline (4D: [1, T_total, H, K]) ----
        k4 = k.unsqueeze(0)
        v4 = v.unsqueeze(0)
        beta4 = beta.unsqueeze(0)
        A4 = A.unsqueeze(0)
        gk4 = gk.unsqueeze(0)
        chunk_indices = prepare_chunk_indices(cu_seqlens.long(), BT)
        w_fla, u_fla, _, kg_fla = fla_recompute_w_u_fwd(
            k4,
            v4,
            beta4,
            A4,
            gk=gk4,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        w_fla = w_fla.squeeze(0)
        u_fla = u_fla.squeeze(0)
        kg_fla = kg_fla.squeeze(0)
        torch.cuda.synchronize()

        # ---- CuTe DSL (3D: [T_total, H, K]) ----
        ci_cute = build_chunk_indices_wu(list(seq_lens), BT=BT, device=k.device)
        w_cute, u_cute, _, kg_cute = recompute_w_u_fwd(
            k,
            v,
            beta,
            A,
            gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=ci_cute,
        )
        torch.cuda.synchronize()

        # ---- Accuracy (mask NaN/inf for meaningful stats) ----
        w_nan = w_cute.isnan().sum().item() + w_cute.isinf().sum().item()
        u_nan = u_cute.isnan().sum().item() + u_cute.isinf().sum().item()
        kg_nan = kg_cute.isnan().sum().item() + kg_cute.isinf().sum().item()

        def _masked_stats(ref, out):
            mask = ref.isfinite() & out.isfinite()
            if mask.sum() == 0:
                return float("nan"), float("nan")
            r, o = ref[mask].double(), out[mask].double()
            diff = (r - o).abs()
            rmse = diff.pow(2).mean().sqrt().item()
            denom = r.abs().max().item()
            rel = diff.max().item() / denom if denom > 0 else 0.0
            return rmse, rel

        w_rmse, w_rel = _masked_stats(w_fla, w_cute)
        u_rmse, u_rel = _masked_stats(u_fla, u_cute)
        kg_rmse, kg_rel = _masked_stats(kg_fla, kg_cute)

        total_nan = w_nan + u_nan + kg_nan
        ok = total_nan == 0 and w_rel < 0.01 and u_rel < 0.01 and kg_rel < 0.01
        status = "PASS" if ok else "FAIL"

        # ---- Performance timing ----
        def run_fla(k4=k4, v4=v4, beta4=beta4, A4=A4, gk4=gk4, cu=cu_seqlens, ci=chunk_indices):
            fla_recompute_w_u_fwd(k4, v4, beta4, A4, gk=gk4, cu_seqlens=cu, chunk_indices=ci)

        def run_cute(k=k, v=v, beta=beta, A=A, gk=gk, cu=cu_seqlens, ci=ci_cute):
            recompute_w_u_fwd(k, v, beta, A, gk, cu_seqlens=cu, chunk_indices=ci)

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        nan_str = f" NaN={total_nan}" if total_nan > 0 else ""
        print(
            f"  N={N:3d} T={total_len:5d} dist={dist:8s} H={H:2d} | "
            f"w: rmse={w_rmse:.2e} rel={w_rel:.2e}  "
            f"u: rmse={u_rmse:.2e} rel={u_rel:.2e}  "
            f"kg: rmse={kg_rmse:.2e} rel={kg_rel:.2e} | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
            f"speedup={speedup:.2f}x | {status}{nan_str}"
        )


# ============================================================
# Main
# ============================================================
def main():
    global NCU_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode (warmup=1, iters=1)")
    parser.add_argument("--mode", choices=["fixed", "varlen", "both"], default="both")
    args = parser.parse_args()
    NCU_MODE = args.ncu

    # Correctness check
    check_correctness()

    # Benchmark configs: (B, T, H)
    if args.mode in ("fixed", "both"):
        configs = [
            (2, 4096, 64),
            (2, 8192, 64),
            (4, 4096, 64),
            (4, 8192, 64),
            (2, 16384, 64),
            (4, 16384, 64),
        ]
        bench(configs)

    if args.mode in ("varlen", "both"):
        bench_varlen()


if __name__ == "__main__":
    main()
