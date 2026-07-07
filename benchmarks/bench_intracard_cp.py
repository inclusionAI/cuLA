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
bench_intracard_cp.py — Benchmark: Intracard Context-Parallel speedup
                        for chunk_kda (KDA forward)

Measures the speedup of cuLA's intracard context-parallel path against the
non-CP baseline across a range of varlen configurations.  Also verifies that
the heuristic does not regress throughput when CP is correctly bypassed.

Usage:
  python bench_intracard_cp.py [--ncu] [--sanitizer]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_intracard_cp.py --ncu
"""

import argparse
import contextlib
import os
import pathlib
import sys

os.environ.setdefault("CULA_INTRACARD_CP", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from benchmarks.utils import (
    SEED,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    set_seed,
)
from cula.kda.chunk_fwd import chunk_kda_fwd
from cula.ops.kda.sm100.cp.chunk_delta_h import (
    compute_subseq_len,
    prepare_subseq_cu_seqlens,
    should_use_intracard_cp,
)
from cula.utils import get_device_sm_count

# ============================================================
# Constants
# ============================================================
BT, D = 64, 128
H_VALUES = [4, 8]
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False

# (tag, seq_lens) — each entry is tested at every H in H_VALUES
CONFIGS = [
    # --- single seq (ascending length) ---
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("T=128K", [131072]),
    # --- equal-length batches (~32K total) ---
    ("8x4K", [4096] * 8),
    ("4x8K", [8192] * 4),
    ("2x16K", [16384] * 2),
    # --- asymmetric multi-seq ---
    ("16K+16K", [16384, 16384]),
    ("24K+8K", [24576, 8192]),
    ("28K+4K", [28672, 4096]),
    ("32K+256+256", [32768, 256, 256]),
    ("40K+1K+8K", [40960, 1024, 8192]),
    ("64K+512+256+128", [65536, 512, 256, 128]),
    ("128K+1K", [131072, 1024]),
    # --- 128K + several short seqs ---
    ("128K+2x1K", [131072, 1024, 1024]),
    ("128K+5x1K", [131072] + [1024] * 5),
    ("128K+10x1K", [131072] + [1024] * 10),
]


# ============================================================
# CP toggle
# ============================================================
@contextlib.contextmanager
def cp_on(enable: bool):
    old = os.environ.get("CULA_INTRACARD_CP")
    os.environ["CULA_INTRACARD_CP"] = "1" if enable else "0"
    try:
        if enable:
            with torch.inference_mode():
                yield
        else:
            yield
    finally:
        if old is None:
            os.environ.pop("CULA_INTRACARD_CP", None)
        else:
            os.environ["CULA_INTRACARD_CP"] = old


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


def run_cp(q, k, v, g, beta, scale, A_log, dt_bias, cu_seqlens, lower_bound, *, enable_cp):
    with cp_on(enable_cp):
        chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens.cpu(),
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )


def predict_cp(seq_lens, H, num_sms):
    cu = torch.tensor(
        exclusive_cumsum(seq_lens),
        dtype=torch.int32,
    )
    if not should_use_intracard_cp(cu, num_sms, H, BT):
        return False, 0
    max_len = int(torch.diff(cu).max().item())
    subseq_len = compute_subseq_len(max_len, num_sms, H, BT, num_seqs=len(seq_lens))
    _, split_info, total_subseqs = prepare_subseq_cu_seqlens(cu, subseq_len, BT)
    return bool(split_info), total_subseqs


# ============================================================
# Benchmark
# ============================================================
def bench_cp(h_values, configs):
    print("\n" + "=" * 100)
    print(" Intracard CP Benchmark: CP-on vs CP-off")
    print("=" * 100)

    device = torch.device("cuda")
    num_sms = get_device_sm_count(device)
    results = []

    for H in h_values:
        for tag, seq_lens in configs:
            set_seed(SEED)
            torch.cuda.empty_cache()

            total_T = sum(seq_lens)
            cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
            inputs = prepare_safe_gate_inputs(1, total_T, H, D, device, cu_seqlens=cu_seqlens, seed=SEED)
            q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
            A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
            scale, lower_bound = inputs["scale"], inputs["lower_bound"]

            pred, n_sub = predict_cp(seq_lens, H, num_sms)

            common = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                lower_bound=lower_bound,
            )

            ms_off = time_kernel(lambda: run_cp(**common, enable_cp=False))
            ms_on = time_kernel(lambda: run_cp(**common, enable_cp=True))

            r = {
                "tag": tag,
                "H": H,
                "total_T": total_T,
                "pred": pred,
                "n_sub": n_sub,
                "ms_off": ms_off,
                "ms_on": ms_on,
                "speedup": ms_off / ms_on if ms_on > 0 else float("inf"),
            }
            results.append(r)

            del q, k, v, g, beta, A_log, dt_bias, inputs
            torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(results, h_values):
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("                       BENCHMARK REPORT: Intracard CP")
    print("                       CP-on vs CP-off (same kernel, different code paths)")
    print(f"                       D={D}  dtype=bf16  safe_gate=True")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                       Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    for H_val in h_values:
        h_results = [r for r in results if r["H"] == H_val]
        if not h_results:
            continue

        print(f"\n  [H={H_val}]")
        print(f"  {'─' * 95}")
        print(
            f"  {'config':<24s} {'T':>7s}  {'pred':>4s} {'sub':>4s}"
            f"  │  {'CP_off(ms)':>10s}  {'CP_on(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 95}")
        for r in h_results:
            pred_s = "Y" if r["pred"] else "N"
            print(
                f"  {r['tag']:<24s} {r['total_T']:>7d}     {pred_s}  {r['n_sub']:>4d}"
                f"  │  {r['ms_off']:>10.4f}  {r['ms_on']:>10.4f}  {r['speedup']:>7.2f}x"
            )
        print(f"  {'─' * 95}")

    # Summary
    triggered = [r for r in results if r["pred"]]
    bypassed = [r for r in results if not r["pred"]]

    if triggered:
        speedups = [r["speedup"] for r in triggered]
        geo = 1.0
        for s in speedups:
            geo *= s
        geo = geo ** (1 / len(speedups))
        print(
            f"\n  CP triggered ({len(triggered)} configs): "
            f"geo-mean={geo:.2f}x  best={max(speedups):.2f}x  worst={min(speedups):.2f}x"
        )

    if bypassed:
        ratios = [r["ms_on"] / r["ms_off"] for r in bypassed]
        print(
            f"  CP bypassed  ({len(bypassed)} configs): "
            f"mean overhead={sum(ratios) / len(ratios):.3f}x  max={max(ratios):.3f}x  "
            f"(1.00 = no regression)"
        )

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_intracard_cp: CP-on vs CP-off benchmark")
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="Sanitizer mode: warmup=1, iters=1",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")

    results = bench_cp(H_VALUES, CONFIGS)
    print_report(results, H_VALUES)
    return results


if __name__ == "__main__":
    main()
