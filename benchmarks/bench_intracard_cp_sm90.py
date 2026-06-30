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

"""bench_intracard_cp_sm90.py — CP-on vs CP-off (vs FLA baseline) for SM90 KDA prefill.

Mirrors benchmarks/bench_intracard_cp.py (SM100 version) but for the Hopper
(SM90) path:

    CP_on  : cula.kda.kda_prefill_hopper_auto
    CP_off : cula.kda.kda_prefill_hopper
    FLA    : fla.ops.kda.chunk_kda (Triton baseline)

Reports per-config `pred` (would CP fire?) and `n_sub` (CP-chunk count). When
`pred=N` we still measure CP_on to confirm the bypass adds no regression. The
`CP_on/FLA` column shows the speedup of cuLA's optimized (CP-on) kernel over
the FLA Triton baseline.

Usage:
  python benchmarks/bench_intracard_cp_sm90.py [--ncu] [--sanitizer]
"""

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))
os.environ.setdefault("FLA_INTRACARD_CP", "1")

from fla.ops.common.intracard_cp import compute_subseq_len, prepare_subseq_cu_seqlens
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    set_seed,
    time_cuda_fn,
)
from cula.kda import kda_prefill_hopper, kda_prefill_hopper_auto
from cula.kda.auto_route import _should_use_opt
from cula.kda.cp_context import _calc_cp_seqs, is_dominant_long_seq
from cula.kda.hopper_fused_fwd_opt import FUSED_GATE_L2NORM_VARLEN_AVG_SEQ, _fused_gate_l2norm_threshold
from cula.utils import get_device_sm_count

# ============================================================
# Constants
# ============================================================
BT, D = 64, 128
H_VALUES = [4, 8]
WARMUP = 10
N_ITERS = 10
NCU_MODE = False
SANITIZER_MODE = False

# (tag, seq_lens) — varlen configs, run with cu_seqlens=cumsum(seq_lens)
CONFIGS = [
    # small varlen — exercises fused gate+l2norm path (packed_T*H <= 65536)
    ("4x256", [256] * 4),
    ("8x256", [256] * 8),
    ("16x256", [256] * 16),
    ("4x1K", [1024] * 4),
    ("8x1K", [1024] * 8),
    ("4x2K", [2048] * 4),
    ("1K+512+256+128", [1024, 512, 256, 128]),
    ("2K+1K+512+256", [2048, 1024, 512, 256]),
    ("1K+1+63+65+129", [1024, 1, 63, 65, 129]),
    # single seq
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("T=128K", [131072]),
    # equal-length batches (~32K total)
    ("8x4K", [4096] * 8),
    ("4x8K", [8192] * 4),
    ("2x16K", [16384] * 2),
    # asymmetric multi-seq
    ("16K+16K", [16384, 16384]),
    ("24K+8K", [24576, 8192]),
    ("28K+4K", [28672, 4096]),
    ("32K+256+256", [32768, 256, 256]),
    ("40K+1K+8K", [40960, 1024, 8192]),
    ("64K+512+256+128", [65536, 512, 256, 128]),
    ("128K+1K", [131072, 1024]),
    ("128K+2x1K", [131072, 1024, 1024]),
    ("128K+5x1K", [131072] + [1024] * 5),
    ("128K+10x1K", [131072] + [1024] * 10),
]


# ============================================================
# Helpers
# ============================================================
def _bench_warmup_iters():
    warmup = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    n_iters = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    return warmup, n_iters


def run_call(q, k, v, g, beta, scale, A_log, dt_bias, cu_seqlens, lower_bound, *, enable_cp, return_state=False):
    fn = kda_prefill_hopper_auto if enable_cp else kda_prefill_hopper
    out = fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=return_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return out


def run_fla_call(q, k, v, g, beta, scale, A_log, dt_bias, cu_seqlens, lower_bound, *, return_state=False):
    # FLA's chunk_kda fuses A_log + dt_bias internally when use_gate_in_kernel=True; pass them via kwargs.
    # Wrap in inference_mode so FLA's IntraCardCPBackend gate (intracard.py) activates.
    with torch.inference_mode():
        return fla_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=return_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=False,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
        )


def accuracy(ref, got):
    if ref is None or got is None:
        return float("nan"), float("nan")
    diff = (ref.float() - got.float()).abs()
    return diff.max().item(), diff.mean().item()


def predict_cp(seq_lens, H, num_sms, device):
    cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    raw_batch = len(seq_lens)
    packed_seq = sum(seq_lens)

    if raw_batch > 1:
        cp_wf = (raw_batch * H <= 16 and packed_seq >= 8192) or (
            packed_seq >= 8192 and H <= 16 and is_dominant_long_seq(seq_lens, H)
        )
    else:
        cp_wf = (H <= 8 and packed_seq >= 4096) or (H <= 16 and packed_seq >= 4096) or (H <= 32 and packed_seq >= 16384)
    if not cp_wf:
        return False, 0

    use_cp, cp_cu, *_ = _calc_cp_seqs(cu, BT, H, num_sms, raw_cu_seqlens_cpu=cu.cpu())
    if not use_cp:
        return False, 0
    n_sub = int(cp_cu.numel() - 1)
    if n_sub == raw_batch:  # no-op split
        return False, 0
    return True, n_sub


def predict_fla_cp(seq_lens, H, num_sms):
    """Mirror fla.ops.common.intracard_cp.intracard_fwd_h gating to predict
    whether FLA's intracard CP fires and how many sub-sequences result.
    HV (num_v_heads) maps to H here."""
    cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)
    seq_lens_t = torch.diff(cu)
    max_seq_len = int(seq_lens_t.max().item())
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, BT)
    if (seq_lens_t < 2 * subseq_len).all():
        return False, 0
    _, split_info, total_subseqs = prepare_subseq_cu_seqlens(cu, subseq_len, BT)
    if not split_info:
        return False, 0
    return True, total_subseqs


def predict_fused_all_pre(q, v, cu_seqlens_for_opt, *, cu_seqlens_is_none, use_gate_in_kernel, use_qk_l2norm_in_kernel):
    if not _should_use_opt(q, cu_seqlens_for_opt):
        return False
    num_qk_heads = q.shape[-2]
    num_v_heads = v.shape[-2]
    if cu_seqlens_is_none:
        avg_seq_ok = True
    else:
        N = cu_seqlens_for_opt.numel() - 1
        packed_T = q.shape[1]
        avg_seq_ok = N <= 1 or packed_T <= N * FUSED_GATE_L2NORM_VARLEN_AVG_SEQ
    return (
        use_gate_in_kernel
        and use_qk_l2norm_in_kernel
        and (q.numel() // q.shape[-1]) <= _fused_gate_l2norm_threshold(cu_seqlens_is_none)
        and num_qk_heads == num_v_heads
        and avg_seq_ok
    )


# ============================================================
# Benchmark
# ============================================================
SEP = "  " + "─" * 180
ROW_HEADER = (
    f"  {'config':<24s} {'T':>7s}  {'pred':>4s} {'sub':>4s} {'fla_cp':>6s} {'fla_sub':>7s} {'fused_pre':>5s}"
    f"  │  {'o max/mean':>17s}  {'ht max/mean':>17s}"
    f"  │  {'FLA(ms)':>9s}  {'CP_off(ms)':>10s}  {'CP_on(ms)':>10s}  {'CP_on/off':>8s}  {'CP_on/FLA':>9s}"
)


def _format_row(r):
    pred_s = "Y" if r["pred"] else "N"
    fla_pred_s = "Y" if r["fla_pred"] else "N"
    fused_s = "Y" if r["fused_all_pre"] else "N"
    return (
        f"  {r['tag']:<24s} {r['total_T']:>7d}     {pred_s}  {r['n_sub']:>4d}      {fla_pred_s}    {r['fla_n_sub']:>4d}     {fused_s}"
        f"  │  {r['o_max']:>7.1e}/{r['o_mean']:>7.1e}  {r['ht_max']:>7.1e}/{r['ht_mean']:>7.1e}"
        f"  │  {r['ms_fla']:>9.4f}  {r['ms_off']:>10.4f}  {r['ms_on']:>10.4f}"
        f"  {r['speedup']:>7.2f}x  {r['speedup_vs_fla']:>8.2f}x"
    )


def bench_cp(h_values, configs):
    print("\n" + "=" * 110)
    print("                       BENCHMARK REPORT: Intracard CP (SM90)")
    print("                       CP-on (kda_prefill_hopper_auto) vs CP-off (kda_prefill_hopper) vs FLA (chunk_kda)")
    print(f"                       D={D}  dtype=bf16  safe_gate=True")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                       Warmup={wu}  Iters={ni}{mode_tag}")
    print("=" * 110)

    device = torch.device("cuda")
    num_sms = get_device_sm_count(device)
    results = []

    for H in h_values:
        print(f"\n  [H={H}]", flush=True)
        print(SEP, flush=True)
        print(ROW_HEADER, flush=True)
        print(SEP, flush=True)

        for tag, seq_lens in configs:
            set_seed(SEED)
            torch.cuda.empty_cache()

            total_T = sum(seq_lens)
            cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
            inputs = prepare_safe_gate_inputs(1, total_T, H, D, device, cu_seqlens=cu_seqlens, seed=SEED)
            q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
            A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
            scale, lower_bound = inputs["scale"], inputs["lower_bound"]

            pred, n_sub = predict_cp(seq_lens, H, num_sms, device)
            fla_pred, fla_n_sub = predict_fla_cp(seq_lens, H, num_sms)
            fused_all_pre = predict_fused_all_pre(
                q,
                v,
                cu_seqlens,
                cu_seqlens_is_none=False,
                use_gate_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
            )

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

            try:
                o_off, ht_off = run_call(**common, enable_cp=False, return_state=True)
                o_on, ht_on = run_call(**common, enable_cp=True, return_state=True)
                o_max, o_mean = accuracy(o_off, o_on)
                ht_max, ht_mean = accuracy(ht_off, ht_on)
                del o_off, ht_off, o_on, ht_on

                ms_off = time_cuda_fn(lambda: run_call(**common, enable_cp=False), *_bench_warmup_iters())
                ms_on = time_cuda_fn(lambda: run_call(**common, enable_cp=True), *_bench_warmup_iters())
                speedup = ms_off / ms_on if ms_on > 0 else float("inf")
                try:
                    ms_fla = time_cuda_fn(lambda: run_fla_call(**common), *_bench_warmup_iters())
                    speedup_vs_fla = ms_fla / ms_on if ms_on > 0 else float("inf")
                except Exception:
                    ms_fla = float("nan")
                    speedup_vs_fla = float("nan")
            except torch.cuda.OutOfMemoryError:
                ms_off = ms_on = speedup = float("nan")
                ms_fla = speedup_vs_fla = float("nan")
                o_max = o_mean = ht_max = ht_mean = float("nan")

            row = {
                "tag": tag,
                "H": H,
                "total_T": total_T,
                "pred": pred,
                "n_sub": n_sub,
                "fla_pred": fla_pred,
                "fla_n_sub": fla_n_sub,
                "fused_all_pre": fused_all_pre,
                "ms_off": ms_off,
                "ms_on": ms_on,
                "ms_fla": ms_fla,
                "speedup": speedup,
                "speedup_vs_fla": speedup_vs_fla,
                "o_max": o_max,
                "o_mean": o_mean,
                "ht_max": ht_max,
                "ht_mean": ht_mean,
            }
            results.append(row)
            print(_format_row(row), flush=True)

            del q, k, v, g, beta, A_log, dt_bias, inputs
            torch.cuda.empty_cache()

        print(SEP, flush=True)

    return results


# ============================================================
# Report (summary only — per-row output is streamed inside bench_cp)
# ============================================================
def print_report(results, h_values):
    sep = "=" * 110
    triggered = [r for r in results if r["pred"]]
    bypassed = [r for r in results if not r["pred"]]

    print()
    print(sep)
    print("  Summary")
    print(sep)

    if triggered:
        speedups = [r["speedup"] for r in triggered if r["speedup"] == r["speedup"]]  # NaN filter
        if speedups:
            geo = 1.0
            for s in speedups:
                geo *= s
            geo = geo ** (1 / len(speedups))
            print(
                f"  CP triggered ({len(triggered)} configs): "
                f"geo-mean={geo:.2f}x  best={max(speedups):.2f}x  worst={min(speedups):.2f}x"
            )

    if bypassed:
        ratios = [r["ms_on"] / r["ms_off"] for r in bypassed if r["ms_off"] == r["ms_off"] and r["ms_off"] > 0]
        if ratios:
            print(
                f"  CP bypassed  ({len(bypassed)} configs): "
                f"mean overhead={sum(ratios) / len(ratios):.3f}x  max={max(ratios):.3f}x  "
                f"(1.00 = no regression)"
            )

    # cuLA (CP-on) vs FLA speedups
    fla_speedups = [r["speedup_vs_fla"] for r in results if r["speedup_vs_fla"] == r["speedup_vs_fla"]]
    if fla_speedups:
        geo = 1.0
        for s in fla_speedups:
            geo *= s
        geo = geo ** (1 / len(fla_speedups))
        print(
            f"  cuLA (CP-on) vs FLA  ({len(fla_speedups)} configs): "
            f"geo-mean={geo:.2f}x  best={max(fla_speedups):.2f}x  worst={min(fla_speedups):.2f}x"
        )
        tri_fla = [r["speedup_vs_fla"] for r in triggered if r["speedup_vs_fla"] == r["speedup_vs_fla"]]
        if tri_fla:
            geo_t = 1.0
            for s in tri_fla:
                geo_t *= s
            geo_t = geo_t ** (1 / len(tri_fla))
            print(
                f"    └─ CP-triggered subset ({len(tri_fla)} configs): "
                f"geo-mean={geo_t:.2f}x  best={max(tri_fla):.2f}x  worst={min(tri_fla):.2f}x"
            )

    o_maxes = [r["o_max"] for r in results if r["o_max"] == r["o_max"]]
    ht_maxes = [r["ht_max"] for r in results if r["ht_max"] == r["ht_max"]]
    if o_maxes:
        print(
            f"  Accuracy (CP-on vs CP-off): "
            f"o  max={max(o_maxes):.2e} avg={sum(o_maxes) / len(o_maxes):.2e}   "
            f"ht max={max(ht_maxes):.2e} avg={sum(ht_maxes) / len(ht_maxes):.2e}"
        )

    print(sep)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_intracard_cp_sm90: CP-on vs CP-off")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    parser.add_argument("--sanitizer", action="store_true", help="Sanitizer mode: warmup=1, iters=1")
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
