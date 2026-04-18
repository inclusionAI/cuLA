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
bench_kda_decode.py — 3-way benchmark for KDA decode (single-token, T=1)

Compares three routes:
  1. cuLA with v-last states  (..., K, V)
  2. cuLA with k-last states  (..., V, K)
  3. FLA  with v-last states  (..., K, V)

Fairness note:
  - All three routes use the fused decode entry point.
  - State buffers are reset before each timed iteration.
  - The reset copy is done outside the CUDA event window and is NOT counted.

Usage:
    python benchmarks/bench_kda_decode.py
    python benchmarks/bench_kda_decode.py --batch-sizes 1 4 16 64 128 256
    python benchmarks/bench_kda_decode.py --Hs 8 32 64
    python benchmarks/bench_kda_decode.py --ncu

Note:
  - This benchmark is currently restricted to K=128 and V=128.
  - By default it reports H=8, H=32, and H=64.
"""

import argparse
import os
import pathlib
import platform
import re
import sys
from datetime import datetime

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda import fused_sigmoid_gating_delta_rule_update as cula_fused
from cula.ops.kda_decode_fla import fused_sigmoid_gating_delta_rule_update as fla_fused


# ──────────────────────────────────────────────────────────────────────
# Timing utility
# ──────────────────────────────────────────────────────────────────────
def benchmark_fn(fn, *, setup_fn=None, warmup=30, rep=200):
    """Benchmark using CUDA events.

    If provided, setup_fn runs before each iteration and is excluded from the
    timing window. This is used to reset the mutable recurrent state fairly.
    """
    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        if setup_fn is not None:
            setup_fn()
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    if n <= 2:
        return sum(times) / max(len(times), 1)
    iqr = times[n // 4 : 3 * n // 4]
    return sum(iqr) / len(iqr)


# ──────────────────────────────────────────────────────────────────────
# Input generation
# ──────────────────────────────────────────────────────────────────────
def make_inputs(N, H, HV, K, V, device="cuda", seed=42):
    """Generate random inputs for KDA decode benchmark."""
    torch.manual_seed(seed)
    q = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


# ──────────────────────────────────────────────────────────────────────
# Accuracy check
# ──────────────────────────────────────────────────────────────────────
def accuracy_stats(ref, out):
    ref_f, out_f = ref.float(), out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    return rmse, rel_max


def to_v_last_state(state: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "kv":
        return state
    if layout == "vk":
        return state.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unsupported layout={layout}")


def normalize_gpu_type(gpu_name: str) -> str:
    """Normalize GPU name to a compact token for report filename."""
    upper = gpu_name.upper()
    tokens = re.findall(r"[A-Z0-9]+", upper)
    ignore = {"NVIDIA", "GEFORCE", "TESLA", "GRAPHICS", "CORPORATION", "INC"}
    tokens = [t for t in tokens if t not in ignore]

    # Handle SKUs like "H20-3e" -> "H203E".
    for i in range(len(tokens) - 1):
        left, right = tokens[i], tokens[i + 1]
        if re.fullmatch(r"H\d+", left) and re.fullmatch(r"\d+[A-Z]+", right):
            return f"{left}{right}"

    if "RTX" in tokens:
        i = tokens.index("RTX")
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            return f"RTX_{tokens[i + 1]}"

    digit_tokens = [t for t in tokens if any(ch.isdigit() for ch in t)]
    if digit_tokens:
        return digit_tokens[-1]

    return "_".join(tokens) if tokens else "UNKNOWN_GPU"


def write_markdown_report(args, gpu_name: str, sections: list[tuple[int, int, list[dict]]], output_path: pathlib.Path):
    """Write benchmark results into a markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_ver = platform.python_version()
    cuda_ver = torch.version.cuda or "unknown"
    torch_ver = torch.__version__

    speedups_v = [r["speedup_v_last"] for _, _, rows in sections for r in rows]
    speedups_k = [r["speedup_k_last"] for _, _, rows in sections for r in rows]

    def summary(vals):
        if not vals:
            return "n/a"
        return f"avg={sum(vals)/len(vals):.2f}x, min={min(vals):.2f}x, max={max(vals):.2f}x"

    lines = []
    lines.append("# Benchmark Results - KDA Decode")
    lines.append("")
    lines.append(f"> Auto-generated by `benchmarks/bench_kda_decode.py` on {now}.")
    lines.append("")
    lines.append(f"> **GPU:** {gpu_name}  |  **CUDA:** {cuda_ver}  |  **PyTorch:** {torch_ver}  |  **Python:** {py_ver}")
    lines.append("")
    lines.append("> Decode setting: single-token (T=1), K=128, V=128.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- v-last speedup (FLA/cuLA): {summary(speedups_v)}")
    lines.append(f"- k-last speedup (FLA/cuLA): {summary(speedups_k)}")
    lines.append(f"- Batch sizes: {args.batch_sizes}")
    lines.append(f"- Q/K heads (H): {args.Hs}")
    lines.append(f"- V heads (HV): {args.HV}")
    lines.append(f"- Timing params: warmup={args.warmup}, rep={args.rep}, ncu_mode={args.ncu}")
    lines.append("")

    for h_dim, v_dim, results in sections:
        lines.append(f"## KDA Decode (H={h_dim}, HV={args.HV}, K={args.K}, V={v_dim})")
        lines.append("")
        lines.append("### Performance")
        lines.append("")
        lines.append("| N | cuLA v-last (ms) | cuLA k-last (ms) | FLA v-last (ms) | v-last speedup | k-last speedup |")
        lines.append("|---|------------------:|------------------:|----------------:|---------------:|---------------:|")
        for r in results:
            lines.append(
                f"| {r['N']} | {r['t_cula_v_last_ms']:.4f} | {r['t_cula_k_last_ms']:.4f} | {r['t_fla_v_last_ms']:.4f} | "
                f"**{r['speedup_v_last']:.2f}x** | **{r['speedup_k_last']:.2f}x** |"
            )
        lines.append("")

        lines.append("### Accuracy (Output)")
        lines.append("")
        lines.append("| N | cuLA v out RMSE | cuLA v out rel | cuLA k out RMSE | cuLA k out rel |")
        lines.append("|---|----------------:|---------------:|----------------:|---------------:|")
        for r in results:
            lines.append(
                f"| {r['N']} | {r['out_v_last_rmse']:.3e} | {r['out_v_last_rel']:.3e} | "
                f"{r['out_k_last_rmse']:.3e} | {r['out_k_last_rel']:.3e} |"
            )
        lines.append("")

        lines.append("### Accuracy (State)")
        lines.append("")
        lines.append("| N | cuLA v state RMSE | cuLA v state rel | cuLA k state RMSE | cuLA k state rel |")
        lines.append("|---|------------------:|-----------------:|------------------:|-----------------:|")
        for r in results:
            lines.append(
                f"| {r['N']} | {r['state_v_last_rmse']:.3e} | {r['state_v_last_rel']:.3e} | "
                f"{r['state_k_last_rmse']:.3e} | {r['state_k_last_rel']:.3e} |"
            )
        lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/bench_kda_decode.py")
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Run one config
# ──────────────────────────────────────────────────────────────────────
def run_config(N, H, HV, K, V, warmup, rep, ncu_mode):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V, device)

    q_4d = q.unsqueeze(1).contiguous()
    k_4d = k.unsqueeze(1).contiguous()
    v_4d = v.unsqueeze(1).contiguous()
    a_flat = a.reshape(N, 1, -1).contiguous()
    b_3d = b.unsqueeze(1).contiguous()
    indices = torch.arange(N, device=device, dtype=torch.int32)

    state_init_k_last = state.clone().contiguous()  # (N, HV, V, K)
    state_init_v_last = state_init_k_last.permute(0, 1, 3, 2).contiguous()  # (N, HV, K, V)

    state_cula_v_last = state_init_v_last.clone()
    state_cula_k_last = state_init_k_last.clone()
    state_fla_v_last = state_init_v_last.clone()

    def call_cula_v_last(state_buf):
        return cula_fused(
            A_log=A_log,
            a=a_flat,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_buf,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="kv",
        )

    def call_cula_k_last(state_buf):
        return cula_fused(
            A_log=A_log,
            a=a_flat,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_buf,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="vk",
        )

    def call_fla_v_last(state_buf):
        return fla_fused(
            A_log=A_log,
            a=a_flat,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_buf,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
        )

    with torch.no_grad():
        o_cula_v_last = call_cula_v_last(state_cula_v_last)
        o_cula_k_last = call_cula_k_last(state_cula_k_last)
        o_fla_v_last = call_fla_v_last(state_fla_v_last)

    out_v_last_rmse, out_v_last_rel = accuracy_stats(o_fla_v_last, o_cula_v_last)
    out_k_last_rmse, out_k_last_rel = accuracy_stats(o_fla_v_last, o_cula_k_last)
    state_v_last_rmse, state_v_last_rel = accuracy_stats(state_fla_v_last, state_cula_v_last)
    state_k_last_rmse, state_k_last_rel = accuracy_stats(state_fla_v_last, to_v_last_state(state_cula_k_last, "vk"))

    if ncu_mode:
        w, r = 1, 1
    else:
        w, r = warmup, rep

    state_bench_cula_v_last = state_init_v_last.clone()
    state_bench_cula_k_last = state_init_k_last.clone()
    state_bench_fla_v_last = state_init_v_last.clone()

    def setup_cula_v_last():
        state_bench_cula_v_last.copy_(state_init_v_last)

    def setup_cula_k_last():
        state_bench_cula_k_last.copy_(state_init_k_last)

    def setup_fla_v_last():
        state_bench_fla_v_last.copy_(state_init_v_last)

    with torch.no_grad():
        t_cula_v_last = benchmark_fn(lambda: call_cula_v_last(state_bench_cula_v_last), setup_fn=setup_cula_v_last, warmup=w, rep=r)
        t_cula_k_last = benchmark_fn(lambda: call_cula_k_last(state_bench_cula_k_last), setup_fn=setup_cula_k_last, warmup=w, rep=r)
        t_fla_v_last = benchmark_fn(lambda: call_fla_v_last(state_bench_fla_v_last), setup_fn=setup_fla_v_last, warmup=w, rep=r)

    return {
        "N": N,
        "H": H,
        "HV": HV,
        "K": K,
        "V": V,
        "t_cula_v_last_ms": t_cula_v_last,
        "t_cula_k_last_ms": t_cula_k_last,
        "t_fla_v_last_ms": t_fla_v_last,
        "speedup_v_last": t_fla_v_last / t_cula_v_last if t_cula_v_last > 0 else float("inf"),
        "speedup_k_last": t_fla_v_last / t_cula_k_last if t_cula_k_last > 0 else float("inf"),
        "out_v_last_rmse": out_v_last_rmse,
        "out_v_last_rel": out_v_last_rel,
        "out_k_last_rmse": out_k_last_rmse,
        "out_k_last_rel": out_k_last_rel,
        "state_v_last_rmse": state_v_last_rmse,
        "state_v_last_rel": state_v_last_rel,
        "state_k_last_rmse": state_k_last_rmse,
        "state_k_last_rel": state_k_last_rel,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark KDA decode: cuLA vs FLA")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--Hs", nargs="+", type=int, default=[8, 32, 64], help="Q/K head counts to benchmark")
    parser.add_argument("--HV", type=int, default=128, help="Number of V heads (GVA)")
    parser.add_argument("--K", type=int, default=128, help="Head dim K (only 128 is supported)")
    parser.add_argument("--V", type=int, default=128, help="Head dim V (only 128 is supported)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--ncu", action="store_true", help="NCU mode: warmup=1, rep=1")
    args = parser.parse_args()

    if args.K != 128 or args.V != 128:
        raise ValueError(f"bench_kda_decode.py currently only supports K=128 and V=128, got K={args.K}, V={args.V}")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: Hs={args.Hs}, HV={args.HV}, K={args.K}, V={args.V}")
    print("Timing note: state reset uses copy_() before each timed iteration and is not counted.")
    print("Accuracy reference: FLA v-last route.")
    print()

    all_sections: list[tuple[int, int, list[dict]]] = []

    def print_section(h_dim: int, v_dim: int):
        print(f"Config: H={h_dim}, HV={args.HV}, K={args.K}, V={v_dim}")
        hdr_perf = (
            f"{'N':>5} | {'cuLA v-last':>12} | {'cuLA k-last':>12} | {'FLA v-last':>12} | "
            f"{'v-last spd':>10} | {'k-last spd':>10}"
        )
        print(hdr_perf)
        print("-" * len(hdr_perf))

        results = []
        for N in args.batch_sizes:
            res = run_config(
                N,
                h_dim,
                args.HV,
                args.K,
                v_dim,
                args.warmup,
                args.rep,
                args.ncu,
            )
            results.append(res)
            print(
                f"{res['N']:5d} | {res['t_cula_v_last_ms']:12.4f} | {res['t_cula_k_last_ms']:12.4f} | "
                f"{res['t_fla_v_last_ms']:12.4f} | {res['speedup_v_last']:9.2f}x | {res['speedup_k_last']:9.2f}x"
            )

        print()
        hdr_out = (
            f"{'N':>5} | {'cuLA v out RMSE':>16} | {'rel':>10} | "
            f"{'cuLA k out RMSE':>16} | {'rel':>10}"
        )
        print(hdr_out)
        print("-" * len(hdr_out))
        for res in results:
            print(
                f"{res['N']:5d} | {res['out_v_last_rmse']:16.3e} | {res['out_v_last_rel']:10.3e} | "
                f"{res['out_k_last_rmse']:16.3e} | {res['out_k_last_rel']:10.3e}"
            )

        print()
        hdr_state = (
            f"{'N':>5} | {'cuLA v state RMSE':>18} | {'rel':>10} | "
            f"{'cuLA k state RMSE':>18} | {'rel':>10}"
        )
        print(hdr_state)
        print("-" * len(hdr_state))
        for res in results:
            print(
                f"{res['N']:5d} | {res['state_v_last_rmse']:18.3e} | {res['state_v_last_rel']:10.3e} | "
                f"{res['state_k_last_rmse']:18.3e} | {res['state_k_last_rel']:10.3e}"
            )

        all_sections.append((h_dim, v_dim, results))

    for h_dim in args.Hs:
        print_section(h_dim, args.V)
        print()

    gpu_type = normalize_gpu_type(gpu_name)
    report_path = pathlib.Path(__file__).resolve().parent.parent / f"BENCHMARK_KDA_DECODE_{gpu_type}.md"
    write_markdown_report(args, gpu_name, all_sections, report_path)
    print(f"Markdown report written to: {report_path}")


if __name__ == "__main__":
    main()
