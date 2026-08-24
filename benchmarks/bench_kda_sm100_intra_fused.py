#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Benchmark/profiling entry for SM100 KDA K123 inverse variants.

Examples:
  python benchmarks/bench_kda_fwd_intra_sm100_fused_inv.py --mode both
  python benchmarks/bench_kda_fwd_intra_sm100_fused_inv.py --mode varlen --seq-lens 288
  /usr/local/cuda-13/bin/ncu --profile-from-start off --set full -o ncu_reports/kda_fwd_intra_sm100_varlen \
    .venv/bin/python benchmarks/bench_kda_fwd_intra_sm100_fused_inv.py --ncu --mode varlen
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

from benchmarks.utils import exclusive_cumsum, gen_random, gen_skewed, gen_uniform, set_seed
from cula.ops.kda.sm100.intra_fused import (
    BT,
    K_DIM,
    chunk_kda_fwd_intra_sm100_equal,
    chunk_kda_fwd_intra_sm100_varlen,
)


@dataclass
class BenchCase:
    name: str
    detail: str
    run: Callable[[], tuple[torch.Tensor, ...]]
    run_fla: Callable[[], tuple[torch.Tensor, ...]]
    compare: Callable[
        [tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], list[tuple[str, tuple[float, float, float, float]]]
    ]


def _l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2.0, dim=-1).to(torch.bfloat16)


def _make_inputs(args: argparse.Namespace, batch: int, total_t: int):
    set_seed(args.seed)
    device = torch.device(args.device)
    q = _l2norm_bf16(torch.randn(batch, total_t, args.H, args.K, device=device, dtype=torch.bfloat16))
    k = _l2norm_bf16(torch.randn(batch, total_t, args.H, args.K, device=device, dtype=torch.bfloat16))
    g = (torch.randn(batch, total_t, args.H, args.K, device=device, dtype=torch.bfloat16) * args.g_scale).bfloat16()
    beta = torch.randn(batch, total_t, args.H, device=device, dtype=torch.float32).sigmoid().bfloat16()
    a_log = torch.randn(args.H, device=device, dtype=torch.float32) * args.alog_scale
    dt_bias = None if args.no_bias else torch.randn(args.H * args.K, device=device, dtype=torch.float32) * args.dt_bias_scale
    return q, k, g, beta, a_log, dt_bias


def _parse_seq_lens(text: str) -> list[int]:
    seq_lens = [int(x) for x in text.replace(",", " ").split()]
    if not seq_lens or any(x <= 0 for x in seq_lens):
        raise ValueError(f"invalid --seq-lens: {text!r}")
    return seq_lens


def _build_seq_lens(args: argparse.Namespace) -> list[int]:
    if args.seq_lens:
        return _parse_seq_lens(args.seq_lens)
    if args.dist == "uniform":
        return gen_uniform(args.num_seqs, args.T)
    if args.dist == "skewed":
        return gen_skewed(args.num_seqs, args.T)
    if args.dist == "random":
        return gen_random(args.num_seqs, args.T, seed=args.seed)
    raise ValueError(f"unknown --dist {args.dist}")


def _varlen_launch_summary(seq_lens: list[int]) -> tuple[int, int, bool]:
    t_launch = 0
    launch_bos = []
    all_bt_aligned = True
    nt = 0
    for seq_len in seq_lens:
        chunks = (seq_len + BT - 1) // BT
        launch_bos.append(t_launch)
        nt += chunks
        t_launch += chunks * BT
        all_bt_aligned = all_bt_aligned and (seq_len % BT == 0)
    cu = exclusive_cumsum(seq_lens)
    pure = all_bt_aligned and (nt % 4) == 0 and launch_bos == cu[:-1]
    return nt, t_launch, pure


def accuracy_stats(
    ref: torch.Tensor, out: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[float, float, float, float]:
    """Return rel_rmse, rel_max, max_abs, and mean_abs."""
    if mask is not None:
        ref = ref[mask]
        out = out[mask]
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.square().mean().sqrt().item()
    ref_rms = ref_f.square().mean().sqrt().item()
    rel_rmse = rmse / (ref_rms + 1e-8)
    max_abs = diff.max().item()
    ref_max = ref_f.abs().max().item()
    rel_max = max_abs / ref_max if ref_max > 0 else 0.0
    mean_abs = diff.mean().item()
    return rel_rmse, rel_max, max_abs, mean_abs


def _valid_lower_mask(batch: int, total_t: int, heads: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(total_t, device=device) % BT
    col = torch.arange(BT, device=device)
    return (col[None, :] <= row[:, None]).view(1, total_t, 1, BT).expand(batch, total_t, heads, BT)


def _valid_lower_mask_varlen(seq_lens: list[int], heads: int, device: torch.device) -> torch.Tensor:
    total_t = sum(seq_lens)
    valid = torch.zeros(1, total_t, heads, BT, device=device, dtype=torch.bool)
    col = torch.arange(BT, device=device)
    offset = 0
    for seq_len in seq_lens:
        row = torch.arange(seq_len, device=device) % BT
        seq_valid = (col[None, :] <= row[:, None]).view(1, seq_len, 1, BT).expand(1, seq_len, heads, BT)
        valid[:, offset : offset + seq_len] = seq_valid
        offset += seq_len
    return valid


def _reference_gk_equal(
    g: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor | None, lower_bound: float
) -> torch.Tensor:
    batch, total_t, _, _ = g.shape
    outs = []
    for b in range(batch):
        cu_seqlens = torch.tensor([0, total_t], dtype=torch.int32, device=g.device)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        outs.append(
            kda_gate_chunk_cumsum(
                g=g[b : b + 1].float(),
                A_log=a_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=BT,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        )
    return torch.cat(outs, dim=0).contiguous()


def _gk_last_exp_equal(gk: torch.Tensor) -> torch.Tensor:
    return torch.exp2(gk[:, BT - 1 :: BT]).contiguous()


def _gk_last_exp_varlen(gk: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    rows = []
    offset = 0
    for seq_len in seq_lens:
        for chunk_start in range(0, seq_len, BT):
            last = offset + min(chunk_start + BT, seq_len) - 1
            rows.append(gk[:, last])
        offset += seq_len
    return torch.stack(rows, dim=1).contiguous()


def _stats_map(stats: list[tuple[str, tuple[float, float, float, float]]]) -> dict[str, tuple[float, float, float, float]]:
    return {name: values for name, values in stats}


def _format_stat(values: tuple[float, float, float, float]) -> str:
    rel_rmse, rel_max, max_abs, mean_abs = values
    return f"rel_rmse={rel_rmse:.3e} rel_max={rel_max:.3e} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"


def make_equal_case(args: argparse.Namespace) -> BenchCase:
    if args.T % (4 * BT) != 0:
        raise NotImplementedError(f"equal mode requires T to be a multiple of {4 * BT}, got {args.T}.")
    q, k, g, beta, a_log, dt_bias = _make_inputs(args, args.B, args.T)
    scale = args.K**-0.5
    if args.cutedsl_variant == "flashinfer-k123-copy":
        cutedsl_fn = chunk_kda_fwd_intra_sm100_equal
    elif args.cutedsl_variant == "flashinfer-k123-copy-pdl-fp32-inv":

        def cutedsl_fn(**kwargs):
            return chunk_kda_fwd_intra_sm100_equal(
                **kwargs,
                pdl_fp32_akk_inv=True,
            )

    else:
        raise NotImplementedError(f"Unsupported CuTeDSL variant: {args.cutedsl_variant}")

    def run():
        return cutedsl_fn(
            q=q,
            k=k,
            g=g,
            beta=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            safe_gate=True,
            lower_bound=args.lower_bound,
        )

    def run_fla_with_gk():
        gk = _reference_gk_equal(g, a_log, dt_bias, args.lower_bound)
        cu_seqlens = None
        chunk_indices = None
        if args.B == 1:
            cu_seqlens = torch.tensor([0, args.T], dtype=torch.int32, device=q.device)
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        fla = fla_chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=k,
            gk=gk,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=BT,
            safe_gate=True,
            disable_recompute=True,
        )
        return (gk, *fla)

    valid_mask = _valid_lower_mask(args.B, args.T, args.H, q.device)

    def compare(cutedsl: tuple[torch.Tensor, ...], fla_with_gk: tuple[torch.Tensor, ...]):
        gk, _, _, _, kg_fla, aqk_fla, akk_fla = fla_with_gk
        exp_gk = torch.exp2(gk)
        k_scaled_ref = (k.float() * exp_gk).to(torch.bfloat16)
        q_scaled_ref = (q.float() * exp_gk).to(torch.bfloat16)
        stats = [
            ("k_scaled", accuracy_stats(k_scaled_ref, cutedsl[0])),
            ("kg", accuracy_stats(kg_fla, cutedsl[1])),
            ("q_scaled", accuracy_stats(q_scaled_ref, cutedsl[2])),
            ("Aqk_valid", accuracy_stats(aqk_fla, cutedsl[4], valid_mask)),
        ]
        if args.cutedsl_variant in ("flashinfer-k123-copy-pdl-fp32-inv",):
            stats.append(("Akk_valid", accuracy_stats(akk_fla, cutedsl[5], valid_mask)))
        return stats

    nt = args.B * (args.T // BT)
    detail = f"B={args.B} T={args.T} H={args.H} K={args.K} NT={nt} bias={not args.no_bias} variant={args.cutedsl_variant}"
    return BenchCase("equal", detail, run, run_fla_with_gk, compare)


def make_varlen_case(args: argparse.Namespace) -> BenchCase:
    if args.cutedsl_variant not in (
        "flashinfer-k123-copy",
        "flashinfer-k123-copy-pdl-fp32-inv",
    ):
        raise NotImplementedError(f"{args.cutedsl_variant} CuTeDSL variant currently supports equal mode only.")
    seq_lens = _build_seq_lens(args)
    total_t = sum(seq_lens)
    q, k, g, beta, a_log, dt_bias = _make_inputs(args, 1, total_t)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=q.device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    scale = args.K**-0.5

    if args.cutedsl_variant == "flashinfer-k123-copy":

        def run():
            return chunk_kda_fwd_intra_sm100_varlen(
                q=q,
                k=k,
                g=g,
                beta=beta,
                A_log=a_log,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                scale=scale,
                dt_bias=dt_bias,
                safe_gate=True,
                lower_bound=args.lower_bound,
                seq_lens=seq_lens,
            )

    elif args.cutedsl_variant == "flashinfer-k123-copy-pdl-fp32-inv":

        def run():
            return chunk_kda_fwd_intra_sm100_varlen(
                q=q,
                k=k,
                g=g,
                beta=beta,
                A_log=a_log,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                scale=scale,
                dt_bias=dt_bias,
                safe_gate=True,
                lower_bound=args.lower_bound,
                seq_lens=seq_lens,
                pdl_fp32_akk_inv=True,
            )

    def run_fla_with_gk():
        gk = kda_gate_chunk_cumsum(
            g=g.float(),
            A_log=a_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=BT,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=args.lower_bound,
        )
        fla = fla_chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=k,
            gk=gk,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=BT,
            safe_gate=True,
            disable_recompute=True,
        )
        return (gk, *fla)

    valid_mask = _valid_lower_mask_varlen(seq_lens, args.H, q.device)

    def compare(cutedsl: tuple[torch.Tensor, ...], fla_with_gk: tuple[torch.Tensor, ...]):
        gk, _, _, _, kg_fla, aqk_fla, akk_fla = fla_with_gk
        exp_gk = torch.exp2(gk)
        k_scaled_ref = (k.float() * exp_gk).to(torch.bfloat16)
        q_scaled_ref = (q.float() * exp_gk).to(torch.bfloat16)
        stats = [
            ("k_scaled", accuracy_stats(k_scaled_ref, cutedsl[0])),
            ("kg", accuracy_stats(kg_fla, cutedsl[1])),
            ("q_scaled", accuracy_stats(q_scaled_ref, cutedsl[2])),
            ("Aqk_valid", accuracy_stats(aqk_fla, cutedsl[4], valid_mask)),
        ]
        if args.cutedsl_variant in ("flashinfer-k123-copy-pdl-fp32-inv",):
            stats.append(("Akk_valid", accuracy_stats(akk_fla, cutedsl[5], valid_mask)))
        return stats

    nt, t_launch, pure = _varlen_launch_summary(seq_lens)
    seq_preview = ",".join(str(x) for x in seq_lens[:8])
    if len(seq_lens) > 8:
        seq_preview += ",..."
    detail = (
        f"seqs={len(seq_lens)} total={total_t} H={args.H} K={args.K} NT={nt} "
        f"T_launch={t_launch} pure={pure} dist={args.dist} bias={not args.no_bias} seq_lens=[{seq_preview}]"
    )
    return BenchCase("varlen", detail, run, run_fla_with_gk, compare)


def time_case(fn: Callable[[], tuple[torch.Tensor, ...]], warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1_000_000.0 / iters
    event_us = start.elapsed_time(end) / iters * 1000.0
    return event_us, wall_us


def run_bench(cases: list[BenchCase], args: argparse.Namespace) -> None:
    print("=" * 100, flush=True)
    print("  SM100 KDA K123 benchmark vs FLA chunk_intra", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  BT={BT} H={args.H} K={args.K} warmup={args.warmup} iters={args.iters}", flush=True)
    print("  FLA baseline: kda_gate_chunk_cumsum + FLA chunk_kda_fwd_intra", flush=True)
    print("=" * 100, flush=True)
    for case in cases:
        print(f"\n  {'-' * 144}", flush=True)
        print(f"  {case.name}: {case.detail}", flush=True)
        print(f"  {'-' * 144}", flush=True)
        print(
            f"  {'FLA(us)':>10} {'CuTeDSL(us)':>12} {'wall/ev':>8}  │ {'FLA/CuTe':>9}  │ {'Aqk rel_rmse':>13} {'Akk rel_rmse':>13}",
            flush=True,
        )
        print(f"  {'-' * 144}", flush=True)

        cutedsl_out = case.run()
        torch.cuda.synchronize()
        cutedsl_event_us, cutedsl_wall_us = time_case(case.run, args.warmup, args.iters)
        # wall/event ~1 means the GPU is the bottleneck and the timing is
        # trustworthy; >>1 means host-side launch overhead is leaking into the
        # measured interval (still unstable).
        cutedsl_ratio = cutedsl_wall_us / cutedsl_event_us if cutedsl_event_us > 0 else float("nan")
        if args.no_fla:
            print(
                f"  {'N/A':>10} {cutedsl_event_us:12.1f} {cutedsl_ratio:8.2f}  │ {'N/A':>9}  │ {'N/A':>13} {'N/A':>13}",
                flush=True,
            )
            continue

        fla_out = case.run_fla()
        torch.cuda.synchronize()
        fla_event_us, _ = time_case(case.run_fla, args.warmup, args.iters)
        speedup = fla_event_us / cutedsl_event_us if cutedsl_event_us > 0 else float("inf")
        stats = case.compare(cutedsl_out, fla_out)
        stats_by_name = _stats_map(stats)
        aqk_rel = stats_by_name["Aqk_valid"][0]
        akk_rel = stats_by_name["Akk_valid"][0] if "Akk_valid" in stats_by_name else float("nan")
        akk_str = f"{akk_rel:13.3e}" if akk_rel == akk_rel else f"{'N/A':>13}"
        print(
            f"  {fla_event_us:10.1f} {cutedsl_event_us:12.1f} {cutedsl_ratio:8.2f}  │ {speedup:8.2f}x  │ {aqk_rel:13.3e} {akk_str}",
            flush=True,
        )
        print("  Accuracy details vs FLA/reference:", flush=True)
        for name, values in stats:
            print(f"    {name:<26} {_format_stat(values)}", flush=True)


def run_ncu(case: BenchCase, args: argparse.Namespace) -> None:
    print(
        f"[NCU profiler] case={case.name} {case.detail} warmup={args.profile_warmup} profile_iters={args.profile_iters}",
        flush=True,
    )
    case.run()
    torch.cuda.synchronize()
    for _ in range(args.profile_warmup):
        case.run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.profile_iters):
        case.run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[NCU profiler] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("equal", "varlen", "both"), default="both")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--K", type=int, default=K_DIM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--dist", choices=("uniform", "random", "skewed"), default="random")
    parser.add_argument("--seq-lens", help="Comma/space separated varlen sequence lengths; overrides --T/--num-seqs/--dist.")
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--g-scale", type=float, default=1.0)
    parser.add_argument("--alog-scale", type=float, default=1.0)
    parser.add_argument("--dt-bias-scale", type=float, default=1.0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--no-fla", action="store_true", help="Only time CuTeDSL; skip FLA timing and accuracy comparison.")
    parser.add_argument(
        "--cutedsl-variant",
        choices=(
            "flashinfer-k123-copy",
            "flashinfer-k123-copy-pdl-fp32-inv",
        ),
        default="flashinfer-k123-copy-pdl-fp32-inv",
        help=("CuTeDSL variant to benchmark. Variants without Akk inverse skip Akk accuracy."),
    )
    parser.add_argument("--ncu", action="store_true", help="Run one case under cudaProfilerStart/Stop for Nsight Compute.")
    args = parser.parse_args()

    if args.K != K_DIM:
        raise NotImplementedError(f"SM100 training intra currently supports K={K_DIM}, got {args.K}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    cases: list[BenchCase] = []
    if args.mode in ("equal", "both"):
        cases.append(make_equal_case(args))
    if args.mode in ("varlen", "both"):
        cases.append(make_varlen_case(args))

    if args.ncu:
        if len(cases) != 1:
            raise ValueError("--ncu requires --mode equal or --mode varlen so the report captures one target case.")
        run_ncu(cases[0], args)
    else:
        run_bench(cases, args)


if __name__ == "__main__":
    main()

