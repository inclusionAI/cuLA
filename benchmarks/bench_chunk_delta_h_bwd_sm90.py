#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark the SM90 CuTe DSL WGMMA bwd_dhu path against FLA Triton.

Current kernel scope:
  - non-varlen
  - K in {64, 128, 256}, BT=64, BV=64
  - state layout [B, NT, H, K, V] or [B, NT, H, V, K]
  - optional gk/dht/h0

Example:
  python benchmarks/bench_chunk_delta_h_bwd_sm90.py --B 1 --T 1024 --H 8 --K 128 --V 64 --gk --dht
"""

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu_sm90


def time_kernel(fn, warmup: int, iters: int) -> float:
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


def make_inputs(args):
    torch.manual_seed(args.seed)
    seq_lens = getattr(args, "seq_lens", None)
    is_varlen = seq_lens is not None
    B = 1 if is_varlen else args.B
    T = sum(seq_lens) if is_varlen else args.T
    N = len(seq_lens) if is_varlen else B
    H, K, V = args.H, args.K, args.V
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    cu_seqlens = None
    if is_varlen:
        cu = [0]
        for seq_len in seq_lens:
            cu.append(cu[-1] + seq_len)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
    g = None
    if args.g:
        if is_varlen:
            g = torch.empty(B, T, H, dtype=torch.float32, device="cuda")
            for i in range(N):
                bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                seg = torch.randn(B, eos - bos, H, dtype=torch.float32, device="cuda") * 0.01
                g[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)
        else:
            g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)
    gk = None
    if args.gk:
        if is_varlen:
            gk = torch.empty(B, T, H, K, dtype=torch.float32, device="cuda")
            for i in range(N):
                bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                seg = torch.randn(B, eos - bos, H, K, dtype=torch.float32, device="cuda") * 0.01
                gk[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)
        else:
            gk = -torch.abs(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)
    state_shape = (N, H, V, K) if args.transpose_state else (N, H, K, V)
    dht = None
    if args.dht:
        dht = torch.randn(state_shape, dtype=torch.float32, device="cuda") * 0.01
    h0 = torch.empty(state_shape, dtype=torch.float32, device="cuda") if args.h0 else None
    return q, k, w, do, dv, g, gk, dht, h0, cu_seqlens


def run_one(args):
    q, k, w, do, dv, g, gk, dht, h0, cu_seqlens = make_inputs(args)
    scale = args.K**-0.5
    is_varlen = cu_seqlens is not None

    def run_fla():
        return fla_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens.long() if cu_seqlens is not None else None,
            chunk_size=64,
            use_exp2=args.use_exp2,
            transpose_state_layout=args.transpose_state,
        )

    def run_cute():
        return chunk_gated_delta_rule_bwd_dhu_sm90(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
            use_exp2=args.use_exp2,
            transpose_state_layout=args.transpose_state,
        )

    ref = run_fla()
    got = run_cute()
    torch.cuda.synchronize()
    max_dh = (ref[0].float() - got[0].float()).abs().max().item()
    max_dv = (ref[2].float() - got[2].float()).abs().max().item()
    max_dh0 = None
    if ref[1] is not None:
        max_dh0 = (ref[1].float() - got[1].float()).abs().max().item()

    fla_ms = time_kernel(run_fla, args.warmup, args.iters)
    cute_ms = time_kernel(run_cute, args.warmup, args.iters)

    shape_tag = f"seq_lens={args.seq_lens}" if is_varlen else f"B={args.B} T={args.T}"
    print(
        f"bwd_dhu SM90 {shape_tag} H={args.H} K={args.K} V={args.V} "
        f"g={args.g} gk={args.gk} dht={args.dht} h0={args.h0} exp2={args.use_exp2} transpose={args.transpose_state}"
    )
    if max_dh0 is None:
        print(f"max_diff dh={max_dh:.6f} dv2={max_dv:.6f}")
    else:
        print(f"max_diff dh={max_dh:.6f} dh0={max_dh0:.6f} dv2={max_dv:.6f}")
    print(f"FLA Triton: {fla_ms:.4f} ms")
    print(f"CuTe DSL : {cute_ms:.4f} ms")
    print(f"speedup  : {fla_ms / cute_ms:.3f}x")
    return {
        "B": args.B,
        "T": args.T,
        "seq_lens": args.seq_lens,
        "H": args.H,
        "K": args.K,
        "V": args.V,
        "g": args.g,
        "gk": args.gk,
        "dht": args.dht,
        "h0": args.h0,
        "exp2": args.use_exp2,
        "transpose": args.transpose_state,
        "max_dh": max_dh,
        "max_dh0": max_dh0,
        "max_dv": max_dv,
        "fla_ms": fla_ms,
        "cute_ms": cute_ms,
        "speedup": fla_ms / cute_ms,
    }


def suite_configs(kind: str):
    quick = [
        dict(B=1, T=512, H=4, K=64, V=64, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=512, H=4, K=128, V=64, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=512, H=4, K=128, V=128, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=512, H=2, K=256, V=64, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
    ]
    if kind == "quick":
        return quick
    return quick + [
        dict(
            seq_lens=[50, 192, 100],
            H=2,
            K=64,
            V=64,
            g=False,
            gk=True,
            dht=True,
            h0=False,
            use_exp2=True,
            transpose_state=False,
        ),
        dict(
            seq_lens=[33, 128, 200],
            H=1,
            K=128,
            V=64,
            g=True,
            gk=False,
            dht=True,
            h0=True,
            use_exp2=True,
            transpose_state=False,
        ),
        dict(B=1, T=512, H=4, K=64, V=64, g=True, gk=False, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=512, H=2, K=128, V=64, g=True, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=2, T=1024, H=4, K=128, V=64, g=False, gk=True, dht=True, h0=True, use_exp2=True, transpose_state=False),
        dict(B=1, T=2048, H=8, K=128, V=64, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=1024, H=8, K=64, V=128, g=False, gk=True, dht=True, h0=False, use_exp2=True, transpose_state=False),
        dict(B=1, T=512, H=4, K=128, V=64, g=False, gk=True, dht=True, h0=True, use_exp2=True, transpose_state=True),
    ]


def _fmt_optional(value):
    return "n/a" if value is None else f"{value:.6f}"


def print_suite(results):
    print("\n" + "=" * 118)
    print(" bwd_dhu SM90 Suite: CuTe DSL vs FLA Triton")
    print("=" * 118)
    for r in results:
        flags = ",".join(name for name in ("g", "gk", "dht", "h0", "exp2", "transpose") if r[name])
        shape = f"seqs={r['seq_lens']!s:<17s}" if r["seq_lens"] is not None else f"B={r['B']:2d} T={r['T']:5d}"
        print(
            f"  {shape} H={r['H']:2d} K={r['K']:3d} V={r['V']:3d} [{flags:<16s}] | "
            f"diff dh={r['max_dh']:.6f} dh0={_fmt_optional(r['max_dh0'])} dv2={r['max_dv']:.6f} | "
            f"FLA={r['fla_ms']:.4f}ms CuTe={r['cute_ms']:.4f}ms speedup={r['speedup']:.3f}x"
        )
    speedups = [r["speedup"] for r in results if r["speedup"] > 0]
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print("-" * 118)
    print(f"  Geometric mean speedup: {geo:.3f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["none", "quick", "full"], default="none")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--seq-lens", type=str, default=None)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--K", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--V", type=int, default=64)
    parser.add_argument("--g", action="store_true")
    parser.add_argument("--gk", action="store_true")
    parser.add_argument("--dht", action="store_true")
    parser.add_argument("--h0", action="store_true")
    parser.add_argument("--use-exp2", action="store_true")
    parser.add_argument("--transpose-state", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seq_lens is not None:
        args.seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an SM90/Hopper GPU.")
    if args.suite == "none":
        if args.seq_lens is None and args.T % 64 != 0:
            raise ValueError("Use T as a multiple of 64 for this benchmark.")
        if args.V % 64 != 0:
            raise ValueError("Use V as a multiple of 64 for the SM90 WGMMA path.")
        run_one(args)
        return

    results = []
    for cfg in suite_configs(args.suite):
        case_args = argparse.Namespace(**vars(args))
        case_args.seq_lens = None
        for key, value in cfg.items():
            setattr(case_args, key, value)
        results.append(run_one(case_args))
    print_suite(results)


if __name__ == "__main__":
    main()
