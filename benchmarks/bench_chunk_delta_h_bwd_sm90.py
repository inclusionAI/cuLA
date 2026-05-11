#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark the SM90 CuTe DSL bwd_dhu prototype against FLA Triton.

Current kernel scope:
  - non-varlen
  - K in {64, 128, 256}, BT=64
  - state layout [B, NT, H, K, V]
  - optional gk/dht/h0

Example:
  python benchmarks/bench_chunk_delta_h_bwd_sm90.py --B 1 --T 1024 --H 8 --K 128 --V 64 --gk --dht
"""

import argparse
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
    B, T, H, K, V = args.B, args.T, args.H, args.K, args.V
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    gk = None
    if args.gk:
        gk = -torch.abs(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)
    dht = None
    if args.dht:
        dht = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.01
    h0 = torch.empty(B, H, K, V, dtype=torch.float32, device="cuda") if args.h0 else None
    return q, k, w, do, dv, gk, dht, h0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--K", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--V", type=int, default=64)
    parser.add_argument("--gk", action="store_true")
    parser.add_argument("--dht", action="store_true")
    parser.add_argument("--h0", action="store_true")
    parser.add_argument("--use-exp2", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an SM90/Hopper GPU.")
    if args.T % 64 != 0:
        raise ValueError("Use T as a multiple of 64 for this prototype benchmark.")

    q, k, w, do, dv, gk, dht, h0 = make_inputs(args)
    scale = args.K**-0.5

    def run_fla():
        return fla_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            chunk_size=64,
            use_exp2=args.use_exp2,
        )

    def run_cute():
        return chunk_gated_delta_rule_bwd_dhu_sm90(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            chunk_size=64,
            use_exp2=args.use_exp2,
        )

    ref = run_fla()
    got = run_cute()
    torch.cuda.synchronize()
    max_dh = (ref[0].float() - got[0].float()).abs().max().item()
    max_dv = (ref[2].float() - got[2].float()).abs().max().item()

    fla_ms = time_kernel(run_fla, args.warmup, args.iters)
    cute_ms = time_kernel(run_cute, args.warmup, args.iters)

    print(
        f"bwd_dhu SM90 B={args.B} T={args.T} H={args.H} K={args.K} V={args.V} "
        f"gk={args.gk} dht={args.dht} h0={args.h0} exp2={args.use_exp2}"
    )
    print(f"max_diff dh={max_dh:.6f} dv2={max_dv:.6f}")
    print(f"FLA Triton: {fla_ms:.4f} ms")
    print(f"CuTe DSL : {cute_ms:.4f} ms")
    print(f"speedup  : {fla_ms / cute_ms:.3f}x")


if __name__ == "__main__":
    main()
