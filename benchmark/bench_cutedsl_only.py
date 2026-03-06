#!/usr/bin/env python3
"""
Standalone CuteDSL benchmark — no FLA/Triton, no precompile.
Tests whether lightning_attn_fwd works correctly across configs.
"""

import torch
import time
import argparse
import sys

sys.path.insert(0, '/ossfs/workspace/flashla')
from flashla.lightning_attn import lightning_attn_fwd


def compute_decay(H, layer_idx=0, num_layers=1):
    return (8 / H * (1 - layer_idx / num_layers)) * torch.arange(
        H, dtype=torch.float32, device='cuda'
    )


def run_one(B, T, H, D, mode, warmup=2, iterations=10):
    """Run CuteDSL for one config, return elapsed_ms or raise on failure."""
    torch.manual_seed(42)
    Q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda')
    decay = compute_decay(H)

    has_h0 = mode == 'h0_ht'
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device='cuda') * 0.01 if has_h0 else None

    def _run():
        return lightning_attn_fwd(
            Q, K, V, decay, scale=1.0,
            initial_state=h0, output_final_state=has_h0,
            chunk_size=64,
        )

    # First call triggers compilation
    t0 = time.time()
    _run()
    torch.cuda.synchronize()
    compile_ms = (time.time() - t0) * 1000

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        O, ht = _run()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / iterations
    return elapsed, compile_ms, O, ht


def main():
    parser = argparse.ArgumentParser(description='CuteDSL-only benchmark')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1])
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[256, 1024, 4096])
    parser.add_argument('--num-heads', nargs='+', type=int, default=[32, 64])
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--modes', nargs='+', default=['no_state', 'h0_ht'])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()

    D = args.head_dim
    configs = []
    for mode in args.modes:
        for B in args.batch_sizes:
            for T in args.seq_lens:
                for H in args.num_heads:
                    configs.append((B, T, H, D, mode))

    header = f"{'Mode':<10} {'B':>3} {'T':>6} {'H':>4} {'D':>4}  {'ms':>8}  {'compile':>9}"
    print(header)
    print('-' * len(header))

    for i, (B, T, H, D, mode) in enumerate(configs):
        try:
            elapsed, compile_ms, O, ht = run_one(
                B, T, H, D, mode,
                warmup=args.warmup, iterations=args.iterations,
            )
            print(f"{mode:<10} {B:>3} {T:>6} {H:>4} {D:>4}  {elapsed:>8.3f}  {compile_ms:>8.1f}ms")
        except Exception as e:
            print(f"{mode:<10} {B:>3} {T:>6} {H:>4} {D:>4}  FAILED: {e}")

    print('\nDone.')


if __name__ == '__main__':
    main()
