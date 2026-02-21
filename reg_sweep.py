#!/usr/bin/env python3
"""
Register allocation sweep: test different (num_regs_cuda, num_regs_others) pairs.

On SM100 with 8 warps (2 warpgroups):
  - Warpgroup 0 (warps 0-3): CUDA core warps → alloc(num_regs_cuda)
  - Warpgroup 1 (warps 4-7): MMA+Load+Store+Empty → dealloc(num_regs_others)

Constraint: num_regs must be multiples of 8, minimum 24.
The pair determines the kernel's register pressure and occupancy.
"""

import sys
import time
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, "/ossfs/workspace/flashla/flashla")
from chunk_delta_h import ChunkDeltaRuleFwdH


def bench_fn(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def run_single_config(num_regs_cuda, num_regs_others, B=4, T=4096, H=64, K=128, V=128, BT=64):
    """Compile and benchmark with given register config."""
    NT = T // BT
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

    g_tensor = torch.zeros(B, T, H, device=device, dtype=torch.float32)
    gk_tensor = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    h0_tensor = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
    v_new_out = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    ht_out = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    # Create kernel with modified registers
    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    kernel.num_regs_cuda = num_regs_cuda
    kernel.num_regs_others = num_regs_others

    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g_tensor), from_dlpack(gk_tensor)
    h0c = from_dlpack(h0_tensor)
    hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)

    args = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        (B, T, H, K, V),
        0, 0, 0, 0, 1,  # no gating, save v_new
        stream,
    )

    t0 = time.time()
    try:
        compiled = cute.compile(kernel, *args)
    except Exception as e:
        return None, f"compile error: {e}"
    compile_t = time.time() - t0

    # Correctness check (quick)
    compiled(*args)
    torch.cuda.synchronize()

    # Quick sanity: h_out should not be all zeros for chunk > 0
    if h_out[:, 1:].abs().max().item() < 1e-6:
        return None, "output all zeros"

    our_ms = bench_fn(lambda: compiled(*args))
    return our_ms, f"compiled in {compile_t:.1f}s"


def main():
    print("=" * 80)
    print("Register Allocation Sweep: (num_regs_cuda, num_regs_others)")
    print("=" * 80)

    # On SM100: 65536 regs / SM, 256 threads (8 warps × 32)
    # Per-thread default = 256. Constraint: sum must allow compilation.
    # setmaxnreg values must be multiples of 8, ≥ 24.
    # Occupancy = floor(65536 / (threads_per_cta * max_regs_per_thread))

    # Configurations to test:
    # (num_regs_cuda, num_regs_others)
    configs = [
        # Baseline
        (232, 40),   # current
        # Reduce others → more CUDA headroom
        (232, 32),
        (232, 24),
        (240, 32),
        (240, 24),
        (248, 24),
        # Increase CUDA significantly
        (256, 24),
        (256, 32),
        (256, 40),
        # Slightly lower CUDA
        (224, 40),
        (224, 32),
        (216, 40),
        (208, 48),
        # Higher others (in case MMA/Load/Store spill)
        (232, 48),
        (232, 56),
        (224, 56),
        (216, 56),
    ]

    # Also get FLA baseline
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

    B, T, H, K, V, BT = 4, 4096, 64, 128, 128, 64
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    def fla_fn():
        fla_fwd_h(k=k, w=w, u=u, g=None, gk=None,
                   initial_state=None, output_final_state=False,
                   chunk_size=BT, save_new_value=True)

    fla_ms = bench_fn(fla_fn)
    print(f"\nFLA baseline: {fla_ms:.3f} ms\n")

    print(f"{'Config (cuda, others)':<25} {'Sum':>5} {'Ours (ms)':>10} {'vs FLA':>8} {'Note'}")
    print("-" * 70)

    results = []
    for (nr_cuda, nr_others) in configs:
        label = f"({nr_cuda:3d}, {nr_others:3d})"
        total = nr_cuda + nr_others
        our_ms, note = run_single_config(nr_cuda, nr_others, B, T, H, K, V, BT)
        if our_ms is not None:
            sp = fla_ms / our_ms
            flag = " <<<" if sp > 1.40 else ""
            print(f"{label:<25} {total:>5} {our_ms:>10.3f} {sp:>7.2f}x {note}{flag}")
            results.append((nr_cuda, nr_others, our_ms, sp))
        else:
            print(f"{label:<25} {total:>5} {'FAIL':>10} {'':>8} {note}")

    print("-" * 70)
    if results:
        best = max(results, key=lambda x: x[3])
        print(f"\nBest: ({best[0]}, {best[1]}) → {best[2]:.3f} ms, {best[3]:.2f}x vs FLA")
        print(f"Current: (232, 40)")

    # Also test best config with gk gating
    if results:
        best_cuda, best_others = best[0], best[1]
        if (best_cuda, best_others) != (232, 40):
            print(f"\n--- Testing best config ({best_cuda}, {best_others}) with gk gating ---")

            def run_gk_config(nr_cuda, nr_others):
                NT = T // BT
                device = "cuda"
                dtype = torch.bfloat16
                torch.manual_seed(42)
                k2 = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
                w2 = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
                u2 = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
                g2 = torch.zeros(B, T, H, device=device, dtype=torch.float32)
                gk2 = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
                gk2 = -torch.abs(gk2).cumsum(dim=1)
                h0_2 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
                h_out2 = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
                vn2 = torch.zeros(B, T, H, V, device=device, dtype=dtype)
                ht2 = torch.zeros(B, H, K, V, device=device, dtype=dtype)

                kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
                kernel.num_regs_cuda = nr_cuda
                kernel.num_regs_others = nr_others
                stream = cutlass_torch.default_stream()
                args = (
                    from_dlpack(k2).iterator, from_dlpack(w2).iterator, from_dlpack(u2).iterator,
                    from_dlpack(g2).iterator, from_dlpack(gk2).iterator,
                    from_dlpack(h_out2).iterator, from_dlpack(vn2).iterator,
                    from_dlpack(h0_2).iterator, from_dlpack(ht2).iterator,
                    (B, T, H, K, V), 0, 1, 0, 0, 1, stream,
                )
                compiled = cute.compile(kernel, *args)
                compiled(*args)
                torch.cuda.synchronize()
                return bench_fn(lambda: compiled(*args))

            # FLA gk baseline
            gk_tensor = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
            gk_tensor = -torch.abs(gk_tensor).cumsum(dim=1)
            fla_gk_ms = bench_fn(lambda: fla_fwd_h(k=k, w=w, u=u, g=None, gk=gk_tensor,
                                                     initial_state=None, output_final_state=False,
                                                     chunk_size=BT, save_new_value=True))

            best_gk_ms = run_gk_config(best_cuda, best_others)
            curr_gk_ms = run_gk_config(232, 40)
            print(f"  FLA gk:          {fla_gk_ms:.3f} ms")
            print(f"  Current (232,40): {curr_gk_ms:.3f} ms → {fla_gk_ms/curr_gk_ms:.2f}x")
            print(f"  Best ({best_cuda},{best_others}):  {best_gk_ms:.3f} ms → {fla_gk_ms/best_gk_ms:.2f}x")


if __name__ == "__main__":
    main()
