#!/usr/bin/env python3
"""
Benchmark: CuTE DSL vs FLA (Triton) — chunk_gla_fwd_o_gk

Compares performance under various settings:
  - Fixed: K=V=128, BT=64, use_exp2=True, scale=K^{-0.5}
  - Sweep: B ∈ {1,2,4,8}, H ∈ {4,16,64}, T ∈ {1024,2048,4096,8192}

Reports latency (ms) and speedup ratio.
"""

import sys
import os
import itertools
import time

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, "/ossfs/workspace/flash-linear-attention")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Direct import to avoid flashla.__init__.py importing CUDA extensions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fwd_o", os.path.join(os.path.dirname(os.path.dirname(__file__)), "flashla", "fwd_o.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ChunkGlaFwdO = _mod.ChunkGlaFwdO

from fla.ops.gla.chunk import chunk_gla_fwd_o_gk


# ─── helpers ────────────────────────────────────────────────────────────────

def bench_triton(fn, warmup=20, repeat=100):
    """Benchmark a callable, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench_cute(compiled, args, warmup=20, repeat=100):
    """Benchmark a compiled CuTE DSL kernel, return median time in ms."""
    for _ in range(warmup):
        compiled(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def make_data(B, T, H, K=128, V=128, BT=64, dtype=torch.bfloat16, device="cuda"):
    NT = (T + BT - 1) // BT
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1
    return q, v, g, h, A


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    K = V = 128
    BT = 64
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    # ── compile CuTE DSL kernel once ───────────────────────────────────────
    print("Compiling CuTE DSL kernel...", flush=True)
    stream = cutlass_torch.default_stream()
    kernel = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale)

    # Use a small tensor for compilation
    q0, v0, g0, h0, A0 = make_data(1, BT, 1, K, V, BT, dtype, device)
    o0 = torch.zeros_like(v0)
    cu0 = torch.zeros(2, dtype=torch.int32, device=device)
    ps0 = (1, BT, 1, K, V)

    q0c = from_dlpack(q0.detach())
    v0c = from_dlpack(v0.detach())
    g0c = from_dlpack(g0.detach())
    h0c = from_dlpack(h0.detach())
    o0c = from_dlpack(o0.detach())
    A0c = from_dlpack(A0.detach())
    cu0c = from_dlpack(cu0.detach())

    compiled = cute.compile(
        kernel,
        q0c.iterator, v0c.iterator, g0c.iterator,
        h0c.iterator, o0c.iterator, A0c.iterator,
        cu0c.iterator, ps0, stream,
    )
    print("Compilation done.\n", flush=True)

    # ── sweep configs ──────────────────────────────────────────────────────
    B_list = [1, 2, 4, 8]
    H_list = [4, 16, 64]
    T_list = [1024, 2048, 4096, 8192]

    header = (
        f"{'B':>3} {'H':>3} {'T':>6} {'NT':>5} │ "
        f"{'Triton(ms)':>11} {'CuTeDSL(ms)':>12} │ "
        f"{'Speedup':>8}"
    )
    sep = "─" * len(header)

    print(f"chunk_gla_fwd_o_gk  K={K} V={V} BT={BT} scale={scale:.4f}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print(header)
    print(sep)

    results = []

    for T_val in T_list:
        for B, H in itertools.product(B_list, H_list):
            NT = (T_val + BT - 1) // BT
            torch.manual_seed(42)
            q, v, g, h, A = make_data(B, T_val, H, K, V, BT, dtype, device)

            # ── Triton ──
            triton_fn = lambda: chunk_gla_fwd_o_gk(
                q=q, v=v, g=g, A=A, h=h,
                scale=scale, chunk_size=BT, use_exp2=True,
            )
            t_triton = bench_triton(triton_fn)

            # ── CuTE DSL ──
            o_out = torch.zeros_like(v)
            cu = torch.zeros(2, dtype=torch.int32, device=device)
            ps = (B, T_val, H, K, V)

            qc = from_dlpack(q.detach())
            vc = from_dlpack(v.detach())
            gc = from_dlpack(g.detach())
            hc = from_dlpack(h.detach())
            oc = from_dlpack(o_out.detach())
            Ac = from_dlpack(A.detach())
            cuc = from_dlpack(cu.detach())

            cute_args = (
                qc.iterator, vc.iterator, gc.iterator,
                hc.iterator, oc.iterator, Ac.iterator,
                cuc.iterator, ps, stream,
            )
            t_cute = bench_cute(compiled, cute_args)

            speedup = t_triton / t_cute if t_cute > 0 else float('inf')
            results.append((B, H, T_val, NT, t_triton, t_cute, speedup))

            print(
                f"{B:>3} {H:>3} {T_val:>6} {NT:>5} │ "
                f"{t_triton:>11.3f} {t_cute:>12.3f} │ "
                f"{speedup:>7.2f}x"
            )

        print(sep)

    # ── summary ────────────────────────────────────────────────────────────
    print()
    speedups = [r[6] for r in results]
    print(f"Speedup range:  {min(speedups):.2f}x – {max(speedups):.2f}x")
    print(f"Speedup mean:   {sum(speedups)/len(speedups):.2f}x")
    geomean = 1.0
    for s in speedups:
        geomean *= s
    geomean = geomean ** (1.0 / len(speedups))
    print(f"Speedup geomean:{geomean:.2f}x")


if __name__ == "__main__":
    main()
