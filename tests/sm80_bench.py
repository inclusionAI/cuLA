#!/usr/bin/env python3
"""Benchmark: CuTe DSL SM80 vs FLA Triton for KDA forward prefill."""

import os, sys, time, torch
sys.path.insert(0, '.')
sys.path.insert(0, 'third_party/flash-linear-attention')
os.environ['CULA_USE_SM80_CUTEDSL'] = '1'

torch.manual_seed(42)
WARMUP, ITERS = 3, 10
H, D = 1, 128  # single head for debug; change to 64 for real benchmark
configs = [
    (1, 512),    # 8 chunks
    (1, 1024),   # 16 chunks
    (1, 2048),   # 32 chunks
]

print(f"{'='*80}")
print(f" SM80 CuTe DSL vs FLA Triton — KDA Forward Prefill")
print(f" H={H}  D={D}  dtype=bf16  warmup={WARMUP}  iters={ITERS}")
print(f"{'='*80}")
print(f" {'B':>3s}  {'T':>5s}  {'FLA(ms)':>9s}  {'CuTe(ms)':>10s}  {'Speedup':>8s}  {'max_diff':>10s}")
print(f" {'─'*55}")

for B, T in configs:
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
    g = torch.randn(B, T, H, D, dtype=torch.float32, device='cuda') * 0.1
    beta = torch.randn(B, T, H, dtype=torch.float32, device='cuda') * 0.1
    initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device='cuda') * 0.01
    scale = D ** -0.5

    # FLA Triton reference
    from fla.ops.kda import chunk_kda as fla_kda
    for _ in range(WARMUP):
        fla_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=initial_state, output_final_state=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o_fla, _ = fla_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                           initial_state=initial_state, output_final_state=True)
    torch.cuda.synchronize()
    ms_fla = (time.perf_counter() - t0) / ITERS * 1000

    # CuTe DSL SM80
    from cula.kda.ampere_fused_fwd import cula_kda_prefill_ampere
    for _ in range(WARMUP):
        cula_kda_prefill_ampere(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                                initial_state=initial_state, output_final_state=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o_cute, _ = cula_kda_prefill_ampere(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                                            initial_state=initial_state, output_final_state=True)
    torch.cuda.synchronize()
    ms_cute = (time.perf_counter() - t0) / ITERS * 1000

    speedup = ms_fla / ms_cute
    max_diff = (o_fla.float() - o_cute.float()).abs().max().item()

    print(f" {B:3d}  {T:5d}  {ms_fla:9.3f}  {ms_cute:10.3f}  {speedup:7.2f}x  {max_diff:10.4f}")
    torch.cuda.empty_cache()

print(f" {'─'*55}")
print(" DONE")
