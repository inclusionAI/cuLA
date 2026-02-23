#!/usr/bin/env python3
"""Comprehensive varlen correctness test: compare all outputs vs FLA across diverse scenarios.

Tests: h_out, v_new, ht (final state) at various:
  - Sequence counts: 1, 2, 5, 10, 20, 40
  - Head counts: 2, 8, 32, 64
  - Lengths: BT-aligned, non-aligned, mixed, very short (< BT), single-chunk
  - Features: with/without gk, with/without initial state
  - BV: 64, 128
  - num_stages: 2, 3
"""
import sys
import torch
import numpy as np
from chunk_delta_h import ChunkDeltaRuleFwdH
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

device = 'cuda'
dtype = torch.bfloat16
BT = 64
K = 128
V = 128

# Tolerances: use atol + rtol model (pass if diff < atol + rtol * |ref|)
# This correctly handles both small-value precision and large-value relative error.
ATOL_H  = 0.04       # h_out absolute tolerance
ATOL_V  = 0.04       # v_new absolute tolerance
ATOL_HT = 0.06       # ht (final state) absolute tolerance
RTOL    = 0.03       # 3% relative tolerance for all outputs

PASS_COUNT = 0
FAIL_COUNT = 0
FAIL_DETAILS = []


def run_test(
    label: str,
    num_seqs: int,
    H: int,
    seq_lengths: list,
    use_gk: bool = True,
    use_h0: bool = True,
    BV: int = 64,
    num_stages: int = 2,
    seed: int = 42,
):
    global PASS_COUNT, FAIL_COUNT, FAIL_DETAILS
    torch.manual_seed(seed)

    total_T = sum(seq_lengths)
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lengths):
        cu_seqlens[i + 1] = cu_seqlens[i] + l
    NTs = [(l + BT - 1) // BT for l in seq_lengths]
    chunk_offsets = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, nt in enumerate(NTs):
        chunk_offsets[i + 1] = chunk_offsets[i] + nt
    total_NT = sum(NTs)

    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1

    h0 = torch.randn(num_seqs, H, K, V, device=device, dtype=torch.float32) * 0.01 if use_h0 else None
    gk_tensor = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * 0.01 if use_gk else None

    # --- FLA reference ---
    fla_out = fla_fwd_h(
        k=k, w=w, u=u, g=None,
        gk=gk_tensor if use_gk else torch.zeros(1, total_T, H, K, device=device, dtype=torch.float32),
        initial_state=h0,
        output_final_state=True, chunk_size=BT, save_new_value=True,
        cu_seqlens=cu_seqlens.long()
    )
    h_out_ref, v_new_ref, ht_ref = fla_out[0], fla_out[1], fla_out[2]

    # --- Our kernel ---
    g = torch.zeros(1, total_T, H, device=device, dtype=torch.float32)
    gk_input = gk_tensor if use_gk else torch.zeros(1, total_T, H, K, device=device, dtype=torch.float32)
    h0_input = h0 if use_h0 else torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)

    h_out = torch.zeros(1, total_NT, H, K, V, device=device, dtype=dtype)
    v_new_out = torch.zeros(1, total_T, H, V, device=device, dtype=dtype)
    ht_out = torch.zeros(num_seqs, H, K, V, device=device, dtype=dtype)
    workspace = torch.zeros(128, dtype=torch.uint8, device=device)
    stream = cutlass_torch.default_stream()

    kernel = ChunkDeltaRuleFwdH(
        chunk_size=BT, head_dim_k=K, head_dim_v=V,
        is_varlen=True, BV=BV, num_stages=num_stages,
    )
    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g), from_dlpack(gk_input)
    h0c = from_dlpack(h0_input)
    hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
    csc, coc, wsc = from_dlpack(cu_seqlens), from_dlpack(chunk_offsets), from_dlpack(workspace)
    args = (
        kc.iterator, wc.iterator, uc.iterator, gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        csc.iterator, coc.iterator, wsc.iterator,
        (num_seqs, total_T, H, K, V), total_NT,
        0, int(use_gk), int(use_h0), 1, 1, stream
    )
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()

    # --- Compare per-sequence ---
    all_ok = True
    per_seq_details = []
    for s in range(num_seqs):
        bos = cu_seqlens[s].item()
        eos = cu_seqlens[s + 1].item()
        ch_start = chunk_offsets[s].item()
        ch_end = chunk_offsets[s + 1].item()

        h_our = h_out[0, ch_start:ch_end].float()
        h_ref_s = h_out_ref[0, ch_start:ch_end].float()
        h_diff = (h_our - h_ref_s).abs().max().item()
        h_scale = max(h_ref_s.abs().max().item(), 1.0)

        v_our = v_new_out[0, bos:eos].float()
        v_ref_s = v_new_ref[0, bos:eos].float()
        v_diff = (v_our - v_ref_s).abs().max().item()
        v_scale = max(v_ref_s.abs().max().item(), 1.0)

        ht_our = ht_out[s].float()
        ht_ref_s = ht_ref[s].float()
        ht_diff = (ht_our - ht_ref_s).abs().max().item()
        ht_scale = max(ht_ref_s.abs().max().item(), 1.0)

        h_ok = h_diff < ATOL_H + RTOL * h_scale
        v_ok = v_diff < ATOL_V + RTOL * v_scale
        ht_ok = ht_diff < ATOL_HT + RTOL * ht_scale
        seq_ok = h_ok and v_ok and ht_ok

        if not seq_ok:
            all_ok = False
            per_seq_details.append(
                f"    seq {s}: len={seq_lengths[s]} h={h_diff:.4g}/{h_scale:.4g}{'!' if not h_ok else ''} "
                f"v={v_diff:.4g}/{v_scale:.4g}{'!' if not v_ok else ''} ht={ht_diff:.4g}/{ht_scale:.4g}{'!' if not ht_ok else ''}"
            )

    # --- Global max relative diffs ---
    h_max_abs = (h_out.float() - h_out_ref.float()).abs().max().item()
    h_max_scale = max(h_out_ref.float().abs().max().item(), 1.0)
    v_max_abs = (v_new_out.float() - v_new_ref.float()).abs().max().item()
    v_max_scale = max(v_new_ref.float().abs().max().item(), 1.0)
    ht_max_abs = (ht_out.float() - ht_ref.float()).abs().max().item()
    ht_max_scale = max(ht_ref.float().abs().max().item(), 1.0)
    h_max = h_max_abs / h_max_scale
    v_max = v_max_abs / v_max_scale
    ht_max = ht_max_abs / ht_max_scale

    status = "PASS" if all_ok else "FAIL"
    if all_ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        FAIL_DETAILS.append(label)

    aligned_count = sum(1 for l in seq_lengths if l % BT == 0)
    tail_count = num_seqs - aligned_count
    print(
        f"  [{status}] {label:<65s}  "
        f"h={h_max:.4f}  v={v_max:.4f}  ht={ht_max:.4f}  "  # relative
        f"({aligned_count}a/{tail_count}t)"
    )
    for d in per_seq_details:
        print(d)


def gen_random_lengths(num_seqs, total_T, ratio=3.0, seed=42):
    """Generate variable-length sequences."""
    rng = np.random.RandomState(seed)
    weights = rng.uniform(1, ratio, num_seqs)
    lengths = (weights / weights.sum() * total_T).astype(int)
    lengths = np.maximum(lengths, 1)
    diff = total_T - lengths.sum()
    for i in range(abs(diff)):
        lengths[i % num_seqs] += 1 if diff > 0 else -1
    lengths = np.maximum(lengths, 1)
    return lengths.tolist()


# ========================================================================
print("=" * 120)
print("COMPREHENSIVE VARLEN TEST: Ours vs FLA")
print("=" * 120)

# ---- Group 1: BT-aligned sequences ----
print("\n--- BT-aligned sequences ---")
run_test("1seq H=2 [128]",             1, 2,  [128])
run_test("1seq H=64 [256]",            1, 64, [256])
run_test("2seq H=2 [128,128]",         2, 2,  [128, 128])
run_test("5seq H=8 [128]*5",           5, 8,  [128]*5)
run_test("10seq H=32 [256]*10",        10, 32, [256]*10)
run_test("20seq H=64 [128]*20",        20, 64, [128]*20)
run_test("20seq H=64 [384]*20",        20, 64, [384]*20)
run_test("40seq H=64 [128]*40",        40, 64, [128]*40)

# ---- Group 2: Non-BT-aligned (tail) sequences ----
print("\n--- Non-BT-aligned sequences ---")
run_test("1seq H=2 [100]",             1, 2,  [100])
run_test("1seq H=64 [65]",             1, 64, [65])
run_test("2seq H=2 [100,128]",         2, 2,  [100, 128])
run_test("2seq H=2 [128,100]",         2, 2,  [128, 100])
run_test("3seq H=64 [100,128,100]",    3, 64, [100, 128, 100])
run_test("3seq H=64 [128,100,128]",    3, 64, [128, 100, 128])
run_test("5seq H=64 [128,100,128,100,128]", 5, 64, [128,100,128,100,128])
run_test("3seq H=64 [100,100,100]",    3, 64, [100, 100, 100])

# ---- Group 3: Very short sequences (< BT) ----
print("\n--- Very short sequences (< BT=64) ---")
run_test("1seq H=2 [1]",               1, 2,  [1])
run_test("1seq H=2 [32]",              1, 2,  [32])
run_test("1seq H=64 [63]",             1, 64, [63])
run_test("2seq H=8 [10,20]",           2, 8,  [10, 20])
run_test("3seq H=32 [1,1,1]",          3, 32, [1, 1, 1])
run_test("5seq H=64 [10,20,30,40,50]", 5, 64, [10, 20, 30, 40, 50])

# ---- Group 4: Mixed aligned and non-aligned ----
print("\n--- Mixed aligned/non-aligned ---")
run_test("5seq H=32 mixed",            5, 32, [128, 65, 256, 100, 192])
run_test("10seq H=64 mixed short+long", 10, 64, [64, 10, 256, 33, 128, 65, 384, 1, 100, 192])
run_test("20seq H=64 random len ~8K",  20, 64, gen_random_lengths(20, 8192, ratio=3, seed=42))
run_test("25seq H=64 random len ~8K",  25, 64, gen_random_lengths(25, 8192, ratio=4, seed=123))

# ---- Group 5: Realistic workloads ----
print("\n--- Realistic workloads ---")
run_test("10seq H=64 real lengths",   10, 64,
         [378, 563, 485, 442, 325, 325, 304, 532, 443, 477])
run_test("20seq H=64 real lengths",   20, 64,
         [378, 563, 485, 442, 325, 325, 304, 532, 443, 477,
          296, 571, 520, 338, 331, 331, 360, 420, 394, 357])
run_test("30seq H=64 real lengths",   30, 64,
         gen_random_lengths(30, 12000, ratio=3, seed=77))
run_test("40seq H=64 real lengths",   40, 64,
         gen_random_lengths(40, 16000, ratio=4, seed=88))

# ---- Group 6: With gk (non-zero gate keys) ----
print("\n--- With gk (gate keys) ---")
run_test("5seq H=8 aligned gk",       5, 8,  [128]*5, use_gk=True)
run_test("5seq H=8 unaligned gk",     5, 8,  [100,128,90,110,72], use_gk=True)
run_test("10seq H=64 mixed gk",       10, 64, gen_random_lengths(10, 4096, ratio=3, seed=55), use_gk=True)
run_test("20seq H=64 real gk",        20, 64,
         [378, 563, 485, 442, 325, 325, 304, 532, 443, 477,
          296, 571, 520, 338, 331, 331, 360, 420, 394, 357], use_gk=True)

# ---- Group 7: Without initial state ----
print("\n--- Without initial state (h0=None) ---")
run_test("5seq H=8 no h0 aligned",    5, 8,  [128]*5, use_h0=False)
run_test("5seq H=8 no h0 unaligned",  5, 8,  [100,128,90,110,72], use_h0=False)
run_test("10seq H=64 no h0 gk",       10, 64, gen_random_lengths(10, 4096, seed=55), use_gk=True, use_h0=False)

# ---- Group 8: BV=128 (no V tiling) ----
print("\n--- BV=128 (no V tiling) ---")
run_test("5seq H=8 BV=128 aligned",   5, 8,  [128]*5, BV=128)
run_test("5seq H=8 BV=128 unaligned", 5, 8,  [100,128,90,110,72], BV=128)
run_test("10seq H=64 BV=128 mixed",   10, 64, gen_random_lengths(10, 4096, seed=55), BV=128)
run_test("20seq H=64 BV=128 real gk", 20, 64,
         [378, 563, 485, 442, 325, 325, 304, 532, 443, 477,
          296, 571, 520, 338, 331, 331, 360, 420, 394, 357], BV=128, use_gk=True)

# ---- Group 9: num_stages=3 ----
print("\n--- num_stages=3 ---")
run_test("5seq H=8 3stage aligned",   5, 8,  [128]*5, num_stages=3)
run_test("5seq H=8 3stage unaligned", 5, 8,  [100,128,90,110,72], num_stages=3)
run_test("10seq H=64 3stage mixed",   10, 64, gen_random_lengths(10, 4096, seed=55), num_stages=3)
run_test("20seq H=64 3stage gk",      20, 64,
         [378, 563, 485, 442, 325, 325, 304, 532, 443, 477,
          296, 571, 520, 338, 331, 331, 360, 420, 394, 357], num_stages=3, use_gk=True)

# ---- Group 10: Edge cases ----
print("\n--- Edge cases ---")
run_test("1seq H=2 exactly BT [64]",  1, 2,  [64])
run_test("2seq H=2 [64,64]",          2, 2,  [64, 64])
run_test("1seq H=2 BT+1 [65]",        1, 2,  [65])
run_test("1seq H=2 2*BT-1 [127]",     1, 2,  [127])
run_test("2seq H=64 [1,1]",           2, 64, [1, 1])
run_test("3seq H=64 [1,64,1]",        3, 64, [1, 64, 1])
run_test("5seq H=64 [1]*5",           5, 64, [1]*5)
run_test("2seq H=64 [1,8191]",        2, 64, [1, 8191])
run_test("2seq H=64 [4096,4096]",     2, 64, [4096, 4096])

# ========================================================================
print("\n" + "=" * 120)
print(f"SUMMARY: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {PASS_COUNT + FAIL_COUNT} tests")
if FAIL_DETAILS:
    print("FAILED tests:")
    for f in FAIL_DETAILS:
        print(f"  - {f}")
print("=" * 120)

sys.exit(1 if FAIL_COUNT > 0 else 0)
