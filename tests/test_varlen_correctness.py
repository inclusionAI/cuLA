"""
Comprehensive correctness test: flashla varlen vs FLA chunk_kda.

Tests flashla output (o) and final state (ht) against FLA across various
sequence length configurations, including:
  - Single short / long sequences
  - Multiple balanced sequences
  - Non-aligned (not multiples of chunk_size=64) sequences
  - Unbalanced sequences (one long + many short)
  - Many tiny sequences
  - Large H (32, 64)
  - With / without initial state
"""

import sys
import pathlib

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda import chunk_kda
from flashla.kda_wrapper import flash_kda_prefill

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
D = 128
SCALE = D ** -0.5


def exclusive_cumsum(seq_lens):
    result = [0]
    for s in seq_lens:
        result.append(result[-1] + s)
    return result


def make_inputs(seq_lens, H, has_initial_state=True):
    total_T = sum(seq_lens)
    N = len(seq_lens)
    cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=DEVICE)

    torch.manual_seed(42)
    q = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
    v = torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float, device=DEVICE)).clamp(-5, 0)
    beta = torch.randn(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()

    h0 = None
    if has_initial_state:
        h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)

    return q, k, v, g, beta, cu, h0


def compare(seq_lens, H, has_initial_state=True, label=""):
    q, k, v, g, beta, cu, h0 = make_inputs(seq_lens, H, has_initial_state)

    # flashla
    o_fl, ht_fl = flash_kda_prefill(
        q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
        initial_state=h0, output_final_state=True,
        safe_gate=True, cu_seqlens=cu,
    )

    # FLA (reference)
    o_fla, ht_fla = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=SCALE,
        initial_state=h0, output_final_state=True,
        safe_gate=True, cu_seqlens=cu,
    )

    # Compute errors
    o_diff = (o_fl.float() - o_fla.float()).abs()
    o_max = o_diff.max().item()
    o_mean = o_diff.mean().item()
    o_ref_norm = o_fla.float().abs().mean().item()
    o_rel = o_mean / (o_ref_norm + 1e-8)

    ht_diff = (ht_fl.float() - ht_fla.float()).abs()
    ht_max = ht_diff.max().item()
    ht_mean = ht_diff.mean().item()
    ht_ref_norm = ht_fla.float().abs().mean().item()
    ht_rel = ht_mean / (ht_ref_norm + 1e-8)

    total_T = sum(seq_lens)
    N = len(seq_lens)

    # Check for NaN/Inf
    o_nan = torch.isnan(o_fl).any().item() or torch.isinf(o_fl).any().item()
    ht_nan = torch.isnan(ht_fl).any().item() or torch.isinf(ht_fl).any().item()

    # Thresholds
    o_ok = o_max < 0.02 and not o_nan
    ht_ok = ht_max < 0.5 and not ht_nan  # state accumulates error

    status = "PASS" if (o_ok and ht_ok) else "FAIL"

    lens_str = str(seq_lens) if len(seq_lens) <= 5 else f"{seq_lens[:3]}...({N} seqs)"
    print(f"  [{status}] {label:<40} H={H:<3} T={total_T:<6} N={N:<4} "
          f"o_max={o_max:.5f} o_rel={o_rel:.6f} "
          f"ht_max={ht_max:.4f} ht_rel={ht_rel:.6f}"
          f"{'  NaN!' if o_nan or ht_nan else ''}")

    return status == "PASS", {
        "label": label, "H": H, "total_T": total_T, "N": N,
        "seq_lens": seq_lens,
        "o_max": o_max, "o_mean": o_mean, "o_rel": o_rel,
        "ht_max": ht_max, "ht_mean": ht_mean, "ht_rel": ht_rel,
        "o_nan": o_nan, "ht_nan": ht_nan,
        "has_h0": has_initial_state,
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}, D={D}, dtype={DTYPE}")
    print()

    results = []
    all_pass = True

    # =========================================================================
    # Category 1: Single sequence (various lengths)
    # =========================================================================
    print("=" * 100)
    print("Category 1: Single sequence (various lengths)")
    print("=" * 100)
    for H in [32, 64]:
        for T in [15, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1000, 1024, 2048, 4096, 8192]:
            ok, r = compare([T], H, label=f"single_T{T}")
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Category 2: Balanced sequences (aligned)
    # =========================================================================
    print()
    print("=" * 100)
    print("Category 2: Balanced sequences (aligned to chunk_size=64)")
    print("=" * 100)
    for H in [32, 64]:
        for N, per_seq in [(2, 2048), (4, 1024), (4, 2048), (8, 512), (8, 1024),
                           (16, 256), (16, 512), (32, 128), (32, 256), (64, 128)]:
            seq_lens = [per_seq] * N
            ok, r = compare(seq_lens, H, label=f"balanced_N{N}_per{per_seq}")
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Category 3: Non-aligned sequences
    # =========================================================================
    print()
    print("=" * 100)
    print("Category 3: Non-aligned sequences (not multiples of 64)")
    print("=" * 100)
    non_aligned_configs = [
        ([15], "tiny_15"),
        ([63], "just_under_64"),
        ([65], "just_over_64"),
        ([15, 100, 300, 1200, 2000], "mixed_nonaligned_5"),
        ([100, 300, 1200, 3000, 4096], "mixed_nonaligned_5b"),
        ([33, 67, 129, 255, 513, 1025], "all_odd_6"),
        ([7, 13, 31, 61, 127, 253, 509], "primes_7"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "tiny_1to10"),
        ([1] * 32, "all_ones_32"),
        ([3] * 64, "all_threes_64"),
        ([63, 65, 63, 65, 63, 65, 63, 65], "alternating_63_65"),
        ([100, 200, 300, 400, 500, 600, 700, 800], "hundreds_8"),
        ([1000, 1, 1000, 1, 1000, 1], "long_short_alt"),
        ([4096, 1], "one_long_one_tiny"),
        ([1, 4096], "one_tiny_one_long"),
    ]
    for H in [32, 64]:
        for seq_lens, label in non_aligned_configs:
            ok, r = compare(seq_lens, H, label=label)
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Category 4: Unbalanced sequences
    # =========================================================================
    print()
    print("=" * 100)
    print("Category 4: Unbalanced sequences")
    print("=" * 100)
    unbalanced_configs = [
        ([4000, 64, 64, 64], "one_long_rest_short_4"),
        ([2048, 128, 128, 128, 128, 128, 128, 128], "one_long_7short_8"),
        ([8000, 32, 32, 32, 32, 32, 32], "very_unbalanced_7"),
        ([256, 500, 1000, 2000, 4000], "increasing_5"),
        ([4000, 2000, 1000, 500, 256], "decreasing_5"),
        ([64, 4096, 64, 4096], "alternating_short_long"),
    ]
    for H in [32, 64]:
        for seq_lens, label in unbalanced_configs:
            ok, r = compare(seq_lens, H, label=label)
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Category 5: Large scale
    # =========================================================================
    print()
    print("=" * 100)
    print("Category 5: Large scale sequences")
    print("=" * 100)
    for H in [32, 64]:
        for T in [16384, 32768]:
            ok, r = compare([T], H, label=f"single_large_T{T}")
            results.append(r)
            if not ok:
                all_pass = False
        for N in [4, 8, 16]:
            T_per = 4096
            ok, r = compare([T_per]*N, H, label=f"large_N{N}_per{T_per}")
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Category 6: Without initial state
    # =========================================================================
    print()
    print("=" * 100)
    print("Category 6: Without initial state (h0=None)")
    print("=" * 100)
    for H in [32, 64]:
        for seq_lens, label in [
            ([1024], "single_1024_no_h0"),
            ([512, 512, 512, 512], "balanced_4x512_no_h0"),
            ([63, 129, 255], "nonaligned_3_no_h0"),
            ([4096, 64, 64], "unbalanced_3_no_h0"),
        ]:
            ok, r = compare(seq_lens, H, has_initial_state=False, label=label)
            results.append(r)
            if not ok:
                all_pass = False

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 100)
    n_pass = sum(1 for r in results if r["o_max"] < 0.02 and r["ht_max"] < 0.5 and not r["o_nan"] and not r["ht_nan"])
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} passed")

    # Show worst cases
    results_sorted_o = sorted(results, key=lambda r: r["o_max"], reverse=True)
    print(f"\nTop 10 worst output max error:")
    for r in results_sorted_o[:10]:
        print(f"  {r['label']:<40} H={r['H']:<3} T={r['total_T']:<6} N={r['N']:<4} o_max={r['o_max']:.5f} o_rel={r['o_rel']:.6f}")

    results_sorted_ht = sorted(results, key=lambda r: r["ht_max"], reverse=True)
    print(f"\nTop 10 worst state max error:")
    for r in results_sorted_ht[:10]:
        print(f"  {r['label']:<40} H={r['H']:<3} T={r['total_T']:<6} N={r['N']:<4} ht_max={r['ht_max']:.4f} ht_rel={r['ht_rel']:.6f}")

    # Failures
    failures = [r for r in results if r["o_max"] >= 0.02 or r["ht_max"] >= 0.5 or r["o_nan"] or r["ht_nan"]]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"  {r['label']:<40} H={r['H']:<3} T={r['total_T']:<6} N={r['N']:<4} "
                  f"o_max={r['o_max']:.5f} ht_max={r['ht_max']:.4f} nan={r['o_nan'] or r['ht_nan']}")
    else:
        print("\nAll tests PASSED!")

    print("=" * 100)


if __name__ == "__main__":
    main()
