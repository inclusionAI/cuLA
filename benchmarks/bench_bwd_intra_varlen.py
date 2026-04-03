"""
Benchmark: kda_bwd_intra CUDA kernel vs FLA Triton baseline
Focused on variable-length (varlen) performance.

Varlen configurations follow flashla/bench_varlen.py:
- Quasi-balanced seq lens (max/min ratio ≤ 2-3x)
- Balanced / unbalanced / non-aligned splits
- T=8k/32k, N=15-25 sequences
- H=32/64, D=128
"""

import argparse
import pathlib
import random
import statistics
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from cula import cudac as cula_cuda
except Exception:
    cula_cuda = None

try:
    from kda.interface import kda_bwd_intra as ext_kda_bwd_intra
except Exception:
    ext_kda_bwd_intra = None

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close

torch.backends.cuda.matmul.allow_tf32 = True

# =============================================================================
# Config
# =============================================================================
K = 128  # head dim
BT = 64  # chunk size
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
REP = 100

# =============================================================================
# Helpers
# =============================================================================


def exclusive_cumsum(a):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r


def generate_quasi_balanced_seqlens(total_tokens, num_seqs, max_ratio=2.5, seed=123):
    rng = random.Random(seed)
    MIN_SEQ = 64
    weights = [rng.uniform(1.0, max_ratio) for _ in range(num_seqs)]
    w_sum = sum(weights)
    raw = [max(MIN_SEQ, int(w / w_sum * total_tokens)) for w in weights]
    diff = total_tokens - sum(raw)
    indices = sorted(range(num_seqs), key=lambda i: raw[i], reverse=True)
    for i in range(abs(diff)):
        idx = indices[i % num_seqs]
        raw[idx] += 1 if diff > 0 else -1
    return raw


def generate_quasi_balanced_seqlens_in_range(
    total_tokens,
    num_seqs,
    min_ratio=2.0,
    max_ratio=3.0,
    seed=123,
    max_tries=256,
):
    """Generate quasi-balanced seq lens whose realized max/min ratio is within [min_ratio, max_ratio]."""
    rng = random.Random(seed)
    best = None
    best_gap = float("inf")
    for _ in range(max_tries):
        sampled_ratio = rng.uniform(min_ratio, max_ratio)
        sampled_seed = rng.randint(0, 2**31 - 1)
        cand = generate_quasi_balanced_seqlens(total_tokens, num_seqs, sampled_ratio, sampled_seed)
        mn, mx = min(cand), max(cand)
        ratio = mx / mn
        if min_ratio <= ratio <= max_ratio:
            return cand
        # Keep closest candidate to avoid hard failure.
        gap = min(abs(ratio - min_ratio), abs(ratio - max_ratio))
        if gap < best_gap:
            best = cand
            best_gap = gap
    return best


def generate_balanced_seqlens(total_tokens, num_seqs):
    base = total_tokens // num_seqs
    remainder = total_tokens % num_seqs
    return [base] * (num_seqs - 1) + [base + remainder]


def generate_unbalanced_seqlens(total_tokens, num_seqs):
    if num_seqs == 1:
        return [total_tokens]
    long_len = total_tokens // 2
    remaining = total_tokens - long_len
    base = remaining // (num_seqs - 1)
    last = remaining - base * (num_seqs - 2)
    return [long_len] + [base] * (num_seqs - 2) + [last]


def generate_nonaligned_seqlens(total_tokens, num_seqs):
    base = total_tokens // num_seqs
    seqlens = []
    remaining = total_tokens
    for i in range(num_seqs - 1):
        offset = 7 if (i % 2 == 0) else -7
        sl = max(1, base + offset)
        seqlens.append(sl)
        remaining -= sl
    seqlens.append(max(1, remaining))
    return seqlens


def make_bwd_intra_inputs(seq_lens, H, beta_dtype=DTYPE):
    """Create inputs for bwd_intra benchmark given variable seq lens."""
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=DEVICE)
    chunk_indices = prepare_chunk_indices(cu_seqlens.to(torch.long), BT).to(torch.int32)

    q = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=DTYPE)
    g = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32) / 10
    beta = torch.randn(1, total_tokens, H, device=DEVICE, dtype=beta_dtype)
    dAqk = torch.randn(1, total_tokens, H, BT, device=DEVICE, dtype=torch.float32)
    dAkk = torch.randn(1, total_tokens, H, BT, device=DEVICE, dtype=torch.float32)
    dq = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32)
    dk = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32)
    db = torch.randn(1, total_tokens, H, device=DEVICE, dtype=torch.float32)
    dg = torch.randn(1, total_tokens, H, K, device=DEVICE, dtype=torch.float32)

    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices


def run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run CUDA kda_bwd_intra kernel."""
    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    if cula_cuda is not None and hasattr(cula_cuda, "chunk_kda_bwd_intra_cuda"):
        tile_counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        cula_cuda.chunk_kda_bwd_intra_cuda(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            db,
            dg,
            cu_seqlens,
            chunk_indices,
            dq_out,
            dk_out,
            db_out,
            dg_out,
            tile_counter,
            BT,
        )
        return dq_out, dk_out, db_out, dg_out

    if ext_kda_bwd_intra is not None:
        return ext_kda_bwd_intra(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            db,
            dg,
            cu_seqlens,
            chunk_indices,
            dq_out,
            dk_out,
            db_out,
            dg_out,
            BT,
        )

    raise RuntimeError("No CUDA bwd_intra backend found: expected cula.cudac or kda.interface")


def run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run FLA Triton kda_bwd_intra."""
    return chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT, True)


def bench_fn(fn, warmup=WARMUP, rep=REP):
    """Benchmark using triton's do_bench (CUDA event timing)."""
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


def print_header(title):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"{'Config':<55} {'CUDA':>9} {'FLA':>9} {'Speedup':>8}")
    print(f"{'-' * 100}")


def print_row(config, cuda_ms, fla_ms):
    speedup = fla_ms / cuda_ms if cuda_ms > 0 else float("inf")
    marker = " <--" if speedup >= 1.5 else ""
    print(f"{config:<55} {cuda_ms:>8.3f}ms {fla_ms:>7.3f}ms {speedup:>7.2f}x{marker}")


# =============================================================================
# Correctness check
# =============================================================================


def check_correctness(H=4, total_T=512, num_seqs=4, beta_dtype=DTYPE):
    """Quick correctness check before benchmarking."""
    torch.manual_seed(42)
    random.seed(42)
    seq_lens = generate_balanced_seqlens(total_T, num_seqs)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices = make_bwd_intra_inputs(
        seq_lens, H, beta_dtype=beta_dtype
    )

    dq_c, dk_c, db_c, dg_c = run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)
    dq_f, dk_f, db_f, dg_f = run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)

    assert_close("dq", dq_f, dq_c, 0.008)
    assert_close("dk", dk_f, dk_c, 0.008)
    assert_close("db", db_f, db_c, 0.02)
    assert_close("dg", dg_f, dg_c, 0.02)
    return True


# =============================================================================
# Benchmark suites
# =============================================================================


def bench_focused_varlen(H):
    """Core benchmark: T=8k/32k, N=16/20/24, quasi-balanced."""
    total_tokens_list = [8192, 32768]
    num_seqs_list = [16, 20, 24]

    print_header(f"Focused varlen: quasi-balanced (H={H})")
    hdr = (
        f"{'Config':<40} {'cuda_base':>9} {'cuda_vl':>9} {'cuda_ovhd':>9} "
        f"{'fla_base':>9} {'fla_vl':>9} {'fla_ovhd':>9} "
        f"{'base_sp':>8} {'vl_sp':>7}"
    )
    print(hdr)
    print(f"{'-' * 115}")

    for total_T in total_tokens_list:
        for N in num_seqs_list:
            torch.manual_seed(42)
            random.seed(42)
            seq_lens = generate_quasi_balanced_seqlens(total_T, N)
            mn, mx = min(seq_lens), max(seq_lens)
            ratio = mx / mn

            # Non-varlen baseline (single seq = total_T)
            base_lens = [total_T]
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_base, ci_base = make_bwd_intra_inputs(base_lens, H)

            cuda_base, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_base, ci_base))
            fla_base, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_base, ci_base))

            # Varlen
            q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl = make_bwd_intra_inputs(seq_lens, H)

            cuda_vl, _, _ = bench_fn(
                lambda: run_cuda_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl)
            )
            fla_vl, _, _ = bench_fn(
                lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl)
            )

            cuda_ovhd = (cuda_vl - cuda_base) / cuda_base * 100
            fla_ovhd = (fla_vl - fla_base) / fla_base * 100
            base_sp = fla_base / cuda_base if cuda_base > 0 else float("inf")
            vl_sp = fla_vl / cuda_vl if cuda_vl > 0 else float("inf")

            tag = f"T={total_T:>5} N={N:>2} ({mn}-{mx}, {ratio:.1f}x)"
            print(
                f"{tag:<40} {cuda_base:>8.3f}ms {cuda_vl:>7.3f}ms {cuda_ovhd:>+8.1f}% "
                f"{fla_base:>8.3f}ms {fla_vl:>7.3f}ms {fla_ovhd:>+8.1f}% "
                f"{base_sp:>7.2f}x {vl_sp:>6.2f}x"
            )
        print()


def bench_scale_total_seqlen(H):
    """Scale total seq len with fixed num_seqs."""
    num_seqs = 4
    print_header(f"Scale total seqlen (N={num_seqs}, balanced, H={H})")
    for total_T in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        per_seq = seq_lens[0]
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
        config = f"T={total_T:>5} ({num_seqs}x{per_seq})"
        cuda_ms, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        print_row(config, cuda_ms, fla_ms)


def bench_scale_num_seqs(H):
    """Scale number of sequences with fixed total length."""
    for total_T in [8192, 32768]:
        print_header(f"Scale num_seqs (total_T={total_T}, balanced, H={H})")
        for num_seqs in [1, 2, 4, 8, 16, 32, 64]:
            torch.manual_seed(42)
            seq_lens = generate_balanced_seqlens(total_T, num_seqs)
            per_seq = seq_lens[0]
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
            config = f"T={total_T:>5} N={num_seqs:<3} (each~{per_seq})"
            cuda_ms, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
            fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
            print_row(config, cuda_ms, fla_ms)


def bench_balanced_vs_unbalanced(H):
    """Compare balanced vs unbalanced seq distributions."""
    for total_T in [8192, 32768]:
        print_header(f"Balanced vs Unbalanced (total_T={total_T}, H={H})")
        for num_seqs in [4, 8, 16]:
            torch.manual_seed(42)
            # Balanced
            seq_lens_b = generate_balanced_seqlens(total_T, num_seqs)
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens_b, H)
            cb, fb = (
                bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0],
                bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0],
            )
            print_row(f"balanced   N={num_seqs:<3} (each={seq_lens_b[0]})", cb, fb)

            # Unbalanced
            seq_lens_u = generate_unbalanced_seqlens(total_T, num_seqs)
            q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2 = make_bwd_intra_inputs(seq_lens_u, H)
            cu_ms, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))
            fu_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))
            longest = max(seq_lens_u)
            shortest = min(seq_lens_u)
            print_row(f"unbalanced N={num_seqs:<3} (max={longest},min={shortest})", cu_ms, fu_ms)


def bench_varlen_overhead(H):
    """Measure varlen overhead: non-varlen vs varlen single-seq."""
    print_header(f"Varlen overhead: non-varlen(single seq) vs varlen N-seq (H={H})")
    hdr = (
        f"{'T':<7} {'cu_base':>9} {'cu_vl':>9} {'cu_ovhd':>8} "
        f"{'fl_base':>9} {'fl_vl':>9} {'fl_ovhd':>9} "
        f"{'base_sp':>8} {'vl_sp':>7}"
    )
    print(hdr)
    print(f"{'-' * 100}")

    for T in [1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)

        # Single seq (non-varlen equivalent)
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1 = make_bwd_intra_inputs([T], H)
        cu_no, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1))
        fl_no, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1))

        # Varlen with 16 quasi-balanced seqs
        seq_lens = generate_quasi_balanced_seqlens(T, 16)
        q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2 = make_bwd_intra_inputs(seq_lens, H)
        cu_vl, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))
        fl_vl, _, _ = bench_fn(lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))

        cu_ovhd = (cu_vl - cu_no) / cu_no * 100 if cu_no > 0 else 0
        fl_ovhd = (fl_vl - fl_no) / fl_no * 100 if fl_no > 0 else 0
        base_sp = fl_no / cu_no if cu_no > 0 else float("inf")
        vl_sp = fl_vl / cu_vl if cu_vl > 0 else float("inf")

        print(
            f"T={T:<5} {cu_no:>8.3f}ms {cu_vl:>7.3f}ms {cu_ovhd:>+7.1f}% "
            f"{fl_no:>8.3f}ms {fl_vl:>7.3f}ms {fl_ovhd:>+8.1f}% "
            f"{base_sp:>7.2f}x {vl_sp:>6.2f}x"
        )


def bench_realistic_prefill(H):
    """Realistic prefill scenarios."""
    print_header(f"Realistic prefill scenarios (H={H})")
    configs = [
        ("4 short prompts", [128, 256, 192, 64]),
        ("4 medium prompts", [512, 1024, 768, 896]),
        ("4 long prompts", [2048, 4096, 2048, 4096]),
        ("mixed short+long", [64, 4096, 128, 2048]),
        ("8 chat turns", [64, 128, 32, 256, 512, 64, 128, 1024]),
        ("batch of 16 (256 each)", [256] * 16),
        ("batch of 32 (128 each)", [128] * 32),
        ("batch of 8 (1024 each)", [1024] * 8),
        ("batch of 4 (4096 each)", [4096] * 4),
        ("batch of 4 (8192 each)", [8192] * 4),
        ("batch of 8 (4096 each)", [4096] * 8),
    ]
    for desc, seq_lens in configs:
        torch.manual_seed(42)
        total = sum(seq_lens)
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
        config = f"{desc} (T={total}, N={len(seq_lens)})"
        cuda_ms, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        print_row(config, cuda_ms, fla_ms)


def bench_varlen_8k_16k_vs_fla(H):
    """
    Expanded varlen benchmark:
    - total tokens in [8k, 16k]
    - number of sequences in [15, 25]
    - realized longest/shortest ratio in [2x, 3x]
    - focus on CUDA vs FLA speedup statistics
    """
    total_tokens_list = [8192, 12288, 16384]
    num_seqs_list = [15, 18, 20, 22, 25]
    ratio_bands = [(2.0, 2.4), (2.4, 3.0)]
    seeds = [42, 123]

    print_header(f"Expanded varlen 8k-16k vs FLA (H={H})")

    speedups = []
    for total_T in total_tokens_list:
        for N in num_seqs_list:
            for r_lo, r_hi in ratio_bands:
                for sd in seeds:
                    seq_lens = generate_quasi_balanced_seqlens_in_range(
                        total_tokens=total_T,
                        num_seqs=N,
                        min_ratio=r_lo,
                        max_ratio=r_hi,
                        seed=sd,
                    )
                    mn, mx = min(seq_lens), max(seq_lens)
                    ratio = mx / mn

                    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
                    cuda_ms, _, _ = bench_fn(lambda: run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
                    fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))

                    sp = fla_ms / cuda_ms if cuda_ms > 0 else float("inf")
                    speedups.append(sp)
                    tag = f"T={total_T:>5} N={N:>2} ({mn:>3}-{mx:<4}, {ratio:.2f}x, s={sd})"
                    print(f"{tag:<52} {cuda_ms:>8.3f}ms {fla_ms:>7.3f}ms {sp:>7.2f}x")

    print(f"{'-' * 100}")
    if speedups:
        avg_sp = statistics.mean(speedups)
        med_sp = statistics.median(speedups)
        min_sp = min(speedups)
        max_sp = max(speedups)
        ge_1p0 = sum(1 for x in speedups if x >= 1.0)
        ge_1p2 = sum(1 for x in speedups if x >= 1.2)
        ge_1p5 = sum(1 for x in speedups if x >= 1.5)
        total = len(speedups)
        print(
            f"Summary(H={H}): avg={avg_sp:.3f}x, median={med_sp:.3f}x, "
            f"min={min_sp:.3f}x, max={max_sp:.3f}x, "
            f">=1.0x {ge_1p0}/{total}, >=1.2x {ge_1p2}/{total}, >=1.5x {ge_1p5}/{total}"
        )


# =============================================================================
# Determinism check
# =============================================================================


def check_determinism(H=4, total_T=512, num_seqs=4, iters=20, beta_dtype=DTYPE):
    """Verify deterministic outputs across repeated runs."""
    torch.manual_seed(42)
    seq_lens = generate_balanced_seqlens(total_T, num_seqs)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H, beta_dtype=beta_dtype)

    ref_dq, ref_dk, ref_db, ref_dg = run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
    for i in range(iters):
        dq_out, dk_out, db_out, dg_out = run_cuda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        assert torch.equal(dq_out, ref_dq), f"dq mismatch at iter {i}"
        assert torch.equal(dk_out, ref_dk), f"dk mismatch at iter {i}"
        assert torch.equal(db_out, ref_db), f"db mismatch at iter {i}"
        assert torch.equal(dg_out, ref_dg), f"dg mismatch at iter {i}"
    return True


# =============================================================================
# Main
# =============================================================================


def run_all_benchmarks(H):
    print(f"\n{'#' * 100}")
    print(f"#  kda_bwd_intra benchmark: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    # Quick correctness + determinism checks
    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness check (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")
    print(f"  Determinism check (H={H}, beta={torch.bfloat16})...", end="", flush=True)
    check_determinism(H, beta_dtype=torch.bfloat16)
    print(" PASS")

    # Run benchmarks
    bench_focused_varlen(H)
    bench_varlen_overhead(H)
    bench_scale_total_seqlen(H)
    bench_scale_num_seqs(H)
    bench_balanced_vs_unbalanced(H)
    bench_realistic_prefill(H)
    bench_varlen_8k_16k_vs_fla(H)


def run_expanded_varlen_only(H):
    """Run only expanded varlen suite with quick validation checks."""
    print(f"\n{'#' * 100}")
    print(f"#  kda_bwd_intra expanded varlen benchmark: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness check (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")

    bench_varlen_8k_16k_vs_fla(H)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark kda_bwd_intra varlen performance vs FLA")
    parser.add_argument(
        "--suite",
        choices=["all", "expanded_varlen"],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Head counts to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warmup iterations for each benchmark")
    parser.add_argument("--rep", type=int, default=REP, help="Measurement repetitions for each benchmark")
    args = parser.parse_args()

    WARMUP = args.warmup
    REP = args.rep

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"K={K}, BT={BT}, dtype={DTYPE}, warmup={WARMUP}, rep={REP}")

    for H in args.heads:
        if args.suite == "expanded_varlen":
            run_expanded_varlen_only(H)
        else:
            run_all_benchmarks(H)

    print(f"\n{'=' * 100}")
    print("  All benchmarks done.")
    print(f"{'=' * 100}")
