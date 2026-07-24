"""
Benchmark: SM100 tcgen05 and portable mma.sync CuTeDSL vs FLA Triton.

Validates correctness and measures both CuTeDSL implementations against the
FLA Triton reference. The mma.sync implementation is shared with SM90.

Input convention (mirrors flashla benchmark):
  - q, k, g:       [1, total_tokens, H, K]  bf16 / f32
  - beta:          [1, total_tokens, H]      bf16 or f32
  - dAqk, dAkk:   [1, total_tokens, H, BT]  f32
  - dq, dk, dg:   [1, total_tokens, H, K]   f32
  - db:            [1, total_tokens, H]      f32
  - cu_seqlens:    [N+1] i32
  - chunk_indices: [num_chunks, 2] i32  (from prepare_chunk_indices)
"""

import argparse
import pathlib
import random
import statistics
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from cutlass.cute.typing import Int32
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close

from cula.ops.kda_bwd_intra_mma import kda_bwd_intra_mma
from cula.ops.kda_bwd_intra_sm100 import compile_kda_bwd_intra

# =============================================================================
# Config
# =============================================================================
K = 128  # head dim
BT = 64  # chunk size (T_TILE)
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
REP = 100

# =============================================================================
# Sequence length generators (mirrors flashla benchmark)
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


def generate_quasi_balanced_seqlens_in_range(total_tokens, num_seqs, min_ratio=2.0, max_ratio=3.0, seed=123, max_tries=256):
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


# =============================================================================
# Input preparation
# =============================================================================


def make_bwd_intra_inputs(seq_lens, H, beta_dtype=torch.float32):
    """Create inputs for bwd_intra benchmark."""
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


# =============================================================================
# Kernel runners
# =============================================================================


def run_cula_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run the CuTeDSL KDABwdIntraSM100 kernel.

    Inputs follow the FLA convention:
      q, k, g:     [1, T, H, K]
      beta, db:    [1, T, H]
      dAqk, dAkk:  [1, T, H, BT]
      dq, dk, dg:  [1, T, H, K]

    Returns (dq_out, dk_out, db_out, dg_out) matching FLA output convention:
      dq_out, dk_out, dg_out: [1, T, H, K]          (bf16 / f32)
      db_out:                 [1, T, H]              (f32)
    """
    # Ensure contiguous (keep batch=1 dim)
    q2 = q.contiguous()  # [1, T, H, K] bf16
    k2 = k.contiguous()  # [1, T, H, K] bf16
    g2 = g.contiguous()  # [1, T, H, K] f32
    beta2 = beta.contiguous()  # [1, T, H]    bf16 or f32
    dAqk2 = dAqk.contiguous()  # [1, T, H, BT] f32
    dAkk2 = dAkk.contiguous()  # [1, T, H, BT] f32
    dq2 = dq.contiguous()  # [1, T, H, K] f32
    dk2 = dk.contiguous()  # [1, T, H, K] f32
    dg2 = dg.contiguous()  # [1, T, H, K] f32
    db2 = db.contiguous()  # [1, T, H]    f32

    _, T, H_sz, K_sz = q2.shape

    # Output tensors in cuLA layout (batch=1)
    dq_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.bfloat16)
    dk_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.bfloat16)
    dg_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.float32)
    db_out = torch.empty(1, T, H_sz, device=DEVICE, dtype=torch.float32)

    # Persistent kernel tile counter: must be zeroed every call
    tile_counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    # chunk_indices: flat [2 * num_chunks]; total_tiles = num_chunks * H
    ci_flat = chunk_indices.reshape(-1).contiguous()
    num_chunks = ci_flat.shape[0] // 2
    num_tiles = num_chunks * H_sz

    compiled_fn = compile_kda_bwd_intra(H_sz, K=K_sz, BT=BT, beta_dtype=beta2.dtype)
    compiled_fn(
        q2,
        k2,
        g2,
        dAqk2,
        dAkk2,
        dq2,
        dk2,
        dg2,
        db2,
        beta2,
        dq_out,
        dk_out,
        dg_out,
        db_out,
        tile_counter,
        cu_seqlens,
        ci_flat,
        (Int32(T), Int32(H_sz), Int32(K_sz)),
        Int32(num_tiles),
    )
    torch.cuda.synchronize()

    return (
        dq_out,  # [1, T, H, K] bf16
        dk_out,  # [1, T, H, K] bf16
        db_out,  # [1, T, H]    f32
        dg_out,  # [1, T, H, K] f32
    )


def _prepare_mma_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Prepare the portable mma.sync CuTeDSL kernel and its outputs."""
    q2 = q.contiguous()
    k2 = k.contiguous()
    g2 = g.contiguous()
    beta2 = beta.contiguous()
    dAqk2 = dAqk.contiguous()
    dAkk2 = dAkk.contiguous()
    dq2 = dq.contiguous()
    dk2 = dk.contiguous()
    db2 = db.contiguous()
    dg2 = dg.contiguous()
    cu_seqlens2 = cu_seqlens.to(torch.int32).contiguous()
    chunk_indices2 = chunk_indices.to(torch.int32).contiguous()

    dq_out = torch.empty_like(q2)
    dk_out = torch.empty_like(k2)
    db_out = torch.empty_like(db2, dtype=torch.float32)
    dg_out = torch.empty_like(dg2, dtype=torch.float32)

    def fn():
        kda_bwd_intra_mma(
            q2,
            k2,
            g2,
            beta2,
            dAqk2,
            dAkk2,
            dq2,
            dk2,
            db2,
            dg2,
            cu_seqlens2,
            chunk_indices2,
            dq_out,
            dk_out,
            db_out,
            dg_out,
            BT,
        )

    return fn, (dq_out, dk_out, db_out, dg_out)


def run_mma_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run the portable mma.sync CuTeDSL kernel on SM90 or SM100."""
    fn, outputs = _prepare_mma_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    fn()
    torch.cuda.synchronize()
    return outputs


def run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run FLA Triton kda_bwd_intra."""
    return chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, BT, True)


# =============================================================================
# Benchmark helpers
# =============================================================================


def _make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Factory for do_bench-compatible cula callable using TVM-FFI compiled kernel.

    The TVM-FFI compiled kernel is compiled once (per H) and reused for any
    input shape — no per-config MLIR re-tracing.
    """
    # Ensure contiguous (keep batch=1 dim)
    q2 = q.contiguous()
    k2 = k.contiguous()
    g2 = g.contiguous()
    beta2 = beta.contiguous()
    dAqk2 = dAqk.contiguous()
    dAkk2 = dAkk.contiguous()
    dq2 = dq.contiguous()
    dk2 = dk.contiguous()
    dg2 = dg.contiguous()
    db2 = db.contiguous()

    if beta2.dtype != torch.bfloat16:
        beta2 = beta2.to(torch.bfloat16)

    _, T, H_sz, K_sz = q2.shape

    dq_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.bfloat16)
    dk_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.bfloat16)
    dg_out = torch.empty(1, T, H_sz, K_sz, device=DEVICE, dtype=torch.float32)
    db_out = torch.empty(1, T, H_sz, device=DEVICE, dtype=torch.float32)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    ci_flat = chunk_indices.reshape(-1).contiguous()
    num_chunks = ci_flat.shape[0] // 2
    num_tiles = num_chunks * H_sz

    problem_size = (Int32(T), Int32(H_sz), Int32(K_sz))
    total_tiles_val = Int32(num_tiles)

    compiled_fn = compile_kda_bwd_intra(H_sz, K=K_sz, BT=BT)

    def fn():
        tile_counter.zero_()
        compiled_fn(
            q2,
            k2,
            g2,
            dAqk2,
            dAkk2,
            dq2,
            dk2,
            dg2,
            db2,
            beta2,
            dq_out,
            dk_out,
            dg_out,
            db_out,
            tile_counter,
            cu_seqlens,
            ci_flat,
            problem_size,
            total_tiles_val,
        )

    return fn


def _make_bench_mma(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Factory for benchmarking the portable mma.sync CuTeDSL kernel."""
    fn, _ = _prepare_mma_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    return fn


def bench_fn(fn, warmup=WARMUP, rep=REP):
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


def print_header(title):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")
    print(f"{'Config':<55} {'cuLA':>9} {'FLA':>9} {'Speedup':>8} {'rRMSE':>9} {'rMAX':>9}")
    print(f"{'-' * 120}")


def print_row(config, cula_ms, fla_ms, acc=None):
    speedup = fla_ms / cula_ms if cula_ms > 0 else float("inf")
    marker = " <--" if speedup >= 1.5 else ""
    if acc is not None:
        rrmse, rmax = _worst_accuracy(acc)
        status = " OK" if rrmse < 0.01 and rmax < 0.05 else " WARN"
        print(f"{config:<55} {cula_ms:>8.3f}ms {fla_ms:>7.3f}ms {speedup:>7.2f}x {rrmse:>9.6f} {rmax:>9.6f}{status}{marker}")
        print(_format_accuracy(acc))
    else:
        print(f"{config:<55} {cula_ms:>8.3f}ms {fla_ms:>7.3f}ms {speedup:>7.2f}x{marker}")


def _print_summary(label, speedups):
    """Print aggregate speedup statistics."""
    if not speedups:
        return
    print(f"{'-' * 120}")
    avg = statistics.mean(speedups)
    med = statistics.median(speedups)
    mn, mx = min(speedups), max(speedups)
    ge1 = sum(1 for x in speedups if x >= 1.0)
    n = len(speedups)
    print(f"  Summary({label}): avg={avg:.3f}x, median={med:.3f}x, min={mn:.3f}x, max={mx:.3f}x, >=1.0x {ge1}/{n}")


# =============================================================================
# Correctness check
# =============================================================================


def _error_metrics(ref, tri):
    """Compute RMSE and relative max diff between ref and tri tensors."""
    diff = (ref.detach().float() - tri.detach().float()).flatten()
    ref_flat = ref.detach().float().flatten()
    # RMSE
    rmse = diff.square().mean().sqrt().item()
    # Relative RMSE (normalized by ref RMS)
    ref_rms = ref_flat.square().mean().sqrt().item()
    rel_rmse = rmse / (ref_rms + 1e-8)
    # Relative max diff: max|diff| / max|ref|
    abs_max_diff = diff.abs().max().item()
    ref_abs_max = ref_flat.abs().max().item()
    rel_max_diff = abs_max_diff / (ref_abs_max + 1e-8)
    return rmse, rel_rmse, abs_max_diff, rel_max_diff


def _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Run both implementations once, return dict of per-output (rel_rmse, rel_max)."""
    dq_c, dk_c, db_c, dg_c = run_cula_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)
    dq_f, dk_f, db_f, dg_f = run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)
    acc = {}
    for name, ref, tri in [
        ("dq", dq_f, dq_c.float()),
        ("dk", dk_f, dk_c.float()),
        ("db", db_f, db_c),
        ("dg", dg_f, dg_c),
    ]:
        _, rel_rmse, _, rel_max = _error_metrics(ref, tri)
        acc[name] = (rel_rmse, rel_max)
    return acc


def _check_mma_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices):
    """Compare the portable mma.sync kernel against FLA."""
    dq_c, dk_c, db_c, dg_c = run_mma_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    dq_f, dk_f, db_f, dg_f = run_fla_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    acc = {}
    for name, ref, tri in [
        ("dq", dq_f, dq_c.float()),
        ("dk", dk_f, dk_c.float()),
        ("db", db_f, db_c),
        ("dg", dg_f, dg_c),
    ]:
        _, rel_rmse, _, rel_max = _error_metrics(ref, tri)
        acc[name] = (rel_rmse, rel_max)
    return acc


def _format_accuracy(acc):
    """Format per-output accuracy dict as a compact sub-line string."""
    parts = []
    for name in ["dq", "dk", "db", "dg"]:
        rrmse, rmax = acc[name]
        parts.append(f"{name}:{rrmse:.2e}/{rmax:.2e}")
    return "  >> " + "  ".join(parts)


def _worst_accuracy(acc):
    """Extract worst-case rRMSE and rMAX from per-output accuracy dict."""
    worst_rrmse = max(v[0] for v in acc.values())
    worst_rmax = max(v[1] for v in acc.values())
    return worst_rrmse, worst_rmax


def check_correctness(H=4, total_T=512, num_seqs=4, beta_dtype=torch.float32):
    """Quick correctness check: compare cuLA vs FLA outputs."""
    torch.manual_seed(42)
    random.seed(42)
    seq_lens = generate_balanced_seqlens(total_T, num_seqs)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices = make_bwd_intra_inputs(
        seq_lens, H, beta_dtype=beta_dtype
    )

    dq_c, dk_c, db_c, dg_c = run_cula_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)
    dq_m, dk_m, db_m, dg_m = run_mma_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    dq_f, dk_f, db_f, dg_f = run_fla_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )

    assert_close("dq", dq_f, dq_c.float(), 0.008)
    assert_close("dk", dk_f, dk_c.float(), 0.008)
    assert_close("db", db_f, db_c, 0.02)
    assert_close("dg", dg_f, dg_c, 0.02)
    assert_close("mma.dq", dq_f, dq_m.float(), 0.008)
    assert_close("mma.dk", dk_f, dk_m.float(), 0.008)
    assert_close("mma.db", db_f, db_m, 0.02)
    assert_close("mma.dg", dg_f, dg_m, 0.02)

    # Print detailed error metrics
    pairs = [
        ("tcgen.dq", dq_f, dq_c.float()),
        ("tcgen.dk", dk_f, dk_c.float()),
        ("tcgen.db", db_f, db_c),
        ("tcgen.dg", dg_f, dg_c),
        ("mma.dq", dq_f, dq_m.float()),
        ("mma.dk", dk_f, dk_m.float()),
        ("mma.db", db_f, db_m),
        ("mma.dg", dg_f, dg_m),
    ]
    print()
    print(f"    {'tensor':>9}  {'RMSE':>10}  {'rel_RMSE':>10}  {'maxdiff':>10}  {'rel_maxdiff':>12}")
    for name, ref, tri in pairs:
        rmse, rel_rmse, abs_max, rel_max = _error_metrics(ref, tri)
        print(f"    {name:>9}  {rmse:>10.6f}  {rel_rmse:>10.6f}  {abs_max:>10.6f}  {rel_max:>12.6f}")
    return True


# =============================================================================
# Determinism check
# =============================================================================


def check_determinism(H=4, total_T=512, num_seqs=4, iters=20, beta_dtype=torch.float32):
    """Verify both CuTeDSL implementations are deterministic."""
    torch.manual_seed(42)
    seq_lens = generate_balanced_seqlens(total_T, num_seqs)
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices = make_bwd_intra_inputs(
        seq_lens, H, beta_dtype=beta_dtype
    )

    ref_dq, ref_dk, ref_db, ref_dg = run_cula_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices)
    ref_mma = run_mma_bwd_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
    )
    for i in range(iters):
        dq_out, dk_out, db_out, dg_out = run_cula_bwd_intra(
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
        )
        assert torch.equal(dq_out, ref_dq), f"dq mismatch at iter {i}"
        assert torch.equal(dk_out, ref_dk), f"dk mismatch at iter {i}"
        assert torch.equal(db_out, ref_db), f"db mismatch at iter {i}"
        assert torch.equal(dg_out, ref_dg), f"dg mismatch at iter {i}"
        mma_outputs = run_mma_bwd_intra(
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices
        )
        for name, output, reference in zip(
            ("dq", "dk", "db", "dg"), mma_outputs, ref_mma
        ):
            assert torch.equal(output, reference), f"mma.{name} mismatch at iter {i}"
    return True


# =============================================================================
# Benchmark suites
# =============================================================================


def bench_focused_varlen(H):
    """Core benchmark: T=8k/32k, N=16/20/24, quasi-balanced."""
    total_tokens_list = [8192, 32768]
    num_seqs_list = [16, 20, 24]

    print(f"\n{'=' * 135}")
    print(f"  Focused varlen: quasi-balanced (H={H})")
    print(f"{'=' * 135}")
    hdr = (
        f"{'Config':<40} {'cula_base':>9} {'cula_vl':>9} {'cula_ovhd':>9} "
        f"{'fla_base':>9} {'fla_vl':>9} {'fla_ovhd':>9} "
        f"{'base_sp':>8} {'vl_sp':>7} {'rRMSE':>9} {'rMAX':>9}"
    )
    print(hdr)
    print(f"{'-' * 135}")

    base_speedups = []
    vl_speedups = []
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
            cula_base, _, _ = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_base, ci_base))
            fla_base, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_base, ci_base))

            # Varlen
            q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl = make_bwd_intra_inputs(seq_lens, H)
            cula_vl, _, _ = bench_fn(_make_bench_cula(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl))
            fla_vl, _, _ = bench_fn(
                lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl)
            )

            acc = _check_accuracy(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu_vl, ci_vl)

            cula_ovhd = (cula_vl - cula_base) / cula_base * 100
            fla_ovhd = (fla_vl - fla_base) / fla_base * 100
            base_sp = fla_base / cula_base if cula_base > 0 else float("inf")
            vl_sp = fla_vl / cula_vl if cula_vl > 0 else float("inf")
            base_speedups.append(base_sp)
            vl_speedups.append(vl_sp)

            tag = f"T={total_T:>5} N={N:>2} ({mn}-{mx}, {ratio:.1f}x)"
            rrmse_w, rmax_w = _worst_accuracy(acc)
            print(
                f"{tag:<40} {cula_base:>8.3f}ms {cula_vl:>7.3f}ms {cula_ovhd:>+8.1f}% "
                f"{fla_base:>8.3f}ms {fla_vl:>7.3f}ms {fla_ovhd:>+8.1f}% "
                f"{base_sp:>7.2f}x {vl_sp:>6.2f}x {rrmse_w:>9.6f} {rmax_w:>9.6f}"
            )
            print(_format_accuracy(acc))
        print()
    print(f"{'-' * 135}")
    print(f"  Summary: base avg={statistics.mean(base_speedups):.3f}x  varlen avg={statistics.mean(vl_speedups):.3f}x")


def bench_scale_total_seqlen(H):
    """Scale total seq len with fixed num_seqs."""
    num_seqs = 4
    print_header(f"Scale total seqlen (N={num_seqs}, balanced, H={H})")
    speedups = []
    for total_T in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        per_seq = seq_lens[0]
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
        config = f"T={total_T:>5} ({num_seqs}x{per_seq})"
        cula_ms, _, _ = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
        acc = _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        print_row(config, cula_ms, fla_ms, acc)
        speedups.append(fla_ms / cula_ms if cula_ms > 0 else float("inf"))
    _print_summary(f"H={H}", speedups)


def bench_scale_num_seqs(H):
    """Scale number of sequences with fixed total length."""
    for total_T in [8192, 32768]:
        print_header(f"Scale num_seqs (total_T={total_T}, balanced, H={H})")
        speedups = []
        for num_seqs in [1, 2, 4, 8, 16, 32, 64]:
            torch.manual_seed(42)
            seq_lens = generate_balanced_seqlens(total_T, num_seqs)
            per_seq = seq_lens[0]
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
            config = f"T={total_T:>5} N={num_seqs:<3} (each~{per_seq})"
            cula_ms, _, _ = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
            fla_ms, _, _ = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))
            acc = _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
            print_row(config, cula_ms, fla_ms, acc)
            speedups.append(fla_ms / cula_ms if cula_ms > 0 else float("inf"))
        _print_summary(f"T={total_T},H={H}", speedups)


def bench_balanced_vs_unbalanced(H):
    """Compare balanced vs unbalanced seq distributions."""
    for total_T in [8192, 32768]:
        print_header(f"Balanced vs Unbalanced (total_T={total_T}, H={H})")
        speedups = []
        for num_seqs in [4, 8, 16]:
            torch.manual_seed(42)
            seq_lens_b = generate_balanced_seqlens(total_T, num_seqs)
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens_b, H)
            cb = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]
            fb = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]
            acc_b = _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
            print_row(f"balanced   N={num_seqs:<3} (each={seq_lens_b[0]})", cb, fb, acc_b)
            speedups.append(fb / cb if cb > 0 else float("inf"))

            seq_lens_u = generate_unbalanced_seqlens(total_T, num_seqs)
            q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2 = make_bwd_intra_inputs(seq_lens_u, H)
            cu_ms = bench_fn(_make_bench_cula(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))[0]
            fu_ms = bench_fn(lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))[0]
            acc_u = _check_accuracy(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2)
            longest = max(seq_lens_u)
            shortest = min(seq_lens_u)
            print_row(f"unbalanced N={num_seqs:<3} (max={longest},min={shortest})", cu_ms, fu_ms, acc_u)
            speedups.append(fu_ms / cu_ms if cu_ms > 0 else float("inf"))
        _print_summary(f"T={total_T},H={H}", speedups)


def bench_varlen_overhead(H):
    """Measure varlen overhead: non-varlen vs varlen."""
    print(f"\n{'=' * 120}")
    print(f"  Varlen overhead: single-seq vs multi-seq (H={H})")
    print(f"{'=' * 120}")
    hdr = (
        f"{'T':<7} {'cu_base':>9} {'cu_vl':>9} {'cu_ovhd':>8} "
        f"{'fl_base':>9} {'fl_vl':>9} {'fl_ovhd':>9} "
        f"{'base_sp':>8} {'vl_sp':>7} {'rRMSE':>9} {'rMAX':>9}"
    )
    print(hdr)
    print(f"{'-' * 120}")

    base_speedups = []
    vl_speedups = []
    for T in [1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)

        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1 = make_bwd_intra_inputs([T], H)
        cu_no = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1))[0]
        fl_no = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu1, ci1))[0]

        seq_lens = generate_quasi_balanced_seqlens(T, 16)
        q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2 = make_bwd_intra_inputs(seq_lens, H)
        cu_vl = bench_fn(_make_bench_cula(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))[0]
        fl_vl = bench_fn(lambda: run_fla_bwd_intra(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2))[0]

        acc = _check_accuracy(q2, k2, g2, beta2, dAqk2, dAkk2, dq2, dk2, db2, dg2, cu2, ci2)

        cu_ovhd = (cu_vl - cu_no) / cu_no * 100 if cu_no > 0 else 0
        fl_ovhd = (fl_vl - fl_no) / fl_no * 100 if fl_no > 0 else 0
        base_sp = fl_no / cu_no if cu_no > 0 else float("inf")
        vl_sp = fl_vl / cu_vl if cu_vl > 0 else float("inf")
        base_speedups.append(base_sp)
        vl_speedups.append(vl_sp)

        rrmse_w, rmax_w = _worst_accuracy(acc)
        print(
            f"T={T:<5} {cu_no:>8.3f}ms {cu_vl:>7.3f}ms {cu_ovhd:>+7.1f}% "
            f"{fl_no:>8.3f}ms {fl_vl:>7.3f}ms {fl_ovhd:>+8.1f}% "
            f"{base_sp:>7.2f}x {vl_sp:>6.2f}x {rrmse_w:>9.6f} {rmax_w:>9.6f}"
        )
        print(_format_accuracy(acc))
    print(f"{'-' * 120}")
    print(f"  Summary: base avg={statistics.mean(base_speedups):.3f}x  varlen avg={statistics.mean(vl_speedups):.3f}x")


def bench_realistic_prefill(H):
    """Realistic prefill scenarios."""
    print_header(f"Realistic prefill scenarios (H={H})")
    speedups = []
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
        cula_ms = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]
        fla_ms = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]
        acc = _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        print_row(config, cula_ms, fla_ms, acc)
        speedups.append(fla_ms / cula_ms if cula_ms > 0 else float("inf"))
    _print_summary(f"H={H}", speedups)


def bench_varlen_8k_16k_vs_fla(H):
    """Expanded varlen: T in [8k,16k], N in [15,25], ratio in [2x,3x]."""
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
                        total_tokens=total_T, num_seqs=N, min_ratio=r_lo, max_ratio=r_hi, seed=sd
                    )
                    mn, mx = min(seq_lens), max(seq_lens)
                    ratio = mx / mn

                    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
                    cula_ms = bench_fn(_make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]
                    fla_ms = bench_fn(lambda: run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci))[0]

                    acc = _check_accuracy(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)

                    sp = fla_ms / cula_ms if cula_ms > 0 else float("inf")
                    speedups.append(sp)
                    tag = f"T={total_T:>5} N={N:>2} ({mn:>3}-{mx:<4}, {ratio:.2f}x, s={sd})"
                    rrmse_w, rmax_w = _worst_accuracy(acc)
                    print(f"{tag:<52} {cula_ms:>8.3f}ms {fla_ms:>7.3f}ms {sp:>7.2f}x {rrmse_w:>9.6f} {rmax_w:>9.6f}")
                    print(_format_accuracy(acc))

    _print_summary(f"H={H}", speedups)


def bench_mma_sync_comparison(H):
    """Compare SM100 tcgen05, portable mma.sync, and FLA directly."""
    configs = [
        ("uniform", [8192]),
        ("uniform", [32768]),
        ("varlen", generate_quasi_balanced_seqlens(8192, 8)),
        ("varlen", generate_quasi_balanced_seqlens(32768, 8)),
    ]

    print(f"\n{'=' * 142}")
    print(f"  SM100 tcgen05 vs portable mma.sync CuTeDSL vs FLA (H={H})")
    print(f"{'=' * 142}")
    print(
        f"{'Config':<39} {'tcgen05':>10} {'mma.sync':>10} {'FLA':>10} "
        f"{'tcgen_sp':>9} {'mma_sp':>9} {'mma/tcgen':>11} {'rRMSE':>9} {'rMAX':>9}"
    )
    print(f"{'-' * 142}")

    for kind, seq_lens in configs:
        torch.manual_seed(42)
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(
            seq_lens, H, beta_dtype=torch.bfloat16
        )
        tcgen_ms = bench_fn(
            _make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        )[0]
        mma_ms = bench_fn(
            _make_bench_mma(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        )[0]
        fla_ms = bench_fn(
            lambda: run_fla_bwd_intra(
                q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci
            )
        )[0]
        acc = _check_mma_accuracy(
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci
        )
        rrmse, rmax = _worst_accuracy(acc)
        total = sum(seq_lens)
        config = f"{kind} T={total} N={len(seq_lens)}"
        print(
            f"{config:<39} {tcgen_ms:>9.3f}ms {mma_ms:>9.3f}ms "
            f"{fla_ms:>9.3f}ms {fla_ms / tcgen_ms:>8.2f}x "
            f"{fla_ms / mma_ms:>8.2f}x {tcgen_ms / mma_ms:>10.2f}x "
            f"{rrmse:>9.6f} {rmax:>9.6f}"
        )
        print(_format_accuracy(acc))


# =============================================================================
# Main runners
# =============================================================================


def run_all_benchmarks(H):
    print(f"\n{'#' * 100}")
    print(f"#  KDABwdIntraSM100 benchmark: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    # Force TVM-FFI compilation before any timing
    compile_kda_bwd_intra(H, K=K, BT=BT)

    # Correctness + determinism checks
    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness check (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")
    print(f"  Determinism check (H={H}, beta={torch.bfloat16})...", end="", flush=True)
    check_determinism(H, beta_dtype=torch.bfloat16)
    print(" PASS")

    bench_focused_varlen(H)
    bench_varlen_overhead(H)
    bench_scale_total_seqlen(H)
    bench_scale_num_seqs(H)
    bench_balanced_vs_unbalanced(H)
    bench_realistic_prefill(H)
    bench_varlen_8k_16k_vs_fla(H)
    bench_mma_sync_comparison(H)


def run_correctness_only(H):
    print(f"\n{'#' * 100}")
    print(f"#  KDABwdIntraSM100 correctness check: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    compile_kda_bwd_intra(H, K=K, BT=BT)
    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")
    print(f"  Determinism (H={H}, beta={torch.bfloat16})...", end="", flush=True)
    check_determinism(H, beta_dtype=torch.bfloat16)
    print(" PASS")


def run_expanded_varlen_only(H):
    print(f"\n{'#' * 100}")
    print(f"#  KDABwdIntraSM100 expanded varlen benchmark: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    compile_kda_bwd_intra(H, K=K, BT=BT)
    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")
    bench_varlen_8k_16k_vs_fla(H)


def run_mma_sync_only(H):
    print(f"\n{'#' * 100}")
    print(f"#  Portable mma.sync comparison on SM100: H={H}, K={K}, BT={BT}")
    print(f"{'#' * 100}")

    compile_kda_bwd_intra(H, K=K, BT=BT)
    for beta_dtype in [torch.bfloat16, torch.float32]:
        print(f"\n  Correctness (H={H}, beta={beta_dtype})...", end="", flush=True)
        check_correctness(H, beta_dtype=beta_dtype)
        print(" PASS")
    print(f"  Determinism (H={H}, beta={torch.bfloat16})...", end="", flush=True)
    check_determinism(H, beta_dtype=torch.bfloat16)
    print(" PASS")
    bench_mma_sync_comparison(H)


def run_ncu(H):
    """NCU profiling mode: warmup=1, rep=1 per config. Run under ncu."""
    print(f"\n{'#' * 100}")
    print(f"#  NCU profiling mode: H={H}, K={K}, BT={BT}")
    print(f"#  Usage: ncu -k regex:'cutlass' python {__file__} --suite ncu --heads {H}")
    print(f"{'#' * 100}")

    configs = [
        ("varlen_8k_N20", generate_quasi_balanced_seqlens(8192, 20)),
        ("varlen_32k_N20", generate_quasi_balanced_seqlens(32768, 20)),
        ("balanced_8k_N4", generate_balanced_seqlens(8192, 4)),
        ("balanced_32k_N4", generate_balanced_seqlens(32768, 4)),
    ]

    for tag, seq_lens in configs:
        total = sum(seq_lens)
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci = make_bwd_intra_inputs(seq_lens, H)
        cula_fn = _make_bench_cula(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)
        mma_fn = _make_bench_mma(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)

        def fla_fn():
            return run_fla_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu, ci)

        # warmup once
        cula_fn()
        mma_fn()
        fla_fn()
        torch.cuda.synchronize()

        # single measured run
        print(f"  [{tag}] T={total} N={len(seq_lens)} running cuLA...", flush=True)
        cula_fn()
        torch.cuda.synchronize()

        print(f"  [{tag}] T={total} N={len(seq_lens)} running mma.sync...", flush=True)
        mma_fn()
        torch.cuda.synchronize()

        print(f"  [{tag}] T={total} N={len(seq_lens)} running FLA...", flush=True)
        fla_fn()
        torch.cuda.synchronize()

    print("  NCU profiling runs complete.")


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Benchmark KDABwdIntraSM100 vs FLA Triton")
    parser.add_argument(
        "--suite",
        choices=["all", "correctness", "expanded_varlen", "mma", "ncu"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Head counts to benchmark (default: [32, 64])",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=REP, help="Measurement repetitions")
    args = parser.parse_args()

    WARMUP = args.warmup
    REP = args.rep

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"K={K}, BT={BT}, dtype={DTYPE}, warmup={WARMUP}, rep={REP}")

    for H in args.heads:
        if args.suite == "correctness":
            run_correctness_only(H)
        elif args.suite == "expanded_varlen":
            run_expanded_varlen_only(H)
        elif args.suite == "ncu":
            run_ncu(H)
        elif args.suite == "mma":
            run_mma_sync_only(H)
        else:
            run_all_benchmarks(H)

    print(f"\n{'=' * 100}")
    print("  Done.")
    print(f"{'=' * 100}")
