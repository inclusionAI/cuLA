"""
Benchmark: flashla (CuTe DSL) vs FLA (Triton) for variable-length KDA.

Tests multiple varlen configurations:
  1. Scaling total sequence length (fixed num_seqs)
  2. Scaling number of sequences (fixed total length)
  3. Balanced vs unbalanced sequence distribution
  4. Aligned vs non-aligned sequence lengths
"""

import sys
import pathlib
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda import chunk_kda
from flashla.kda_wrapper import flash_kda_prefill
from benchmark.utils import set_seed, exclusive_cumsum

# =============================================================================
# Config
# =============================================================================
H = 32          # num heads (typical for models like Mamba-2 / HGRN-2)
D = 128         # head dim (fixed for flashla kernel)
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
REP = 100

# =============================================================================
# Helpers
# =============================================================================

def make_varlen_inputs(
    seq_lens: list[int],
    H: int,
    D: int,
    safe_gate: bool = True,
    has_initial_state: bool = False,
):
    """Create varlen inputs for both flashla and FLA."""
    total_tokens = sum(seq_lens)
    num_seqs = len(seq_lens)
    cu_seqlens = torch.tensor(
        exclusive_cumsum(seq_lens), dtype=torch.long, device=DEVICE
    )

    q = torch.randn(1, total_tokens, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, total_tokens, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, total_tokens, H, D, dtype=DTYPE, device=DEVICE)
    g = F.logsigmoid(torch.randn(1, total_tokens, H, D, dtype=torch.float, device=DEVICE))
    if safe_gate:
        g = g.clamp(-5, 0)
    beta = torch.randn(1, total_tokens, H, dtype=torch.float32, device=DEVICE).sigmoid()

    # Pre-normalize q, k (both impls use use_qk_l2norm_in_kernel=True)
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)

    h0 = None
    if has_initial_state:
        h0 = torch.randn(num_seqs, H, D, D, dtype=torch.float32, device=DEVICE)

    return q_norm, k_norm, v, g, beta, cu_seqlens, h0


def bench_fn(fn, warmup=WARMUP, rep=REP):
    """Benchmark a function using triton's do_bench (CUDA event timing)."""
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


def generate_balanced_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Split total_tokens evenly across num_seqs (last seq absorbs remainder)."""
    base = total_tokens // num_seqs
    remainder = total_tokens % num_seqs
    return [base] * (num_seqs - 1) + [base + remainder]


def generate_unbalanced_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Generate unbalanced seq lens: one long seq + many short seqs."""
    if num_seqs == 1:
        return [total_tokens]
    # Give half tokens to first seq, split rest evenly
    long_len = total_tokens // 2
    remaining = total_tokens - long_len
    base = remaining // (num_seqs - 1)
    last = remaining - base * (num_seqs - 2)
    return [long_len] + [base] * (num_seqs - 2) + [last]


def generate_nonaligned_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Generate non-aligned seq lens (not multiples of chunk_size=64)."""
    # Start from balanced, then add/subtract small offsets
    base = total_tokens // num_seqs
    seqlens = []
    remaining = total_tokens
    for i in range(num_seqs - 1):
        # Alternate +7 and -7 to keep total roughly correct
        offset = 7 if (i % 2 == 0) else -7
        sl = max(1, base + offset)
        seqlens.append(sl)
        remaining -= sl
    seqlens.append(max(1, remaining))
    return seqlens


# =============================================================================
# Benchmark runners
# =============================================================================

def run_single_config(
    seq_lens: list[int],
    safe_gate: bool = True,
    has_initial_state: bool = True,
    output_final_state: bool = True,
):
    """Run a single benchmark config, return (flashla_ms, fla_ms)."""
    set_seed(42)
    scale = D ** -0.5

    q, k, v, g, beta, cu_seqlens, h0 = make_varlen_inputs(
        seq_lens, H, D, safe_gate=safe_gate, has_initial_state=has_initial_state
    )

    # --- flashla (CuTe DSL) ---
    def run_flashla():
        return flash_kda_prefill(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=h0,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,  # already normalized
            safe_gate=safe_gate,
            cu_seqlens=cu_seqlens,
        )

    # --- FLA (Triton) ---
    def run_fla():
        return chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=h0,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,  # already normalized
            safe_gate=safe_gate,
            cu_seqlens=cu_seqlens,
        )

    # Warmup + benchmark
    flashla_ms, flashla_min, flashla_max = bench_fn(run_flashla)
    fla_ms, fla_min, fla_max = bench_fn(run_fla)

    return flashla_ms, fla_ms


def print_header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Config':<45} {'flashla(ms)':>11} {'FLA(ms)':>11} {'Speedup':>8}")
    print(f"{'-'*80}")


def print_row(config: str, flashla_ms: float, fla_ms: float):
    speedup = fla_ms / flashla_ms if flashla_ms > 0 else float("inf")
    print(f"{config:<45} {flashla_ms:>11.3f} {fla_ms:>11.3f} {speedup:>7.2f}x")


# =============================================================================
# Benchmark 1: Scale total sequence length (fixed num_seqs)
# =============================================================================

def bench_scale_total_seqlen():
    print_header("Benchmark 1: Scale total seqlen (num_seqs=4, balanced)")
    num_seqs = 4
    for total_T in [1024, 2048, 4096, 8192, 16384, 32768]:
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        config = f"T={total_T}, seqs={seq_lens}"
        flashla_ms, fla_ms = run_single_config(seq_lens)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Benchmark 2: Scale number of sequences (fixed total length)
# =============================================================================

def bench_scale_num_seqs():
    print_header("Benchmark 2: Scale num_seqs (total_T=8192, balanced)")
    total_T = 8192
    for num_seqs in [1, 2, 4, 8, 16, 32]:
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        config = f"N={num_seqs}, lens~{seq_lens[0]}"
        flashla_ms, fla_ms = run_single_config(seq_lens)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Benchmark 3: Balanced vs Unbalanced
# =============================================================================

def bench_balanced_vs_unbalanced():
    print_header("Benchmark 3: Balanced vs Unbalanced (total_T=8192)")
    for num_seqs in [4, 8, 16]:
        # Balanced
        seq_lens_b = generate_balanced_seqlens(8192, num_seqs)
        config_b = f"balanced N={num_seqs}, lens={seq_lens_b}"
        fb, flb = run_single_config(seq_lens_b)
        print_row(config_b, fb, flb)

        # Unbalanced
        seq_lens_u = generate_unbalanced_seqlens(8192, num_seqs)
        config_u = f"unbalanced N={num_seqs}, lens={seq_lens_u[:3]}..."
        fu, flu = run_single_config(seq_lens_u)
        print_row(config_u, fu, flu)


# =============================================================================
# Benchmark 4: Aligned vs Non-aligned
# =============================================================================

def bench_aligned_vs_nonaligned():
    print_header("Benchmark 4: Aligned vs Non-aligned (total_T=8192, num_seqs=8)")
    num_seqs = 8

    # Aligned (all multiples of 64)
    seq_lens_a = generate_balanced_seqlens(8192, num_seqs)
    config_a = f"aligned, lens={seq_lens_a[:3]}..."
    fa, fla_a = run_single_config(seq_lens_a)
    print_row(config_a, fa, fla_a)

    # Non-aligned
    seq_lens_na = generate_nonaligned_seqlens(8192, num_seqs)
    config_na = f"non-aligned, lens={seq_lens_na[:3]}..."
    fna, fla_na = run_single_config(seq_lens_na)
    print_row(config_na, fna, fla_na)

    # Extreme non-aligned: lots of tiny seqs
    seq_lens_ex = [65, 63, 129, 127, 33, 95, 513, 8192 - (65+63+129+127+33+95+513)]
    config_ex = f"extreme non-aligned, N={len(seq_lens_ex)}"
    fex, fla_ex = run_single_config(seq_lens_ex)
    print_row(config_ex, fex, fla_ex)


# =============================================================================
# Benchmark 5: Varlen vs non-varlen (overhead measurement)
# =============================================================================

def bench_varlen_overhead():
    print_header("Benchmark 5: Non-varlen baseline + varlen overhead")
    print(f"{'T':<8} {'flashla':>10} {'flashla_vl':>12} {'fl_ovhd%':>9} {'FLA':>10} {'FLA_vl':>10} {'fla_ovhd%':>10} {'base_spdup':>11} {'vl_spdup':>10}")
    print(f"{'-'*100}")
    for T in [1024, 2048, 4096, 8192, 16384, 32768]:
        set_seed(42)
        scale = D ** -0.5

        q = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
        v = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
        g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float, device=DEVICE)).clamp(-5, 0)
        beta = torch.randn(1, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        cu = torch.tensor([0, T], dtype=torch.long, device=DEVICE)
        h0 = torch.randn(1, H, D, D, dtype=torch.float32, device=DEVICE)

        # flashla without cu_seqlens
        def run_flashla_no_vl():
            return flash_kda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=h0, output_final_state=True,
                safe_gate=True, cu_seqlens=None,
            )

        # flashla with cu_seqlens (single seq)
        def run_flashla_vl():
            return flash_kda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=h0, output_final_state=True,
                safe_gate=True, cu_seqlens=cu,
            )

        # FLA without cu_seqlens
        def run_fla_no_vl():
            return chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=h0, output_final_state=True,
                safe_gate=True, cu_seqlens=None,
            )

        # FLA with cu_seqlens (single seq)
        def run_fla_vl():
            return chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=h0, output_final_state=True,
                safe_gate=True, cu_seqlens=cu,
            )

        fl_no, _, _ = bench_fn(run_flashla_no_vl)
        fl_vl, _, _ = bench_fn(run_flashla_vl)
        fla_no, _, _ = bench_fn(run_fla_no_vl)
        fla_vl, _, _ = bench_fn(run_fla_vl)

        fl_ovhd = (fl_vl - fl_no) / fl_no * 100
        fla_ovhd = (fla_vl - fla_no) / fla_no * 100
        base_spdup = fla_no / fl_no if fl_no > 0 else float("inf")
        vl_spdup = fla_vl / fl_vl if fl_vl > 0 else float("inf")

        print(f"T={T:<6} {fl_no:>9.3f}ms {fl_vl:>10.3f}ms {fl_ovhd:>+8.1f}% {fla_no:>8.3f}ms {fla_vl:>8.3f}ms {fla_ovhd:>+9.1f}% {base_spdup:>9.2f}x {vl_spdup:>9.2f}x")


# =============================================================================
# Benchmark 6: Realistic prefill scenario
# =============================================================================

def bench_realistic_prefill():
    print_header("Benchmark 6: Realistic prefill scenarios (mixed seq lens)")
    configs = [
        # (description, seq_lens)
        ("4 short prompts", [128, 256, 192, 64]),
        ("4 medium prompts", [512, 1024, 768, 896]),
        ("4 long prompts", [2048, 4096, 2048, 4096]),
        ("mixed short+long", [64, 4096, 128, 2048]),
        ("8 chat turns", [64, 128, 32, 256, 512, 64, 128, 1024]),
        ("batch of 16", [256] * 16),
        ("batch of 32 short", [128] * 32),
    ]
    for desc, seq_lens in configs:
        total = sum(seq_lens)
        config = f"{desc} (T={total}, N={len(seq_lens)})"
        flashla_ms, fla_ms = run_single_config(seq_lens)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"H={H}, D={D}, dtype={DTYPE}, warmup={WARMUP}, rep={REP}")

    bench_scale_total_seqlen()
    bench_scale_num_seqs()
    bench_balanced_vs_unbalanced()
    bench_aligned_vs_nonaligned()
    bench_varlen_overhead()
    bench_realistic_prefill()

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")
