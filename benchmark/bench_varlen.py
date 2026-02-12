"""
Benchmark: flashla (CuTe DSL) vs FLA (Triton) for variable-length KDA.

Tests multiple varlen configurations:
  1. Scale total tokens (fixed num_seqs, balanced)
  2. Scale number of sequences (fixed total tokens)
  3. Balanced vs Unbalanced distribution
  4. Aligned vs Non-aligned sequence lengths
  5. Varlen overhead (non-varlen vs varlen single-seq)
  6. Realistic prefill scenarios
"""

import sys
import os
import pathlib
import contextlib
import io

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda import chunk_kda
from flashla.kda_wrapper import flash_kda_prefill
from benchmark.utils import set_seed, exclusive_cumsum

# =============================================================================
# Config
# =============================================================================
D = 128             # head dim (fixed for flashla kernel)
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
REP = 100

# =============================================================================
# Suppress kernel debug prints during compilation
# =============================================================================
@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to /dev/null to suppress kernel compilation debug prints."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def precompile_kernels(H: int):
    """Trigger kernel compilation for all needed variants with debug prints suppressed."""
    scale = D ** -0.5
    T = 256
    num_seqs = 2
    cu = torch.tensor([0, 128, 256], dtype=torch.long, device=DEVICE)
    q = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float, device=DEVICE)).clamp(-5, 0)
    beta = torch.randn(1, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    h0 = torch.randn(num_seqs, H, D, D, dtype=torch.float32, device=DEVICE)

    with suppress_stdout():
        # Compile: varlen with init+output state
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=h0, output_final_state=True,
                          use_qk_l2norm_in_kernel=False, safe_gate=True,
                          cu_seqlens=cu)
        # Compile: varlen without init/output state
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, safe_gate=True,
                          cu_seqlens=cu)
        # Compile: non-varlen with init+output state
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=h0[:1], output_final_state=True,
                          use_qk_l2norm_in_kernel=False, safe_gate=True,
                          cu_seqlens=None)
        # Compile: non-varlen without init/output state
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, safe_gate=True,
                          cu_seqlens=None)
        # Also trigger FLA Triton compilation
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                  initial_state=h0, output_final_state=True,
                  use_qk_l2norm_in_kernel=False, safe_gate=True,
                  cu_seqlens=cu)
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                  initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, safe_gate=True,
                  cu_seqlens=None)

    torch.cuda.synchronize()


# =============================================================================
# Helpers
# =============================================================================

def make_varlen_inputs(
    seq_lens: list[int],
    H: int,
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

    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)

    h0 = None
    if has_initial_state:
        h0 = torch.randn(num_seqs, H, D, D, dtype=torch.float32, device=DEVICE)

    return q_norm, k_norm, v, g, beta, cu_seqlens, h0


def bench_fn(fn, warmup=WARMUP, rep=REP):
    """Benchmark using triton's do_bench (CUDA event timing)."""
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms


def generate_balanced_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Split total_tokens evenly across num_seqs."""
    base = total_tokens // num_seqs
    remainder = total_tokens % num_seqs
    return [base] * (num_seqs - 1) + [base + remainder]


def generate_unbalanced_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """One long seq + many short seqs."""
    if num_seqs == 1:
        return [total_tokens]
    long_len = total_tokens // 2
    remaining = total_tokens - long_len
    base = remaining // (num_seqs - 1)
    last = remaining - base * (num_seqs - 2)
    return [long_len] + [base] * (num_seqs - 2) + [last]


def generate_nonaligned_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Non-aligned seq lens (not multiples of chunk_size=64)."""
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


# =============================================================================
# Core benchmark runner
# =============================================================================

def run_single_config(
    seq_lens: list[int],
    H: int,
    safe_gate: bool = True,
    has_initial_state: bool = True,
    output_final_state: bool = True,
):
    """Run a single benchmark config, return (flashla_ms, fla_ms)."""
    set_seed(42)
    scale = D ** -0.5

    q, k, v, g, beta, cu_seqlens, h0 = make_varlen_inputs(
        seq_lens, H, safe_gate=safe_gate, has_initial_state=has_initial_state
    )

    def run_flashla():
        return flash_kda_prefill(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=h0, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False, safe_gate=safe_gate,
            cu_seqlens=cu_seqlens,
        )

    def run_fla():
        return chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=h0, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False, safe_gate=safe_gate,
            cu_seqlens=cu_seqlens,
        )

    flashla_ms, _, _ = bench_fn(run_flashla)
    fla_ms, _, _ = bench_fn(run_fla)
    return flashla_ms, fla_ms


# =============================================================================
# Formatting helpers
# =============================================================================

def print_header(title: str):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Config':<50} {'flashla':>9} {'FLA':>9} {'Speedup':>8}")
    print(f"{'-'*90}")


def print_row(config: str, flashla_ms: float, fla_ms: float):
    speedup = fla_ms / flashla_ms if flashla_ms > 0 else float("inf")
    marker = " <--" if speedup >= 1.5 else ""
    print(f"{config:<50} {flashla_ms:>8.3f}ms {fla_ms:>7.3f}ms {speedup:>7.2f}x{marker}")


# =============================================================================
# Benchmark 1: Scale total sequence length (fixed num_seqs)
# =============================================================================

def bench_scale_total_seqlen(H: int):
    num_seqs = 4
    print_header(f"1. Scale total seqlen (N={num_seqs}, balanced, H={H})")
    for total_T in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        per_seq = seq_lens[0]
        config = f"T={total_T:>5} ({num_seqs}x{per_seq})"
        flashla_ms, fla_ms = run_single_config(seq_lens, H)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Benchmark 2: Scale number of sequences (fixed total length)
# =============================================================================

def bench_scale_num_seqs(H: int):
    total_T = 8192
    print_header(f"2. Scale num_seqs (total_T={total_T}, balanced, H={H})")
    for num_seqs in [1, 2, 4, 8, 16, 32, 64]:
        seq_lens = generate_balanced_seqlens(total_T, num_seqs)
        per_seq = seq_lens[0]
        config = f"N={num_seqs:<3} (each~{per_seq})"
        flashla_ms, fla_ms = run_single_config(seq_lens, H)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Benchmark 3: Balanced vs Unbalanced
# =============================================================================

def bench_balanced_vs_unbalanced(H: int):
    print_header(f"3. Balanced vs Unbalanced (total_T=8192, H={H})")
    for num_seqs in [4, 8, 16]:
        # Balanced
        seq_lens_b = generate_balanced_seqlens(8192, num_seqs)
        fb, flb = run_single_config(seq_lens_b, H)
        print_row(f"balanced   N={num_seqs:<3} (each={seq_lens_b[0]})", fb, flb)

        # Unbalanced
        seq_lens_u = generate_unbalanced_seqlens(8192, num_seqs)
        fu, flu = run_single_config(seq_lens_u, H)
        longest = max(seq_lens_u)
        shortest = min(seq_lens_u)
        print_row(f"unbalanced N={num_seqs:<3} (max={longest},min={shortest})", fu, flu)


# =============================================================================
# Benchmark 4: Aligned vs Non-aligned
# =============================================================================

def bench_aligned_vs_nonaligned(H: int):
    print_header(f"4. Aligned vs Non-aligned (total_T=8192, H={H})")
    for num_seqs in [4, 8]:
        # Aligned (balanced → multiples of chunk_size)
        seq_lens_a = generate_balanced_seqlens(8192, num_seqs)
        fa, fla_a = run_single_config(seq_lens_a, H)
        print_row(f"aligned     N={num_seqs} (each={seq_lens_a[0]})", fa, fla_a)

        # Non-aligned
        seq_lens_na = generate_nonaligned_seqlens(8192, num_seqs)
        fna, fla_na = run_single_config(seq_lens_na, H)
        print_row(f"non-aligned N={num_seqs} (e.g. {seq_lens_na[0]},{seq_lens_na[1]})", fna, fla_na)

    # Extreme: widely varying sizes
    seq_lens_ex = [65, 63, 129, 127, 33, 95, 513, 8192 - (65+63+129+127+33+95+513)]
    fex, fla_ex = run_single_config(seq_lens_ex, H)
    print_row(f"extreme N={len(seq_lens_ex)} (65,63,129,...,{seq_lens_ex[-1]})", fex, fla_ex)


# =============================================================================
# Benchmark 5: Varlen overhead (non-varlen baseline vs varlen single-seq)
# =============================================================================

def bench_varlen_overhead(H: int):
    print_header(f"5. Varlen overhead: non-varlen vs varlen single-seq (H={H})")
    hdr = (f"{'T':<7} {'fl_base':>9} {'fl_vl':>9} {'fl_ovhd':>8} "
           f"{'fla_base':>9} {'fla_vl':>9} {'fla_ovhd':>9} "
           f"{'base_sp':>8} {'vl_sp':>7}")
    print(hdr)
    print(f"{'-'*90}")

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

        def mk_flashla(cu_seqlens):
            def fn():
                return flash_kda_prefill(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    initial_state=h0, output_final_state=True,
                    safe_gate=True, cu_seqlens=cu_seqlens,
                )
            return fn

        def mk_fla(cu_seqlens):
            def fn():
                return chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    initial_state=h0, output_final_state=True,
                    safe_gate=True, cu_seqlens=cu_seqlens,
                )
            return fn

        fl_no, _, _ = bench_fn(mk_flashla(None))
        fl_vl, _, _ = bench_fn(mk_flashla(cu))
        fla_no, _, _ = bench_fn(mk_fla(None))
        fla_vl, _, _ = bench_fn(mk_fla(cu))

        fl_ovhd = (fl_vl - fl_no) / fl_no * 100
        fla_ovhd = (fla_vl - fla_no) / fla_no * 100
        base_sp = fla_no / fl_no if fl_no > 0 else float("inf")
        vl_sp = fla_vl / fl_vl if fl_vl > 0 else float("inf")

        print(f"T={T:<5} {fl_no:>8.3f}ms {fl_vl:>7.3f}ms {fl_ovhd:>+7.1f}% "
              f"{fla_no:>8.3f}ms {fla_vl:>7.3f}ms {fla_ovhd:>+8.1f}% "
              f"{base_sp:>7.2f}x {vl_sp:>6.2f}x")


# =============================================================================
# Benchmark 6: Realistic prefill scenarios
# =============================================================================

def bench_realistic_prefill(H: int):
    print_header(f"6. Realistic prefill scenarios (H={H})")
    configs = [
        ("4 short prompts",        [128, 256, 192, 64]),
        ("4 medium prompts",       [512, 1024, 768, 896]),
        ("4 long prompts",         [2048, 4096, 2048, 4096]),
        ("mixed short+long",       [64, 4096, 128, 2048]),
        ("8 chat turns",           [64, 128, 32, 256, 512, 64, 128, 1024]),
        ("batch of 16 (256 each)", [256] * 16),
        ("batch of 32 (128 each)", [128] * 32),
        ("batch of 8 (1024 each)", [1024] * 8),
        ("batch of 4 (4096 each)", [4096] * 4),
    ]
    for desc, seq_lens in configs:
        total = sum(seq_lens)
        config = f"{desc} (T={total}, N={len(seq_lens)})"
        flashla_ms, fla_ms = run_single_config(seq_lens, H)
        print_row(config, flashla_ms, fla_ms)


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks(H: int):
    """Run all benchmark suites for a given H."""
    print(f"\n{'#'*90}")
    print(f"#  H={H}, D={D}")
    print(f"{'#'*90}")

    # Pre-compile all kernel variants with prints suppressed
    print(f"  Pre-compiling kernels for H={H}...", end="", flush=True)
    precompile_kernels(H)
    print(" done.")

    bench_scale_total_seqlen(H)
    bench_scale_num_seqs(H)
    bench_balanced_vs_unbalanced(H)
    bench_aligned_vs_nonaligned(H)
    bench_varlen_overhead(H)
    bench_realistic_prefill(H)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"D={D}, dtype={DTYPE}, warmup={WARMUP}, rep={REP}")

    for H in [32, 64]:
        run_all_benchmarks(H)

    print(f"\n{'='*90}")
    print("  All benchmarks done.")
    print(f"{'='*90}")
