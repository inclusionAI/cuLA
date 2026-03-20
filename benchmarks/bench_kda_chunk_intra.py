import torch
import triton
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from benchmarks.utils import set_seed, exclusive_cumsum, generate_random_seq_lens, SEED

from cula.kda.chunk_intra import chunk_kda_fwd_intra as cula_chunk_kda_fwd_intra

# Constant params
B, H, D = 2, 64, 128
BT = 64  # chunk size

# Varlen benchmark params
NUM_SEQS = 8
TOTAL_LEN = 8192
MIN_SEQ_LEN = 63
VARIANCE = 1.0


def prepare_intra_inputs(batch_size, T, H, D, device, cu_seqlens=None):
    """Prepare preprocessed inputs ready for chunk_kda_fwd_intra.

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    chunk_size = BT
    scale = D ** (-0.5)

    set_seed(SEED)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    g_raw = torch.randn(batch_size, T, H, D, dtype=torch.float, device=device)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid()

    # l2norm q, k
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g_raw, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g_raw, beta))

    # gate preprocessing
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    g = kda_gate_chunk_cumsum(
        g=g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=-5.0,
    )

    return q, k, v, g, beta, scale, cu_seqlens, chunk_indices


def accuracy_stats(a, b):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    a, b = a.float(), b.float()
    diff = a - b
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.abs().max().item()
    denom = b.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.abs().mean().item()
    return rmse, rel_max, mean_diff


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
def benchmark_chunk_intra_uniform():
    device = torch.device("cuda")
    chunk_size = BT
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 90)
    print(f"  Uniform-Length ChunkIntra Benchmark: cuLA vs FLA Triton  B={B} H={H} D={D}")
    print("=" * 90)
    print(f"{'B':>4} {'T':>7} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}")
    print("─" * 90)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
            B, T, H, D, device, cu_seqlens=cu_seqlens
        )

        # Accuracy: run once and compare
        out_fla = fla_chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            safe_gate=True,
        )
        out_cula = cula_chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            safe_gate=True,
        )
        # Compare the first output tensor (o)
        o_fla = out_fla[0] if isinstance(out_fla, (tuple, list)) else out_fla
        o_cula = out_cula[0] if isinstance(out_cula, (tuple, list)) else out_cula
        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: cula_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float('inf')

        print(f"{B:>4} {T:>7} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x")

    print("─" * 90)


# ==============================================================================
# Varlen benchmark
# ==============================================================================
def benchmark_chunk_intra_varlen():
    device = torch.device("cuda")
    chunk_size = BT
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 100)
    print(f"  Varlen ChunkIntra Benchmark: cuLA vs FLA Triton  NUM_SEQS={NUM_SEQS} H={H} D={D}")
    print("=" * 100)
    print(f"{'total_len':>10} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}")
    print("─" * 100)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
            B, T, H, D, device, cu_seqlens=cu_seqlens
        )

        # Accuracy
        out_fla = fla_chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            safe_gate=True,
        )
        out_cula = cula_chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            safe_gate=True,
        )
        o_fla = out_fla[0] if isinstance(out_fla, (tuple, list)) else out_fla
        o_cula = out_cula[0] if isinstance(out_cula, (tuple, list)) else out_cula
        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: cula_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float('inf')

        print(f"{total_len:>10} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x")

    print("─" * 100)


if __name__ == "__main__":
    benchmark_chunk_intra_uniform()
    benchmark_chunk_intra_varlen()
