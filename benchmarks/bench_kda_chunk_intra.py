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

from flashla.kda.chunk_intra import chunk_kda_fwd_intra as flat_chunk_kda_fwd_intra

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


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['flat_ops', 'fla'],
        line_names=['flat_ops', 'fla'],
        styles=[('blue', '-'), ('red', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"ChunkIntra_uniform_B{B}_H{H}_D{D}",
        args={},
    ),
)
def benchmark_chunk_intra_uniform(T, provider):
    device = torch.device("cuda")
    chunk_size = BT

    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
        B, T, H, D, device, cu_seqlens=cu_seqlens
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flat_ops':
        results = triton.testing.do_bench(
            lambda: flat_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
            quantiles=quantiles,
        )
    elif provider == 'fla':
        results = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
            quantiles=quantiles,
        )

    return results


# ==============================================================================
# Varlen benchmark
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['total_len'],
        x_vals=[8192, 16384, 32768, 65536],
        line_arg='provider',
        line_vals=['flat_ops', 'fla'],
        line_names=['flat_ops', 'fla'],
        styles=[('blue', '-'), ('red', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"ChunkIntra_varlen_NSEQ{NUM_SEQS}_H{H}_D{D}",
        args={},
    ),
)
def benchmark_chunk_intra_varlen(total_len, provider):
    device = torch.device("cuda")
    chunk_size = BT

    seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
    T = total_len
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    # hardcoded real-world training traces
    # varlen_traces = {
    #     8192:  [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192],
    #     16384: [0, 315, 973, 1283, 2162, 2459, 2678, 2998, 3781, 4096, 4503, 5459, 6318, 6669, 6979, 7583, 8192],
    #     32768: [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192],
    #     65536: [0, 652, 1255, 1600, 2083, 2345, 2756, 3172, 3767, 4096, 4891, 5236, 5543, 6255, 6480, 6947, 7616, 8192],
    # }
    # T = 8192
    # cu_seqlens = torch.tensor(varlen_traces[total_len], dtype=torch.int32, device=device)

    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
        B, T, H, D, device, cu_seqlens=cu_seqlens
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flat_ops':
        results = triton.testing.do_bench(
            lambda: flat_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
            quantiles=quantiles,
        )
    elif provider == 'fla':
        results = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=chunk_size, chunk_indices=chunk_indices,
                safe_gate=True,
            ),
            quantiles=quantiles,
        )

    return results


if __name__ == "__main__":
    # Uniform-length benchmark
    benchmark_chunk_intra_uniform.run(print_data=True, save_path='./bench_chunk_intra')

    # Varlen benchmark
    benchmark_chunk_intra_varlen.run(print_data=True, save_path='./bench_chunk_intra_varlen')
