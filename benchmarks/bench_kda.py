import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import os

import torch
import triton

from fla.ops.kda import chunk_kda as fla_chunk_kda
from benchmarks.utils import (
    set_seed, exclusive_cumsum, generate_random_seq_lens,
    prepare_safe_gate_inputs,
    SEED,
)

from flashla.kda.chunk import chunk_kda as flashla_chunk_kda

# Constant params
B, H, D = 2, 64, 128

# Varlen benchmark params
NUM_SEQS = 8 # 序列个数
TOTAL_LEN = 8192  # 总长度
MIN_SEQ_LEN = 63  # 最小序列长度
VARIANCE = 1.0  # 方差控制: 0.0=均衡, 1.0=正常随机, >1.0=更不均衡

# hardcoded real-world training traces
VARLEN_TRACES = {
    4096: [0, 652, 1255, 1600, 2083, 2345, 2756, 3172, 3767, 4096, 4891, 5236, 5543, 6255, 6480, 6947, 7616, 8192],
    8192:  [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192],
    16384: [0, 315, 973, 1283, 2162, 2459, 2678, 2998, 3781, 4096, 4503, 5459, 6318, 6669, 6979, 7583, 8192],
    32768: [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192],
}


# ==============================================================================
# Benchmark 1: safe_gate (use_gate_in_kernel=True, safe_gate=True), uniform seqlen
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['flashla', 'fla'],
        line_names=['flashla', 'fla'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-.'),
                ('orange', '-.'), ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"Performance_B{B}_H{H}",
        args={},
    ),
)
def benchmark_safe_gate(T, provider):
    set_seed(SEED)
    device = torch.device("cuda")

    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flashla':
        results = triton.testing.do_bench(
            lambda: flashla_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )
    elif provider == 'fla':
        results = triton.testing.do_bench(
            lambda: fla_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )

    return results

# ==============================================================================
# Benchmark 2: varlen safe_gate
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['total_len'],
        x_vals=[4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['flashla', 'fla'],
        line_names=['flashla', 'fla'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"Performance_varlen_NSEQ{NUM_SEQS}_H{H}_VAR{VARIANCE}",
        args={},
    ),
)
def benchmark_varlen_safe_gate(total_len, provider):
    """
    Varlen 版本的 benchmark，支持配置：
    - NUM_SEQS: 序列个数 (使用全局变量)
    - total_len: 总长度 (x轴)
    - MIN_SEQ_LEN: 最小序列长度 (使用全局变量)
    - VARIANCE: 方差控制 (使用全局变量)
    """
    set_seed(SEED)
    device = torch.device("cuda")

    seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, 42)
    T = total_len
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    # hardcoded real-world training traces
    # T = 8192
    # cu_seqlens = torch.tensor(VARLEN_TRACES[total_len], dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flashla':
        results = triton.testing.do_bench(
            lambda: flashla_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )
    elif provider == 'fla':
        results = triton.testing.do_bench(
            lambda: fla_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )

    return results

if __name__ == "__main__":
    # run_state_combo_sweep(B=2)
    # run_safe_gate_sweep(B_list=[1, 2, 4])
    # Fixed-length benchmarks
    output_dir = "./bench_safe_gate"
    os.makedirs(output_dir, exist_ok=True)
    benchmark_safe_gate.run(print_data=True, save_path=output_dir)
    # Varlen benchmarks
    output_dir = "./bench_varlen_safe_gate"
    os.makedirs(output_dir, exist_ok=True)
    benchmark_varlen_safe_gate.run(print_data=True, save_path=output_dir)