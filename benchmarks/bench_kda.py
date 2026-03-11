import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import triton

from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.kda import chunk_kda as fla_chunk_kda
from benchmarks.utils import (
    set_seed, exclusive_cumsum, generate_random_seq_lens,
    prepare_safe_gate_inputs, prepare_no_gate_inputs,
    SEED, CHUNK_SIZE,
)

from flashla.kda_wrapper import flash_kda_prefill
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
    8192:  [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192],
    16384: [0, 315, 973, 1283, 2162, 2459, 2678, 2998, 3781, 4096, 4503, 5459, 6318, 6669, 6979, 7583, 8192],
    32768: [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192],
    65536: [0, 652, 1255, 1600, 2083, 2345, 2756, 3172, 3767, 4096, 4891, 5236, 5543, 6255, 6480, 6947, 7616, 8192],
}


# ==============================================================================
# Benchmark 1: safe_gate (use_gate_in_kernel=True, safe_gate=True), uniform seqlen
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        line_arg='provider',
        line_vals=['flashla_fully_fused', 'flashla', 'fla'],
        line_names=['flashla_fully_fused', 'flashla', 'fla'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-.'),
                ('orange', '-.'), ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"Performance_B{B}_H{H}",
        args={},
    ),
)
def benchmark_safe_gate(T, provider):
    device = torch.device("cuda")

    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flashla_fully_fused':
        results = triton.testing.do_bench(
            lambda: flash_kda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )
    elif provider == 'flashla':
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
# Benchmark 2: use_gate_in_kernel=False, uniform seqlen
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[2048, 4096, 8192, 16384, 32768, 65536],
        line_arg='provider',
        line_vals=['flashla_fully_fused', 'fla'],
        line_names=['flashla_fully_fused', 'fla'],
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",
        plot_name=f"Performance_B{B}_H{H}_kda_use_gate_false",
        args={},
    ),
)
def benchmark(T, provider):
    device = torch.device("cuda")

    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)

    inputs = prepare_no_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    scale = inputs['scale']

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flashla_fully_fused':
        results = triton.testing.do_bench(
            lambda: flash_kda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=None, dt_bias=None,
                initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=False, safe_gate=False,
            ),
            quantiles=quantiles,
        )
    elif provider == 'fla':
        results = triton.testing.do_bench(
            lambda: fla_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=None, dt_bias=None,
                initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=False,
            ),
            quantiles=quantiles,
        )

    return results


# ==============================================================================
# Benchmark 3: varlen safe_gate
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['total_len'],
        x_vals=[8192, 16384, 32768, 65536],
        line_arg='provider',
        line_vals=['flashla_fully_fused', 'flashla', 'fla'],
        line_names=['flashla_fully_fused', 'flashla', 'fla'],
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
    device = torch.device("cuda")

    # seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, 42)
    # T = total_len
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    # hardcoded real-world training traces
    T = 8192
    cu_seqlens = torch.tensor(VARLEN_TRACES[total_len], dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'flashla_fully_fused':
        results = triton.testing.do_bench(
            lambda: flash_kda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                A_log=A_log, dt_bias=dt_bias,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
            ),
            quantiles=quantiles,
        )
    elif provider == 'flashla':
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

def run_safe_gate_sweep(B_list=[1, 2], H=64, D=128,
                        T_list=[128, 256, 512, 1024, 2048, 4096, 8192, 16384]):
    """Run safe_gate benchmark for multiple B values and print a combined table with speedup."""
    dtype = torch.bfloat16
    device = torch.device("cuda")
    scale = D ** (-0.5)
    use_gate_in_kernel = True
    safe_gate = True
    lower_bound = -5.0
    quantiles = [0.5, 0.2, 0.8]

    all_results = {}  # (B, T) -> {'flashla': ms, 'fla': ms}

    for B in B_list:
        for T in T_list:
            set_seed(SEED)
            q = torch.randn(B, T, H, D, dtype=dtype, device=device)
            k = torch.randn(B, T, H, D, dtype=dtype, device=device)
            v = torch.randn(B, T, H, D, dtype=dtype, device=device)
            g = torch.randn(B, T, H, D, dtype=torch.float, device=device)
            beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid()
            A_log = torch.randn(H, dtype=torch.float, device=device)
            dt_bias = torch.randn(H * D, dtype=torch.float, device=device)

            flashla_ms = triton.testing.do_bench(
                lambda: flash_kda_prefill(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                    initial_state=None, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
                    safe_gate=safe_gate, lower_bound=lower_bound,
                ),
                quantiles=quantiles,
            )[0]

            fla_ms = triton.testing.do_bench(
                lambda: chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                    initial_state=None, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
                    safe_gate=safe_gate, lower_bound=lower_bound,
                ),
                quantiles=quantiles,
            )[0]

            all_results[(B, T)] = {'flashla': flashla_ms, 'fla': fla_ms}
            speedup = fla_ms / flashla_ms
            print(f"  B={B}, T={T:>5}: flashla={flashla_ms:7.3f}ms  fla={fla_ms:7.3f}ms  speedup={speedup:.2f}x")

    # Print combined table
    col_w = 33  # width per B column
    total_w = 8 + len(B_list) * (col_w + 2)
    print("\n" + "=" * total_w)
    print(f"{'':>8}", end="")
    for B in B_list:
        print(f" |{'B=' + str(B):^{col_w}}", end="")
    print()
    print(f"{'T':>8}", end="")
    for B in B_list:
        print(f" | {'flashla':>8}  {'fla':>8}  {'speedup':>8}", end="")
    print()
    print("-" * total_w)
    for T in T_list:
        print(f"{T:>8}", end="")
        for B in B_list:
            r = all_results[(B, T)]
            speedup = r['fla'] / r['flashla']
            print(f" | {r['flashla']:>7.3f}ms {r['fla']:>7.3f}ms {speedup:>7.2f}x", end="")
        print()
    print("=" * total_w)


def run_state_combo_sweep(B=2, H=64, D=128,
                          T_list=[128, 256, 512, 1024, 2048, 4096, 8192, 16384]):
    """Run benchmark for all 4 combinations of (has_init_state, output_final_state).

    Columns: (init=F,out=F), (init=F,out=T), (init=T,out=F), (init=T,out=T)
    Each column shows flashla ms, fla ms, and speedup.
    """
    dtype = torch.bfloat16
    device = torch.device("cuda")
    scale = D ** (-0.5)
    use_gate_in_kernel = True
    safe_gate = True
    lower_bound = -5.0
    quantiles = [0.5, 0.2, 0.8]

    combos = [
        (False, False),  # no init, no output
        (False, True),   # no init, output final
        (True, False),   # has init, no output
        (True, True),    # has init, output final
    ]
    combo_labels = [
        "init=N,out=N",
        "init=N,out=Y",
        "init=Y,out=N",
        "init=Y,out=Y",
    ]

    # all_results[(has_init, out_final, T)] -> {'flashla': ms, 'fla': ms}
    all_results = {}

    # Warmup: compile all kernel variants before timing
    print("Warming up all kernel variants...")
    for has_init, out_final in combos:
        set_seed(SEED)
        T_warmup = T_list[0]
        q = torch.randn(B, T_warmup, H, D, dtype=dtype, device=device)
        k = torch.randn(B, T_warmup, H, D, dtype=dtype, device=device)
        v = torch.randn(B, T_warmup, H, D, dtype=dtype, device=device)
        g = torch.randn(B, T_warmup, H, D, dtype=torch.float, device=device)
        beta = torch.randn(B, T_warmup, H, dtype=torch.float, device=device).sigmoid()
        A_log = torch.randn(H, dtype=torch.float, device=device)
        dt_bias = torch.randn(H * D, dtype=torch.float, device=device)
        init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device) if has_init else None
        # Trigger compilation
        flash_kda_prefill(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            A_log=A_log.clone(), dt_bias=dt_bias.clone(),
            initial_state=init_state, output_final_state=out_final,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate, lower_bound=lower_bound,
        )
        chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            A_log=A_log.clone(), dt_bias=dt_bias.clone(),
            initial_state=init_state, output_final_state=out_final,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate, lower_bound=lower_bound,
        )
        tag = f"init={'Y' if has_init else 'N'}, out={'Y' if out_final else 'N'}"
        print(f"  {tag} compiled")
    torch.cuda.synchronize()
    print("Warmup done.\n")

    for has_init, out_final in combos:
        tag = f"init={'Y' if has_init else 'N'}, out={'Y' if out_final else 'N'}"
        print(f"\n--- {tag} ---")
        for T in T_list:
            set_seed(SEED)
            q = torch.randn(B, T, H, D, dtype=dtype, device=device)
            k = torch.randn(B, T, H, D, dtype=dtype, device=device)
            v = torch.randn(B, T, H, D, dtype=dtype, device=device)
            g = torch.randn(B, T, H, D, dtype=torch.float, device=device)
            beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid()
            A_log = torch.randn(H, dtype=torch.float, device=device)
            dt_bias = torch.randn(H * D, dtype=torch.float, device=device)

            if has_init:
                init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
            else:
                init_state = None

            flashla_ms = triton.testing.do_bench(
                lambda: flash_kda_prefill(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                    initial_state=init_state, output_final_state=out_final,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
                    safe_gate=safe_gate, lower_bound=lower_bound,
                ),
                quantiles=quantiles,
            )[0]

            fla_ms = triton.testing.do_bench(
                lambda: chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                    initial_state=init_state, output_final_state=out_final,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
                    safe_gate=safe_gate, lower_bound=lower_bound,
                ),
                quantiles=quantiles,
            )[0]

            all_results[(has_init, out_final, T)] = {'flashla': flashla_ms, 'fla': fla_ms}
            speedup = fla_ms / flashla_ms
            print(f"  T={T:>5}: flashla={flashla_ms:7.3f}ms  fla={fla_ms:7.3f}ms  speedup={speedup:.2f}x")

    # Print combined table
    col_w = 33  # width per combo column
    total_w = 8 + len(combos) * (col_w + 2)
    print(f"\n  State combo sweep  B={B}, H={H}, D={D}")
    print("=" * total_w)
    print(f"{'':>8}", end="")
    for label in combo_labels:
        print(f" |{label:^{col_w}}", end="")
    print()
    print(f"{'T':>8}", end="")
    for _ in combos:
        print(f" | {'flashla':>8}  {'fla':>8}  {'speedup':>8}", end="")
    print()
    print("-" * total_w)
    for T in T_list:
        print(f"{T:>8}", end="")
        for has_init, out_final in combos:
            r = all_results[(has_init, out_final, T)]
            speedup = r['fla'] / r['flashla']
            print(f" | {r['flashla']:>7.3f}ms {r['fla']:>7.3f}ms {speedup:>7.2f}x", end="")
        print()
    print("=" * total_w)

    # Also print a delta table: how much each combo costs relative to (init=N, out=N) baseline
    print(f"\n  Overhead vs baseline (init=N, out=N)  B={B}")
    print("=" * (8 + len(combos) * 24))
    print(f"{'T':>8}", end="")
    for label in combo_labels:
        print(f" | {'flashla':>8}  {'fla':>8}", end="")
    print()
    print("-" * (8 + len(combos) * 24))
    for T in T_list:
        print(f"{T:>8}", end="")
        base_fl = all_results[(False, False, T)]['flashla']
        base_fla = all_results[(False, False, T)]['fla']
        for has_init, out_final in combos:
            r = all_results[(has_init, out_final, T)]
            delta_fl = (r['flashla'] / base_fl - 1) * 100
            delta_fla = (r['fla'] / base_fla - 1) * 100
            print(f" | {delta_fl:>+7.1f}%  {delta_fla:>+7.1f}%", end="")
        print()
    print("=" * (8 + len(combos) * 24))


if __name__ == "__main__":
    # run_state_combo_sweep(B=2)
    # run_safe_gate_sweep(B_list=[1, 2, 4])
    # Fixed-length benchmarks
    benchmark_safe_gate.run(print_data=True, save_path='./bench_safe_gate')
    # Varlen benchmarks
    benchmark_varlen_safe_gate.run(print_data=True, save_path='./bench_varlen_safe_gate')