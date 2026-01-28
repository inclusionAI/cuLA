import torch
import triton
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch.nn.functional as F
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.kda import chunk_kda
from benchmark.utils import set_seed, exclusive_cumsum

from flashla.kda_wrapper import flash_kda_prefill

# Constant params
B, H, D = 1, 64, 128

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['flashla_cutedsl', 'fla'],
        # label name for the lines
        line_names=['flashla_cutedsl', 'fla'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name=f"Performance_B{B}_H{H}_kda_safe_gate",
        args={},
    ),
)
def benchmark_safe_gate(T, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)

    use_gate_in_kernel = True
    safe_gate = True
    has_init_state = False
    # normalize for gate to test numerical stablity
    gate_logit_normalizer = 1

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    set_seed(42)

    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(B, T, H, D, dtype=torch.float, device=device).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    if use_gate_in_kernel:
      A_log = torch.randn(H, dtype=torch.float)
      dt_bias = torch.randn(H * D, dtype=torch.float)
      A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    else:
      g = F.logsigmoid(g)
      g = g / gate_logit_normalizer

    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
    else:
        lower_bound = None

    if has_init_state:
      init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device).requires_grad_(False)
    else:
       init_state = None

    if provider == 'flashla_cutedsl':
        results = triton.testing.do_bench(
          lambda: flash_kda_prefill(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              scale=scale,
              A_log=(A_log.clone() if use_gate_in_kernel else None),
              dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
              initial_state=init_state,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=use_gate_in_kernel,
              safe_gate=safe_gate,
              lower_bound=lower_bound
          ),
          quantiles=quantiles,
        )
    elif provider == 'fla':
      results = triton.testing.do_bench(
          lambda: chunk_kda(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              A_log=(A_log.clone() if use_gate_in_kernel else None),
              dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
              scale=scale,
              initial_state=init_state,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=use_gate_in_kernel,
              safe_gate=safe_gate,
              lower_bound=lower_bound
          ),
          quantiles=quantiles,
      )

    return results


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[2048, 4096, 8192, 16384, 32768, 65536],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['flat_ops', 'fla'],
        # label name for the lines
        line_names=['flat_ops', 'fla'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name=f"Performance_B{B}_H{H}_kda_use_gate_false",
        args={},
    ),
)
def benchmark(T, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)

    use_gate_in_kernel = False

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    set_seed(42)

    use_gate_in_kernel = False

    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(B, T, H, D, dtype=torch.float, device=device).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    if use_gate_in_kernel:
      A_log = torch.randn(H, dtype=torch.float)
      dt_bias = torch.randn(H * D, dtype=torch.float)
      A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    else:
      g = F.logsigmoid(g)

    if provider == 'flat_ops':
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
        # set batch size to 1 for compatibility with the preprocessing kernel
        if B != 1:
          q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        results = triton.testing.do_bench(
          lambda: flash_kda_prefill(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              scale=scale,
              A_log=(A_log.clone() if use_gate_in_kernel else None),
              dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
              initial_state=None,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              cu_seqlens=cu_seqlens,
              use_gate_in_kernel=use_gate_in_kernel,
              safe_gate=False
          ),
          quantiles=quantiles,
        )
    elif provider == 'fla':
      results = triton.testing.do_bench(
          lambda: chunk_kda(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              A_log=(A_log.clone() if use_gate_in_kernel else None),
              dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
              scale=scale,
              initial_state=None,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=use_gate_in_kernel,
          ),
          quantiles=quantiles,
      )

    return results

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[2048, 4096, 8192, 16384, 32768, 65536],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['flat_ops', 'fla'],
        # label name for the lines
        line_names=['flat_ops', 'fla'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name=f"Performance_B{B}_H{H}_kda_use_gate_in_kernel",
        args={},
    ),
)
def benchmark_kernel(T, provider):
    try:
        from fla.ops.kda import chunk_kda_fwd
    except ImportError:
        raise ImportError("Please modify the fla/ops/kda/__init__.py to export chunk_kda_fwd.")

    dtype = torch.bfloat16
    device = torch.device("cuda")

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    scale = D ** (-0.5)
    use_qk_l2norm_in_kernel = True
    output_final_state = True

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    set_seed(42)

    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    chunk_indices = None
    q_rstd, k_rstd = None, None
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)

    chunk_size = 64

    if provider == 'flat_ops':
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
        # set batch size to 1 for compatibility with the preprocessing kernel
        if B != 1:
          q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices
        )
        workspace_buffer = torch.zeros(132 * 1024, dtype=torch.uint8, device=q.device)

        results = triton.testing.do_bench(
          lambda: ops.kda_fwd_prefill(
            None, None, q, k, v, None, g, beta, cu_seqlens, workspace_buffer, scale
          ),
          quantiles=quantiles,
        )
    elif provider == 'fla':
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
        )
        results = triton.testing.do_bench(
          lambda: chunk_kda_fwd(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              scale=scale,
              initial_state=None,
              output_final_state=output_final_state,
          ),
          quantiles=quantiles,
        )

    return results

if __name__ == "__main__":
  benchmark_safe_gate.run(print_data=True, save_path='./benchmarks_safe_gate')
  # benchmark.run(print_data=True, save_path='./benchmarks')
  # benchmark_kernel.run(print_data=True, save_path='./benchmark_kernel')
