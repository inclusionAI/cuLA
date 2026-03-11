"""
Profiling script for nsys: flashla vs FLA varlen KDA.
Reduced warmup/rep for clean nsys traces.
"""

import sys
import pathlib
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda import chunk_kda
from flashla.kda_wrapper import flash_kda_prefill
from benchmarks.utils import set_seed, exclusive_cumsum

D = 128
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 3
REP = 5

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_inputs(total_T, N, H):
    set_seed(42)
    base = total_T // N
    rem = total_T % N
    seq_lens = [base] * (N - 1) + [base + rem]
    cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=DEVICE)
    q = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
    v = torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEVICE)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float, device=DEVICE)).clamp(-5, 0)
    beta = torch.randn(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)
    return q, k, v, g, beta, cu, h0


def run_config(total_T, N, H):
    q, k, v, g, beta, cu, h0 = make_inputs(total_T, N, H)
    scale = D ** -0.5

    # Warmup
    for _ in range(WARMUP):
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=h0, output_final_state=True,
                          safe_gate=True, cu_seqlens=cu)
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                  initial_state=h0, output_final_state=True,
                  safe_gate=True, cu_seqlens=cu)
    torch.cuda.synchronize()

    # Profiled region: flashla
    torch.cuda.nvtx.range_push(f"flashla_H{H}_T{total_T}_N{N}")
    for _ in range(REP):
        flash_kda_prefill(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                          initial_state=h0, output_final_state=True,
                          safe_gate=True, cu_seqlens=cu)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Profiled region: FLA
    torch.cuda.nvtx.range_push(f"FLA_H{H}_T{total_T}_N{N}")
    for _ in range(REP):
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                  initial_state=h0, output_final_state=True,
                  safe_gate=True, cu_seqlens=cu)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}, D={D}, warmup={WARMUP}, rep={REP}")

    configs = [
        # (total_T, N, H)
        (16384, 4, 32),
        (16384, 4, 64),
        (32768, 8, 32),
        (32768, 8, 64),
        (32768, 16, 64),
    ]

    for total_T, N, H in configs:
        label = f"H={H}, T={total_T}, N={N}, CTAs={N*H}"
        print(f"Profiling: {label}")
        run_config(total_T, N, H)

    print("Done.")
