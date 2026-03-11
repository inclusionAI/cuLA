"""
Profile script: non-varlen vs varlen (balanced) with ncu.
Creates two entry points that can be profiled separately.

Usage:
  # Profile non-varlen
  ncu --set full -o /tmp/ncu_nonvarlen python profile_varlen_overhead.py nonvarlen
  # Profile varlen (single seq, same work)
  ncu --set full -o /tmp/ncu_varlen python profile_varlen_overhead.py varlen
  # Profile varlen (4 balanced seqs, same total work)
  ncu --set full -o /tmp/ncu_varlen_bal python profile_varlen_overhead.py varlen_balanced
"""

import sys
import pathlib

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from flashla.kda_wrapper import flash_kda_prefill
from benchmarks.utils import set_seed, exclusive_cumsum

H = 32
D = 128
T = 8192
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

def setup_inputs():
    set_seed(42)
    scale = D ** -0.5
    q = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T, H, D, dtype=DTYPE, device=DEVICE)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float, device=DEVICE)).clamp(-5, 0)
    beta = torch.randn(1, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    h0 = torch.randn(1, H, D, D, dtype=torch.float32, device=DEVICE)
    return q, k, v, g, beta, h0, scale

def run_nonvarlen(q, k, v, g, beta, h0, scale):
    return flash_kda_prefill(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False, safe_gate=True,
        cu_seqlens=None,
    )

def run_varlen_single(q, k, v, g, beta, h0, scale):
    cu = torch.tensor([0, T], dtype=torch.long, device=DEVICE)
    return flash_kda_prefill(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False, safe_gate=True,
        cu_seqlens=cu,
    )

def run_varlen_balanced(q, k, v, g, beta, h0_multi, scale):
    num_seqs = 4
    per_seq = T // num_seqs
    cu = torch.tensor(exclusive_cumsum([per_seq] * num_seqs), dtype=torch.long, device=DEVICE)
    return flash_kda_prefill(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_multi, output_final_state=True,
        use_qk_l2norm_in_kernel=False, safe_gate=True,
        cu_seqlens=cu,
    )

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nonvarlen"
    
    q, k, v, g, beta, h0, scale = setup_inputs()
    
    # Warmup (compile kernels)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        if mode == "nonvarlen":
            run_nonvarlen(q, k, v, g, beta, h0, scale)
        elif mode == "varlen":
            run_varlen_single(q, k, v, g, beta, h0, scale)
        elif mode == "varlen_balanced":
            h0_multi = torch.randn(4, H, D, D, dtype=torch.float32, device=DEVICE)
            run_varlen_balanced(q, k, v, g, beta, h0_multi, scale)
    torch.cuda.synchronize()
    
    # Profiled run
    if mode == "nonvarlen":
        for _ in range(3):
            run_nonvarlen(q, k, v, g, beta, h0, scale)
    elif mode == "varlen":
        for _ in range(3):
            run_varlen_single(q, k, v, g, beta, h0, scale)
    elif mode == "varlen_balanced":
        h0_multi = torch.randn(4, H, D, D, dtype=torch.float32, device=DEVICE)
        for _ in range(3):
            run_varlen_balanced(q, k, v, g, beta, h0_multi, scale)
    
    torch.cuda.synchronize()
    print(f"Done: mode={mode}")
