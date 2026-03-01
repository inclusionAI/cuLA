#!/usr/bin/env python3
"""Clean isolated test: single cute.compile() call, no cache pollution."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# Import directly to avoid flashla __init__ importing cuda extensions
import importlib.util
spec = importlib.util.spec_from_file_location("fwd_o", os.path.join(os.path.dirname(__file__), "flashla", "fwd_o.py"))
fwd_o_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fwd_o_mod)
ChunkGlaFwdO = fwd_o_mod.ChunkGlaFwdO
reference_chunk_gla_fwd_o = fwd_o_mod.reference_chunk_gla_fwd_o

BT = 64
K = V = 128
H = 4
dtype = torch.bfloat16
device = "cuda"
scale = K ** -0.5

def test_config(seq_lens, seed=42):
    """Test a single varlen config with fresh state."""
    torch.manual_seed(seed)
    num_seqs = len(seq_lens)
    T_total = sum(seq_lens)

    cu_seqlens_list = [0]
    chunk_offsets_list = [0]
    for sl in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        chunk_offsets_list.append(chunk_offsets_list[-1] + (sl + BT - 1) // BT)
    total_nt = chunk_offsets_list[-1]

    cu_t = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    ch_t = torch.tensor(chunk_offsets_list, dtype=torch.int32, device=device)
    cu_c = from_dlpack(cu_t.detach())
    ch_c = from_dlpack(ch_t.detach())
    ps = (num_seqs, T_total, H, K, V)
    stream = cutlass_torch.default_stream()

    q = torch.randn(T_total, H, K, dtype=dtype, device=device)
    v = torch.randn(T_total, H, V, dtype=dtype, device=device)
    g = torch.randn(T_total, H, K, dtype=dtype, device=device) * 0.1
    h = torch.randn(total_nt, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1

    # Reference
    o_ref = torch.zeros(T_total, H, V, dtype=dtype, device=device)
    for si, sl in enumerate(seq_lens):
        s, e = cu_seqlens_list[si], cu_seqlens_list[si + 1]
        co = chunk_offsets_list[si]
        nt_s = (sl + BT - 1) // BT
        o_s = reference_chunk_gla_fwd_o(
            q[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
            g[s:e].unsqueeze(0), h[co:co+nt_s],
            A[s:e].unsqueeze(0), scale, BT)
        o_ref[s:e] = o_s[0]

    # Kernel
    o_out = torch.zeros(T_total, H, V, dtype=dtype, device=device)
    kernel = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
    compiled = cute.compile(
        kernel,
        from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
        from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
        from_dlpack(o_out.detach()).iterator, from_dlpack(A.detach()).iterator,
        cu_c.iterator, ch_c.iterator,
        ps, total_nt, stream,
    )
    compiled(
        from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
        from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
        from_dlpack(o_out.detach()).iterator, from_dlpack(A.detach()).iterator,
        cu_c.iterator, ch_c.iterator,
        ps, total_nt, stream,
    )
    torch.cuda.synchronize()

    # Per-chunk analysis
    tok_offs = [cu_seqlens_list[i] for i in range(num_seqs)]
    aligned = all(t % BT == 0 for t in tok_offs)
    max_diff_all = 0
    for si, sl in enumerate(seq_lens):
        s = cu_seqlens_list[si]
        nt_s = (sl + BT - 1) // BT
        for ci in range(nt_s):
            cs = s + ci * BT
            ce = min(cs + BT, cu_seqlens_list[si + 1])
            cd = (o_ref[cs:ce].float() - o_out[cs:ce].float()).abs().max().item()
            max_diff_all = max(max_diff_all, cd)
            status = "PASS" if cd < 0.02 else "FAIL"
            if cd > 0.02:
                # Per-head breakdown
                hdiffs = [(o_ref[cs:ce, hh].float() - o_out[cs:ce, hh].float()).abs().max().item() for hh in range(H)]
                hstr = " ".join(f"h{i}={d:.4f}" for i, d in enumerate(hdiffs))
                print(f"  Seq{si} c{ci} tok={cs} rem={sl-ci*BT}: diff={cd:.4f} [{status}]  {hstr}")

    status = "PASS" if max_diff_all < 0.02 else "FAIL"
    print(f"seq_lens={seq_lens} T={T_total} aligned={aligned}: max_diff={max_diff_all:.6f} [{status}]")
    return max_diff_all < 0.02

# Run test specified via command line or default
if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "100,128"
    seq_lens = [int(x) for x in config_name.split(",")]
    print(f"Testing seq_lens={seq_lens} (single fresh compile, no cache pollution)")
    test_config(seq_lens)
