#!/usr/bin/env python3
"""Diagnostic: isolate which input causes the varlen misalignment error."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

import importlib.util
spec = importlib.util.spec_from_file_location("fwd_o", os.path.join(os.path.dirname(__file__), "flashla", "fwd_o.py"))
fwd_o_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fwd_o_mod)
ChunkGlaFwdO = fwd_o_mod.ChunkGlaFwdO
reference_chunk_gla_fwd_o = fwd_o_mod.reference_chunk_gla_fwd_o

BT = 64; K = V = 128; H = 4
dtype = torch.bfloat16; device = "cuda"; scale = K ** -0.5

def run_test(seq_lens, label="", h_zero=False, v_zero=False, a_zero=False, q_zero=False, g_zero=False):
    torch.manual_seed(42)
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

    if h_zero: h.zero_()
    if v_zero: v.zero_()
    if a_zero: A.zero_()
    if q_zero: q.zero_()
    if g_zero: g.zero_()

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

    # Check seq 1 chunk 0 only
    si = 1; sl = seq_lens[1]
    s = cu_seqlens_list[1]
    cs = s; ce = min(s + BT, cu_seqlens_list[2])
    cd = (o_ref[cs:ce].float() - o_out[cs:ce].float()).abs().max().item()
    hdiffs = [(o_ref[cs:ce, hh].float() - o_out[cs:ce, hh].float()).abs().max().item() for hh in range(H)]
    hstr = " ".join(f"h{i}={d:.4f}" for i, d in enumerate(hdiffs))
    status = "PASS" if cd < 0.02 else "FAIL"
    print(f"{label:30s} Seq1c0 diff={cd:.4f} [{status}]  {hstr}")

seq_lens = [100, 128]
print(f"seq_lens={seq_lens}, tok_offset={sum(seq_lens[:1])}")
print("="*120)
run_test(seq_lens, label="baseline (all inputs random)")
run_test(seq_lens, label="h=0 (only AV term)", h_zero=True)
run_test(seq_lens, label="A=0,v=0 (only QH term)", a_zero=True, v_zero=True)
run_test(seq_lens, label="q=0 (only AV term via q)", q_zero=True)
run_test(seq_lens, label="v=0 (only QH term via v)", v_zero=True)
