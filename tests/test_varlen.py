#!/usr/bin/env python3
"""Test varlen support for ChunkDeltaRuleFwdH."""

import sys
import pathlib
import torch
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashla.chunk_delta_h import ChunkDeltaRuleFwdH, reference_bf16_roundtrip


def make_varlen_data(seq_lens, H, K, V, BT, device="cuda", dtype=torch.bfloat16):
    """Create varlen-packed tensors from per-sequence lengths.

    All sequence lengths must be divisible by BT.
    Returns packed tensors and indexing metadata.
    """
    num_seqs = len(seq_lens)
    total_T = sum(seq_lens)
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    # chunk_offsets: prefix sum of per-sequence number of chunks
    NTs = [(l + BT - 1) // BT for l in seq_lens]
    chunk_offsets = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i, nt in enumerate(NTs):
        chunk_offsets[i + 1] = chunk_offsets[i] + nt
    total_NT = chunk_offsets[-1].item()

    torch.manual_seed(42)
    # Packed data: (1, total_T, H, D)
    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1

    return k, w, u, cu_seqlens, chunk_offsets, total_T, total_NT, NTs


def run_varlen_kernel(k, w, u, gk, h0, cu_seqlens, chunk_offsets,
                      num_seqs, total_T, total_NT, H, K, V, BT,
                      use_gk=0, use_h0=0, store_ht=0, save_vnew=1):
    """Run varlen kernel and return outputs."""
    NT = total_NT
    device = k.device
    dtype = k.dtype

    h_out = torch.zeros(1, total_NT, H, K, V, device=device, dtype=dtype)
    v_new = torch.zeros(1, total_T, H, V, device=device, dtype=dtype)
    ht = torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)

    g = torch.zeros(1, total_T, H, device=device, dtype=torch.float32)
    gk_t = gk if gk is not None else torch.zeros(1, total_T, H, K, device=device, dtype=torch.float32)
    h0_t = h0 if h0 is not None else torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V, is_varlen=True)
    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g), from_dlpack(gk_t)
    h0c = from_dlpack(h0_t)
    hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new), from_dlpack(ht)
    csc = from_dlpack(cu_seqlens)
    coc = from_dlpack(chunk_offsets)

    # Workspace for TensorMapManager (128 bytes per sequence)
    workspace = torch.zeros(num_seqs * 128, dtype=torch.uint8, device=device)
    wsc = from_dlpack(workspace)

    args = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        csc.iterator, coc.iterator, wsc.iterator,
        (num_seqs, total_T, H, K, V), total_NT,
        0, int(use_gk), int(use_h0), int(store_ht), int(save_vnew),
        stream,
    )

    print("  Compiling varlen kernel...")
    t0 = time.time()
    compiled = cute.compile(kernel, *args)
    print(f"  Compiled in {time.time()-t0:.2f}s")

    compiled(*args)
    torch.cuda.synchronize()

    return h_out, v_new, ht, compiled, args


def reference_varlen(k_packed, w_packed, u_packed, cu_seqlens, seq_lens,
                     H, K, V, BT, gk_packed=None, h0=None):
    """Per-sequence reference with full head support (all heads tracked)."""
    num_seqs = len(seq_lens)
    device = k_packed.device
    NTs = [(l + BT - 1) // BT for l in seq_lens]
    total_NT = sum(NTs)

    h_out_ref = torch.zeros(1, total_NT, H, K, V, device=device, dtype=torch.bfloat16)
    v_new_ref = torch.zeros(1, cu_seqlens[-1].item(), H, V, device=device, dtype=torch.bfloat16)
    ht_ref = torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.bfloat16)

    chunk_off = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        NT = NTs[seq_idx]

        # Extract per-sequence data: (1, seq_len, H, D)
        k_seq = k_packed[:, bos:eos]
        w_seq = w_packed[:, bos:eos]
        u_seq = u_packed[:, bos:eos]

        gk_seq = gk_packed[:, bos:eos] if gk_packed is not None else None
        h0_seq = h0[seq_idx:seq_idx+1] if h0 is not None else None

        # Run chunk loop manually (like reference_bf16_roundtrip but track ALL heads)
        h = torch.zeros(1, H, K, V, device=device, dtype=torch.float32)
        if h0_seq is not None:
            h = h0_seq.clone().float()

        for t in range(NT):
            # Store h at beginning of chunk (h_out[chunk_off + t] = state BEFORE chunk t)
            h_out_ref[0, chunk_off + t] = h[0].to(torch.bfloat16)

            s, e = t * BT, min((t + 1) * BT, seq_len)
            wc = w_seq[:, s:e].permute(0, 2, 1, 3).float()
            kc = k_seq[:, s:e].permute(0, 2, 1, 3).float()
            uc = u_seq[:, s:e].permute(0, 2, 1, 3).float()

            h_bf16 = h.to(torch.bfloat16).float()
            wh = torch.matmul(wc, h_bf16)
            vnc = uc - wh

            v_new_ref[:, bos + s:bos + e] = vnc.permute(0, 2, 1, 3).to(torch.bfloat16)

            if gk_seq is not None:
                gkc = gk_seq[:, s:e].permute(0, 2, 1, 3).float()
                gkl = gkc[:, :, -1, :].float()
                h = h * torch.exp(gkl).unsqueeze(-1)

            vn_bf16 = vnc.to(torch.bfloat16).float()
            h = h + torch.matmul(kc.transpose(-2, -1), vn_bf16)

        # Final state for this sequence
        ht_ref[seq_idx] = h[0].to(torch.bfloat16)
        chunk_off += NT

    return h_out_ref, v_new_ref, ht_ref


def test_varlen_basic():
    """Test varlen with equal-length sequences (simplest case)."""
    print("\n" + "="*60)
    print("Test Varlen 1: 2 equal-length sequences (128 tokens each)")
    H, K, V, BT = 1, 128, 128, 64
    seq_lens = [128, 128]
    k, w, u, cu_seqlens, chunk_offsets, total_T, total_NT, NTs = \
        make_varlen_data(seq_lens, H, K, V, BT)

    h_out, v_new, ht, _, _ = run_varlen_kernel(
        k, w, u, None, None, cu_seqlens, chunk_offsets,
        len(seq_lens), total_T, total_NT, H, K, V, BT,
    )

    h_out_ref, v_new_ref, ht_ref = reference_varlen(
        k, w, u, cu_seqlens, seq_lens, H, K, V, BT
    )

    # Compare h_out (skip first chunk per sequence since it's zero/h0)
    max_diff_h = 0.0
    chunk_off = 0
    for seq_idx, (sl, nt) in enumerate(zip(seq_lens, NTs)):
        for t in range(1, nt):
            d = (h_out[0, chunk_off + t, 0].float() - h_out_ref[0, chunk_off + t, 0].float()).abs().max().item()
            max_diff_h = max(max_diff_h, d)
        chunk_off += nt
    print(f"  h_out max diff: {max_diff_h:.6f}")

    d_vn = (v_new.float() - v_new_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vn:.6f}")

    passed = max_diff_h < 0.5 and d_vn < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_varlen_mixed():
    """Test varlen with different-length sequences."""
    print("\n" + "="*60)
    print("Test Varlen 2: 3 sequences of different lengths (50, 192, 100)")
    H, K, V, BT = 1, 128, 128, 64
    seq_lens = [50, 192, 100]  # 50 and 100 are NOT multiples of BT=64
    k, w, u, cu_seqlens, chunk_offsets, total_T, total_NT, NTs = \
        make_varlen_data(seq_lens, H, K, V, BT)

    h_out, v_new, ht, _, _ = run_varlen_kernel(
        k, w, u, None, None, cu_seqlens, chunk_offsets,
        len(seq_lens), total_T, total_NT, H, K, V, BT,
    )

    h_out_ref, v_new_ref, ht_ref = reference_varlen(
        k, w, u, cu_seqlens, seq_lens, H, K, V, BT
    )

    max_diff_h = 0.0
    chunk_off = 0
    for seq_idx, (sl, nt) in enumerate(zip(seq_lens, NTs)):
        for t in range(1, nt):
            d = (h_out[0, chunk_off + t, 0].float() - h_out_ref[0, chunk_off + t, 0].float()).abs().max().item()
            max_diff_h = max(max_diff_h, d)
        chunk_off += nt
    print(f"  h_out max diff: {max_diff_h:.6f}")

    d_vn = (v_new.float() - v_new_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vn:.6f}")

    passed = max_diff_h < 0.5 and d_vn < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_varlen_with_gk_h0_ht():
    """Test varlen with gk gating, h0, and ht."""
    print("\n" + "="*60)
    print("Test Varlen 3: With gk + h0 + ht (3 seqs, H=2, non-aligned lengths)")
    H, K, V, BT = 2, 128, 128, 64
    seq_lens = [100, 256, 30]  # 100 and 30 are NOT multiples of BT=64
    num_seqs = len(seq_lens)
    k, w, u, cu_seqlens, chunk_offsets, total_T, total_NT, NTs = \
        make_varlen_data(seq_lens, H, K, V, BT)

    torch.manual_seed(123)
    gk = torch.randn(1, total_T, H, K, device="cuda", dtype=torch.float32) * 0.1
    gk = -torch.abs(gk).cumsum(dim=1)
    # Per-sequence cumsum: reset at sequence boundaries
    gk_proper = torch.zeros_like(gk)
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        bos, eos = bos.item(), eos.item()
        gk_seq = torch.randn(1, eos - bos, H, K, device="cuda", dtype=torch.float32) * 0.1
        gk_proper[:, bos:eos] = -torch.abs(gk_seq).cumsum(dim=1)
    gk = gk_proper

    h0 = torch.randn(num_seqs, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    h_out, v_new, ht, _, _ = run_varlen_kernel(
        k, w, u, gk, h0, cu_seqlens, chunk_offsets,
        num_seqs, total_T, total_NT, H, K, V, BT,
        use_gk=1, use_h0=1, store_ht=1, save_vnew=1,
    )

    h_out_ref, v_new_ref, ht_ref = reference_varlen(
        k, w, u, cu_seqlens, seq_lens, H, K, V, BT,
        gk_packed=gk, h0=h0,
    )

    max_diff_h = 0.0
    chunk_off = 0
    for seq_idx, (sl, nt) in enumerate(zip(seq_lens, NTs)):
        for t in range(1, nt):
            d = (h_out[0, chunk_off + t, 0].float() - h_out_ref[0, chunk_off + t, 0].float()).abs().max().item()
            max_diff_h = max(max_diff_h, d)
        chunk_off += nt
    print(f"  h_out max diff: {max_diff_h:.6f}")

    d_vn = (v_new.float() - v_new_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vn:.6f}")

    d_ht = (ht.float() - ht_ref.float()).abs().max().item()
    print(f"  ht max diff: {d_ht:.6f}")

    passed = max_diff_h < 0.5 and d_vn < 0.5 and d_ht < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_varlen_multi_head():
    """Test varlen with multiple heads and V-tiles."""
    print("\n" + "="*60)
    print("Test Varlen 4: Multi-head H=4, 4 sequences (non-aligned)")
    H, K, V, BT = 4, 128, 128, 64
    seq_lens = [33, 128, 200, 95]  # 33, 200, 95 are NOT multiples of BT=64
    k, w, u, cu_seqlens, chunk_offsets, total_T, total_NT, NTs = \
        make_varlen_data(seq_lens, H, K, V, BT)

    h_out, v_new, ht, _, _ = run_varlen_kernel(
        k, w, u, None, None, cu_seqlens, chunk_offsets,
        len(seq_lens), total_T, total_NT, H, K, V, BT,
    )

    h_out_ref, v_new_ref, ht_ref = reference_varlen(
        k, w, u, cu_seqlens, seq_lens, H, K, V, BT
    )

    max_diff_h = 0.0
    chunk_off = 0
    for seq_idx, (sl, nt) in enumerate(zip(seq_lens, NTs)):
        for t in range(1, nt):
            for h in range(H):
                d = (h_out[0, chunk_off + t, h].float() - h_out_ref[0, chunk_off + t, h].float()).abs().max().item()
                max_diff_h = max(max_diff_h, d)
        chunk_off += nt
    print(f"  h_out max diff: {max_diff_h:.6f}")

    d_vn = (v_new.float() - v_new_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vn:.6f}")

    passed = max_diff_h < 0.5 and d_vn < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_varlen_vs_nonvarlen():
    """Test that varlen with a single sequence matches non-varlen."""
    print("\n" + "="*60)
    print("Test Varlen 5: Single sequence matches non-varlen")
    H, K, V, BT = 2, 128, 128, 64
    B, T = 1, 256
    NT = T // BT

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    # Non-varlen run
    g_z = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk_z = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0_z = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    h_out_nv = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new_nv = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht_nv = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    kernel_nv = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V, is_varlen=False)
    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g_z), from_dlpack(gk_z)
    h0c = from_dlpack(h0_z)
    hc, vnc, htc = from_dlpack(h_out_nv), from_dlpack(v_new_nv), from_dlpack(ht_nv)
    cu_seqlens_d = torch.zeros(2, dtype=torch.int32, device="cuda")
    chunk_offsets_d = torch.zeros(2, dtype=torch.int32, device="cuda")
    workspace_d = torch.zeros(128, dtype=torch.uint8, device="cuda")
    csd = from_dlpack(cu_seqlens_d)
    cod = from_dlpack(chunk_offsets_d)
    wsd = from_dlpack(workspace_d)

    args_nv = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        csd.iterator, cod.iterator, wsd.iterator,
        (B, T, H, K, V), NT,
        0, 0, 0, 0, 1,
        stream,
    )
    print("  Compiling non-varlen kernel...")
    compiled_nv = cute.compile(kernel_nv, *args_nv)
    compiled_nv(*args_nv)
    torch.cuda.synchronize()

    # Varlen run with same data as single sequence
    cu_seqlens_v = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    chunk_offsets_v = torch.tensor([0, NT], dtype=torch.int32, device="cuda")
    h_out_v = torch.zeros(1, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new_v = torch.zeros(1, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht_v = torch.zeros(1, H, K, V, device="cuda", dtype=torch.float32)

    kernel_v = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V, is_varlen=True)

    hc_v, vnc_v, htc_v = from_dlpack(h_out_v), from_dlpack(v_new_v), from_dlpack(ht_v)
    csv = from_dlpack(cu_seqlens_v)
    cov = from_dlpack(chunk_offsets_v)
    workspace_v = torch.zeros(128, dtype=torch.uint8, device="cuda")
    wsv = from_dlpack(workspace_v)

    args_v = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc_v.iterator, vnc_v.iterator, h0c.iterator, htc_v.iterator,
        csv.iterator, cov.iterator, wsv.iterator,
        (1, T, H, K, V), NT,
        0, 0, 0, 0, 1,
        stream,
    )
    print("  Compiling varlen kernel...")
    compiled_v = cute.compile(kernel_v, *args_v)
    compiled_v(*args_v)
    torch.cuda.synchronize()

    d_h = (h_out_nv.float() - h_out_v.float()).abs().max().item()
    d_vn = (v_new_nv.float() - v_new_v.float()).abs().max().item()
    print(f"  h_out diff (varlen vs non-varlen): {d_h:.6f}")
    print(f"  v_new diff (varlen vs non-varlen): {d_vn:.6f}")

    passed = d_h < 1e-6 and d_vn < 1e-6
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []
    names = []

    names.append("Equal-length seqs")
    results.append(test_varlen_basic())

    names.append("Mixed-length seqs")
    results.append(test_varlen_mixed())

    names.append("gk + h0 + ht")
    results.append(test_varlen_with_gk_h0_ht())

    names.append("Multi-head H=4")
    results.append(test_varlen_multi_head())

    names.append("Varlen vs non-varlen")
    results.append(test_varlen_vs_nonvarlen())

    print("\n" + "="*60)
    print("VARLEN TEST SUMMARY:")
    for i, (name, r) in enumerate(zip(names, results)):
        print(f"  Test {i+1} ({name}): {'PASS' if r else 'FAIL'}")
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} tests passed")
    print("ALL PASS" if n_pass == len(results) else "SOME FAILED")
