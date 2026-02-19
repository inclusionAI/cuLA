#!/usr/bin/env python3
"""Debug test for h0 initial state loading into KV TMEM."""

import os
os.environ["CUTLASS_DSL_DEBUG_LEVEL"] = "0"

import torch
import cutlass
import cutlass.torch as cutlass_torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from chunk_delta_h import ChunkDeltaRuleFwdH

def test_h0_identity():
    """Test h0 with zero inputs — kernel should output h0 as h_out."""
    B, T, H, K, V = 1, 64, 1, 128, 128  # Single chunk
    BT = 64
    NT = 1
    
    torch.manual_seed(42)
    
    # All zeros for k, w, u → v_new = 0, h_out = h0
    k = torch.zeros(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    u = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    
    # Random h0 (bf16 for kernel)
    h0_bf16 = (torch.randn(B, H, K, V, device="cuda", dtype=torch.bfloat16) * 0.5)
    
    # Output tensors
    h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new_out = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht_out = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    
    # Convert to CuTe
    k_cute = from_dlpack(k)
    w_cute = from_dlpack(w)
    u_cute = from_dlpack(u)
    g_cute = from_dlpack(g)
    gk_cute = from_dlpack(gk)
    h0_cute = from_dlpack(h0_bf16)
    h_out_cute = from_dlpack(h_out)
    v_new_cute = from_dlpack(v_new_out)
    ht_cute = from_dlpack(ht_out)
    
    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    stream = cutlass_torch.default_stream()
    problem_size = (B, T, H, K, V)
    
    print("Compiling...")
    compiled = cute.compile(
        kernel,
        k_cute.iterator, w_cute.iterator, u_cute.iterator, g_cute.iterator, gk_cute.iterator,
        h_out_cute.iterator, v_new_cute.iterator, h0_cute.iterator, ht_cute.iterator,
        problem_size,
        False, False, True, True, True,  # use_g=F, use_gk=F, use_initial_state=T, store_final=T, save_vnew=T
        stream,
    )
    
    print("Running...")
    compiled(
        k_cute.iterator, w_cute.iterator, u_cute.iterator, g_cute.iterator, gk_cute.iterator,
        h_out_cute.iterator, v_new_cute.iterator, h0_cute.iterator, ht_cute.iterator,
        problem_size,
        False, False, True, True, True,
        stream,
    )
    torch.cuda.synchronize()
    
    # With all-zero inputs: v_new should be 0, h_out should be h0
    print(f"\nh0 (first 8 elements): {h0_bf16[0,0,0,:8]}")
    print(f"h_out[0] (first 8 elements): {h_out[0,0,0,0,:8]}")
    print(f"v_new (first 8 elements): {v_new_out[0,0,0,:8]}")
    
    # h_out[0] should equal h0 (since k=0, w=0, u=0 → v_new=0, KV=0, h_out = h0 + 0 = h0)
    h0_fp32 = h0_bf16.float()
    h_out_fp32 = h_out.float()
    
    diff_h = (h_out_fp32[0, 0] - h0_fp32[0, 0]).abs().max().item()
    diff_vnew = v_new_out.abs().max().item()
    
    print(f"\nMax |h_out - h0|: {diff_h}")
    print(f"Max |v_new|: {diff_vnew}")
    
    if diff_h < 0.01 and diff_vnew < 0.01:
        print("✅ h0 identity test PASSED")
    else:
        print("❌ h0 identity test FAILED")
        # Print more details
        print(f"\nh0[0,0,:4,:4]:\n{h0_bf16[0,0,:4,:4]}")
        print(f"\nh_out[0,0,0,:4,:4]:\n{h_out[0,0,0,:4,:4]}")
        print(f"\nh_out is all zeros: {(h_out == 0).all().item()}")

if __name__ == "__main__":
    test_h0_identity()
