"""Test: does chunk_delta_h_sm80 compile?"""
import sys
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA")
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA/third_party/flash-linear-attention")

import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cula.ops.chunk_delta_h_sm80 import ChunkDeltaHFwdSM80

print(f"GPU: {torch.cuda.get_device_name(0)} SM{torch.cuda.get_device_capability(0)}")
print("[1] Import OK")

kda_h = ChunkDeltaHFwdSM80()
print("[2] Instance OK")

B,S,H,V,K = 1,64,1,64,128
k=torch.randn(S,H,K,dtype=torch.bfloat16,device="cuda")*0.1
w=torch.randn(S,H,K,dtype=torch.bfloat16,device="cuda")*0.1
g=torch.zeros(S,H,K,dtype=torch.float32,device="cuda")
h0=torch.zeros(B,H,V,K,dtype=torch.float32,device="cuda")
ho=torch.zeros(B,H,V,K,dtype=torch.float32,device="cuda")
beta=torch.zeros(S,H,dtype=torch.float32,device="cuda")

print("[3] Compiling...")
try:
    compiled = cute.compile(
        kda_h,
        from_dlpack(k,16), from_dlpack(w,16), from_dlpack(g,16),
        from_dlpack(h0,16), from_dlpack(ho,16), from_dlpack(beta,16),
        (B,S,H,V,K),
        decay=0.99,
        stream=make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    print("    COMPILED!")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {str(e)[:500]}")