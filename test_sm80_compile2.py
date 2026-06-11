"""Test: does SM80 CuTe DSL kernel with MmaF16BF16Op compile?"""
import sys
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA")
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA/third_party/flash-linear-attention")

import torch, cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op
from cutlass.cute.typing import BFloat16, Float32
from cutlass.cute.runtime import from_dlpack, make_fake_stream

print(f"GPU: {torch.cuda.get_device_name(0)} SM{torch.cuda.get_device_capability(0)}")

# Test 1: Can we create the MMA op?
print("\n[1] Creating MmaF16BF16Op...")
op = MmaF16BF16Op(ab_dtype=BFloat16, acc_dtype=Float32, shape_mnk=(16,8,16))
print("    OK:", str(op).split('\n')[0])

# Test 2: MMA creation happens inside @cute.jit context — skip module-level test
print("\n[2] Skipped (MMA requires CuTe DSL context, created inside kernel)")

# Test 3: Can we import our kernel?
print("\n[3] Importing KDAFusedFwdSM80...")
from cula.ops.kda_fused_fwd_sm80 import KDAFusedFwdSM80
kda = KDAFusedFwdSM80()
print("    OK")

# Test 4: Compile via cute.compile (standard CuTe DSL pattern)
print("\n[4] Compiling kernel (cute.compile)...")
try:
    B, S, H, D = 1, 64, 1, 128
    q_c = from_dlpack(torch.randn(S, H, D, dtype=torch.bfloat16, device="cuda"), assumed_align=16)
    k_c = from_dlpack(torch.randn(S, H, D, dtype=torch.bfloat16, device="cuda"), assumed_align=16)
    g_c = from_dlpack(torch.randn(S, H, D, dtype=torch.float32, device="cuda"), assumed_align=16)
    o_c = from_dlpack(torch.empty(D, S, H*B, dtype=torch.bfloat16, device="cuda"), assumed_align=16)
    b_c = from_dlpack(torch.randn(S, H*B, dtype=torch.float32, device="cuda"), assumed_align=16)
    s_c = from_dlpack(torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda"), assumed_align=16)
    cu_c = from_dlpack(torch.tensor([0,S], dtype=torch.int32, device="cuda"), assumed_align=16)
    ws_c = from_dlpack(torch.zeros(128, dtype=torch.uint8, device="cuda"), assumed_align=16)
    stream = make_fake_stream()

    compiled = cute.compile(
        kda,
        q_c, k_c, g_c, o_c, b_c, s_c, s_c, cu_c, ws_c,
        problem_size=(B, S, H, D),
        stream=stream,
        options="--enable-tvm-ffi",
    )
    print("    COMPILED SUCCESSFULLY!")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {str(e)[:500]}")
