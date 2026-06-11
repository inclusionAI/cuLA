"""Runtime test: run SM80 QK MMA kernel via KDAFusedFwdSM80."""
import sys, json
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA")
sys.path.insert(0, "/mnt/d/Programming/New folder (2)/cuLA/third_party/flash-linear-attention")

OUT = "/tmp/kda_qk_runtime.json"

try:
    import torch, cutlass, cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
    from cula.ops.kda_fused_fwd_sm80 import KDAFusedFwdSM80, BT

    torch.manual_seed(42)
    device = "cuda"
    B, S, H, D = 1, 64, 1, 128

    q = torch.randn(S, H, D, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(S, H, D, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.zeros(S, H, D, dtype=torch.float32, device=device)
    o = torch.zeros(D, S, H * B, dtype=torch.bfloat16, device=device)
    beta = torch.zeros(S, H * B, dtype=torch.float32, device=device)
    s0 = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
    cu = torch.tensor([0, S], dtype=torch.int32, device=device)
    ws = torch.zeros(128, dtype=torch.uint8, device=device)

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    ref = q_f32 @ k_f32.T
    print(f"Reference Q@K^T: shape={ref.shape}, range=[{ref.min():.4f}, {ref.max():.4f}]")

    kda = KDAFusedFwdSM80(chunk_size=64, head_dim=128, scale=1.0)

    q_c = from_dlpack(q, assumed_align=16)
    k_c = from_dlpack(k, assumed_align=16)
    g_c = from_dlpack(g, assumed_align=16)
    o_c = from_dlpack(o, assumed_align=16)
    b_c = from_dlpack(beta, assumed_align=16)
    s_c = from_dlpack(s0, assumed_align=16)
    cu_c = from_dlpack(cu, assumed_align=16)
    ws_c = from_dlpack(ws, assumed_align=16)

    print("Compiling...")
    compiled = cute.compile(
        kda, q_c, k_c, g_c, o_c, b_c, s_c, s_c, cu_c, ws_c,
        problem_size=(B, S, H, D),
        stream=make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    print("Compiled OK")

    print("Running...")
    import cuda.bindings.driver as cuda_drv
    real_stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(q, k, g, o, beta, s0, s0, cu, ws, (B, S, H, D), real_stream)
    torch.cuda.synchronize()
    print(f"Ran OK. o[0,0,0]={o[0,0,0].item()}")

    o_cpu = o[:BT, 0, :BT].float().cpu()
    ref_cpu = ref.cpu()
    diff = (o_cpu - ref_cpu).abs()
    correct = diff.max() < 0.05
    msg = "✅ CORRECT" if correct else "❌ MISMATCH"
    print(f"{msg} | max_err={diff.max():.4f} mean_err={diff.mean():.4f}")

    with open(OUT, "w") as f:
        json.dump({
            "status": "SUCCESS",
            "max_error": float(diff.max().item()),
            "mean_error": float(diff.mean().item()),
            "correct": correct,
        }, f)

except Exception as e:
    with open(OUT, "w") as f:
        json.dump({"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:500]}"}, f)
    print(f"FAIL: {e}")