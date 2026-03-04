"""Quick persistent kernel test — 1-stage, grid=SM_count."""
import os, sys
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, '/ossfs/workspace/fwd_o/flashla/flashla')
from fwd_o import ChunkGlaFwdO, build_chunk_indices

def reference_fwd_o(q, v, g, h, A, scale, BT):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = (T + BT - 1) // BT
    o = torch.zeros(B, T, H, V, dtype=q.dtype, device=q.device)
    for b in range(B):
        for i in range(NT):
            s = i * BT
            e = min(s + BT, T)
            L = e - s
            for hh in range(H):
                qg = (q[b, s:e, hh] * torch.exp2(g[b, s:e, hh])).float() * scale
                inter = qg @ h[b * NT + i if h.shape[0] > NT else i, hh].float()
                A_chunk = A[b, s:e, hh, :L].float()
                mask = torch.tril(torch.ones(L, L, device=A.device))
                A_masked = A_chunk * mask
                intra = A_masked @ v[b, s:e, hh].float()
                o[b, s:e, hh] = (inter + intra).to(q.dtype)
    return o

device = 'cuda'
dtype = torch.bfloat16
BT, K, V = 64, 128, 128
scale = K ** -0.5
stream = cutlass_torch.default_stream()

sm_count = torch.cuda.get_device_properties(0).multi_processor_count
print(f"SM count: {sm_count}")

def build_chunk_offsets(seq_lens, BT):
    offsets = [0]
    for s in seq_lens:
        offsets.append(offsets[-1] + (s + BT - 1) // BT)
    return offsets

def test_varlen(seq_lens, H=1, seed=42):
    torch.manual_seed(seed)
    T_total = sum(seq_lens)
    cu_list = [0]
    for s in seq_lens:
        cu_list.append(cu_list[-1] + s)
    total_nt = sum((s + BT - 1) // BT for s in seq_lens)
    ch_off = build_chunk_offsets(seq_lens, BT)

    ci = build_chunk_indices(seq_lens, BT=BT, device=device)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=device)

    q = torch.randn(T_total, H, K, dtype=dtype, device=device)
    v = torch.randn(T_total, H, V, dtype=dtype, device=device)
    g = torch.randn(T_total, H, K, dtype=torch.float32, device=device) * 0.1
    h = torch.randn(total_nt, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1
    o = torch.zeros(T_total, H, V, dtype=dtype, device=device)

    # Reference
    o_ref = torch.zeros_like(o)
    for si, sl in enumerate(seq_lens):
        s, e = cu_list[si], cu_list[si+1]
        co = ch_off[si]
        nt_s = (sl + BT - 1) // BT
        o_s = reference_fwd_o(
            q[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
            g[s:e].unsqueeze(0), h[co:co+nt_s],
            A[s:e].unsqueeze(0), scale, BT)
        o_ref[s:e] = o_s[0]

    # Kernel
    ps = (len(seq_lens), T_total, H, K, V)
    num_wu = total_nt * H
    num_iters_max = (num_wu + sm_count - 1) // sm_count
    kernel = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale,
                          is_varlen=True, persistent=True)
    print(f"  Compiling: seqs={seq_lens} H={H} WUs={num_wu} max_iters/CTA={num_iters_max} "
          f"persistent={kernel.persistent} occ={kernel.min_occupancy} "
          f"stages=q{kernel.q_stage}/g{kernel.g_stage}/h{kernel.h_stage}/v{kernel.v_stage}/a{kernel.a_stage}")
    sys.stdout.flush()

    compiled = cute.compile(kernel,
        from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
        from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
        from_dlpack(o.detach()).iterator, from_dlpack(A.detach()).iterator,
        from_dlpack(cu.detach()).iterator, from_dlpack(ci.detach()).iterator,
        ps, total_nt, stream)
    print("  Compiled. Running...")
    sys.stdout.flush()

    compiled(
        from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
        from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
        from_dlpack(o.detach()).iterator, from_dlpack(A.detach()).iterator,
        from_dlpack(cu.detach()).iterator, from_dlpack(ci.detach()).iterator,
        ps, total_nt, stream)
    torch.cuda.synchronize()

    max_diff = (o_ref.float() - o.float()).abs().max().item()
    status = "PASS" if max_diff < 0.02 else "FAIL"
    print(f"  Result: max_diff={max_diff:.6f} [{status}]")
    sys.stdout.flush()

    if max_diff >= 0.02:
        for si, sl in enumerate(seq_lens):
            s = cu_list[si]
            nt_s = (sl + BT - 1) // BT
            for c in range(nt_s):
                cs = s + c * BT
                ce = min(cs + BT, cu_list[si+1])
                cd = (o_ref[cs:ce].float() - o[cs:ce].float()).abs().max().item()
                if cd > 0.01:
                    print(f"    Seq{si} c{c} tok={cs} rem={sl-c*BT}: diff={cd:.4f}")
    return max_diff < 0.02

# Phase 1: simplest case — num_iters=1 for all CTAs
print("\n=== Phase 1: 1 WU (num_iters=1) ===")
sys.stdout.flush()
ok1 = test_varlen([64], H=1)

# Phase 2: still mostly num_iters=1 but 2 WUs
print("\n=== Phase 2: 2 WUs ===")
sys.stdout.flush()
ok2 = test_varlen([128], H=1)

# Phase 3: num_iters > 1 for some CTAs — this is the critical test
print("\n=== Phase 3: Many WUs (tests persistent loop) ===")
sys.stdout.flush()
ok3 = test_varlen([100, 128], H=4)

# Phase 4: Large test
print("\n=== Phase 4: Large test ===")
sys.stdout.flush()
ok4 = test_varlen([400, 300, 500], H=8)

# Phase 5: Very large
print("\n=== Phase 5: Very large (H=64) ===")
sys.stdout.flush()
ok5 = test_varlen([400, 300, 500, 350, 450, 380, 420, 370, 410, 390,
                   430, 360, 440, 350, 460, 380, 420, 370, 410, 362], H=64)

all_pass = ok1 and ok2 and ok3 and ok4 and ok5
print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
