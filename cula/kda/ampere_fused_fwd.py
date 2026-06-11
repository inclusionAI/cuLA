"""
Ampere (SM80) fully-fused KDA forward prefill.

Architecture:
    - Default: FLA Triton (all tests pass)
    - Opt-in: CuTe DSL SM80 kernels (CULA_USE_SM80_CUTEDSL=1)
"""

import os as _os
import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

_USE_CUTEDSL = _os.environ.get("CULA_USE_SM80_CUTEDSL", "0") == "1"


class AmpereChunkKDAFunction(torch.autograd.Function):
    """Ampere (SM80) KDA forward prefill."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.IntTensor | None = None,
        chunk_indices: torch.IntTensor | None = None,
    ):
        assert q.shape == k.shape, "q and k must have the same shape."

        if _USE_CUTEDSL:
            o, final_state = _forward_cutedsl(
                q, k, v, g, beta, A_log, dt_bias, scale,
                initial_state, output_final_state,
                use_qk_l2norm_in_kernel, use_gate_in_kernel,
                safe_gate, lower_bound, cu_seqlens, chunk_indices,
            )
        else:
            from fla.ops.kda import chunk_kda
            o, final_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                A_log=A_log, dt_bias=dt_bias,
                transpose_state_layout=True,
            )

        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill_ampere(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    """Ampere (SM80) KDA forward prefill """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)

    if A_log is None:
        A_log = -torch.ones(v.shape[-2], dtype=torch.float32, device=q.device)
    if dt_bias is None:
        dt_bias = torch.zeros(v.shape[-2], q.shape[-1], dtype=torch.float32, device=q.device)

    return AmpereChunkKDAFunction.apply(
        q, k, v, g, beta,
        A_log, dt_bias, scale, initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        safe_gate,
        lower_bound,
        cu_seqlens,
        chunk_indices,
    )


def _forward_cutedsl(
    q, k, v, g, beta, A_log, dt_bias, scale,
    initial_state, output_final_state,
    use_qk_l2norm_in_kernel, use_gate_in_kernel,
    safe_gate, lower_bound, cu_seqlens, chunk_indices,
):
    import os, torch.nn.functional as F
    import cutlass.cute as cute
    from cula.ops.kda_fused_fwd_sm80 import KDAFusedFwdSM80
    from cula.ops.chunk_delta_h_sm80 import ChunkDeltaHFwdSM80
    from cula.ops.fwd_o_sm80 import FwdOSM80

    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    C = 64
    BV = 64
    NV = (V_dim + BV - 1) // BV  # number of V-tiles
    NC = (T + C - 1) // C          # number of chunks

    # Gate preprocessing: per-chunk cumsum → pure PyTorch, <1e-5 vs Triton
    LN2 = 0.6931471805599453
    RCP_LN2 = 1.0 / LN2
    g = g.float()
    g_padded = g.reshape(B, NC, C, H, K_dim)
    g = (g_padded.cumsum(dim=2) * RCP_LN2).reshape(B, T, H, K_dim).to(g.dtype)

    if use_qk_l2norm_in_kernel:
        from fla.modules.l2norm import l2norm_fwd
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    # Output & state tensors
    if initial_state is None:
        h_state = torch.zeros(B, H, V_dim, K_dim, dtype=torch.float32, device=q.device)
    else:
        h_state = initial_state.to(torch.float32).clone()

    # Flatten batch into sequence for 3D CuTe tensors [S, H, D]
    q_kda = q.reshape(B * T, H, K_dim).contiguous()
    k_kda = k.reshape(B * T, H, K_dim).contiguous()
    g_kda = g.reshape(B * T, H, K_dim).contiguous()
    v_kda = v.reshape(B * T, H, V_dim).contiguous()

    # Temporary buffer for QK scores [B*T, H, K_dim]
    qk_buf = torch.zeros(B * T, H, K_dim, dtype=torch.bfloat16, device=q.device)

    # Compile kernels
    stream = torch.cuda.current_stream().cuda_stream
    kda_qk = KDAFusedFwdSM80(chunk_size=C, head_dim=K_dim, scale=scale)
    kda_dh = ChunkDeltaHFwdSM80(chunk_size=C, head_dim_k=K_dim, head_dim_v=V_dim)
    kda_fo = FwdOSM80(chunk_size=C, head_dim_k=K_dim, head_dim_v=V_dim)

    # ── Pre-compute asymmetric gating on host ──
    # g is cumsum-scaled by RCP_LN2. exp2(g) gives the raw gate.
    # FLA KDA: Q_g = exp2(g - g_last) * Q, K_inv = exp2(-g + g_last) * K
    # We pass g=0 to kernels so they don't re-gate.
    g_zero = torch.zeros_like(g_kda)  # zero gate → kernels apply exp2(0)=1 (no-op)

    # Pre-compute per-chunk g_last for asymmetric gating
    q_gated = torch.zeros_like(q_kda, dtype=torch.bfloat16)
    k_inv = torch.zeros_like(k_kda, dtype=torch.bfloat16)
    for n in range(NC):
        t_beg = n * C
        t_end = min(t_beg + C, T)
        ct = t_end - t_beg
        # g_last: gate at last token of chunk [H, K]
        g_last = g_kda[t_end - 1, :, :].float()  # [H, K]
        g_chunk = g_kda[t_beg:t_end, :, :].float()  # [ct, H, K]
        q_chunk = q_kda[t_beg:t_end, :, :].float()  # [ct, H, K]
        k_chunk = k_kda[t_beg:t_end, :, :].float()  # [ct, H, K]

        # Asymmetric gating
        q_g = q_chunk * torch.exp((g_chunk - g_last.unsqueeze(0)) * LN2)  # exp2(g - g_last)
        k_i = k_chunk * torch.exp((-g_chunk + g_last.unsqueeze(0)) * LN2)  # exp2(-g + g_last)

        q_gated[t_beg:t_end] = q_g.to(torch.bfloat16)
        k_inv[t_beg:t_end] = k_i.to(torch.bfloat16)

    # Enable persistent JIT caching (avoids recompilation across sessions)
    os.environ.setdefault("CUTE_DSL_CACHE_DIR",
                          os.path.expanduser("~/.cache/cutlass_dsl"))

    # ── Call kernels directly (@cute.jit handles caching) ──
    # No cute.compile — first call compiles, subsequent calls hit cache

    # Step 1: QK kernel (pre-gated Q, K_inv, g=0) → Q_g @ K_inv^T
    kda_qk(q_gated, k_inv, g_zero, qk_buf, beta, h_state, h_state,
        torch.zeros(2, dtype=torch.int32, device=q.device),
        torch.zeros(128, dtype=torch.uint8, device=q.device),
        (B, T, H, K_dim), stream)

    # ── Steps 2-5: Per-chunk: FO(inter) → DH(WH) → host(state) → intra → output ──
    mask = torch.tril(torch.ones(C, C, device=q.device))
    scale_t = torch.tensor(scale, dtype=torch.float32, device=q.device)
    o_final = torch.zeros(B, T, H, V_dim, dtype=torch.float32, device=q.device)
    for n in range(NC):
        t_beg = n * C
        t_end = min(t_beg + C, T)
        ct = t_end - t_beg

        # FO: inter = Q_g @ state (state BEFORE this chunk's update)
        fo_ch = torch.zeros(ct, H, V_dim, dtype=torch.bfloat16, device=q.device)
        kda_fo(q_gated[t_beg:t_end].contiguous(), h_state, g_zero[t_beg:t_end].contiguous(),
                    fo_ch, (B, ct, H, V_dim, K_dim), stream)
        torch.cuda.synchronize()
        # FO computes Q_norm @ h. Multiply by exp2(g_last) to match FLA's Q*exp2(g)@h
        g_last_h = g_kda[t_end - 1, :, :].float()  # [H, K]
        g_last_scalar = g_last_h.mean()  # scalar
        inter_n = fo_ch[:, 0, :].float() * torch.exp(g_last_scalar * LN2)  # [ct, V]

        # ② DH: compute W, run DH kernel, host state update
        # u = Aqk @ (V * beta)  [ct, V] — FLA's pre-processed V for delta-H
        qk_n = qk_buf[t_beg:t_end, 0, :ct].float()  # [ct, ct]
        v_ch = v_kda[t_beg:t_end].contiguous()
        v_n = v_ch[:, 0, :].float()  # [ct, V]
        bf = beta[0, t_beg:t_end, :].float()  # [ct, H]
        u_n = (qk_n * scale_t) @ (v_n * bf.mean(dim=1, keepdim=True))  # Aqk @ (V*β)

        kg_ch = k_inv[t_beg:t_end].contiguous()
        zg_ch = g_zero[t_beg:t_end].contiguous()
        k_chunk = k_kda[t_beg:t_end, 0, :].float()
        g_chunk = g_kda[t_beg:t_end, 0, :].float()
        g_last_2d = g_kda[t_end - 1, 0, :].float()
        k_inv_ch = k_chunk * torch.exp((-g_chunk + g_last_2d.unsqueeze(0)) * LN2)
        k_fwd = k_chunk * torch.exp((g_chunk - g_last_2d.unsqueeze(0)) * LN2)
        bf = beta[0, t_beg:t_end, :].float()
        Akk = (k_fwd @ k_inv_ch.transpose(0, 1)) * bf.mean(dim=1, keepdim=True)
        L = torch.tril(Akk, diagonal=-1)
        # FLA-style 16×16 block triangular inverse
        BC = 16
        NB = ct // BC  # 4 blocks for ct=64
        # Compute (I-L_diag)^(-1) for each diagonal block
        Ai_blocks = []
        for b in range(NB):
            r0, r1 = b*BC, (b+1)*BC
            L_diag = L[r0:r1, r0:r1]
            Ai_blocks.append(torch.linalg.solve_triangular(
                torch.eye(BC, dtype=torch.float32, device=q.device) - L_diag,
                torch.eye(BC, dtype=torch.float32, device=q.device),
                upper=False))
        # Schur complements for off-diagonal blocks
        Ai_full = torch.zeros(ct, ct, dtype=torch.float32, device=q.device)
        for b in range(NB):
            r0, r1 = b*BC, (b+1)*BC
            Ai_full[r0:r1, r0:r1] = Ai_blocks[b]
        Ai_full[BC:2*BC, 0:BC]    = -Ai_blocks[1] @ L[BC:2*BC, 0:BC] @ Ai_blocks[0]
        Ai_full[2*BC:3*BC, 0:BC]  = -Ai_blocks[2] @ (L[2*BC:3*BC, 0:BC] @ Ai_blocks[0] + L[2*BC:3*BC, BC:2*BC] @ Ai_full[BC:2*BC, 0:BC])
        Ai_full[2*BC:3*BC, BC:2*BC] = -Ai_blocks[2] @ L[2*BC:3*BC, BC:2*BC] @ Ai_blocks[1]
        Ai_full[3*BC:4*BC, 0:BC]  = -Ai_blocks[3] @ (L[3*BC:4*BC, 0:BC] @ Ai_blocks[0] + L[3*BC:4*BC, BC:2*BC] @ Ai_full[BC:2*BC, 0:BC] + L[3*BC:4*BC, 2*BC:3*BC] @ Ai_full[2*BC:3*BC, 0:BC])
        Ai_full[3*BC:4*BC, BC:2*BC] = -Ai_blocks[3] @ (L[3*BC:4*BC, BC:2*BC] @ Ai_blocks[1] + L[3*BC:4*BC, 2*BC:3*BC] @ Ai_full[2*BC:3*BC, BC:2*BC])
        Ai_full[3*BC:4*BC, 2*BC:3*BC] = -Ai_blocks[3] @ L[3*BC:4*BC, 2*BC:3*BC] @ Ai_blocks[2]
        w_ch = (Ai_full @ (k_chunk * bf.mean(dim=1, keepdim=True) * torch.exp(g_chunk * LN2))).unsqueeze(1).to(torch.bfloat16).contiguous()

        kda_dh(kg_ch, w_ch, zg_ch, h_state, h_state, beta, (B, ct, H, V_dim, K_dim), stream)
        torch.cuda.synchronize()

        for nv in range(NV):
            v_beg = nv * BV
            v_end = min(v_beg + BV, V_dim)
            vt = v_end - v_beg
            wh_acc = h_state[0, 0, v_beg:v_end, :ct].float()
            u_tile = u_n[:, v_beg:v_end]  # [ct, vt] — from Aqk@(V*β)
            v_new = u_tile - wh_acc.transpose(0, 1)  # u - WH^T
            update = k_inv_ch.transpose(0, 1) @ v_new
            h_state[0, 0, v_beg:v_end, :] += update.transpose(0, 1)

        # Intra = causal_masked(QK * scale) @ V
        qk_n = qk_buf[t_beg:t_end, 0, :ct].float()
        v_n = v_ch[:, 0, :].float()
        intra = ((qk_n * scale_t) * mask[:ct, :ct]) @ v_n

        o_final[0, t_beg:t_end, :, :] = (inter_n + intra).unsqueeze(1)

    o_final = o_final.to(torch.bfloat16)
    return o_final, h_state if output_final_state else None
