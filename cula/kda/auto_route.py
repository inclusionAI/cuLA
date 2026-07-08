from __future__ import annotations

import torch

from cula.kda.cp_context import is_dominant_long_seq
from cula.kda.hopper_fused_fwd import cula_kda_prefill as _basic
from cula.kda.hopper_fused_fwd_opt import FUSED_GATE_L2NORM_TH_VARLEN
from cula.kda.hopper_fused_fwd_opt import cula_kda_prefill_opt as _opt


def _should_use_opt(q: torch.Tensor, cu_seqlens: torch.Tensor | None) -> bool:
    """Pick opt vs basic based on H100 measurements."""
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]

    if cu_seqlens is not None:
        N = cu_seqlens.numel() - 1
        if N > 1:
            packed_T = q.shape[1]
            if packed_T * H <= FUSED_GATE_L2NORM_TH_VARLEN:
                return True
            if N * H <= 16 and T >= 8192:
                return True
            if H <= 16 and T >= 32768 + N - 1:
                cu_list = cu_seqlens.tolist()
                seqlens = [cu_list[i + 1] - cu_list[i] for i in range(N)]
                if is_dominant_long_seq(seqlens, H):
                    return True
            return False
        # N == 1 falls through to the single-sequence logic below.

    # Fused gate+l2norm reliably wins at very small T*H even with B>1.
    if T * H <= 6000:
        return True

    if B == 1:
        # T=1024 H=8/16 gets a small win from fused l2norm_qk (T*H<10000).
        if H <= 16 and T <= 1024:
            return True
        if H <= 8:
            return T >= 4096  # CP kicks in
        elif H <= 16:
            return T >= 4096
        elif H <= 32:
            return T >= 16384
        else:  # H >= 64
            return False  # base ties or wins

    if B == 2:
        if H == 8:
            return T >= 4096
        return False  # B=2 H>=16 mostly ties

    # B >= 4 : B*H >= 32 already saturates a sizable fraction of SMs, CP
    # buys little; basic and opt tie. Default to basic (cheaper wrapper).
    return False


def cula_kda_prefill_auto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    if _should_use_opt(q, cu_seqlens):
        return _opt(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            auto_cp=True,
            **kwargs,
        )
    return _basic(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **kwargs,
    )
