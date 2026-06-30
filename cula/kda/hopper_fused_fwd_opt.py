# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimized Hopper KDA prefill: fused gate+l2norm preprocessing + intra-card CP."""

import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import cula.cudac as cula_cuda
from cula.kda.cp_context import intra_card_cp_preprocess, is_dominant_long_seq
from cula.kda.gate_l2norm_fused import gate_l2norm_fused_fwd
from cula.kda.l2norm_qk_fused import l2norm_fwd_qk
from cula.utils import _get_cache_buf, assert_hopper, get_device_sm_count, prepare_uniform_cu_seqlens

FUSED_L2NORM_QK_TH_MAX = 10000

FUSED_GATE_L2NORM_TH_FIXED = 16384

FUSED_GATE_L2NORM_TH_VARLEN = 65536

FUSED_GATE_L2NORM_VARLEN_AVG_SEQ = 256


def _fused_gate_l2norm_threshold(cu_seqlens_is_none):
    return FUSED_GATE_L2NORM_TH_FIXED if cu_seqlens_is_none else FUSED_GATE_L2NORM_TH_VARLEN


FUSED_GATE_L2NORM_TH_MAX = FUSED_GATE_L2NORM_TH_VARLEN


def _inference_forward(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    safe_gate,
    lower_bound,
    cu_seqlens,
    chunk_indices,
    auto_cp,
    cu_seqlens_cpu=None,
):
    chunk_size = 64
    batch_size, seq_len, num_qk_heads, head_dim = q.shape
    num_v_heads = v.shape[-2]

    cu_seqlens_is_none = cu_seqlens is None
    if cu_seqlens_is_none:
        cu_seqlens = prepare_uniform_cu_seqlens(batch_size, seq_len, q.device, torch.int32)
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g, beta))

    if cu_seqlens_is_none:
        avg_seq_ok = True
    else:
        N = cu_seqlens.numel() - 1
        packed_T = q.shape[1]
        avg_seq_ok = N <= 1 or packed_T <= N * FUSED_GATE_L2NORM_VARLEN_AVG_SEQ

    fused_all_pre = (
        use_gate_in_kernel
        and use_qk_l2norm_in_kernel
        and (q.numel() // q.shape[-1]) <= _fused_gate_l2norm_threshold(cu_seqlens_is_none)
        and num_qk_heads == num_v_heads
        and avg_seq_ok
    )

    if fused_all_pre:
        if chunk_indices is None and not cu_seqlens_is_none:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        g_out, yq, yk, _, _ = gate_l2norm_fused_fwd(
            g=g,
            q=q,
            k=k,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound if safe_gate else None,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens if not cu_seqlens_is_none else None,
            chunk_indices=chunk_indices if not cu_seqlens_is_none else None,
        )
        g, q, k = g_out, yq, yk
    else:
        if chunk_indices is None and not cu_seqlens_is_none:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        else:
            g = chunk_local_cumsum(
                g=g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        if use_qk_l2norm_in_kernel:
            D = q.shape[-1]
            n_rows = q.numel() // D
            if n_rows <= FUSED_L2NORM_QK_TH_MAX:
                q_flat = q.view(-1, D)
                k_flat = k.view(-1, D)
                yq_flat, yk_flat, _, _ = l2norm_fwd_qk(q_flat, k_flat)
                q = yq_flat.view_as(q)
                k = yk_flat.view_as(k)
            else:
                q, _ = l2norm_fwd(q)
                k, _ = l2norm_fwd(k)

    packed_seq = batch_size * seq_len
    q = q.reshape(packed_seq, num_qk_heads, head_dim).contiguous()
    k = k.reshape(packed_seq, num_qk_heads, head_dim).contiguous()
    v = v.reshape(packed_seq, num_v_heads, head_dim).contiguous()
    g = g.reshape(packed_seq, num_v_heads, head_dim).contiguous()
    beta = beta.reshape(packed_seq, num_v_heads).contiguous()

    cp_seq_map = None
    raw_cu_seqlens_for_cp = None

    def _dominant_long_seq_gate() -> bool:
        if cu_seqlens is None:
            return False
        N_ = cu_seqlens.numel() - 1
        if N_ <= 1 or N_ * num_v_heads <= 16:
            return False  # falls under the main branch
        if num_v_heads > 16 or packed_seq < 8192:
            return False
        cu_src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        cu_list = cu_src.tolist()
        seqlens = [cu_list[i + 1] - cu_list[i] for i in range(N_)]
        return is_dominant_long_seq(seqlens, num_v_heads)

    cp_would_fire = auto_cp and (
        # Multi-seq varlen: CP only when grid is starved AND total T amortizes.
        (
            cu_seqlens is not None
            and (cu_seqlens.numel() - 1) > 1
            and (cu_seqlens.numel() - 1) * num_v_heads <= 16
            and packed_seq >= 8192
        )
        # Multi-seq dominant-seq exception.
        or _dominant_long_seq_gate()
        or
        # Single sequence: per-H T thresholds matching _calc_cp_seqs.
        (
            (cu_seqlens is None or cu_seqlens.numel() - 1 == 1)
            and (
                (num_v_heads <= 8 and packed_seq >= 4096)
                or (num_v_heads <= 16 and packed_seq >= 4096)
                or (num_v_heads <= 32 and packed_seq >= 16384)
            )
        )
    )
    if cp_would_fire:
        q4 = q.view(1, packed_seq, num_qk_heads, head_dim)
        k4 = k.view(1, packed_seq, num_qk_heads, head_dim)
        v4 = v.view(1, packed_seq, num_v_heads, head_dim)
        g4 = g.view(1, packed_seq, num_v_heads, head_dim)
        beta4 = beta.view(1, packed_seq, num_v_heads)
        cp_h0, cp_cu_seqlens, cp_seq_map, raw_cu_seqlens_for_cp = intra_card_cp_preprocess(
            q=q4,
            k=k4,
            v=v4,
            g=g4,
            beta=beta4,
            scale=scale,
            raw_h0=initial_state,
            raw_cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            raw_cu_seqlens_cpu=cu_seqlens_cpu,
        )
        del q4, k4, v4, g4, beta4
        if cp_seq_map is not None:
            cu_seqlens = cp_cu_seqlens
            initial_state = cp_h0

    sm_count = get_device_sm_count(q.device)
    workspace_buffer = _get_cache_buf("hopper_kda_fwd_workspace", sm_count * 128, q.device)

    o, final_state = cula_cuda.kda_fwd_prefill(
        None,
        None,
        q,
        k,
        v,
        initial_state,
        g,
        beta,
        cu_seqlens,
        workspace_buffer,
        scale,
        output_final_state,
        safe_gate,
        cp_seq_map_=cp_seq_map,
        raw_cu_seqlens_=raw_cu_seqlens_for_cp,
    )
    o = rearrange(o, "(b t) h d -> b t h d", b=batch_size)
    return o.to(q.dtype), final_state


class HopperChunkKDAFunctionOpt(torch.autograd.Function):
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
        auto_cp: bool = True,
        cu_seqlens_cpu: torch.IntTensor | None = None,
    ):
        return _inference_forward(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            safe_gate,
            lower_bound,
            cu_seqlens,
            chunk_indices,
            auto_cp,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill_opt(
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
    auto_cp: bool = True,
    cu_seqlens_cpu: torch.IntTensor | None = None,
    **kwargs,
):
    assert_hopper()
    assert safe_gate, "Only support safe_gate=True."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if safe_gate:
            if lower_bound is None:
                raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
            if not (-5 <= lower_bound < 0):
                raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape, "q and k must have the same shape."
    assert q.shape[:2] == v.shape[:2] == g.shape[:2], "q, k, v, g must share batch and sequence dimensions."
    batch_size, seq_len, num_qk_heads, head_dim = q.shape
    num_v_heads = v.shape[-2]
    assert num_qk_heads > 0 and num_v_heads > 0
    assert num_v_heads % num_qk_heads == 0
    assert g.shape == (batch_size, seq_len, num_v_heads, head_dim)
    assert v.shape == (batch_size, seq_len, num_v_heads, head_dim)
    assert beta.shape == (batch_size, seq_len, num_v_heads)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert beta.dtype == torch.bfloat16 or beta.dtype == torch.float32, "beta must be in bfloat16 or float32."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently we only support head dim of 128 for KDA"
    if scale is None:
        scale = k.shape[-1] ** -0.5

    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, g, beta) if t is not None)
    if not needs_grad:
        o, final_state = _inference_forward(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            safe_gate,
            lower_bound,
            cu_seqlens,
            chunk_indices,
            auto_cp,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        return o, (final_state if output_final_state else None)

    o, final_state = HopperChunkKDAFunctionOpt.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        safe_gate,
        lower_bound,
        cu_seqlens,
        chunk_indices,
        auto_cp,
        cu_seqlens_cpu,
    )

    return o, (final_state if output_final_state else None)
