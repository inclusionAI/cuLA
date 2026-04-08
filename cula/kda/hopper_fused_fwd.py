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


import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import cula.cudac as cula_cuda
from cula.utils import _get_cache_buf, assert_hopper, get_device_sm_count, prepare_uniform_cu_seqlens


class HopperChunkKDAFunction(torch.autograd.Function):
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
        chunk_size = 64
        assert q.shape[-2] == v.shape[-2] == k.shape[-2], "Number of heads must be the same for q, k, v."

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is None:
            cu_seqlens = prepare_uniform_cu_seqlens(batch_size, seq_len, q.device, torch.int32)

        # set batch size to 1 after handling cu_seqlens
        if batch_size != 1:
            q, k, v, g, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g, beta))

        # gate preprocessing
        if use_gate_in_kernel:
            if safe_gate:
                assert lower_bound is not None, "lower_bound must be set when use safe_gate"
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

        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        # reshape to packed [T, H, K] for the C++ kernel
        packed_seq = batch_size * seq_len
        q = q.reshape(packed_seq, num_heads, head_dim).contiguous()
        k = k.reshape(packed_seq, num_heads, head_dim).contiguous()
        v = v.reshape(packed_seq, num_heads, head_dim).contiguous()
        g = g.reshape(packed_seq, num_heads, head_dim).contiguous()
        beta = beta.reshape(packed_seq, num_heads).contiguous()

        # workspace buffer for TMA Store O tensormap
        sm_count = get_device_sm_count(q.device)
        workspace_size = sm_count * 128
        workspace_buffer = _get_cache_buf("hopper_kda_fwd_workspace", workspace_size, q.device)

        # call the C++ kernel
        # Signature: kda_fwd_prefill(output_, output_state_, q, k, v, input_state_, alpha_, beta_, cu_seqlens, workspace, scale, safe_gate)
        o, final_state = cula_cuda.kda_fwd_prefill(
            None,  # output_ (auto-allocate)
            None,  # output_state_ (auto-allocate)
            q,
            k,
            v,
            initial_state,  # input_state_
            g,  # alpha_
            beta,  # beta_
            cu_seqlens,
            workspace_buffer,
            scale,
            safe_gate,
        )

        # reshape back
        o = rearrange(o, "(b t) h d -> b t h d", b=batch_size)

        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill(
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
    r"""
    Hopper (SM90) fully-fused KDA forward prefill using CUTLASS TMA warp-specialized kernel.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally. Default: `False`.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            When `True`, the kernel can use M=16 TensorCore acceleration.
            The safe range is approximately [-5, 0). Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]`, int32.
        chunk_indices (torch.IntTensor):
            Chunk indices for variable-length training.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert_hopper()
    assert safe_gate, "Only support safe_gate=True."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
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

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Only support head dim of 128 for inputs."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = HopperChunkKDAFunction.apply(
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
    )
    return o, final_state
