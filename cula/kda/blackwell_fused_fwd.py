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

import pathlib
import sys
import warnings

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from fla.modules.l2norm import l2norm_fwd

# from fla.ops.kda.chunk_inter import chunk_kda_bwd_dqkwg
from fla.ops.kda.gate import kda_gate_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from cula.ops.kda_fully_fused_wip import KDAChunkwise
from cula.utils import USE_FAST_MATH, assert_blackwell

# Global kernel cache
compiled_kernel_cache = {}
COMPILE_OPTIONS = "--generate-line-info --ptxas-options '--verbose'"

# Cached dummy tensors to avoid per-call allocation overhead (~0.12ms)
# Key: device -> {cu_seqlens, state_dummy, cu_seqlens_cute, state_cute}
_dummy_cache = {}


class ChunkKDAFunction(torch.autograd.Function):
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

        global compiled_kernel_cache

        B, S, H, D = q.shape
        is_varlen = cu_seqlens is not None
        if is_varlen:
            assert B == 1, "For varlen, batch size must be 1. Flatten variable-length inputs first."
            num_seqs = cu_seqlens.shape[0] - 1
        else:
            num_seqs = B

        # No output padding needed — the kernel handles tail tiles via
        # TMA descriptor modification (like flashkda), preventing writes
        # into the next sequence's output region.

        g_org = None
        if use_gate_in_kernel:
            try:
                from fla.ops.kda.gate import kda_gate_chunk_cumsum

                g_org = g
                if safe_gate:
                    assert lower_bound is not None, "lower_bound must be set when use safe_gate"
                g = kda_gate_chunk_cumsum(
                    g=g_org,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=RCP_LN2,
                    chunk_size=chunk_size,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    lower_bound=lower_bound,
                )
            except ImportError:
                warnings.warn("Can't use safe gate due to older FLA version, worse numerical issues.")
                g_org = g
                g = kda_gate_fwd(
                    g=g_org,
                    A_log=A_log,
                    dt_bias=dt_bias,
                )
        # only in safe_gate && use_gate_in_kernel, cumsum is fused into kda_gate_chunk_cumsum
        if not (safe_gate and use_gate_in_kernel):
            g = chunk_local_cumsum(
                g=g, chunk_size=chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
            )
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        q_cute = from_dlpack(q.detach())
        k_cute = from_dlpack(k.detach())
        v_cute = from_dlpack(v.detach())
        g_cute = from_dlpack(g.detach())
        beta_cute = from_dlpack(beta.detach())

        o = torch.empty_like(q)
        o_cute = from_dlpack(o.detach())

        stream = cutlass_torch.default_stream()

        has_initial_state = initial_state is not None
        g_dtype = g.dtype  # must be in cache_key: g_dtype affects SMEM layout and TMA atom
        cache_key = (has_initial_state, output_final_state, safe_gate, is_varlen, scale, chunk_size, D, USE_FAST_MATH, g_dtype)

        # Prepare cu_seqlens as int32 for kernel
        if is_varlen:
            cu_seqlens_i32 = cu_seqlens.to(torch.int32).contiguous()
            cu_seqlens_cute = from_dlpack(cu_seqlens_i32.detach())
        else:
            # Use cached dummy cu_seqlens to avoid per-call allocation overhead
            dev = q.device
            if dev not in _dummy_cache:
                _dummy_cu = torch.zeros(2, dtype=torch.int32, device=dev)
                _dummy_st = torch.empty(1, dtype=torch.float32, device=dev)
                _dummy_cache[dev] = {
                    "cu_seqlens": _dummy_cu,
                    "cu_seqlens_cute": from_dlpack(_dummy_cu.detach()),
                    "state_dummy": _dummy_st,
                    "state_cute": from_dlpack(_dummy_st.detach()),
                }
            dc = _dummy_cache[dev]
            cu_seqlens_i32 = dc["cu_seqlens"]
            cu_seqlens_cute = dc["cu_seqlens_cute"]

        # Workspace buffer for TMA descriptor modification (varlen tail tiles)
        # Same approach as flashkda: per-CTA slot for modified TMA descriptors
        # 128 bytes per TMA descriptor, indexed by bidx (sequence index)
        dev = q.device
        if dev not in _dummy_cache:
            _dummy_cu = torch.zeros(2, dtype=torch.int32, device=dev)
            _dummy_st = torch.empty(1, dtype=torch.float32, device=dev)
            _dummy_cache[dev] = {
                "cu_seqlens": _dummy_cu,
                "cu_seqlens_cute": from_dlpack(_dummy_cu.detach()),
                "state_dummy": _dummy_st,
                "state_cute": from_dlpack(_dummy_st.detach()),
            }
        dc = _dummy_cache[dev]
        if is_varlen:
            ws_size = num_seqs * 128
            # Allocate/reuse workspace (grow if needed)
            if "workspace" not in dc or dc["workspace"].numel() < ws_size:
                ws_buf = torch.zeros(ws_size, dtype=torch.uint8, device=dev)
                dc["workspace"] = ws_buf
                dc["workspace_cute"] = from_dlpack(ws_buf.detach())
            workspace_cute = dc["workspace_cute"]
        else:
            if "workspace" not in dc:
                ws_buf = torch.zeros(128, dtype=torch.uint8, device=dev)
                dc["workspace"] = ws_buf
                dc["workspace_cute"] = from_dlpack(ws_buf.detach())
            workspace_cute = dc["workspace_cute"]

        # State shape: [num_seqs, H, D, D]
        # Prepare initial_state and final_state tensors
        if has_initial_state:
            initial_state_f32 = initial_state.to(torch.float32).contiguous()
            initial_state_cute = from_dlpack(initial_state_f32.detach())
        else:
            # Use cached tiny dummy (pointer won't be dereferenced when has_initial_state=False)
            initial_state_f32 = None
            initial_state_cute = _dummy_cache[q.device]["state_cute"]

        if output_final_state:
            final_state_f32 = torch.zeros(num_seqs, H, D, D, dtype=torch.float32, device=q.device)
            final_state_cute = from_dlpack(final_state_f32.detach())
        else:
            # Use cached tiny dummy (pointer won't be dereferenced when output_final_state=False)
            final_state_f32 = None
            final_state_cute = _dummy_cache[q.device]["state_cute"]

        # problem_size: (num_seqs, total_tokens_or_seq_len, H, D)
        problem_size = (num_seqs, S, H, D)

        if cache_key in compiled_kernel_cache:
            compiled_kernel = compiled_kernel_cache[cache_key]
        else:
            attn_kernel = KDAChunkwise(
                chunk_size=chunk_size,
                qk_acc_dtype=cutlass.Float32,
                kv_acc_dtype=cutlass.Float32,
                io_dtype=cutlass.BFloat16,
                scale=scale,
                safe_gate=safe_gate,
                has_initial_state=has_initial_state,
                output_final_state=output_final_state,
                is_varlen=is_varlen,
                use_fast_math=USE_FAST_MATH,
            )
            compiled_kernel = cute.compile(
                attn_kernel,
                q_cute.iterator,
                k_cute.iterator,
                v_cute.iterator,
                g_cute.iterator,
                o_cute.iterator,
                beta_cute.iterator,
                initial_state_cute.iterator,
                final_state_cute.iterator,
                cu_seqlens_cute.iterator,
                workspace_cute.iterator,
                problem_size,
                stream,
                options=COMPILE_OPTIONS,
            )
            compiled_kernel_cache[cache_key] = compiled_kernel

        compiled_kernel(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            initial_state_cute.iterator,
            final_state_cute.iterator,
            cu_seqlens_cute.iterator,
            workspace_cute.iterator,
            problem_size,
            stream,
            options=COMPILE_OPTIONS,
        )

        if use_gate_in_kernel:
            g = None

        return o.to(q.dtype), final_state_f32 if output_final_state else None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        raise NotImplementedError("Backward pass is not implemented yet.")


# TODO: Blackwell fused prefill is still under development
@torch.compiler.disable
def flash_kda_prefill(
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
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If `True`:
              The passed `g` acts as the raw input for `-exp(A_log).view(H, -1) * softplus(g + dt_bias.view(H, K))`.
              Note that as part of the input arguments,
              `A_log` (shape `[H]`) and the optional `dt_bias` (shape `[H * K]`) should be provided.
            - If `False`, `g` is expected to be the pre-computed decay value.
            Default: `False`.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            When `True`, the kernel can use M=16 TensorCore acceleration.
            The safe range is approximately [-5, 0). Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function when `use_gate_in_kernel=True`.
            This parameter modifies the internal forget gate activation and is recommended
            to be set to `-5` when `safe_gate` is enabled. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        chunk_indices (torch.IntTensor):
            Chunk indices used for variable-length training,

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert_blackwell()
    # initial_state is now supported
    assert cu_seqlens is None or q.shape[0] == 1, "For varlen, batch size must be 1. Flatten sequences first."
    # assert output_final_state == False, "output_final_state=True is not supported in cutedsl_kda_prefill yet."
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
        # Non-aligned sequence lengths are handled natively by the kernel
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
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently we only support head dim of 128 for KDA"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkKDAFunction.apply(
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
