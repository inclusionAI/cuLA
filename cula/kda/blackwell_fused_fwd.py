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

from cula.ops.kda_fully_fused_sm100_wip import KDAChunkwise
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
        cache_key = (has_initial_state, output_final_state, safe_gate, is_varlen, scale, chunk_size, D, USE_FAST_MATH)

        if is_varlen:
            cu_seqlens_i32 = cu_seqlens.to(torch.int32).contiguous()
            cu_seqlens_cute = from_dlpack(cu_seqlens_i32.detach())
        else:
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

        if has_initial_state:
            initial_state_f32 = initial_state.to(torch.float32).contiguous()
            initial_state_cute = from_dlpack(initial_state_f32.detach())
        else:
            initial_state_f32 = None
            initial_state_cute = _dummy_cache[q.device]["state_cute"]

        if output_final_state:
            final_state_f32 = torch.zeros(num_seqs, H, D, D, dtype=torch.float32, device=q.device)
            final_state_cute = from_dlpack(final_state_f32.detach())
        else:
            final_state_f32 = None
            final_state_cute = _dummy_cache[q.device]["state_cute"]

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
    assert_blackwell()
    assert cu_seqlens is None or q.shape[0] == 1, "For varlen, batch size must be 1. Flatten sequences first."
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

    # Validate head dimensions for GVA
    B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got q={q.shape} vs k={k.shape}"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert beta.dtype == torch.bfloat16 or beta.dtype == torch.float32, "beta must be in bfloat16 or float32."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently we only support head dim of 128 for KDA"
    assert HV % H == 0, (
        f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by num_qk_heads (H={H}), but got HV % H = {HV % H}"
    )
    assert g.shape == (B, T, HV, K), f"g must have shape [B, T, HV, K]={[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape [B, T, HV]={[B, T, HV]}, got {list(beta.shape)}"

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
