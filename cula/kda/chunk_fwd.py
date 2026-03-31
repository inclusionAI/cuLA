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

import importlib

import torch

# from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
)
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

from cula.kda.chunk_intra import chunk_kda_fwd_intra
from cula.utils import assert_blackwell

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_delta_h_mod = importlib.import_module("cula.ops.chunk_delta_h")
chunk_gated_delta_rule_fwd_h = _delta_h_mod.chunk_gated_delta_rule_fwd_h
_fwd_o_mod = importlib.import_module("cula.ops.fwd_o")
chunk_gla_fwd_o = _fwd_o_mod.chunk_gla_fwd_o


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.IntTensor | None = None,
    cu_seqlens_cpu: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: FLACPContext | None = None,
    use_tf32_inverse: bool = True,
    unified_gref: bool = False,  # Set True for ~5% extra perf (slightly lower precision)
):
    assert_blackwell(q.device)

    # Apply gate activation
    g_org = None
    if use_gate_in_kernel:
        g_org = g
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
    else:
        g = chunk_local_cumsum(g=g, scale=RCP_LN2, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    # qg = None if disable_recompute is False
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        use_tf32_inverse=use_tf32_inverse,
        unified_gref=unified_gref,
    )

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            use_exp2=True,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        # In Context Parallel (CP) mode, global initial states are not supported at the entry point.
        # The `initial_state` here is computed internally via inter-rank communication.
        # Since only the first sequence in the local batch can be a continuation of a cross-rank sequence,
        # only the first state in the tensor is relevant. We compress it to optimize memory for `save_for_backward`.
        initial_state = compress_h0(initial_state, context=cp_context)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
    chunk_gla_fwd_o(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        is_varlen=cu_seqlens is not None,
    )
    if disable_recompute is False:
        # Delete to save memory
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            # Only delete h if not requested for inference
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
