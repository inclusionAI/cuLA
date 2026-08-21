# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host launch adapter for the Hopper SM90a GDN2 prefill kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .config import (
    MAX_SEQUENCES,
    SM90_BACKEND_ID,
    SUPPORTED_Q_HEADS,
    SUPPORTED_V_HEADS,
)
from .prefill_kernel import GDN2PrefillKernel

if TYPE_CHECKING:
    from cula.gdn2.prefill import _GDN2Inputs

_compiled: dict[tuple[int, int, bool, bool, bool, bool], object] = {}


@dataclass(frozen=True)
class GDN2ExecutionInfo:
    """Metadata receipt for one product GDN2 launch."""

    backend_id: str
    total_tokens: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    has_initial_state: bool
    store_final_state: bool
    sequence_policy: str
    compile_cache_entries: int
    fallback: bool


def _device_key(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _dynamic_mode0(tensor: torch.Tensor):
    return from_dlpack(
        tensor,
        assumed_align=16,
    ).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tensor.dim_order(),
    )


def _resolve_support(inputs: _GDN2Inputs) -> None:
    if inputs.num_q_heads != SUPPORTED_Q_HEADS:
        raise NotImplementedError(
            f"GDN2 SM90a prefill requires Hq={SUPPORTED_Q_HEADS}",
        )
    if inputs.num_v_heads not in SUPPORTED_V_HEADS:
        raise NotImplementedError(
            f"GDN2 SM90a prefill requires Hv in {SUPPORTED_V_HEADS}",
        )
    if not 1 <= inputs.num_sequences <= MAX_SEQUENCES:
        raise NotImplementedError(
            f"GDN2 SM90a prefill requires 1 <= N <= {MAX_SEQUENCES}",
        )


def _compile(
    inputs: _GDN2Inputs,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    stream: cuda.CUstream,
):
    has_initial_state = inputs.initial_state is not None
    store_final_state = inputs.output_final_state
    use_n1_hv16_v64 = (
        inputs.num_sequences == 1
        and inputs.num_v_heads == 16
        and has_initial_state
        and store_final_state
        and inputs.total_tokens > 64
    )
    retain_final_tail = store_final_state and not (inputs.num_sequences == 1 and inputs.total_tokens <= 64)
    key = (
        _device_key(inputs.q.device),
        inputs.num_v_heads,
        has_initial_state,
        store_final_state,
        use_n1_hv16_v64,
        retain_final_tail,
    )
    compiled = _compiled.get(key)
    if compiled is not None:
        return compiled

    kernel = GDN2PrefillKernel(
        has_initial_state=has_initial_state,
        store_final_state=store_final_state,
        value_tile=64 if use_n1_hv16_v64 else 128,
        single_state_owner=use_n1_hv16_v64,
        retain_final_tail=retain_final_tail,
    )
    compiled = cute.compile(
        kernel,
        *(
            _dynamic_mode0(tensor)
            for tensor in (
                inputs.q,
                inputs.k,
                inputs.v,
                inputs.b,
                inputs.w,
                inputs.cu_seqlens,
                inputs.g,
                inputs.q,
                inputs.q,
                initial_state,
                inputs.output,
                final_state,
            )
        ),
        cutlass.Int32(inputs.num_sequences),
        cutlass.Int32(inputs.num_q_heads),
        cutlass.Int32(inputs.num_v_heads),
        cutlass.Int32(inputs.total_tokens),
        cutlass.Float32(inputs.scale),
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled


def launch_sm90_gdn2(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> GDN2ExecutionInfo | None:
    """Launch the product GDN2 backend without a fallback."""

    _resolve_support(inputs)
    initial_state = inputs.initial_state if inputs.initial_state is not None else inputs.q
    final_state = inputs.output_state if inputs.output_state is not None else inputs.output

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            initial_state,
            final_state,
            stream,
        )
        compiled(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.b,
            inputs.w,
            inputs.cu_seqlens,
            inputs.g,
            inputs.q,
            inputs.q,
            initial_state,
            inputs.output,
            final_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return GDN2ExecutionInfo(
        backend_id=SM90_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        has_initial_state=inputs.initial_state is not None,
        store_final_state=inputs.output_final_state,
        sequence_policy="stable_lpt32",
        compile_cache_entries=len(_compiled),
        fallback=False,
    )
