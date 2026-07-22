# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public packed-varlen GDN prefill API for NVIDIA Hopper SM90."""

from __future__ import annotations

import functools
import importlib.metadata
import math
from dataclasses import dataclass

import torch

from cula.ops.gdn.sm90.config import (
    CHUNK_SIZE,
    EXPECTED_CUTLASS_DSL_VERSION,
    HEAD_SIZE,
    SM90_BACKEND_ID,
    classify_head_mode,
)

__all__ = [
    "chunk_gated_delta_rule",
    "get_sm90_gdn_prefill_backend",
    "get_sm90_gdn_prefill_backend_identity",
    "is_sm90_gdn_prefill_available",
]

_INT32_MAX = 2**31 - 1
_TMA_ALIGNMENT = 16


@dataclass(frozen=True)
class _GDNPrefillInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    output: torch.Tensor
    initial_state: torch.Tensor | None
    output_state: torch.Tensor | None
    cu_seqlens: torch.Tensor
    num_seqs: int
    scale: float


@functools.cache
def _installed_cutlass_dsl_version() -> str | None:
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return None


def get_sm90_gdn_prefill_backend() -> str:
    """Return the production backend selected for SM90 GDN prefill.

    Returns:
        The backend name ``"dsl"``.
    """

    return "dsl"


def get_sm90_gdn_prefill_backend_identity() -> str:
    """Return the implementation identity used by the SM90 dispatch.

    Returns:
        A stable identifier for the SM90 CuTe DSL implementation.
    """

    return SM90_BACKEND_ID


def is_sm90_gdn_prefill_available(
    device: torch.device | int | str | None = None,
) -> bool:
    """Check whether SM90 GDN prefill is available on a device.

    Args:
        device: CUDA device to query. ``None`` selects the current CUDA device.

    Returns:
        ``True`` when the device has compute capability 9.0 and the required
        CuTe DSL version is installed; otherwise ``False``.
    """

    if device is not None and not isinstance(device, int):
        device = torch.device(device)
        if device.type != "cuda":
            return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        return False
    return _installed_cutlass_dsl_version() == EXPECTED_CUTLASS_DSL_VERSION


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    ndim: int,
    dtype: torch.dtype | tuple[torch.dtype, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got shape {tuple(tensor.shape)}")
    valid_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in valid_dtypes:
        expected = ", ".join(str(item) for item in valid_dtypes)
        raise TypeError(f"{name} must use {expected}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % _TMA_ALIGNMENT != 0:
        raise ValueError(f"{name} data pointer must be {_TMA_ALIGNMENT}-byte aligned")


def _check_cu_seqlens_metadata(
    cu_seqlens: torch.Tensor | None,
    *,
    device: torch.device,
) -> int:
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for packed-varlen GDN prefill")
    _check_tensor(
        "cu_seqlens",
        cu_seqlens,
        device=device,
        ndim=1,
        dtype=(torch.int32, torch.int64),
    )
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
    return cu_seqlens.numel() - 1


def _validate_device_contents(
    cu_seqlens: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    total_tokens: int,
) -> None:
    """Synchronously validate CUDA-resident values for explicit diagnostics."""

    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if offsets[0] != 0:
        raise ValueError(f"cu_seqlens[0] must be 0, got {offsets[0]}")
    if offsets[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens[-1] must equal total_tokens={total_tokens}, got {offsets[-1]}",
        )
    if offsets[-1] > _INT32_MAX:
        raise ValueError(f"packed token count must not exceed {_INT32_MAX}")

    seq_lens = tuple(end - start for start, end in zip(offsets, offsets[1:]))
    if any(length <= 0 for length in seq_lens):
        raise ValueError("zero-length or decreasing sequences are outside the SM90 GDN contract")
    if not bool(torch.isfinite(alpha).all()) or not bool((alpha > 0).all()):
        raise ValueError("g must contain finite, strictly positive forget factors")
    if not bool(torch.isfinite(beta).all()):
        raise ValueError("beta must contain finite update factors")


def _prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    output: torch.Tensor | None,
    output_state: torch.Tensor | None,
    state_checkpoints: torch.Tensor | None,
    checkpoint_cu_starts: torch.Tensor | None,
    checkpoint_every_n_tokens: int,
    use_qk_l2norm_in_kernel: bool,
    validate_inputs: bool,
) -> _GDNPrefillInputs:
    device = q.device if isinstance(q, torch.Tensor) else torch.device("cuda")
    _check_tensor("q", q, device=device, ndim=3, dtype=torch.bfloat16)
    _check_tensor("k", k, device=device, ndim=3, dtype=torch.bfloat16)
    _check_tensor("v", v, device=device, ndim=3, dtype=torch.bfloat16)

    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("SM90 GDN does not normalize Q/K inside the kernel")
    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint_every_n_tokens must be non-negative")
    if checkpoint_every_n_tokens % CHUNK_SIZE != 0:
        raise ValueError(f"checkpoint_every_n_tokens must be a multiple of {CHUNK_SIZE}")
    if checkpoint_every_n_tokens or state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise NotImplementedError("SM90 GDN state checkpoints are not implemented")

    total_tokens, num_q_heads, head_size = q.shape
    if not 0 < total_tokens <= _INT32_MAX:
        raise ValueError(f"packed token count must be in [1, {_INT32_MAX}]")
    if k.shape[0] != total_tokens or v.shape[0] != total_tokens:
        raise ValueError("q, k, and v must share the packed token dimension")
    if q.shape[2] != HEAD_SIZE or k.shape[2] != HEAD_SIZE or v.shape[2] != HEAD_SIZE:
        raise ValueError(f"q, k, and v head size must be {HEAD_SIZE}")
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    classify_head_mode(num_q_heads, num_k_heads, num_v_heads)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_seqs = _check_cu_seqlens_metadata(
        cu_seqlens,
        device=device,
    )

    gate_shape = (total_tokens, num_o_heads)
    if alpha is None:
        alpha = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _check_tensor("g", alpha, device=device, ndim=2, dtype=torch.float32)
        if tuple(alpha.shape) != gate_shape:
            raise ValueError(f"g must have shape {gate_shape}, got {tuple(alpha.shape)}")

    if beta is None:
        beta = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _check_tensor("beta", beta, device=device, ndim=2, dtype=torch.float32)
        if tuple(beta.shape) != gate_shape:
            raise ValueError(f"beta must have shape {gate_shape}, got {tuple(beta.shape)}")

    if not isinstance(validate_inputs, bool):
        raise TypeError(f"validate_inputs must be a bool, got {type(validate_inputs).__name__}")
    if validate_inputs:
        _validate_device_contents(
            cu_seqlens,
            alpha,
            beta,
            total_tokens=total_tokens,
        )

    output_shape = (total_tokens, num_o_heads, head_size)
    if output is None:
        output = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
    else:
        _check_tensor("output", output, device=device, ndim=3, dtype=torch.bfloat16)
        if tuple(output.shape) != output_shape:
            raise ValueError(f"output must have shape {output_shape}, got {tuple(output.shape)}")

    state_shape = (num_seqs, num_o_heads, head_size, head_size)
    if initial_state is not None:
        _check_tensor(
            "initial_state",
            initial_state,
            device=device,
            ndim=4,
            dtype=torch.float32,
        )
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(
                f"initial_state must have [sequence, output_head, V, K] shape {state_shape}, got {tuple(initial_state.shape)}",
            )

    if output_state is not None and not output_final_state:
        raise ValueError("output_state requires output_final_state=True")
    if output_final_state and output_state is None:
        output_state = torch.empty(state_shape, dtype=torch.float32, device=device)
    elif output_state is not None:
        _check_tensor(
            "output_state",
            output_state,
            device=device,
            ndim=4,
            dtype=torch.float32,
        )
        if tuple(output_state.shape) != state_shape:
            raise ValueError(
                f"output_state must have [sequence, output_head, V, K] shape {state_shape}, got {tuple(output_state.shape)}",
            )

    scale_value = 1.0 / math.sqrt(head_size) if scale is None or scale == 0.0 else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    return _GDNPrefillInputs(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        output=output,
        initial_state=initial_state,
        output_state=output_state,
        cu_seqlens=cu_seqlens,
        num_seqs=num_seqs,
        scale=scale_value,
    )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    validate_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run packed variable-length GDN prefill on Hopper SM90.

    Args:
        q: Contiguous BF16 queries with shape
            ``[total_tokens, num_q_heads, 128]``.
        k: Contiguous BF16 keys with shape
            ``[total_tokens, num_k_heads, 128]``.
        v: Contiguous BF16 values with shape
            ``[total_tokens, num_v_heads, 128]``.
        g: Optional FP32 forget factors with shape
            ``[total_tokens, num_output_heads]``. Values must be finite and
            strictly positive. ``None`` uses ones.
        beta: Optional FP32 update factors with shape
            ``[total_tokens, num_output_heads]``. Values must be finite.
            ``None`` uses ones.
        scale: Attention scale. ``None`` or ``0`` uses ``1 / sqrt(128)``.
        initial_state: Optional FP32 state with shape
            ``[num_sequences, num_output_heads, 128, 128]`` in public
            ``[V, K]`` orientation.
        output_final_state: Whether to return the final recurrent state.
        cu_seqlens: CUDA INT32 or INT64 cumulative sequence lengths with shape
            ``[num_sequences + 1]``. Every sequence must contain at least one
            token.
        use_qk_l2norm_in_kernel: Must be ``False``; in-kernel Q/K L2
            normalization is unsupported.
        output: Optional preallocated contiguous BF16 output with shape
            ``[total_tokens, num_output_heads, 128]``.
        output_state: Optional preallocated contiguous FP32 final-state buffer
            with shape ``[num_sequences, num_output_heads, 128, 128]``.
        state_checkpoints: Must be ``None``; intermediate state checkpoints are
            unsupported.
        checkpoint_cu_starts: Must be ``None``; intermediate state checkpoints
            are unsupported.
        checkpoint_every_n_tokens: Must be ``0``; intermediate state
            checkpoints are unsupported.
        validate_inputs: Whether to synchronously copy and validate
            CUDA-resident sequence offsets and gate values. Leave disabled on
            latency-sensitive paths.

    Returns:
        The BF16 output tensor, or ``(output, final_state)`` when
        ``output_final_state=True``.

    Raises:
        TypeError: If an input has an unsupported type or dtype.
        ValueError: If an input shape, layout, device, or value contract is
            invalid.
        NotImplementedError: If an unsupported normalization or checkpoint
            option is requested.
        RuntimeError: If the GPU or CuTe DSL runtime does not satisfy the SM90
            backend requirements.
    """

    inputs = _prepare_inputs(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        output,
        output_state,
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel,
        validate_inputs,
    )
    props = torch.cuda.get_device_properties(inputs.q.device)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError(
            f"SM90 GDN requires compute capability 9.0, got {props.major}.{props.minor}",
        )
    installed = _installed_cutlass_dsl_version()
    if installed != EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            f"SM90 GDN requires nvidia-cutlass-dsl=={EXPECTED_CUTLASS_DSL_VERSION}, found {installed}",
        )

    from cula.ops.gdn.sm90.launch import launch_sm90_gdn_prefill

    launch_sm90_gdn_prefill(inputs)
    if output_final_state:
        assert inputs.output_state is not None
        return inputs.output, inputs.output_state
    return inputs.output
