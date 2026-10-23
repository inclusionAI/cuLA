# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single-dispatch host wrapper for the bitwise-aligned SM100 KDA port."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32

from cula.ops.kda.sm100.akk_inv_fp32 import akk_inv_fp32_physical_host
from cula.ops.kda.sm100.intra_fused import (
    BT,
    K_DIM,
    _acquire_csrc_port_outputs,
    _export_csrc_port_outputs,
    _get_csrc_port_dummy_alog,
    _get_csrc_port_dummy_bias,
    _get_csrc_port_dummy_outputs,
    make_host_function,
)
from cula.ops.kda.sm100.recompute_wu import (
    KDARecomputeWU,
    _acquire_output_buffers,
    _export_output_buffers,
)
from cula.utils import USE_FAST_MATH

_FULL_PORT_CACHE = {}
_FULL_METADATA_CACHE = {}


def _get_full_metadata(device):
    key = device.index if device.index is not None else 0
    tensors = _FULL_METADATA_CACHE.get(key)
    if tensors is None:
        tensors = (
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([[0, 0]], dtype=torch.int32, device=device),
        )
        _FULL_METADATA_CACHE[key] = tensors
    return tensors


def _make_full_host(batch, length, h_qk, h_v, beta_dtype, scale):
    chunks = length // BT
    intra_host = make_host_function(
        batch,
        chunks,
        h_qk,
        h_v,
        is_varlen=False,
        fp32_akk_workspace=True,
        input_gk_fp32=True,
        beta_fp32=beta_dtype == torch.float32,
    )
    wu = KDARecomputeWU(
        K=K_DIM,
        V=K_DIM,
        chunk_size=BT,
        beta_dtype=cutlass.Float32 if beta_dtype == torch.float32 else cutlass.BFloat16,
        use_fast_math=USE_FAST_MATH,
    )
    problem_size = (Int32(batch), Int32(length), Int32(h_qk), Int32(h_v), Int32(K_DIM), Int32(K_DIM))
    total_nt = Int32(batch * chunks)

    @cute.jit
    def full_host(
        q,
        k,
        v,
        gk,
        beta,
        Aqk,
        Akk_workspace,
        Akk,
        w,
        u,
        kg,
        dummy_alog,
        dummy_bf16,
        dummy_fp32,
        cu_seqlens,
        chunk_indices,
        dummy_bias,
        stream,
    ):
        intra_host(
            q,
            k,
            gk,
            dummy_alog,
            beta,
            cutlass.Float32(scale),
            dummy_bf16,
            dummy_bf16,
            dummy_bf16,
            dummy_fp32,
            Aqk,
            Akk_workspace,
            cu_seqlens,
            chunk_indices,
            dummy_bias,
            cutlass.Float32(0.0),
            stream,
        )
        akk_inv_fp32_physical_host(
            Akk_workspace,
            Akk,
            batch,
            chunks,
            h_v,
            cu_seqlens,
            chunk_indices,
            0,
            length,
            stream,
        )
        wu(
            k,
            v,
            beta,
            Akk,
            gk,
            w,
            u,
            kg,
            cu_seqlens,
            chunk_indices,
            problem_size,
            total_nt,
            stream,
        )

    return full_host


def _compile_full_port(device, batch, length, h_qk, h_v, beta_dtype, scale):
    key = (device.index, batch, length, h_qk, h_v, beta_dtype, float(scale), USE_FAST_MATH)
    compiled = _FULL_PORT_CACHE.get(key)
    if compiled is not None:
        return compiled

    bf16_4d_qk = make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, length, h_qk, K_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    bf16_4d_v = make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, length, h_v, K_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    fp32_gk = make_fake_compact_tensor(
        cutlass.Float32,
        (batch, length, h_v, K_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    beta_fake = make_fake_compact_tensor(
        cutlass.Float32 if beta_dtype == torch.float32 else cutlass.BFloat16,
        (batch, length, h_v),
        stride_order=(2, 1, 0),
        assumed_align=128,
    )
    bf16_A = make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, length, h_v, BT),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    fp32_A = make_fake_compact_tensor(
        cutlass.Float32,
        (batch, length, h_v, BT),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    alog_fake = make_fake_compact_tensor(cutlass.Float32, (h_v,), assumed_align=128)
    dummy_bf16_fake = make_fake_compact_tensor(cutlass.BFloat16, (1,), assumed_align=16)
    dummy_fp32_fake = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=16)
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=16)
    ci_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (1, 2),
        stride_order=(1, 0),
        assumed_align=16,
    )
    bias_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (h_v, K_DIM),
        stride_order=(1, 0),
        assumed_align=128,
    )
    full_host = _make_full_host(batch, length, h_qk, h_v, beta_dtype, scale)
    compiled = cute.compile(
        full_host,
        bf16_4d_qk,
        bf16_4d_qk,
        bf16_4d_v,
        fp32_gk,
        beta_fake,
        bf16_A,
        fp32_A,
        bf16_A,
        bf16_4d_v,
        bf16_4d_v,
        bf16_4d_v,
        alog_fake,
        dummy_bf16_fake,
        dummy_fp32_fake,
        cu_fake,
        ci_fake,
        bias_fake,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _FULL_PORT_CACHE[key] = compiled
    return compiled


def chunk_kda_fwd_full_sm100_from_gk(q, k, v, gk, beta, scale):
    """Run exact intra, inverse, and WU through one host dispatch."""
    batch, length, h_qk, dim = q.shape
    h_v = v.shape[2]
    if dim != K_DIM or v.shape[-1] != K_DIM:
        raise NotImplementedError("the SM100 full port requires K=V=128")
    if length % (4 * BT):
        raise NotImplementedError("the full-port fast path requires T to be divisible by 256")

    intra_entry, (Aqk, Akk, Akk_workspace) = _acquire_csrc_port_outputs(q.device, batch, length, h_v)
    wu_entry, (w, u, kg) = _acquire_output_buffers(
        q.device,
        (
            ((batch, length, h_v, K_DIM), q.dtype),
            (tuple(v.shape), v.dtype),
            ((batch, length, h_v, K_DIM), q.dtype),
        ),
    )
    dummy_alog = _get_csrc_port_dummy_alog(q.device, h_v)
    dummy_bf16, dummy_fp32 = _get_csrc_port_dummy_outputs(q.device)
    cu_seqlens, chunk_indices = _get_full_metadata(q.device)
    dummy_bias = _get_csrc_port_dummy_bias(q.device, h_v)
    compiled = _compile_full_port(q.device, batch, length, h_qk, h_v, beta.dtype, scale)
    try:
        compiled(
            q,
            k,
            v,
            gk,
            beta,
            Aqk,
            Akk_workspace,
            Akk,
            w,
            u,
            kg,
            dummy_alog,
            dummy_bf16,
            dummy_fp32,
            cu_seqlens,
            chunk_indices,
            dummy_bias,
        )
    except Exception:
        if intra_entry is not None:
            intra_entry.release_after_error()
        if wu_entry is not None:
            wu_entry.release_after_error()
        raise

    Aqk, Akk = _export_csrc_port_outputs(intra_entry, Aqk, Akk)
    w, u, kg = _export_output_buffers(wu_entry, (w, u, kg))
    return w, u, None, kg, Aqk, Akk
