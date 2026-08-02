#!/usr/bin/env python3
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

import torch
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.qwen35_conv1d_prefill import qwen35_conv1d_prefill
from cula.ops.qwen35_fused_kda_prefill import has_qwen35_fused_kda_prefill, qwen35_fused_kda_prefill
from cula.ops.qwen35_layout_prefill import qwen35_layout_prefill, qwen35_layout_prefill_reference
from cula.ops.qwen35_scalar_kda_prefill import qwen35_scalar_kda_prefill
from cula.qwen35.common import Qwen35LinearAttentionConfig
from cula.qwen35.runtime import qwen35_linear_attention_prefill

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def _manual_scalar_prefill(q, k, v, a, b, A_log, dt_bias, initial_state=None, cu_seqlens=None):
    B, T, HV, K = q.shape
    state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    state = torch.zeros(state_count, HV, K, K, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        state = initial_state.float().clone()
    out = torch.empty_like(v)
    q_f = torch.nn.functional.normalize(q.float(), dim=-1) * (K**-0.5)
    k_f = torch.nn.functional.normalize(k.float(), dim=-1)

    def run_seq(batch_idx, state_idx, start, end):
        for t in range(start, end):
            for hv in range(HV):
                state_kv = state[state_idx, hv]
                decay = torch.exp(-torch.exp(A_log[hv].float()) * torch.nn.functional.softplus(a[batch_idx, t, hv].float() + dt_bias[hv].float()))
                beta = torch.sigmoid(b[batch_idx, t, hv].float())
                k_vec = k_f[batch_idx, t, hv]
                q_vec = q_f[batch_idx, t, hv]
                proj = decay * (state_kv.transpose(0, 1) @ k_vec)
                v_new = beta * (v[batch_idx, t, hv].float() - proj)
                state_new = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
                out[batch_idx, t, hv] = (state_new.transpose(0, 1) @ q_vec).to(out.dtype)
                state[state_idx, hv] = state_new

    if cu_seqlens is None:
        for batch_idx in range(B):
            run_seq(batch_idx, batch_idx, 0, T)
    else:
        for state_idx in range(state_count):
            run_seq(0, state_idx, int(cu_seqlens[state_idx].item()), int(cu_seqlens[state_idx + 1].item()))
    return out, state


def _local_config(local_v_heads: int) -> Qwen35LinearAttentionConfig:
    return Qwen35LinearAttentionConfig(num_k_heads=local_v_heads // 3, num_v_heads=local_v_heads)


def test_qwen35_scalar_kda_prefill_reference_matches_manual():
    torch.manual_seed(0)
    B, T, HV, K = 2, 3, 2, 128
    q = torch.randn(B, T, HV, K, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
    initial_state = torch.randn(B, HV, K, K, dtype=torch.float32) * 0.01

    out_ref, state_ref = _manual_scalar_prefill(q, k, v, a, b, A_log, dt_bias, initial_state)
    out, state = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="reference",
    )

    torch.testing.assert_close(out.float(), out_ref.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-4)


def test_qwen35_scalar_kda_prefill_varlen_reference_matches_manual():
    torch.manual_seed(1)
    T, HV, K = 4, 2, 128
    q = torch.randn(1, T, HV, K, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(1, T, HV, dtype=torch.bfloat16)
    b = torch.randn(1, T, HV, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    out_ref, state_ref = _manual_scalar_prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens)
    out, state = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        backend="reference",
    )

    torch.testing.assert_close(out.float(), out_ref.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-4)


def test_qwen35_scalar_kda_prefill_cuda_matches_reference():
    if not torch.cuda.is_available() or cula_cuda is None or not hasattr(cula_cuda, "qwen35_scalar_kda_prefill"):
        import pytest

        pytest.skip("qwen35_scalar_kda_prefill CUDA extension is not available")

    torch.manual_seed(10)
    device = torch.device("cuda")
    B, T, HV, K = 1, 8, 48, 128
    q = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    initial_state = torch.randn(B, HV, K, K, device=device, dtype=torch.float32) * 0.01

    out_ref, state_ref = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="reference",
    )
    out, state = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state, state_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_layout_prefill_cuda_supports_local_tp_shards(local_v_heads: int):
    if not torch.cuda.is_available() or cula_cuda is None or not hasattr(cula_cuda, "qwen35_layout_prefill"):
        pytest.skip("qwen35_layout_prefill CUDA extension is not available")

    torch.manual_seed(20 + local_v_heads)
    device = torch.device("cuda")
    config = _local_config(local_v_heads)
    tokens = 5
    mixed_qkv = torch.randn(tokens, config.conv_dim, device=device, dtype=torch.bfloat16)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=torch.bfloat16)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=torch.bfloat16)

    ref = qwen35_layout_prefill_reference(mixed_qkv, a, b, config=config)
    out = qwen35_layout_prefill(mixed_qkv, a, b, config=config, backend="cudac")

    torch.cuda.synchronize()
    for out_tensor, ref_tensor in zip(out, ref, strict=True):
        torch.testing.assert_close(out_tensor, ref_tensor)


@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_scalar_kda_prefill_cuda_supports_local_tp_shards(local_v_heads: int):
    if not torch.cuda.is_available() or cula_cuda is None or not hasattr(cula_cuda, "qwen35_scalar_kda_prefill"):
        pytest.skip("qwen35_scalar_kda_prefill CUDA extension is not available")

    torch.manual_seed(30 + local_v_heads)
    device = torch.device("cuda")
    B, T, HV, K = 1, 4, local_v_heads, 128
    q = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    initial_state = torch.randn(B, HV, K, K, device=device, dtype=torch.float32) * 0.01

    out_ref, state_ref = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="reference",
    )
    out, state = qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state, state_ref, atol=2e-2, rtol=2e-2)


def test_qwen35_chunk_qk_prefill_sm90_matches_torch():
    if not torch.cuda.is_available() or cula_cuda is None or not hasattr(cula_cuda, "qwen35_chunk_qk_prefill_sm90"):
        import pytest

        pytest.skip("qwen35_chunk_qk_prefill_sm90 CUDA extension is not available")

    torch.manual_seed(11)
    device = torch.device("cuda")
    B, T, HV, K = 1, 64, 48, 128
    q = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    out = torch.empty(B, HV, T, T, device=device, dtype=torch.float32)

    cula_cuda.qwen35_chunk_qk_prefill_sm90(q.contiguous(), k.contiguous(), out)
    torch.cuda.synchronize()

    ref = torch.einsum("bthd,bshd->bhts", q.float(), k.float())
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_chunk_qk_prefill_sm90_supports_local_tp_shards(local_v_heads: int):
    if not torch.cuda.is_available() or cula_cuda is None or not hasattr(cula_cuda, "qwen35_chunk_qk_prefill_sm90"):
        pytest.skip("qwen35_chunk_qk_prefill_sm90 CUDA extension is not available")

    torch.manual_seed(40 + local_v_heads)
    device = torch.device("cuda")
    B, T, HV, K = 1, 32, local_v_heads, 128
    q = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    out = torch.empty(B, HV, T, T, device=device, dtype=torch.float32)

    cula_cuda.qwen35_chunk_qk_prefill_sm90(q.contiguous(), k.contiguous(), out)
    torch.cuda.synchronize()

    ref = torch.einsum("bthd,bshd->bhts", q.float(), k.float())
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("T", [64, 128])
def test_qwen35_fused_kda_prefill_matches_reference(T: int):
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is not available")
    if not has_qwen35_fused_kda_prefill(torch.device("cuda")):
        import pytest

        pytest.skip("Qwen3.5 fused KDA prefill backend is not available")

    torch.manual_seed(12)
    device = torch.device("cuda")
    B, H, HV, K = 1, 16, 48, 128
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, K, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    initial_state = torch.randn(B, HV, K, K, device=device, dtype=torch.float32) * 0.01

    out_ref, state_ref = qwen35_scalar_kda_prefill(
        q.repeat_interleave(HV // H, dim=2),
        k.repeat_interleave(HV // H, dim=2),
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="reference",
    )
    out, state = qwen35_fused_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(state, state_ref, atol=3e-2, rtol=3e-2)


def test_qwen35_conv1d_prefill_flattened_state():
    x = torch.arange(5 * 3, dtype=torch.bfloat16).reshape(5, 3)
    weight = torch.ones(3, 4, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    y, state = qwen35_conv1d_prefill(x, weight, cu_seqlens=cu_seqlens, output_final_state=True)

    assert y.shape == x.shape
    assert state.shape == (2, 3, 4)
    torch.testing.assert_close(state[0, :, -2:], x[:2].transpose(0, 1))
    torch.testing.assert_close(state[1, :, -3:], x[2:5].transpose(0, 1))


def test_qwen35_layout_prefill_reference():
    torch.manual_seed(2)
    config = Qwen35LinearAttentionConfig(num_k_heads=1, num_v_heads=2)
    tokens = 3
    mixed_qkv = torch.randn(tokens, config.conv_dim, dtype=torch.bfloat16)
    a = torch.randn(tokens, config.num_v_heads, dtype=torch.bfloat16)
    b = torch.randn(tokens, config.num_v_heads, dtype=torch.bfloat16)

    ref = qwen35_layout_prefill_reference(mixed_qkv, a, b, config=config)
    out = qwen35_layout_prefill(mixed_qkv, a, b, config=config, backend="reference")

    for out_tensor, ref_tensor in zip(out, ref, strict=True):
        assert torch.equal(out_tensor, ref_tensor)


def test_qwen35_linear_attention_prefill_reference_shapes():
    torch.manual_seed(2)
    config = Qwen35LinearAttentionConfig(num_k_heads=1, num_v_heads=2)
    tokens = 3
    mixed_qkv = torch.randn(tokens, config.conv_dim, dtype=torch.bfloat16)
    a = torch.randn(tokens, config.num_v_heads, dtype=torch.bfloat16)
    b = torch.randn(tokens, config.num_v_heads, dtype=torch.bfloat16)
    conv_weight = torch.randn(config.conv_dim, config.conv_kernel_size, dtype=torch.bfloat16)
    A_log = -torch.rand(config.num_v_heads, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, dtype=torch.float32) * 0.1
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)

    out, conv_state, recurrent_state = qwen35_linear_attention_prefill(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        config=config,
        cu_seqlens=cu_seqlens,
        backend="reference",
    )

    assert out.shape == (tokens, config.value_dim)
    assert conv_state.shape == (2, config.conv_dim, config.conv_kernel_size)
    assert recurrent_state.shape == (2, config.num_v_heads, config.head_k_dim, config.head_v_dim)
