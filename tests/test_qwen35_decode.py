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

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.qwen35_layout_decode import qwen35_layout_decode, qwen35_layout_decode_reference
from cula.ops.qwen35_scalar_kda_decode import qwen35_layout_scalar_kda_decode, qwen35_scalar_kda_decode
from cula.ops.qwen35_conv1d_decode import qwen35_conv1d_decode_reference, qwen35_conv1d_decode_update
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG, Qwen35LinearAttentionConfig
from cula.qwen35.runtime import qwen35_linear_attention_decode

try:
    from cula.ops.kda_decode_fla import fused_sigmoid_gating_delta_rule_update as triton_fused_sigmoid_update
except ImportError:
    triton_fused_sigmoid_update = None

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _has_qwen35_cudac():
    return (
        torch.cuda.is_available()
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_conv1d_decode")
        and hasattr(cula_cuda, "qwen35_layout_decode")
        and hasattr(cula_cuda, "qwen35_scalar_kda_decode")
    )


def _has_qwen35_fused_layout_kda_cudac():
    return _has_qwen35_cudac() and hasattr(cula_cuda, "qwen35_layout_scalar_kda_decode")


def make_inputs(
    tokens: int = 2,
    pool_size: int = 3,
    device: torch.device | None = None,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
):
    device = _device() if device is None else device
    torch.manual_seed(0)
    mixed_qkv = torch.randn(tokens, config.conv_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    conv_weight = torch.randn(config.conv_dim, config.conv_kernel_size, device=device, dtype=config.qkv_dtype)
    conv_state = torch.randn(tokens, config.conv_dim, config.conv_kernel_size, device=device, dtype=config.qkv_dtype)
    recurrent_state = torch.randn(
        pool_size,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32) % pool_size
    return mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices


def manual_conv_decode(x_t: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor):
    state_tail = conv_state[..., 1:].float()
    window = torch.cat([state_tail, x_t.unsqueeze(-1).float()], dim=-1)
    conv = (window * weight.float().unsqueeze(0)).sum(dim=-1)
    y = torch.nn.functional.silu(conv).to(dtype=x_t.dtype)
    state_new = conv_state.clone()
    state_new[..., 0] = conv_state[..., 1]
    state_new[..., 1] = conv_state[..., 2]
    state_new[..., 2] = conv_state[..., 3]
    state_new[..., 3] = x_t
    return y, state_new


def manual_qwen35_decode_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
):
    conv_out, conv_state_out = manual_conv_decode(mixed_qkv, conv_state, conv_weight)
    q_end = config.key_dim
    k_end = q_end + config.key_dim
    q = conv_out[:, :q_end].view(mixed_qkv.shape[0], config.num_k_heads, config.head_k_dim)
    k = conv_out[:, q_end:k_end].view(mixed_qkv.shape[0], config.num_k_heads, config.head_k_dim)
    v = conv_out[:, k_end:].view(mixed_qkv.shape[0], config.num_v_heads, config.head_v_dim)
    q_rep = q.repeat_interleave(config.qk_repeat_factor, dim=1)
    k_rep = k.repeat_interleave(config.qk_repeat_factor, dim=1)

    scale = config.head_k_dim**-0.5
    q_f = torch.nn.functional.normalize(q_rep.float(), dim=-1) * scale
    k_f = torch.nn.functional.normalize(k_rep.float(), dim=-1)
    v_f = v.float()
    state_out = recurrent_state.clone()
    out = torch.empty(mixed_qkv.shape[0], config.value_dim, device=mixed_qkv.device, dtype=mixed_qkv.dtype)

    for token_idx in range(mixed_qkv.shape[0]):
        per_token = []
        pool_idx = int(state_indices[token_idx].item())
        for hv in range(config.num_v_heads):
            state_kv = state_out[pool_idx, hv]
            decay = torch.exp(-torch.exp(A_log[hv]) * torch.nn.functional.softplus(a[token_idx, hv].float() + dt_bias[hv]))
            beta = torch.sigmoid(b[token_idx, hv].float())
            k_vec = k_f[token_idx, hv]
            q_vec = q_f[token_idx, hv]
            proj = decay * (state_kv.transpose(0, 1) @ k_vec)
            v_new = beta * (v_f[token_idx, hv] - proj)
            state_new_kv = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
            per_token.append((state_new_kv.transpose(0, 1) @ q_vec).to(mixed_qkv.dtype))
            state_out[pool_idx, hv] = state_new_kv
        out[token_idx] = torch.cat(per_token, dim=0)
    return out, conv_state_out, state_out


def manual_qwen35_scalar_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
):
    if a.ndim == 2:
        a = a.unsqueeze(1)
    if b.ndim == 2:
        b = b.unsqueeze(1)
    N, _, HV, K = q.shape
    scale = K**-0.5
    q_f = torch.nn.functional.normalize(q.squeeze(1).float(), dim=-1) * scale
    k_f = torch.nn.functional.normalize(k.squeeze(1).float(), dim=-1)
    v_f = v.squeeze(1).float()
    state_out = recurrent_state.clone()
    out = torch.empty(N, 1, HV, v.shape[-1], device=q.device, dtype=v.dtype)

    for token_idx in range(N):
        pool_idx = int(state_indices[token_idx].item())
        for hv in range(HV):
            state_kv = state_out[pool_idx, hv]
            decay = torch.exp(-torch.exp(A_log[hv]) * torch.nn.functional.softplus(a[token_idx, 0, hv].float() + dt_bias[hv]))
            beta = torch.sigmoid(b[token_idx, 0, hv].float())
            k_vec = k_f[token_idx, hv]
            q_vec = q_f[token_idx, hv]
            proj = decay * (state_kv.transpose(0, 1) @ k_vec)
            v_new = beta * (v_f[token_idx, hv] - proj)
            state_new_kv = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
            out[token_idx, 0, hv] = (state_new_kv.transpose(0, 1) @ q_vec).to(v.dtype)
            state_out[pool_idx, hv] = state_new_kv
    return out, state_out


def manual_qwen35_layout_scalar_kda_reference(
    mixed_qkv_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
):
    q_rep, k_rep, v, a_ref, b_ref = qwen35_layout_decode_reference(mixed_qkv_conv, a, b, config=config)

    scale = config.head_k_dim**-0.5
    q_f = torch.nn.functional.normalize(q_rep.float(), dim=-1) * scale
    k_f = torch.nn.functional.normalize(k_rep.float(), dim=-1)
    v_f = v.float()
    state_out = recurrent_state.clone()
    out = torch.empty(
        mixed_qkv_conv.shape[0],
        q_rep.shape[1],
        config.head_v_dim,
        device=mixed_qkv_conv.device,
        dtype=mixed_qkv_conv.dtype,
    )

    for token_idx in range(mixed_qkv_conv.shape[0]):
        pool_idx = int(state_indices[token_idx].item())
        for hv in range(q_rep.shape[1]):
            state_kv = state_out[pool_idx, hv]
            decay = torch.exp(-torch.exp(A_log[hv]) * torch.nn.functional.softplus(a_ref[token_idx, hv].float() + dt_bias[hv]))
            beta = torch.sigmoid(b_ref[token_idx, hv].float())
            k_vec = k_f[token_idx, hv]
            q_vec = q_f[token_idx, hv]
            proj = decay * (state_kv.transpose(0, 1) @ k_vec)
            v_new = beta * (v_f[token_idx, hv] - proj)
            state_new_kv = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
            out[token_idx, hv] = (state_new_kv.transpose(0, 1) @ q_vec).to(mixed_qkv_conv.dtype)
            state_out[pool_idx, hv] = state_new_kv
    return out.unsqueeze(1), state_out


def _local_config(local_v_heads: int) -> Qwen35LinearAttentionConfig:
    return Qwen35LinearAttentionConfig(num_k_heads=local_v_heads // 3, num_v_heads=local_v_heads)


@pytest.mark.parametrize("tokens", [1, 2])
def test_qwen35_conv_decode_reference(tokens: int):
    mixed_qkv, _, _, conv_weight, conv_state, _, _, _, _ = make_inputs(tokens=tokens)
    y_ref, state_ref = manual_conv_decode(mixed_qkv, conv_state, conv_weight)
    y_op, state_op = qwen35_conv1d_decode_update(mixed_qkv, conv_state, conv_weight, backend="reference")
    assert torch.equal(y_ref, y_op)
    assert torch.equal(state_ref, state_op)
    y_ref2, state_ref2 = qwen35_conv1d_decode_reference(mixed_qkv, conv_state, conv_weight)
    assert torch.equal(y_ref, y_ref2)
    assert torch.equal(state_ref, state_ref2)


def test_qwen35_layout_decode_reference():
    mixed_qkv, a, b, _, _, _, _, _, _ = make_inputs(tokens=2)
    q_rep_ref, k_rep_ref, v_ref, a_ref, b_ref = qwen35_layout_decode_reference(mixed_qkv, a, b)
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(mixed_qkv, a, b, backend="reference")
    assert torch.equal(q_rep_ref, q_rep)
    assert torch.equal(k_rep_ref, k_rep)
    assert torch.equal(v_ref, v)
    assert torch.equal(a_ref, a_kernel)
    assert torch.equal(b_ref, b_kernel)


@pytest.mark.parametrize("tokens", [1, 2])
def test_qwen35_decode_reference_chain(tokens: int):
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(tokens=tokens)
    out_ref, conv_state_ref, recurrent_state_ref = manual_qwen35_decode_reference(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state,
        recurrent_state,
        state_indices,
    )
    out, conv_state_out, recurrent_state_out = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="reference",
    )

    assert torch.allclose(out_ref.float(), out.float(), atol=1e-5, rtol=1e-5)
    assert torch.equal(conv_state_ref, conv_state_out)
    assert torch.allclose(recurrent_state_ref, recurrent_state_out, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_qwen35_decode_cudac_matches_reference(tokens: int):
    # Decode batches represent distinct active sequences, so keep state rows unique
    # to avoid intentionally racing multiple token updates against one cache row.
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=tokens,
        pool_size=max(tokens, 3),
        device=torch.device("cuda"),
    )
    out_ref, conv_state_ref, recurrent_state_ref = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="reference",
    )
    out, conv_state_out, recurrent_state_out = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )

    torch.cuda.synchronize()
    assert torch.allclose(out_ref.float(), out.float(), atol=3e-2, rtol=3e-2)
    assert torch.equal(conv_state_ref, conv_state_out)
    assert torch.allclose(recurrent_state_ref, recurrent_state_out, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_conv_decode_cudac_supports_local_tp_shapes(local_v_heads: int):
    config = _local_config(local_v_heads)
    mixed_qkv, _, _, conv_weight, conv_state, _, _, _, _ = make_inputs(
        tokens=3,
        pool_size=3,
        device=torch.device("cuda"),
        config=config,
    )
    y_ref, state_ref = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        backend="reference",
    )
    y, state = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(state, state_ref)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_scalar_kda_decode_cudac_supports_local_tp_shapes(local_v_heads: int):
    torch.manual_seed(3)
    config = _local_config(local_v_heads)
    tokens = 3
    device = torch.device("cuda")
    q = torch.randn(tokens, 1, config.num_v_heads, config.head_k_dim, device=device, dtype=config.qkv_dtype)
    k = torch.randn_like(q)
    v = torch.randn(tokens, 1, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, 1, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, 1, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    recurrent_state = torch.randn(
        tokens,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)

    out_ref, state_ref = manual_qwen35_scalar_kda_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices,
    )
    out, state = qwen35_scalar_kda_decode(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(state, state_ref, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_decode_cudac_supports_local_tp_shapes(local_v_heads: int):
    config = _local_config(local_v_heads)
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=3,
        pool_size=3,
        device=torch.device("cuda"),
        config=config,
    )
    out_ref, conv_state_ref, recurrent_state_ref = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        config=config,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="reference",
    )
    out, conv_state_out, recurrent_state_out = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        config=config,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv_state_out, conv_state_ref)
    torch.testing.assert_close(recurrent_state_out, recurrent_state_ref, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not _has_qwen35_fused_layout_kda_cudac(), reason="Qwen3.5 fused layout+KDA CUDA backend is not available")
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_qwen35_fused_layout_kda_cudac_matches_reference_unfused_and_triton(tokens: int):
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=tokens,
        pool_size=max(tokens, 3),
        device=torch.device("cuda"),
    )
    conv_out, _ = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        backend="cudac",
    )
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(conv_out, a, b, backend="cudac")
    out_ref, state_ref = manual_qwen35_layout_scalar_kda_reference(
        conv_out,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices,
    )
    out_unfused, state_unfused = qwen35_scalar_kda_decode(
        q=q_rep.unsqueeze(1),
        k=k_rep.unsqueeze(1),
        v=v.unsqueeze(1),
        a=a_kernel,
        b=b_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )
    if triton_fused_sigmoid_update is not None:
        state_triton = recurrent_state.clone()
        out_triton = triton_fused_sigmoid_update(
            A_log=A_log,
            a=a_kernel.unsqueeze(1).contiguous(),
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_rep.unsqueeze(1).contiguous(),
            k=k_rep.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            b=b_kernel.unsqueeze(1).contiguous(),
            initial_state_source=state_triton,
            initial_state_indices=state_indices,
            scale=DEFAULT_QWEN35_LINEAR_ATTN_CONFIG.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            is_kda=False,
        )
    out_fused, state_fused = qwen35_layout_scalar_kda_decode(
        mixed_qkv_conv=conv_out,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        config=DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
        backend="cudac",
    )

    torch.cuda.synchronize()
    assert torch.allclose(out_ref.float(), out_fused.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(state_ref, state_fused, atol=3e-5, rtol=3e-5)
    assert torch.equal(out_unfused, out_fused)
    assert torch.equal(state_unfused, state_fused)
    if triton_fused_sigmoid_update is not None:
        assert torch.allclose(out_triton.float(), out_fused.float(), atol=3e-2, rtol=3e-2)
        assert torch.allclose(state_triton, state_fused, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(
    not _has_qwen35_fused_layout_kda_cudac(),
    reason="Qwen3.5 fused layout+KDA CUDA backend is not available",
)
@pytest.mark.parametrize("tokens", [64, 128])
def test_qwen35_fused_layout_kda_cudac_long_matches_reference(tokens: int):
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=tokens,
        pool_size=tokens,
        device=torch.device("cuda"),
    )
    conv_out, _ = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        backend="cudac",
    )
    out_ref, state_ref = manual_qwen35_layout_scalar_kda_reference(
        conv_out, a, b, A_log, dt_bias, recurrent_state, state_indices
    )
    out_fused, state_fused = qwen35_layout_scalar_kda_decode(
        mixed_qkv_conv=conv_out,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        config=DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(state_fused, state_ref, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not _has_qwen35_fused_layout_kda_cudac(), reason="Qwen3.5 fused layout+KDA CUDA backend is not available")
@pytest.mark.parametrize("local_v_heads", [48, 24, 12, 6])
def test_qwen35_layout_scalar_kda_cudac_supports_local_tp_shards(local_v_heads: int):
    config = _local_config(local_v_heads)
    mixed_qkv, a, b, _, _, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=2,
        pool_size=3,
        device=torch.device("cuda"),
        config=config,
    )
    out_ref, state_ref = manual_qwen35_layout_scalar_kda_reference(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices,
        config=config,
    )
    q_rep_ref, k_rep_ref, v_ref, a_ref, b_ref = qwen35_layout_decode_reference(mixed_qkv, a, b, config=config)
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(mixed_qkv, a, b, config=config, backend="cudac")
    out, state = qwen35_layout_scalar_kda_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices=state_indices,
        config=config,
        backend="cudac",
    )
    out_3d_gate, state_3d_gate = qwen35_layout_scalar_kda_decode(
        mixed_qkv,
        a.unsqueeze(1),
        b.unsqueeze(1),
        A_log,
        dt_bias,
        recurrent_state,
        state_indices=state_indices,
        config=config,
        backend="cudac",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(q_rep, q_rep_ref)
    torch.testing.assert_close(k_rep, k_rep_ref)
    torch.testing.assert_close(v, v_ref)
    torch.testing.assert_close(a_kernel, a_ref)
    torch.testing.assert_close(b_kernel, b_ref)
    torch.testing.assert_close(out.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(state, state_ref, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(out_3d_gate, out)
    torch.testing.assert_close(state_3d_gate, state)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
def test_qwen35_decode_cudac_rejects_duplicate_state_indices():
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, _ = make_inputs(
        tokens=2,
        pool_size=3,
        device=torch.device("cuda"),
    )
    state_indices = torch.zeros(2, device=mixed_qkv.device, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires unique state_indices"):
        qwen35_linear_attention_decode(
            mixed_qkv,
            a,
            b,
            conv_weight,
            A_log,
            dt_bias,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
            backend="cudac",
        )
