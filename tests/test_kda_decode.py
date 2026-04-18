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

"""
Unit tests for kda_decode (CuTe DSL KDA decode kernel).

Compares against a pure PyTorch reference implementation of the
delta-rule state update:
    gate  = exp(-exp(A_log) * softplus(a + dt_bias))
    v_new = sigmoid(b) * (v - H^T @ (gate * k_norm))
    H_new = diag(gate) @ H + k_norm @ v_new^T
    o     = H_new^T @ (l2norm(q) * scale)
"""

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda import kda_decode, fused_sigmoid_gating_delta_rule_update


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def torch_kda_decode_ref(
    q,  # (N, H, K) float32
    k,  # (N, H, K) float32   (H here is the query/key head count)
    v,  # (N, HV, V) float32
    a,  # (N, HV, K) float32
    b,  # (N, HV) float32
    A_log,  # (HV,) float32
    dt_bias,  # (HV, K) float32
    state,  # (N, HV, V, K) float32
    scale,  # float
    use_l2norm=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """
    Pure PyTorch reference for single-token KDA decode.

    Returns:
        o:         (N, HV, V) float32
        state_new: (N, HV, V, K) float32
    """
    N, HV, V, K = state.shape
    H = q.shape[1]
    heads_per_group = HV // H  # GQA ratio

    A = torch.exp(A_log)  # (HV,)

    state_new = state.clone()
    o = torch.zeros(N, HV, V, dtype=torch.float32, device=q.device)

    for n in range(N):
        for hv in range(HV):
            i_h = hv // heads_per_group

            # Gate: exp(-A * softplus(a + dt_bias))
            x = a[n, hv, :] + dt_bias[hv, :]  # (K,)
            sp = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
            gate = torch.exp(-A[hv] * sp)  # (K,)

            # L2 normalize q and k
            if use_l2norm:
                q_vec = F.normalize(q[n, i_h, :], dim=0) * scale
                k_vec = F.normalize(k[n, i_h, :], dim=0)
            else:
                q_vec = q[n, i_h, :] * scale
                k_vec = k[n, i_h, :]

            # H^T @ (gate * k_norm) — sum over K dim
            # state is (V, K), gate*k_vec is (K,)
            Hk = state[n, hv] @ (gate * k_vec)  # (V,)

            # Delta correction
            beta_val = torch.sigmoid(b[n, hv])  # scalar
            v_new = beta_val * (v[n, hv, :] - Hk)  # (V,)

            # State update: diag(gate) @ H + outer(v_new, k_norm)
            state_new[n, hv] = gate[None, :] * state[n, hv] + v_new[:, None] * k_vec[None, :]

            # Output: H_new @ q_scaled
            o[n, hv, :] = state_new[n, hv] @ q_vec  # (V,)

    return o, state_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(N, H, HV, K, V, device="cuda", seed=42):
    """Generate random inputs for KDA decode."""
    torch.manual_seed(seed)
    q = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2  # negative → A < 1
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def run_kda_decode_dense(q, k, v, a, b, A_log, dt_bias, state, scale):
    """Run kda_decode in dense layout: (N, 1, H/HV, dim)."""
    N, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]

    # Reshape to dense layout: (N, 1, H, K), etc.
    q_4d = q.unsqueeze(1).contiguous()  # (N, 1, H, K)
    k_4d = k.unsqueeze(1).contiguous()  # (N, 1, H, K) — note: H not HV for k
    v_4d = v.unsqueeze(1).contiguous()  # (N, 1, HV, V)
    a_4d = a.unsqueeze(1).contiguous()  # (N, 1, HV, K)
    b_3d = b.unsqueeze(1).contiguous()  # (N, 1, HV)

    state_source = state.clone().contiguous()  # (N, HV, V, K)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    o = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q_4d.to(torch.bfloat16),
        k=k_4d.to(torch.bfloat16),
        v=v_4d.to(torch.bfloat16),
        a=a_4d.to(torch.bfloat16),
        b=b_3d.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    # o shape: (N, 1, HV, V), state_source modified in-place
    return o.squeeze(1), state_source  # (N, HV, V), (N, HV, V, K)


def run_kda_decode_varlen(q, k, v, a, b, A_log, dt_bias, state, scale):
    """Run kda_decode in varlen layout: (1, N, H/HV, dim)."""
    N, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]

    # Reshape to varlen layout: (1, N, H, K), etc.
    q_4d = q.unsqueeze(0).contiguous()  # (1, N, H, K)
    k_4d = k.unsqueeze(0).contiguous()  # (1, N, H, K)
    v_4d = v.unsqueeze(0).contiguous()  # (1, N, HV, V)
    a_3d = a.contiguous()  # (N, HV, K) — varlen uses 3D
    b_2d = b.contiguous()  # (N, HV) — varlen uses 2D

    state_source = state.clone().contiguous()  # (N, HV, V, K)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, device=q.device, dtype=torch.int32)

    o = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q_4d.to(torch.bfloat16),
        k=k_4d.to(torch.bfloat16),
        v=v_4d.to(torch.bfloat16),
        a=a_3d.to(torch.bfloat16),
        b=b_2d.to(torch.bfloat16),
        initial_state_source=state_source,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    # o shape: (1, N, HV, V), state_source modified in-place
    return o.squeeze(0), state_source  # (N, HV, V), (N, HV, V, K)


def _assert_close(name, ref, actual, atol=3e-2, rtol=2e-2):
    """Assert tensors are close with informative error message."""
    diff = (ref.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(ref.float(), actual.float(), atol=atol, rtol=rtol)
    assert ok, f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, atol={atol}, rtol={rtol}"


def run_kda_decode_triton_compatible(q, k, v, a, b, A_log, dt_bias, state_kv, scale, fused_fn=None):
    """Run the Triton-compatible cuLA decode API."""
    N = q.shape[0]
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    out = q.new_empty(N, 1, v.shape[1], v.shape[2], dtype=torch.bfloat16)
    fused_fn = fused_sigmoid_gating_delta_rule_update if fused_fn is None else fused_fn
    o = fused_fn(
        A_log=A_log,
        a=a.reshape(N, 1, -1).to(torch.bfloat16),
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.unsqueeze(1).to(torch.bfloat16),
        k=k.unsqueeze(1).to(torch.bfloat16),
        v=v.unsqueeze(1).to(torch.bfloat16),
        b=b.unsqueeze(1).to(torch.bfloat16),
        initial_state_source=state_kv,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        is_kda=True,
        out=out,
    )
    return o.squeeze(1), state_kv.permute(0, 1, 3, 2).contiguous()


# ---------------------------------------------------------------------------
# Tests: Dense layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 4, 16, 32, 64, 128])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
def test_kda_decode_dense(N, H, HV):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    # Reference (fp32)
    o_ref, state_ref = torch_kda_decode_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    # Kernel
    o_kernel, state_kernel = run_kda_decode_dense(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        scale,
    )

    _assert_close("output", o_ref, o_kernel.float())
    _assert_close("state", state_ref, state_kernel)


# ---------------------------------------------------------------------------
# Tests: Varlen layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 16, 32, 64, 128])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
def test_kda_decode_varlen(N, H, HV):
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    # Reference (fp32)
    o_ref, state_ref = torch_kda_decode_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    # Kernel
    o_kernel, state_kernel = run_kda_decode_varlen(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        scale,
    )

    _assert_close("output", o_ref, o_kernel.float())
    _assert_close("state", state_ref, state_kernel)


# ---------------------------------------------------------------------------
# Tests: Different V dimension (V=256)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 32])
def test_kda_decode_large_v(N):
    H, HV, K, V = 8, 16, 128, 256
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = torch_kda_decode_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    o_kernel, state_kernel = run_kda_decode_dense(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        scale,
    )

    _assert_close("output", o_ref, o_kernel.float())
    _assert_close("state", state_ref, state_kernel)


# ---------------------------------------------------------------------------
# Tests: Zero initial state
# ---------------------------------------------------------------------------
def test_kda_decode_zero_state():
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, _ = make_inputs(N, H, HV, K, V)
    state = torch.zeros(N, HV, V, K, device="cuda", dtype=torch.float32)

    o_ref, state_ref = torch_kda_decode_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    o_kernel, state_kernel = run_kda_decode_dense(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        scale,
    )

    _assert_close("output", o_ref, o_kernel.float())
    _assert_close("state", state_ref, state_kernel)


def test_kda_decode_layout_defaults_to_vk():
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state_vk = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = torch_kda_decode_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state_vk.clone(),
        scale,
    )

    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    out = q.new_empty(N, 1, HV, V, dtype=torch.bfloat16)
    state_default = state_vk.clone().contiguous()
    state_v_last = state_vk.permute(0, 1, 3, 2).contiguous()

    o_default = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a.reshape(N, 1, -1).to(torch.bfloat16),
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.unsqueeze(1).to(torch.bfloat16),
        k=k.unsqueeze(1).to(torch.bfloat16),
        v=v.unsqueeze(1).to(torch.bfloat16),
        b=b.unsqueeze(1).to(torch.bfloat16),
        initial_state_source=state_default,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        out=out,
    )

    o_v_last = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a.reshape(N, 1, -1).to(torch.bfloat16),
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.unsqueeze(1).to(torch.bfloat16),
        k=k.unsqueeze(1).to(torch.bfloat16),
        v=v.unsqueeze(1).to(torch.bfloat16),
        b=b.unsqueeze(1).to(torch.bfloat16),
        initial_state_source=state_v_last,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        state_layout="kv",
    )

    _assert_close("default-vk output", o_ref, o_default.squeeze(1).float())
    _assert_close("default-vk state", state_ref, state_default)
    _assert_close("kv output", o_ref, o_v_last.squeeze(1).float())
    _assert_close("kv state", state_ref, state_v_last.permute(0, 1, 3, 2).contiguous())


def test_kda_decode_layout_mismatch_raises():
    N, H, HV, K, V = 4, 8, 16, 128, 256
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state_vk = make_inputs(N, H, HV, K, V)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    state_kv = state_vk.permute(0, 1, 3, 2).contiguous()

    with pytest.raises(ValueError, match="State layout mismatch"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a.reshape(N, 1, -1).to(torch.bfloat16),
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q.unsqueeze(1).to(torch.bfloat16),
            k=k.unsqueeze(1).to(torch.bfloat16),
            v=v.unsqueeze(1).to(torch.bfloat16),
            b=b.unsqueeze(1).to(torch.bfloat16),
            initial_state_source=state_kv,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="vk",
        )

    with pytest.raises(ValueError, match="State layout mismatch"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a.reshape(N, 1, -1).to(torch.bfloat16),
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q.unsqueeze(1).to(torch.bfloat16),
            k=k.unsqueeze(1).to(torch.bfloat16),
            v=v.unsqueeze(1).to(torch.bfloat16),
            b=b.unsqueeze(1).to(torch.bfloat16),
            initial_state_source=state_vk,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="kv",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
