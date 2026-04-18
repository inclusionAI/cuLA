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
Unit tests for la_decode (CuTe DSL Lightning Attention decode kernel).

Compares against:
  1. PyTorch reference implementation
  2. fla fused_recurrent_fwd (if available)
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


from cula.ops.la_decode import linear_attention_decode

try:
    from fla.ops.common.fused_recurrent import fused_recurrent_fwd

    HAS_FLA = True
except ImportError:
    HAS_FLA = False


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def torch_la_decode_ref(q, k, v, state, decay_scales, scale):
    """
    Pure PyTorch reference for single-token linear attention decode.

    Args:
        q, k, v: [B, H, D] bf16
        state: [B, H, D, D] fp32  (K x V layout)
        decay_scales: [H] fp32 (positive values; kernel does exp(-decay))
        scale: float

    Returns:
        o: [B, H, D] bf16
        state_new: [B, H, D, D] fp32
    """
    B, H, D = q.shape
    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()

    decay = torch.exp(-decay_scales).view(1, H, 1, 1)  # [1, H, 1, 1]
    state_new = state * decay + k_f.unsqueeze(-1) * v_f.unsqueeze(-2)  # [B,H,D,D]
    o = torch.einsum("bhk,bhkv->bhv", q_f, state_new)  # [B,H,D]
    return o.to(torch.bfloat16), state_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(B, H, D, device="cuda", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device=device, dtype=torch.bfloat16)
    state = torch.randn(B, H, D, D, device=device, dtype=torch.float32) * 0.01
    return q, k, v, state


def run_la_decode(q, k, v, state_4d, decay_scales, scale):
    """Run la_decode with proper state layout conversion."""
    B, H, D, _ = state_4d.shape
    # la_decode state layout: [B*H, V, K] (pretransposed)
    state_cute = (
        state_4d.clone()
        .permute(0, 1, 3, 2)  # [B, H, V, K]
        .reshape(B * H, D, D)
        .contiguous()
    )
    out = torch.zeros(B, H, D, device=q.device, dtype=torch.bfloat16)
    s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)

    linear_attention_decode(
        q,
        k,
        v,
        state_cute,
        out,
        softmax_scale=scale,
        stride_q=0,
        stride_k=0,
        stride_v=0,
        stride_s=0,
        stride_o=0,
        s_offsets=s_offsets,
        decay_scales=decay_scales,
        HEAD_DIM=D,
        K_SPLIT_DIM=D,
        V_SPLIT_DIM=D,
    )
    # Convert state back to [B, H, K, V]
    state_out = state_cute.reshape(B, H, D, D).permute(0, 1, 3, 2).contiguous()
    return out, state_out


# ---------------------------------------------------------------------------
# Tests vs PyTorch reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_output_vs_torch_ref(B):
    H, D = 32, 128
    scale = D**-0.5
    layer_idx, num_layers = 12, 24
    decay_scales = (8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, device="cuda", dtype=torch.float32)

    q, k, v, state = make_inputs(B, H, D)
    o_ref, state_ref = torch_la_decode_ref(q, k, v, state, decay_scales, scale)
    o_cute, state_cute = run_la_decode(q, k, v, state, decay_scales, scale)

    # Output check
    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel_err = rmse / (max_ref + 1e-8)
    assert rel_err < 0.01, f"B={B}: output relative RMSE {rel_err:.6f} too large"

    # State check
    state_rmse = torch.sqrt(torch.mean((state_cute - state_ref) ** 2)).item()
    state_max = torch.abs(state_ref).max().item()
    state_rel = state_rmse / (state_max + 1e-8)
    assert state_rel < 0.001, f"B={B}: state relative RMSE {state_rel:.6f} too large"


@pytest.mark.parametrize("H", [8, 16, 32, 64])
def test_different_heads(H):
    B, D = 4, 128
    scale = D**-0.5
    decay_scales = 0.5 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, H, D)
    o_ref, state_ref = torch_la_decode_ref(q, k, v, state, decay_scales, scale)
    o_cute, state_cute = run_la_decode(q, k, v, state, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, f"H={H}: output mismatch"

    state_rmse = torch.sqrt(torch.mean((state_cute - state_ref) ** 2)).item()
    state_max = torch.abs(state_ref).max().item()
    assert state_rmse / (state_max + 1e-8) < 0.001, f"H={H}: state mismatch"


def test_zero_decay():
    """With decay=0, state_new = state_old + k⊗v (no decay applied)."""
    B, H, D = 2, 32, 128
    scale = D**-0.5
    decay_scales = torch.zeros(H, device="cuda", dtype=torch.float32)

    q, k, v, state = make_inputs(B, H, D)
    o_ref, state_ref = torch_la_decode_ref(q, k, v, state, decay_scales, scale)
    o_cute, state_cute = run_la_decode(q, k, v, state, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, "zero decay: output mismatch"


def test_zero_state():
    """With zero initial state, output = q @ (k⊗v) * scale."""
    B, H, D = 4, 32, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.ones(H, device="cuda", dtype=torch.float32)

    q, k, v, _ = make_inputs(B, H, D)
    state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)

    o_ref, state_ref = torch_la_decode_ref(q, k, v, state, decay_scales, scale)
    o_cute, state_cute = run_la_decode(q, k, v, state, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, "zero state: output mismatch"


# ---------------------------------------------------------------------------
# Tests vs fla fused_recurrent
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FLA, reason="fla not available")
@pytest.mark.parametrize("B", [1, 8, 32, 128])
def test_vs_fla(B):
    H, D = 32, 128
    scale = D**-0.5
    g_gamma = -(8 / H * 0.5) * torch.arange(H, device="cuda", dtype=torch.float32)
    decay_scales = -g_gamma

    q, k, v, state = make_inputs(B, H, D)

    # fla
    q_4d = q.unsqueeze(1)  # [B,1,H,D]
    k_4d = k.unsqueeze(1)
    v_4d = v.unsqueeze(1)
    with torch.no_grad():
        o_fla, _ = fused_recurrent_fwd(
            q_4d,
            k_4d,
            v_4d,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=state.clone(),
            output_final_state=True,
        )
    o_fla = o_fla.squeeze(1).to(torch.bfloat16)  # [B,H,D]

    # la_decode
    o_cute, _ = run_la_decode(q, k, v, state, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_fla.float()) ** 2)).item()
    max_ref = torch.abs(o_fla.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.005, f"B={B}: vs fla mismatch, rel_rmse={rmse / (max_ref + 1e-8):.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
