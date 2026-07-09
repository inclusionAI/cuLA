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

"""Unit tests for the fused causal-conv1d + MTP verify decode kernel.

Covers both dispatch variants (small_batch small/medium-batch, large_batch warp-spec large-batch) against
a pure-torch reference (depthwise causal conv1d + SiLU + loop delta-rule
recurrence + per-step conv-window/SSM snapshots). Shapes follow the real KDA
config where num_q_heads == num_v_heads (H == HV); H=HV=8 is the TP=4 per-GPU
shape and H=HV=32 the full model. GVA (H=8, HV=16) is included for robustness
(the kernel supports HV >= H even though the model uses H == HV).
"""

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.kda.decode.mtp_conv import NWARP, kda_conv_decode_mtp_verify

W = 4  # KDA conv width (short_conv_kernel_size)


# --------------------------------------------------------------------------- #
# torch reference: conv + loop recurrence + per-step snapshots
# --------------------------------------------------------------------------- #
def _torch_reference(inp, N, T, H, HV, K, V, scale, lower_bound,
                     softplus_beta=1.0, softplus_threshold=20.0):
    dev = inp["mixed_qkv"].device
    D = 2 * H * K + HV * V
    mixed = inp["mixed_qkv"].float()
    w = inp["conv_weight"]
    bias = inp["conv_bias"]
    hist0 = inp["conv_state_native"].float()
    a = inp["a"].float()
    b = inp["b"].float()
    A_log = inp["A_log"]
    dt_bias = inp["dt_bias"]
    ssm0 = inp["ssm_states"].float()
    cidx = inp["cache_indices"]

    qk_dim = H * K
    o = torch.zeros(N, T, HV, V, device=dev, dtype=torch.float32)
    ssm_snap = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)
    win_snap = torch.zeros(N, T, W - 1, D, device=dev, dtype=torch.float32)
    conv_state_out = torch.zeros_like(hist0)

    for n in range(N):
        slot = int(cidx[n].item())
        x = mixed[n * T:(n + 1) * T]
        hist = hist0[slot]
        xfull = torch.cat([hist, x], dim=0)  # [W-1+T, D]

        y = torch.zeros(T, D, device=dev, dtype=torch.float32)
        for t in range(T):
            acc = bias.clone()
            for j in range(W):
                acc = acc + w[:, j] * xfull[t + j]
            y[t] = torch.nn.functional.silu(acc)
            win_snap[n, t] = xfull[t + 1:t + 1 + (W - 1)]
        y = y.to(torch.bfloat16).float()  # bf16 round-trip
        conv_state_out[slot] = xfull[-(W - 1):]

        qy = y[:, 0:qk_dim].view(T, H, K)
        ky = y[:, qk_dim:2 * qk_dim].view(T, H, K)
        vy = y[:, 2 * qk_dim:2 * qk_dim + HV * V].view(T, HV, V)

        for hv in range(HV):
            ih = hv // (HV // H)
            S = ssm0[slot, hv].clone()
            eA = torch.exp(A_log[hv])
            for t in range(T):
                qb = qy[t, ih].clone()
                kb = ky[t, ih].clone()
                vb = vy[t, hv].clone()
                gx = a[n, t, hv] + dt_bias[hv]
                if lower_bound is not None:
                    g = lower_bound * torch.sigmoid(eA * gx)
                else:
                    beta_x = softplus_beta * gx
                    sp = torch.where(beta_x <= softplus_threshold,
                                     (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
                                     gx)
                    g = -eA * sp
                beta = torch.sigmoid(b[n, t, hv])
                qb = qb / torch.sqrt((qb * qb).sum() + 1e-6) * scale
                kb = kb / torch.sqrt((kb * kb).sum() + 1e-6)
                S = S * torch.exp(g)[None, :]
                v_new = (vb - S @ kb) * beta
                S = S + v_new[:, None] * kb[None, :]
                o[n, t, hv] = S @ qb
                ssm_snap[n, t, hv] = S

    return dict(o=o, ssm_snap=ssm_snap, win_snap=win_snap, conv_state_out=conv_state_out)


def _make_inputs(N, T, H, HV, K, V, gate, seed, device="cuda"):
    torch.manual_seed(seed)
    D = 2 * H * K + HV * V
    f32, bf16 = torch.float32, torch.bfloat16
    return dict(
        D=D,
        mixed_qkv=(torch.randn(N * T, D, device=device, dtype=f32) * 0.5).to(bf16),
        conv_weight=torch.randn(D, W, device=device, dtype=f32) * 0.3,
        conv_bias=torch.randn(D, device=device, dtype=f32) * 0.1,
        conv_state_native=torch.randn(N, W - 1, D, device=device, dtype=f32) * 0.3,
        a=(torch.randn(N, T, HV, K, device=device, dtype=f32) * 0.5).to(bf16),
        b=(torch.randn(N, T, HV, device=device, dtype=f32) * 0.5).to(bf16),
        A_log=-torch.rand(HV, device=device, dtype=f32) * 2.0,
        dt_bias=torch.randn(HV, K, device=device, dtype=f32) * 0.1,
        ssm_states=torch.randn(N, HV, V, K, device=device, dtype=f32) * 0.01,
        cache_indices=torch.arange(N, device=device, dtype=torch.int32),
    )


def _run_cula(inp, N, T, H, HV, K, V, scale, lower_bound, variant):
    dev = inp["mixed_qkv"].device
    D = inp["D"]
    conv_state = inp["conv_state_native"].permute(0, 2, 1).contiguous()  # [lines, D, W-1]
    conv_window = torch.zeros(N, T, D, W - 1, device=dev, dtype=torch.float32)
    inter_states = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)
    idx = inp["cache_indices"]
    o = kda_conv_decode_mtp_verify(
        mixed_qkv=inp["mixed_qkv"], conv_weight=inp["conv_weight"], conv_bias=inp["conv_bias"],
        conv_state=conv_state, conv_state_indices=idx, intermediate_conv_window=conv_window,
        intermediate_state_indices=idx, a=inp["a"], b=inp["b"], A_log=inp["A_log"],
        dt_bias=inp["dt_bias"], ssm_states=inp["ssm_states"].clone(), cache_indices=idx,
        intermediate_states_buffer=inter_states, scale=scale, T=T, num_q_heads=H,
        num_v_heads=HV, head_k_dim=K, head_v_dim=V, lower_bound=lower_bound, variant=variant,
    )
    return dict(o=o.view(N, T, HV, V), conv_state=conv_state, conv_window=conv_window,
                inter_states=inter_states)


def _assert_close(name, ref, actual, atol, rtol):
    diff = (ref.float() - actual.float()).abs()
    max_diff = diff.max().item()
    print(f"    [{name}] max_diff={max_diff:.6e} (atol={atol}, rtol={rtol})")
    assert torch.allclose(ref.float(), actual.float(), atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6e}")


def _check(N, T, H, HV, variant, gate, seed=0):
    K, V = 128, 128
    scale = K ** -0.5
    lower_bound = -5.0 if gate == "safe" else None
    inp = _make_inputs(N, T, H, HV, K, V, gate, seed)
    ref = _torch_reference(inp, N, T, H, HV, K, V, scale, lower_bound)
    act = _run_cula(inp, N, T, H, HV, K, V, scale, lower_bound, variant)
    # conv snapshots are raw-input copies -> bit-exact; o/ssm carry bf16+reduction noise.
    _assert_close("conv_window", ref["win_snap"], act["conv_window"].transpose(-1, -2), 0.0, 0.0)
    _assert_close("conv_state", ref["conv_state_out"], act["conv_state"].transpose(-1, -2), 0.0, 0.0)
    _assert_close("o", ref["o"], act["o"], 3e-2, 2e-2)
    _assert_close("inter_ssm", ref["ssm_snap"], act["inter_states"], 5e-2, 3e-2)


# --------------------------------------------------------------------------- #
# real KDA shape: H == HV. small_batch (small/medium batch) and large_batch (large batch) variants.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,HV", [(8, 8), (32, 32)], ids=["h8", "h32"])
@pytest.mark.parametrize("N,T", [(4, 4), (8, 4), (2, 2), (4, 8)])
@pytest.mark.parametrize("variant", ["small_batch", "large_batch"])
def test_conv_mtp_hv_eq_h_safe(N, T, H, HV, variant):
    _check(N, T, H, HV, variant, "safe")


@pytest.mark.parametrize("H,HV", [(8, 8), (32, 32)], ids=["h8", "h32"])
@pytest.mark.parametrize("N,T", [(4, 4), (8, 4)])
@pytest.mark.parametrize("variant", ["small_batch", "large_batch"])
def test_conv_mtp_hv_eq_h_softplus(N, T, H, HV, variant):
    _check(N, T, H, HV, variant, "softplus")


# --------------------------------------------------------------------------- #
# GVA robustness: HV > H (kernel supports HV>=H even though the model uses H==HV).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["small_batch", "large_batch"])
@pytest.mark.parametrize("gate", ["safe", "softplus"])
def test_conv_mtp_gva(variant, gate):
    _check(N=8, T=4, H=8, HV=16, variant=variant, gate=gate)


# --------------------------------------------------------------------------- #
# small_batch and large_batch must agree (same math, different threading).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,HV", [(8, 8), (32, 32)], ids=["h8", "h32"])
def test_conv_mtp_vk_ws_agree(H, HV):
    N, T, K, V = 8, 4, 128, 128
    scale = K ** -0.5
    inp = _make_inputs(N, T, H, HV, K, V, "safe", seed=1)
    small_batch = _run_cula(inp, N, T, H, HV, K, V, scale, -5.0, "small_batch")
    large_batch = _run_cula(inp, N, T, H, HV, K, V, scale, -5.0, "large_batch")
    _assert_close("vk_vs_ws_o", small_batch["o"], large_batch["o"], 2e-2, 1e-2)
    _assert_close("vk_vs_ws_conv_state", small_batch["conv_state"], large_batch["conv_state"], 0.0, 0.0)


# --------------------------------------------------------------------------- #
# auto dispatch: small/medium batch -> small_batch, large batch -> large_batch (both correct).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", [1, 4, 32])
def test_conv_mtp_auto_dispatch(N):
    _check(N, T=4, H=8, HV=8, variant="auto", gate="safe")


# --------------------------------------------------------------------------- #
# determinism: identical inputs must give bit-identical outputs across runs.
# A residual race (e.g. on the shared conv_state) would surface as non-determinism.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,HV", [(8, 8), (32, 32), (8, 16)], ids=["h8", "h32", "gva"])
@pytest.mark.parametrize("variant", ["small_batch", "large_batch"])
def test_conv_mtp_determinism(variant, H, HV):
    N, T, K, V = 16, 4, 128, 128
    scale = K ** -0.5
    inp = _make_inputs(N, T, H, HV, K, V, "safe", seed=7)
    ref = _run_cula(inp, N, T, H, HV, K, V, scale, -5.0, variant)
    for r in range(int(os.environ.get("KDA_DET_ITERS", "6"))):
        act = _run_cula(inp, N, T, H, HV, K, V, scale, -5.0, variant)
        for name in ("o", "inter_states", "conv_state", "conv_window"):
            assert torch.equal(ref[name], act[name]), f"{name} not deterministic (run {r})"



# --------------------------------------------------------------------------- #
# large_batch extreme small batch: work_units=8 -> bvw=1 (tile_v=8), the smallest v-tile
# (only lane 0 computes v-conv + broadcast). Correctness at the NWARP=8 edge.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate", ["safe", "softplus"])
def test_conv_mtp_ws_bvw1(gate):
    _check(N=1, T=4, H=8, HV=8, variant="large_batch", gate=gate)


def test_nwarp_is_eight():
    # large_batch fixed at NWARP=8 (2026-07-14): 8 warps/CTA raise occupancy at the
    # grid=128-CTA tiers (work=64 valley filled) with no shared-q/k redundancy.
    assert NWARP == 8
