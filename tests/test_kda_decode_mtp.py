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

import os
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for sibling test import

from test_kda_decode import torch_kda_decode_ref  # trusted single-token reference

from cula.kda import kda_decode
from cula.ops.kda.decode.mtp import (
    _select_mtp_config,
    _select_mtp_tile_v,
    kda_decode_mtp_recurrent,
    kda_decode_mtp_recurrent_ws,
)
from cula.ops.kda.decode.mtp_kvbuffer import (
    _kvbuffer_prefer_tensor_core,
    _select_kvb_tile_v,
    _select_shuffle_kvb_ilp_rows,
    kda_decode_mtp_kvbuffer,
    kda_decode_mtp_shuffle_kvbuffer,
    kda_decode_mtp_tensor_core_kvbuffer,
    kda_flush_kvbuffer,
)


def torch_kda_mtp_ref(
    q, k, v, a, b, A_log, dt_bias, state, scale, use_l2norm=True, softplus_beta=1.0, softplus_threshold=20.0, lower_bound=None
):
    """fp32 ground truth: the single-token KDA recurrence threaded over T. Returns (o, final_state)."""
    N, T, HV, V = v.shape
    H = q.shape[2]
    heads_per_group = HV // H
    A = torch.exp(A_log)
    state_cur = state.clone()
    o = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        for n in range(N):
            for hv in range(HV):
                i_h = hv // heads_per_group
                x = a[n, t, hv, :] + dt_bias[hv, :]
                if lower_bound is not None:
                    # safe gate: g = lower_bound * sigmoid(exp(A_log) * x)
                    gate = torch.exp(lower_bound * torch.sigmoid(A[hv] * x))
                else:
                    sp = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
                    gate = torch.exp(-A[hv] * sp)
                if use_l2norm:
                    q_vec = F.normalize(q[n, t, i_h, :], dim=0) * scale
                    k_vec = F.normalize(k[n, t, i_h, :], dim=0)
                else:
                    q_vec = q[n, t, i_h, :] * scale
                    k_vec = k[n, t, i_h, :]
                Hk = state_cur[n, hv] @ (gate * k_vec)
                beta_val = torch.sigmoid(b[n, t, hv])
                v_new = beta_val * (v[n, t, hv, :] - Hk)
                state_cur[n, hv] = gate[None, :] * state_cur[n, hv] + v_new[:, None] * k_vec[None, :]
                o[n, t, hv, :] = state_cur[n, hv] @ q_vec
    return o, state_cur


def make_inputs_mtp(N, T, H, HV, K, V, device="cuda", seed=42):
    """Random MTP inputs (q/k/v/a/b bf16, A_log/dt_bias/state fp32)."""
    torch.manual_seed(seed)
    q = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2  # negative -> A < 1
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale):
    """The "loop" baseline: T sequential single-token kda_decode calls, state carried across tokens."""
    N, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    state_source = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o_all = torch.empty(N, T, HV, V, device=q.device, dtype=torch.bfloat16)
    for t in range(T):
        q_t = q[:, t].unsqueeze(1).contiguous()
        k_t = k[:, t].unsqueeze(1).contiguous()
        v_t = v[:, t].unsqueeze(1).contiguous()
        a_t = a[:, t].unsqueeze(1).contiguous()
        b_t = b[:, t].unsqueeze(1).contiguous()
        o_t = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q_t.to(torch.bfloat16),
            k=k_t.to(torch.bfloat16),
            v=v_t.to(torch.bfloat16),
            a=a_t.to(torch.bfloat16),
            b=b_t.to(torch.bfloat16),
            initial_state_source=state_source,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        o_all[:, t] = o_t.squeeze(1)
    return o_all, state_source


def _assert_close(name, ref, actual, atol=3e-2, rtol=2e-2):
    """allclose, printing the observed max/mean margin (pytest -s)."""
    diff = (ref.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"    [{name}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} (atol={atol}, rtol={rtol})")
    ok = torch.allclose(ref.float(), actual.float(), atol=atol, rtol=rtol)
    assert ok, f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, atol={atol}, rtol={rtol}"


def oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state, scale):
    """fp32 per-token state snapshots [N,T,HV,V,K] from the trusted single-token reference."""
    N, T = q.shape[0], q.shape[1]
    HV, V, K = v.shape[2], v.shape[3], q.shape[3]
    state_cur = state.clone()
    inter = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device=q.device)
    for t in range(T):
        _, state_cur = torch_kda_decode_ref(
            q[:, t].float(),
            k[:, t].float(),
            v[:, t].float(),
            a[:, t],
            b[:, t].float(),
            A_log,
            dt_bias,
            state_cur,
            scale,
        )
        inter[:, t] = state_cur
    return inter


def run_recurrent(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    scale,
    *,
    variant,
    bv=-1,
    k_split=-1,
    disable_state_update=False,
    intermediate=False,
    lower_bound=None,
):
    """Run kda_decode_mtp_recurrent; state fed/returned in vk layout (kv transposed in and back)."""
    N = q.shape[0]
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    T = q.shape[1]
    HV, V, K = v.shape[2], v.shape[3], q.shape[3]
    inter = torch.zeros(N, T, HV, V, K, device=q.device, dtype=torch.float32) if intermediate else None
    st = state.clone().contiguous()
    if variant == "kv":
        st = st.transpose(-2, -1).contiguous()  # vk -> kv
    rec_kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=st,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        variant=variant,
        k_split=k_split,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=inter,
        lower_bound=lower_bound,
    )
    if variant == "vk":
        rec_kwargs["bv"] = bv  # kv is fixed 1-warp; bv stays at the WARP_BV default
    o = kda_decode_mtp_recurrent(**rec_kwargs)
    state_vk = st.transpose(-2, -1).contiguous() if variant == "kv" else st
    return (o, state_vk, inter) if intermediate else (o, state_vk)


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_mtp_ref_is_threaded_single_token(T):
    """Pure-torch: the MTP oracle equals the trusted single-token ref threaded over T."""
    N, H, HV, K, V = 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_mtp, st_mtp = torch_kda_mtp_ref(q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale)
    st_cur = state.clone()
    o_manual = torch.zeros(N, T, HV, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        o_t, st_cur = torch_kda_decode_ref(
            q[:, t].float(), k[:, t].float(), v[:, t].float(), a[:, t], b[:, t].float(), A_log, dt_bias, st_cur, scale
        )
        o_manual[:, t] = o_t
    torch.testing.assert_close(o_mtp, o_manual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_mtp, st_cur, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("zero_state", [False, True], ids=["randstate", "zerostate"])
@pytest.mark.parametrize(
    "N,T,H,HV",
    [
        pytest.param(*c, id="N{}-T{}-H{}-HV{}".format(*c))
        for c in [(1, 1, 8, 16), (4, 4, 8, 16), (16, 8, 8, 16), (64, 2, 16, 32), (4, 4, 16, 32)]
    ],
)
def test_oracle_vs_loop(N, T, H, HV, zero_state):
    """The looped single-token kernel matches the fp32 oracle (small N)."""
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    if zero_state:
        state = torch.zeros_like(state)
    o_ref, st_ref = torch_kda_mtp_ref(q.float(), k.float(), v.float(), a, b.float(), A_log, dt_bias, state.clone(), scale)
    o_loop, st_loop = run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale)
    _assert_close("loop output", o_ref, o_loop.float())
    _assert_close("loop final state", st_ref, st_loop)


@pytest.mark.parametrize(
    "N,T,H,HV,variant,bv,k_split",
    [
        pytest.param(*c, id="N{}-T{}-H{}-HV{}-{}-bv{}-ks{}".format(*c))
        for c in [
            # vk: bv sweep + auto, incl T=1 and GQA
            (1, 1, 8, 16, "vk", -1, 1),
            (4, 4, 8, 16, "vk", -1, 1),
            (8, 2, 8, 16, "vk", -1, 1),
            (4, 4, 8, 16, "vk", 8, 1),
            (4, 4, 8, 16, "vk", 16, 1),
            (4, 2, 8, 16, "vk", 32, 1),
            (16, 4, 16, 32, "vk", -1, 1),
            # kv: k_split sweep + auto, incl T=1 and GQA
            (1, 1, 8, 16, "kv", 32, -1),
            (4, 4, 8, 16, "kv", 32, -1),
            (8, 2, 8, 16, "kv", 32, -1),
            (4, 4, 8, 16, "kv", 32, 1),
            (4, 4, 8, 16, "kv", 32, 2),
            (4, 4, 8, 16, "kv", 32, 4),
            (16, 4, 16, 32, "kv", 32, -1),
        ]
    ],
)
def test_recurrent_decode(N, T, H, HV, variant, bv, k_split):
    """recurrent vk + kv vs loop: bv / k_split / auto / GQA in one table."""
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_loop, st_loop = run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_sb, st_sb = run_recurrent(q, k, v, a, b, A_log, dt_bias, state, scale, variant=variant, bv=bv, k_split=k_split)
    tag = f"recurrent {variant} bv={bv} ks={k_split}"
    _assert_close(f"{tag} output", o_loop.float(), o_sb.float())
    _assert_close(f"{tag} final state", st_loop, st_sb)


@pytest.mark.parametrize(
    "kernel", ["recurrent_ws", "recurrent_ws_ilp4", "recurrent_ws_smem_v", "recurrent_vk", "recurrent_kv"]
)
@pytest.mark.parametrize(
    "N,T,H,HV",
    [
        pytest.param(*c, id="N{}-T{}-H{}-HV{}".format(*c))
        for c in [
            (1, 1, 8, 16),
            (4, 4, 8, 16),
            (8, 4, 8, 16),
            (16, 4, 16, 32),
        ]
    ],
)
def test_lower_bound_safe_gate(kernel, N, T, H, HV):
    """Safe-gate path g = lower_bound * sigmoid(exp(A_log) * x): the MTP kernels must
    match the fp32 oracle (the single-token loop kernel has no safe-gate path)."""
    K, V = 128, 128
    scale = K**-0.5
    lower_bound = -4.0
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_ref, st_ref = torch_kda_mtp_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
        lower_bound=lower_bound,
    )
    if kernel == "recurrent_ws":
        o, st = run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, lower_bound=lower_bound)
    elif kernel == "recurrent_ws_ilp4":
        o, st = run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=16, ilp_rows=4, lower_bound=lower_bound)
    elif kernel == "recurrent_ws_smem_v":
        o, st = run_recurrent_ws(
            q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=32, ilp_rows=4, use_smem_v=True, lower_bound=lower_bound
        )
    elif kernel == "recurrent_vk":
        o, st = run_recurrent(q, k, v, a, b, A_log, dt_bias, state, scale, variant="vk", lower_bound=lower_bound)
    else:  # recurrent_kv
        o, st = run_recurrent(q, k, v, a, b, A_log, dt_bias, state, scale, variant="kv", lower_bound=lower_bound)
    tag = f"lb {kernel} N={N} T={T} HV={HV}"
    _assert_close(f"{tag} output", o_ref, o.float())
    _assert_close(f"{tag} final state", st_ref, st)


@pytest.mark.parametrize("kernel", ["recurrent_ws", "recurrent_ws_ilp4", "recurrent_vk", "recurrent_kv"])
def test_disable_state_update(kernel):
    """disable_state_update leaves the state pool unchanged while output still matches the loop."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_loop, _ = run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale)

    if kernel == "recurrent_ws":
        o, st = run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, disable_state_update=True)
    elif kernel == "recurrent_ws_ilp4":
        o, st = run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=32, ilp_rows=4, disable_state_update=True)
    else:
        variant = "vk" if kernel == "recurrent_vk" else "kv"
        o, st = run_recurrent(q, k, v, a, b, A_log, dt_bias, state, scale, variant=variant, disable_state_update=True)

    assert torch.equal(st, state), f"{kernel}: state pool modified despite disable_state_update=True"
    _assert_close(f"{kernel} dsu output", o_loop.float(), o.float())


@pytest.mark.parametrize("kernel", ["recurrent_ws", "recurrent_ws_smem_v", "recurrent_vk", "recurrent_kv"])
def test_determinism(kernel):
    """Bit-exact determinism: repeat the state-writeback launch, assert identical output + state."""
    N, T, H, HV, K, V = 16, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)

    def launch():
        if kernel == "recurrent_ws":
            return run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=64, ilp_rows=4, use_packed_fma=False)
        if kernel == "recurrent_ws_smem_v":
            return run_recurrent_ws(
                q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=64, ilp_rows=4, use_packed_fma=False, use_smem_v=True
            )
        variant = "vk" if kernel == "recurrent_vk" else "kv"
        return run_recurrent(q, k, v, a, b, A_log, dt_bias, state, scale, variant=variant)

    o_ref, st_ref = launch()
    o_ref = o_ref.clone()
    n_iters = int(os.environ.get("KDA_MTP_DET_ITERS", "100000"))
    for i in range(n_iters):
        o_i, st_i = launch()
        assert torch.equal(o_i, o_ref), f"{kernel} output non-deterministic at iter {i}"
        assert torch.equal(st_i, st_ref), f"{kernel} state non-deterministic at iter {i}"


def test_intermediate_disable_state_update():
    """disable_state_update leaves the pool untouched; snapshots still fire and match the oracle."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)

    _o, st_vk, inter = run_recurrent(
        q, k, v, a, b, A_log, dt_bias, state, scale, variant="vk", disable_state_update=True, intermediate=True
    )
    assert torch.equal(st_vk, state), "pool modified despite disable_state_update=True"
    for t in range(T):
        _assert_close(f"inter+dsu snapshot[t={t}]", inter_ref[:, t], inter[:, t])


def test_intermediate_buffer_validation():
    """Bad intermediate_states_buffer shape / dtype must raise."""
    N, T, H, HV, K, V = 4, 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    st = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)

    def _call(buf):
        return kda_decode_mtp_recurrent(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.to(torch.bfloat16),
            k=k.to(torch.bfloat16),
            v=v.to(torch.bfloat16),
            a=a.to(torch.bfloat16),
            b=b.to(torch.bfloat16),
            initial_state_source=st,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            variant="vk",
            intermediate_states_buffer=buf,
        )

    with pytest.raises((ValueError, AssertionError)):
        _call(torch.zeros(N, T + 1, HV, V, K, device="cuda", dtype=torch.float32))
    with pytest.raises((ValueError, AssertionError)):
        _call(torch.zeros(N, T, HV, V, K, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("N,T", [(1, 2), (4, 4), (8, 8), (4, 2), (16, 6)])
def test_intermediate_recurrent_vk(N, T):
    """vk per-token snapshot == fp32 oracle; t=T-1 snapshot == final state pool."""
    H, HV, K, V = 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)
    o, st_vk, inter = run_recurrent(
        q, k, v, a, b, A_log, dt_bias, state.clone(), scale, variant="vk", disable_state_update=False, intermediate=True
    )
    for t in range(T):
        _assert_close(f"sbvk inter snapshot[t={t}]", inter_ref[:, t], inter[:, t])
    assert torch.equal(inter[:, T - 1], st_vk), "sbvk: t=T-1 snapshot != final state"


def run_recurrent_ws(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    scale,
    *,
    tile_v=None,
    ilp_rows=None,
    use_packed_fma=None,
    use_smem_v=None,
    disable_state_update=False,
    intermediate=False,
    lower_bound=None,
):
    """Run kda_decode_mtp_recurrent_ws (vk). Returns (o, state) or (o, state, inter)."""
    N, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    st = state.clone().contiguous()
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    inter = torch.zeros(N, T, HV, V, K, device=q.device, dtype=torch.float32) if intermediate else None
    o = kda_decode_mtp_recurrent_ws(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=st,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        use_packed_fma=use_packed_fma,
        use_smem_v=use_smem_v,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=inter,
        lower_bound=lower_bound,
    )
    return (o, st, inter) if intermediate else (o, st)


@pytest.mark.parametrize(
    "N,T,H,HV,tile_v,ilp_rows,use_smem_v",
    [
        pytest.param(*c, id="N{}-T{}-H{}-HV{}-tv{}-ilp{}-smem{}".format(*c))
        for c in [
            # auto (None) across N incl GQA and large batch
            (1, 2, 8, 16, None, None, None),
            (4, 4, 8, 16, None, None, None),
            (16, 4, 16, 32, None, None, None),
            (64, 8, 8, 16, None, None, None),
            (1024, 2, 8, 16, None, None, None),
            (2048, 2, 8, 16, None, None, None),
            # explicit tile_v sweep, ilp=2
            (4, 4, 8, 16, 8, 2, False),
            (4, 4, 8, 16, 16, 2, False),
            (4, 4, 8, 16, 32, 2, False),
            (4, 2, 8, 16, 64, 2, False),
            # ilp=4 (tile_v % 16 == 0), fused steps + double-accumulator
            (4, 4, 8, 16, 16, 4, False),
            (4, 4, 8, 16, 32, 4, False),
            (4, 2, 8, 16, 64, 4, False),
            # use_smem_v on
            (4, 4, 8, 16, 32, 4, True),
            (16, 2, 16, 32, 64, 4, True),
        ]
    ],
)
def test_recurrent_ws_decode(N, T, H, HV, tile_v, ilp_rows, use_smem_v):
    """ws warp-spec vs loop: auto / tile_v / ilp 2,4 / use_smem_v / large N in one table."""
    K, V = 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_loop, st_loop = run_kda_decode_mtp_via_loop_dense(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_ws, st_ws = run_recurrent_ws(
        q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=tile_v, ilp_rows=ilp_rows, use_smem_v=use_smem_v
    )
    tag = f"ws tv={tile_v} ilp={ilp_rows} smem={use_smem_v}"
    _assert_close(f"{tag} output", o_loop.float(), o_ws.float())
    _assert_close(f"{tag} final state", st_loop, st_ws)


@pytest.mark.parametrize("tile_v,ilp_rows", [(8, 2), (16, 2), (32, 2), (64, 2), (16, 4), (32, 4), (64, 4)])
def test_recurrent_ws_smem_v_bit_identical(tile_v, ilp_rows):
    """use_smem_v is pure data movement: byte-for-byte identical to the GMEM path."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    o_g, st_g = run_recurrent_ws(
        q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=False
    )
    o_s, st_s = run_recurrent_ws(
        q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=tile_v, ilp_rows=ilp_rows, use_packed_fma=False, use_smem_v=True
    )
    assert torch.equal(o_s, o_g), f"smem_v output != GMEM (tile_v={tile_v}, ilp={ilp_rows})"
    assert torch.equal(st_s, st_g), f"smem_v state != GMEM (tile_v={tile_v}, ilp={ilp_rows})"


def test_recurrent_ws_ilp4_rejects_bad_tile_v():
    """ilp=4 requires tile_v % 16 == 0; tile_v=8 must raise."""
    N, T, H, HV, K, V = 4, 2, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    with pytest.raises(AssertionError):
        run_recurrent_ws(q, k, v, a, b, A_log, dt_bias, state, scale, tile_v=8, ilp_rows=4, use_packed_fma=False)


@pytest.mark.parametrize(
    "N,HV,V,T,expected",
    [
        (1, 16, 128, 2, (8, 2, False)),
        (4, 16, 128, 4, (8, 2, False)),
        (1, 65, 128, 2, (16, 4, False)),
        (8, 16, 128, 2, (16, 4, False)),
        (16, 16, 128, 2, (16, 2, False)),
        (16, 16, 128, 4, (32, 4, False)),
        (7, 64, 128, 2, (16, 2, False)),
        (7, 64, 128, 8, (32, 4, False)),
        (16, 64, 128, 2, (32, 4, False)),
        (64, 16, 128, 8, (32, 4, False)),
        (17, 64, 128, 2, (64, 4, True)),
        (256, 64, 128, 8, (64, 4, True)),
        (8, 16, 8, 2, (8, 2, False)),
        (8, 16, 16, 2, (16, 4, False)),
    ],
)
def test_select_mtp_config(N, HV, V, T, expected):
    """The joint (tile_v, ilp_rows, use_smem_v) heuristic returns the expected config."""
    assert _select_mtp_config(N, HV, V, T) == expected
    assert _select_mtp_tile_v(N, HV, V, T) == expected[0]


def test_select_mtp_config_ilp_capped_at_4():
    """ilp is capped at 4 (no ilp=8 path) in every bucket."""
    for N in (1, 8, 16, 64, 256, 4096):
        for HV in (16, 64):
            for T in (1, 2, 4, 8):
                for dsu in (False, True):
                    _, ilp, _ = _select_mtp_config(N, HV, 128, T, disable_state_update=dsu)
                    assert ilp in (2, 4), f"N={N},HV={HV},T={T},dsu={dsu} -> ilp={ilp}"


@pytest.mark.parametrize("use_smem_v", [False, True])
@pytest.mark.parametrize("tile_v,ilp_rows", [(16, 2), (32, 4), (64, 4)])
def test_intermediate_vs_oracle_and_final(use_smem_v, tile_v, ilp_rows):
    """Each per-token snapshot == fp32 oracle state; the t=T-1 snapshot == final state pool."""
    N, T, H, HV, K, V = 4, 4, 8, 16, 128, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K, V)
    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)
    _o, st_final, inter = run_recurrent_ws(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        scale,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        use_packed_fma=False,
        use_smem_v=use_smem_v,
        intermediate=True,
    )
    tag = f"inter smem={use_smem_v} tv={tile_v} ilp={ilp_rows}"
    for t in range(T):
        _assert_close(f"{tag} snapshot[t={t}]", inter_ref[:, t], inter[:, t])
    assert torch.equal(inter[:, T - 1], st_final), f"{tag}: t=T-1 snapshot != final state pool"


K_DIM = 128  # kvbuffer ops hard-require K=128


def _alloc_ubufs(N, T, HV, V, device="cuda"):
    """d_buffer [N,T,HV,V], k/g_buffer [N,T,HV,K] — fp32, matching the kernel contract."""
    return (
        torch.zeros(N, T, HV, V, dtype=torch.float32, device=device),
        torch.zeros(N, T, HV, K_DIM, dtype=torch.float32, device=device),
        torch.zeros(N, T, HV, K_DIM, dtype=torch.float32, device=device),
    )


def _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, *, ubufs=None, lower_bound=None):
    """Run a kvbuffer verify op (disable_state_update=True). Returns output o [N,T,HV,V]."""
    N = q.shape[0]
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    d_b, k_b, g_b = ubufs if ubufs is not None else (None, None, None)
    op = kda_decode_mtp_shuffle_kvbuffer if which == "shuffle" else kda_decode_mtp_tensor_core_kvbuffer
    return op(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state.clone().contiguous(),
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        emit_output=True,
        d_buffer=d_b,
        k_buffer=k_b,
        g_buffer=g_b,
        lower_bound=lower_bound,
    )


def _kvb_oracle_out(q, k, v, a, b, A_log, dt_bias, state, scale):
    o_ref, _ = torch_kda_mtp_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state,
        scale,
    )
    return o_ref


def _check_kvb_verify_and_flush(which, N, T, H, HV):
    """verify output == oracle, u-buffer populated; flush(m) == m-th oracle snapshot (m=full/half/one)."""
    V = K_DIM
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K_DIM, V)
    scale = K_DIM**-0.5
    o_ref = _kvb_oracle_out(q, k, v, a, b, A_log, dt_bias, state, scale)
    inter_ref = oracle_intermediate_states(q, k, v, a, b, A_log, dt_bias, state.clone(), scale)

    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    ubufs = _alloc_ubufs(N, T, HV, V)
    o = _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, ubufs=ubufs)
    _assert_close(f"{which}_verify N{N}T{T}", o_ref, o)
    assert ubufs[0].abs().sum() > 0, f"{which}: d_buffer was not written"

    # flush each accept length m -> rebuilt S_m == oracle state after m tokens (snapshot m-1)
    for m in sorted({T, max(1, T // 2), 1}):
        pool = state.clone().contiguous()
        kda_flush_kvbuffer(pool, indices, ubufs[0], ubufs[1], ubufs[2], accept_len=m)
        _assert_close(f"{which}_flush N{N}T{T}m{m}", inter_ref[:, m - 1], pool)


@pytest.mark.parametrize("N,T,H,HV", [(2, 2, 16, 16), (4, 4, 16, 16), (2, 4, 32, 32)])
def test_shuffle_kvbuffer_verify_and_flush(N, T, H, HV):
    """shuffle-kvbuffer (token-parallel SIMT) verify output + rank-m flush match the fp32 oracle."""
    _check_kvb_verify_and_flush("shuffle", N, T, H, HV)


@pytest.mark.parametrize("N,T,H,HV", [(2, 3, 16, 16), (4, 6, 16, 16), (1, 8, 32, 32)])
def test_tensor_core_kvbuffer_verify_and_flush(N, T, H, HV):
    """tensor_core-kvbuffer (CuTe tensor-core gemm) verify output + rank-m flush match the fp32 oracle."""
    _check_kvb_verify_and_flush("tensor_core", N, T, H, HV)


@pytest.mark.parametrize(
    "which,N,T,H,HV",
    [("shuffle", 2, 2, 16, 16), ("shuffle", 4, 2, 16, 16), ("tensor_core", 2, 4, 16, 16), ("tensor_core", 1, 8, 32, 32)],
)
def test_lower_bound_kvbuffer(which, N, T, H, HV):
    """kvbuffer (shuffle/tensor_core) safe-gate path: verify output matches the fp32 oracle with lower_bound."""
    V = K_DIM
    scale = K_DIM**-0.5
    lower_bound = -4.0
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K_DIM, V)
    o_ref, _ = torch_kda_mtp_ref(
        q.float(),
        k.float(),
        v.float(),
        a,
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
        lower_bound=lower_bound,
    )
    ubufs = _alloc_ubufs(N, T, HV, V)
    o = _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, ubufs=ubufs, lower_bound=lower_bound)
    _assert_close(f"lb {which} N{N}T{T}HV{HV}", o_ref, o)


@pytest.mark.parametrize(
    "N,HV,T,routed",
    [(2, 16, 2, "shuffle"), (8, 16, 4, "tensor_core")],  # S=HV*N: 32 -> shuffle @T2 ; 128 -> tensor_core @T4
)
def test_kvbuffer_dispatch_output_matches_oracle(N, HV, T, routed):
    """kda_decode_mtp_kvbuffer auto-dispatch (S=HV*N + T rule) routes as expected and the
    output matches the oracle whichever kvbuffer kernel it picks."""
    H, V = HV, K_DIM
    assert _kvbuffer_prefer_tensor_core(N, HV, T) is (routed == "tensor_core")
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K_DIM, V)
    scale = K_DIM**-0.5
    o_ref = _kvb_oracle_out(q, k, v, a, b, A_log, dt_bias, state, scale)
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    o = kda_decode_mtp_kvbuffer(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=v.to(torch.bfloat16),
        a=a.to(torch.bfloat16),
        b=b.to(torch.bfloat16),
        initial_state_source=state.clone().contiguous(),
        initial_state_indices=indices,
        scale=scale,
    )
    _assert_close(f"dispatch N{N} HV{HV} T{T}->{routed}", o_ref, o)


def test_kvbuffer_prefer_tensor_core_matches_bench():
    """_kvbuffer_prefer_tensor_core reproduces the kvbuffer-family winner from the kernel-level chain
    bench at grid points spanning the S=HV*N collapse (tensor_core iff T >= t_tc(S))."""
    cases = [
        # (HV, N, T, expect_tensor_core) -- kvbuffer-family winner per the kernel_level speedup table
        (8, 1, 6, True),
        (8, 4, 4, False),
        (8, 4, 6, True),
        (8, 8, 3, False),
        (8, 8, 4, True),
        (8, 32, 2, False),
        (8, 32, 3, True),
        (16, 2, 6, True),
        (16, 4, 4, True),
        (16, 16, 3, True),
        (32, 1, 6, True),
        (32, 2, 4, True),
        (64, 1, 3, False),
        (64, 1, 4, True),
        (64, 4, 3, True),
        (64, 128, 2, False),
    ]
    for hv, n, t, exp in cases:
        assert _kvbuffer_prefer_tensor_core(n, hv, t) is exp, f"HV={hv} N={n} T={t}: want tensor_core={exp}"


@pytest.mark.parametrize("which,N,T,H,HV", [("shuffle", 4, 4, 16, 16), ("tensor_core", 4, 6, 16, 16)])
def test_kvbuffer_verify_determinism(which, N, T, H, HV):
    """Repeated kvbuffer verify launches produce a bit-identical output (and u-buffer)."""
    V = K_DIM
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K_DIM, V)
    scale = K_DIM**-0.5
    ub_ref = _alloc_ubufs(N, T, HV, V)
    o_ref = _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, ubufs=ub_ref)
    for i in range(int(os.environ.get("KDA_MTP_DET_ITERS", "100000"))):
        ub_i = _alloc_ubufs(N, T, HV, V)
        o_i = _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, ubufs=ub_i)
        assert torch.equal(o_i, o_ref), f"{which} verify output non-deterministic at iter {i}"
        assert torch.equal(ub_i[0], ub_ref[0]), f"{which} u-buffer non-deterministic at iter {i}"


@pytest.mark.parametrize("which,N,T,H,HV", [("shuffle", 4, 4, 16, 16), ("tensor_core", 4, 6, 16, 16)])
def test_kvbuffer_flush_determinism(which, N, T, H, HV):
    """Repeated flush launches rebuild a bit-identical state."""
    V = K_DIM
    q, k, v, a, b, A_log, dt_bias, state = make_inputs_mtp(N, T, H, HV, K_DIM, V)
    scale = K_DIM**-0.5
    indices = torch.arange(N, device=q.device, dtype=torch.int32)
    ubufs = _alloc_ubufs(N, T, HV, V)
    _kvb_verify(which, q, k, v, a, b, A_log, dt_bias, state, scale, ubufs=ubufs)
    pool_ref = state.clone().contiguous()
    kda_flush_kvbuffer(pool_ref, indices, ubufs[0], ubufs[1], ubufs[2], accept_len=T)
    for i in range(int(os.environ.get("KDA_MTP_DET_ITERS", "100000"))):
        pool_i = state.clone().contiguous()
        kda_flush_kvbuffer(pool_i, indices, ubufs[0], ubufs[1], ubufs[2], accept_len=T)
        assert torch.equal(pool_i, pool_ref), f"{which} flush state non-deterministic at iter {i}"


@pytest.mark.parametrize("V,N,HV", [(128, 1, 16), (128, 4, 32), (128, 16, 64)])
def test_select_kvb_tile_v_invariants(V, N, HV):
    """The auto tile_v must divide V and be a multiple of 4 (4-warp consumer)."""
    tile_v = _select_kvb_tile_v(V, N, HV)
    assert V % tile_v == 0 and tile_v % 4 == 0, f"tile_v={tile_v} violates V%tile_v==0 & tile_v%4==0"


@pytest.mark.parametrize("tile_v,T", [(64, 2), (32, 4), (64, 8), (16, 6)])
def test_select_shuffle_kvb_ilp_rows_invariants(tile_v, T):
    """ilp_rows must divide rows_per_group = tile_v/4 (the wrapper asserts this)."""
    ilp = _select_shuffle_kvb_ilp_rows(tile_v, T)
    assert ilp >= 1 and (tile_v // 4) % ilp == 0, f"ilp_rows={ilp} must divide tile_v/4={tile_v // 4}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
