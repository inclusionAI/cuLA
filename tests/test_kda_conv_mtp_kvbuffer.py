#!/usr/bin/env python3
"""Correctness tests for fused causal-conv1d + KDA KVBuffer verify."""

import os

import pytest
import torch

from cula.ops.kda.decode.mtp_conv_kvbuffer import (
    _select_conv_kvb_variant,
    kda_conv_decode_mtp_kvbuffer,
    kda_conv_decode_mtp_shuffle_kvbuffer,
    kda_conv_decode_mtp_tensor_core_kvbuffer,
)
from cula.ops.kda.decode.mtp_kvbuffer import (
    kda_decode_mtp_shuffle_kvbuffer,
    kda_decode_mtp_tensor_core_kvbuffer,
    kda_flush_kvbuffer,
)

W = 4
K = 128
V = 128


@pytest.mark.parametrize("T", [9, 16, 32])
def test_auto_dispatch_uses_shuffle_above_tensor_core_limit(T):
    assert _select_conv_kvb_variant(N=128, HV=64, T=T) == "shuffle"


def _make_inputs(N, T, H, HV, *, seed=0, pool_size=None, dynamic_stride=False):
    torch.manual_seed(seed)
    pool_size = N + 3 if pool_size is None else pool_size
    D = 2 * H * K + HV * V
    device = "cuda"
    conv_indices = torch.tensor([pool_size - 1 - i for i in range(N)], device=device, dtype=torch.int32)
    cache_indices = torch.tensor([(2 * i + 1) % pool_size for i in range(N)], device=device, dtype=torch.int32)
    inter_indices = torch.tensor([(i + 2) % pool_size for i in range(N)], device=device, dtype=torch.int32)
    mixed_qkv = (torch.randn(N * T, D + (8 if dynamic_stride else 0), device=device) * 0.5).to(torch.bfloat16)[:, :D]
    return {
        "mixed_qkv": mixed_qkv,
        "conv_weight": torch.randn(D, W, device=device) * 0.3,
        "conv_bias": torch.randn(D, device=device) * 0.1,
        "conv_state_native": torch.randn(pool_size, W - 1, D, device=device) * 0.3,
        "conv_indices": conv_indices,
        "cache_indices": cache_indices,
        "inter_indices": inter_indices,
        "a": (torch.randn(N, T, HV, K, device=device) * 0.5).to(torch.bfloat16),
        "b": (torch.randn(N, T, HV, device=device) * 0.5).to(torch.bfloat16),
        "A_log": -torch.rand(HV, device=device) * 2.0,
        "dt_bias": torch.randn(HV, K, device=device) * 0.1,
        "ssm_states": torch.randn(pool_size, HV, V, K, device=device) * 0.01,
        "D": D,
        "pool_size": pool_size,
    }


def _torch_conv(inp, N, T, H, HV, bias=True):
    mixed = inp["mixed_qkv"].float()
    weight = inp["conv_weight"]
    conv_bias = inp["conv_bias"] if bias else torch.zeros_like(inp["conv_bias"])
    state = inp["conv_state_native"]
    D = inp["D"]
    y = torch.empty(N, T, D, device="cuda", dtype=torch.float32)
    windows = torch.zeros(inp["pool_size"], T, D, W - 1, device="cuda", dtype=torch.float32)
    rolled = state.clone()
    for n in range(N):
        cs_idx = int(inp["conv_indices"][n])
        xfull = torch.cat((state[cs_idx], mixed[n * T : (n + 1) * T]), dim=0)
        for t in range(T):
            acc = conv_bias.clone()
            for w in range(W):
                acc = acc + weight[:, w] * xfull[t + w]
            y[n, t] = torch.nn.functional.silu(acc)
            windows[inp["inter_indices"][n], t] = xfull[t + 1 : t + W].T
        rolled[cs_idx] = xfull[-(W - 1) :]
    y = y.to(torch.bfloat16)
    q_end = H * K
    q = y[..., :q_end].view(N, T, H, K)
    k = y[..., q_end : 2 * q_end].view(N, T, H, K)
    v = y[..., 2 * q_end :].view(N, T, HV, V)
    return q, k, v, windows, rolled


def _alloc_ubufs(N, T, HV):
    return (
        torch.empty(N, T, HV, V, device="cuda", dtype=torch.float32),
        torch.empty(N, T, HV, K, device="cuda", dtype=torch.float32),
        torch.empty(N, T, HV, K, device="cuda", dtype=torch.float32),
    )


def _run_fused(
    inp,
    N,
    T,
    H,
    HV,
    variant,
    conv_state,
    conv_windows,
    bufs,
    out,
    lower_bound=-5.0,
    num_v_tiles=-1,
):
    if variant == "shuffle":
        fused_op = kda_conv_decode_mtp_shuffle_kvbuffer
        extra = {}
    elif variant == "tensor_core":
        fused_op = kda_conv_decode_mtp_tensor_core_kvbuffer
        extra = {"num_v_tiles": num_v_tiles}
    else:
        fused_op = kda_conv_decode_mtp_kvbuffer
        extra = {"variant": "auto"}
    return fused_op(
        mixed_qkv=inp["mixed_qkv"],
        conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"],
        conv_state=conv_state,
        conv_state_indices=inp["conv_indices"],
        intermediate_conv_window=conv_windows,
        intermediate_state_indices=inp["inter_indices"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=inp["ssm_states"],
        cache_indices=inp["cache_indices"],
        scale=K**-0.5,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        out=out,
        disable_state_update=True,
        d_buffer=bufs[0],
        k_buffer=bufs[1],
        g_buffer=bufs[2],
        lower_bound=lower_bound,
        **extra,
    )


def _run_pair(
    N,
    T,
    H,
    HV,
    *,
    variant="shuffle",
    gate="safe",
    bias=True,
    seed=0,
    dynamic_stride=False,
    num_v_tiles=-1,
):
    inp = _make_inputs(N, T, H, HV, seed=seed, dynamic_stride=dynamic_stride)
    q, k, v, windows_ref, rolled_ref = _torch_conv(inp, N, T, H, HV, bias=bias)
    lower_bound = -5.0 if gate == "safe" else None
    scale = K**-0.5
    base_bufs = _alloc_ubufs(N, T, HV)
    fused_bufs = _alloc_ubufs(N, T, HV)
    out_base = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    out_fused = torch.empty_like(out_base)
    base_state = inp["ssm_states"].clone()
    conv_state = inp["conv_state_native"].transpose(-1, -2).contiguous()
    conv_windows = torch.zeros_like(windows_ref)

    baseline_variant = _select_conv_kvb_variant(N, HV, T) if variant == "auto" else variant
    baseline_op = kda_decode_mtp_shuffle_kvbuffer if baseline_variant == "shuffle" else kda_decode_mtp_tensor_core_kvbuffer
    baseline_op(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=q,
        k=k,
        v=v,
        a=inp["a"],
        b=inp["b"],
        initial_state_source=base_state,
        initial_state_indices=inp["cache_indices"],
        scale=scale,
        out=out_base,
        disable_state_update=True,
        d_buffer=base_bufs[0],
        k_buffer=base_bufs[1],
        g_buffer=base_bufs[2],
        lower_bound=lower_bound,
    )
    fused_inp = inp if bias else dict(inp, conv_bias=None)
    _run_fused(
        fused_inp,
        N,
        T,
        H,
        HV,
        variant,
        conv_state,
        conv_windows,
        fused_bufs,
        out_fused,
        lower_bound,
        num_v_tiles,
    )
    return inp, out_base, out_fused, base_bufs, fused_bufs, conv_state, conv_windows, rolled_ref, windows_ref


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("gate", ["safe", "softplus"])
@pytest.mark.parametrize("bias", [True, False])
def test_fused_shuffle_matches_unfused(T, gate, bias):
    result = _run_pair(2, T, 8, 8, gate=gate, bias=bias, seed=10 + T)
    _, out_base, out_fused, base_bufs, fused_bufs, conv_state, windows, rolled, windows_ref = result
    assert torch.equal(conv_state.transpose(-1, -2), rolled)
    assert torch.equal(windows, windows_ref)
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(fused_bufs[0], base_bufs[0], atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(fused_bufs[1], base_bufs[1], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(fused_bufs[2], base_bufs[2], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("H,HV", [(8, 16), (16, 16), (32, 32)])
def test_fused_shuffle_gva_and_heads(H, HV):
    result = _run_pair(2, 4, H, HV, gate="safe", seed=30 + H + HV)
    _, out_base, out_fused, base_bufs, fused_bufs, *_ = result
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    for actual, expected in zip(fused_bufs, base_bufs):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-3)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("gate", ["safe", "softplus"])
@pytest.mark.parametrize("H,HV", [(8, 8), (8, 16), (32, 32)])
def test_fused_tensor_core_matches_unfused(T, gate, H, HV):
    result = _run_pair(2, T, H, HV, variant="tensor_core", gate=gate, seed=50 + T + HV)
    _, out_base, out_fused, base_bufs, fused_bufs, conv_state, windows, rolled, windows_ref = result
    assert torch.equal(conv_state.transpose(-1, -2), rolled)
    assert torch.equal(windows, windows_ref)
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(fused_bufs[0], base_bufs[0], atol=4e-3, rtol=3e-3)
    # Rare conv FMA -> bf16 boundary flips amplify through L2 normalization.
    # The L20X sweep observed max_abs=8.34e-4 and max_rel=5.46e-3.
    torch.testing.assert_close(fused_bufs[1], base_bufs[1], atol=1e-3, rtol=6e-3)
    torch.testing.assert_close(fused_bufs[2], base_bufs[2], atol=2e-5, rtol=2e-5)


def test_fused_tensor_core_single_qk_cta():
    result = _run_pair(2, 4, 8, 8, variant="tensor_core", num_v_tiles=1, seed=64)
    _, out_base, out_fused, base_bufs, fused_bufs, conv_state, windows, rolled, windows_ref = result
    assert torch.equal(conv_state.transpose(-1, -2), rolled)
    assert torch.equal(windows, windows_ref)
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    for actual, expected in zip(fused_bufs, base_bufs):
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=6e-3)


def test_fused_tensor_core_auto_two_v_tiles():
    result = _run_pair(4, 8, 32, 32, variant="tensor_core", seed=65)
    _, out_base, out_fused, base_bufs, fused_bufs, conv_state, windows, rolled, windows_ref = result
    assert torch.equal(conv_state.transpose(-1, -2), rolled)
    assert torch.equal(windows, windows_ref)
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    for actual, expected in zip(fused_bufs, base_bufs):
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=6e-3)


def test_fused_auto_above_tensor_core_limit_matches_shuffle():
    result = _run_pair(1, 9, 8, 8, variant="auto", seed=67)
    _, out_base, out_fused, base_bufs, fused_bufs, *_ = result
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    for actual, expected in zip(fused_bufs, base_bufs):
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=3e-3)


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core"])
def test_fused_kvbuffer_dynamic_mixed_stride(variant):
    result = _run_pair(2, 4, 8, 16, variant=variant, dynamic_stride=True, seed=66)
    _, out_base, out_fused, base_bufs, fused_bufs, conv_state, windows, rolled, windows_ref = result
    assert torch.equal(conv_state.transpose(-1, -2), rolled)
    assert torch.equal(windows, windows_ref)
    torch.testing.assert_close(out_fused, out_base, atol=3e-2, rtol=2e-2)
    for actual, expected in zip(fused_bufs, base_bufs):
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=3e-3)


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core"])
@pytest.mark.parametrize("accept_mode", ["scalar", "per_request"])
def test_fused_kvbuffer_flush_compatible(accept_mode, variant):
    N, T, H, HV = 3, 6, 8, 16
    inp, _, _, base_bufs, fused_bufs, *_ = _run_pair(N, T, H, HV, variant=variant, seed=71)
    accept_len = 3
    if accept_mode == "per_request":
        accept_len = torch.tensor([1, 3, T], device="cuda", dtype=torch.int64)
    base_state = inp["ssm_states"].clone()
    fused_state = inp["ssm_states"].clone()
    kda_flush_kvbuffer(base_state, inp["cache_indices"], *base_bufs, accept_len=accept_len)
    kda_flush_kvbuffer(fused_state, inp["cache_indices"], *fused_bufs, accept_len=accept_len)
    torch.testing.assert_close(fused_state, base_state, atol=5e-2, rtol=3e-2)


def test_fused_shuffle_requires_complete_ubuffer_triplet():
    inp = _make_inputs(1, 2, 8, 8)
    d_buffer, _, _ = _alloc_ubufs(1, 2, 8)
    with pytest.raises(ValueError, match="must be supplied together"):
        kda_conv_decode_mtp_shuffle_kvbuffer(
            mixed_qkv=inp["mixed_qkv"],
            conv_weight=inp["conv_weight"],
            conv_bias=inp["conv_bias"],
            conv_state=inp["conv_state_native"].transpose(-1, -2).contiguous(),
            conv_state_indices=inp["conv_indices"],
            intermediate_conv_window=torch.empty(inp["pool_size"], 2, inp["D"], 3, device="cuda"),
            intermediate_state_indices=inp["inter_indices"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            ssm_states=inp["ssm_states"],
            cache_indices=inp["cache_indices"],
            scale=K**-0.5,
            T=2,
            num_q_heads=8,
            num_v_heads=8,
            head_k_dim=K,
            head_v_dim=V,
            d_buffer=d_buffer,
        )


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core", "auto"])
def test_fused_kvbuffer_cuda_graph_replay(variant):
    N, T, H, HV = 2, 4, 8, 16
    inp = _make_inputs(N, T, H, HV, seed=81)
    initial_state = inp["conv_state_native"].transpose(-1, -2).contiguous()
    state = initial_state.clone()
    windows = torch.zeros(inp["pool_size"], T, inp["D"], 3, device="cuda")
    bufs = _alloc_ubufs(N, T, HV)
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.copy_(initial_state)
        _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
    graph.replay()
    torch.cuda.synchronize()
    reference = (out.clone(), state.clone(), windows.clone(), *(x.clone() for x in bufs))
    graph.replay()
    torch.cuda.synchronize()
    current = (out, state, windows, *bufs)
    for name, actual, expected in zip(("out", "conv_state", "conv_window", "d", "k", "g"), current, reference):
        assert torch.equal(actual, expected), f"{name} changed across graph replays"


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core"])
def test_fused_kvbuffer_multistream(variant):
    N, T, H, HV = 2, 4, 8, 16
    inputs = [_make_inputs(N, T, H, HV, seed=84 + i) for i in range(2)]
    expected = [_run_pair(N, T, H, HV, variant=variant, seed=84 + i) for i in range(2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    actual = []
    for stream, inp in zip(streams, inputs):
        state = inp["conv_state_native"].transpose(-1, -2).contiguous()
        windows = torch.zeros(inp["pool_size"], T, inp["D"], 3, device="cuda")
        bufs = _alloc_ubufs(N, T, HV)
        out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
        actual.append((out, state, windows, bufs))
    torch.cuda.synchronize()
    for got, ref in zip(actual, expected):
        out, state, windows, bufs = got
        _, _, out_ref, _, bufs_ref, state_ref, windows_ref, *_ = ref
        assert torch.equal(state, state_ref)
        assert torch.equal(windows, windows_ref)
        torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=2e-2)
        for actual_buf, expected_buf in zip(bufs, bufs_ref):
            torch.testing.assert_close(actual_buf, expected_buf, atol=4e-3, rtol=3e-3)


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core"])
def test_fused_kvbuffer_repeated_chain(variant):
    N, T, H, HV = 2, 4, 8, 16
    inp = _make_inputs(N, T, H, HV, seed=88)
    state = inp["conv_state_native"].transpose(-1, -2).contiguous()
    windows = torch.zeros(inp["pool_size"], T, inp["D"], 3, device="cuda")
    bufs = _alloc_ubufs(N, T, HV)
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)

    next_inp = dict(inp, conv_state_native=state.transpose(-1, -2).clone())
    q, k, v, expected_windows, expected_state = _torch_conv(next_inp, N, T, H, HV)
    base_bufs = _alloc_ubufs(N, T, HV)
    base_out = torch.empty_like(out)
    baseline_op = kda_decode_mtp_shuffle_kvbuffer if variant == "shuffle" else kda_decode_mtp_tensor_core_kvbuffer
    baseline_op(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        q=q,
        k=k,
        v=v,
        a=inp["a"],
        b=inp["b"],
        initial_state_source=inp["ssm_states"],
        initial_state_indices=inp["cache_indices"],
        scale=K**-0.5,
        out=base_out,
        d_buffer=base_bufs[0],
        k_buffer=base_bufs[1],
        g_buffer=base_bufs[2],
        lower_bound=-5.0,
    )
    _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
    assert torch.equal(state.transpose(-1, -2), expected_state)
    assert torch.equal(windows, expected_windows)
    torch.testing.assert_close(out, base_out, atol=3e-2, rtol=2e-2)
    for actual_buf, expected_buf in zip(bufs, base_bufs):
        torch.testing.assert_close(actual_buf, expected_buf, atol=4e-3, rtol=6e-3)


@pytest.mark.parametrize("variant", ["shuffle", "tensor_core"])
@pytest.mark.parametrize("T", [4, 8])
def test_fused_kvbuffer_deterministic(T, variant):
    N, H, HV = 4, 8, 16
    inp = _make_inputs(N, T, H, HV, seed=91)
    initial_conv_state = inp["conv_state_native"].transpose(-1, -2).contiguous()
    state = initial_conv_state.clone()
    windows = torch.zeros(inp["pool_size"], T, inp["D"], 3, device="cuda")
    bufs = _alloc_ubufs(N, T, HV)
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
    reference = (out.clone(), state.clone(), windows.clone(), *(x.clone() for x in bufs))
    for iteration in range(int(os.environ.get("KDA_CONV_KVB_DET_ITERS", "100000"))):
        state.copy_(initial_conv_state)
        _run_fused(inp, N, T, H, HV, variant, state, windows, bufs, out)
        current = (out, state, windows, *bufs)
        for name, actual, expected in zip(("out", "conv_state", "conv_window", "d", "k", "g"), current, reference):
            assert torch.equal(actual, expected), f"{name} differs at iteration {iteration}"
