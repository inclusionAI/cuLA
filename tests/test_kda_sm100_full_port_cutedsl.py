# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

from cula.ops.kda.sm100.full_port import chunk_kda_fwd_full_sm100_from_gk
from cula.ops.kda.sm100.intra_fused import chunk_kda_fwd_intra_sm100_from_gk
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd

pytestmark = pytest.mark.cula_slow


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


def _inputs(seed, h_qk=4, h_v=8):
    torch.manual_seed(seed)
    batch, length, dim = 2, 512, 128
    q = torch.randn(batch, length, h_qk, dim, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(batch, length, h_v, dim, dtype=torch.bfloat16, device="cuda") * 0.1
    gk = torch.randn(batch, length, h_v, dim, dtype=torch.float32, device="cuda") * 0.02
    beta = torch.rand(batch, length, h_v, dtype=torch.bfloat16, device="cuda")
    return q, k, v, gk, beta


def _separate(inputs):
    q, k, v, gk, beta = inputs
    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(q, k, gk, beta)
    w, u, qg, kg = recompute_w_u_fwd(k, v, beta, Akk, gk)
    return w, u, qg, kg, Aqk, Akk


def _assert_bitwise(expected, actual):
    for reference, observed in zip(expected, actual):
        if reference is None:
            assert observed is None
        else:
            assert torch.equal(reference, observed)


def test_full_port_matches_separate_exact_path():
    _requires_sm100()
    inputs = _inputs(17)
    reference = _separate(inputs)
    actual = chunk_kda_fwd_full_sm100_from_gk(*inputs, 128**-0.5)
    _assert_bitwise(reference, actual)


def test_full_port_keeps_live_outputs_disjoint():
    _requires_sm100()
    inputs = _inputs(18)
    first = chunk_kda_fwd_full_sm100_from_gk(*inputs, 128**-0.5)
    second = chunk_kda_fwd_full_sm100_from_gk(*inputs, 128**-0.5)
    for lhs, rhs in zip(first, second):
        if lhs is not None:
            assert lhs.data_ptr() != rhs.data_ptr()
    _assert_bitwise(first, second)


def test_full_port_isolates_concurrent_streams():
    _requires_sm100()
    inputs_a = _inputs(19)
    inputs_b = _inputs(20)
    reference_a = _separate(inputs_a)
    reference_b = _separate(inputs_b)
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    with torch.cuda.stream(stream_a):
        actual_a = chunk_kda_fwd_full_sm100_from_gk(*inputs_a, 128**-0.5)
    with torch.cuda.stream(stream_b):
        actual_b = chunk_kda_fwd_full_sm100_from_gk(*inputs_b, 128**-0.5)
    stream_a.synchronize()
    stream_b.synchronize()
    _assert_bitwise(reference_a, actual_a)
    _assert_bitwise(reference_b, actual_b)


def test_full_port_does_not_retain_per_call_outputs():
    _requires_sm100()
    inputs = _inputs(21)
    for _ in range(5):
        chunk_kda_fwd_full_sm100_from_gk(*inputs, 128**-0.5)
    torch.cuda.synchronize()
    gc.collect()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(100):
        chunk_kda_fwd_full_sm100_from_gk(*inputs, 128**-0.5)
    torch.cuda.synchronize()
    gc.collect()
    allocated_after = torch.cuda.memory_allocated()
    assert allocated_after <= allocated_before + 1024 * 1024
