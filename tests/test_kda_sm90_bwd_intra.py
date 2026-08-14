# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_bwd_intra
from fla.ops.utils import prepare_chunk_indices

import cula.kda.chunk_intra as chunk_intra_module
from cula.kda.chunk_intra import _is_mma_bwd_intra_supported
from cula.kda.chunk_intra import chunk_kda_bwd_intra as cula_bwd_intra
from cula.ops.kda.sm90.bwd_intra import kda_bwd_intra_mma

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_SUPPORTED_CAPABILITIES = {(9, 0), (10, 0), (10, 3)}
_LIMITS = (8.0e-3, 8.0e-3, 2.0e-2, 2.0e-2)


def _require_supported_device() -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in _SUPPORTED_CAPABILITIES:
        pytest.skip(f"mma.sync kernel does not support SM{capability[0]}{capability[1]}")


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    rmse = (actual - expected).square().mean().sqrt()
    return (rmse / (expected.square().mean().sqrt() + 1e-8)).item()


def _make_inputs(lengths: list[int], heads: int = 4):
    torch.manual_seed(42)
    total = sum(lengths)
    dim, chunk_size = 128, 64
    device = torch.device("cuda")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens.to(torch.long), chunk_size).to(torch.int32).contiguous()

    q = torch.randn(1, total, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = torch.randn(1, total, heads, dim, device=device, dtype=torch.float32) / 10
    beta = torch.randn(1, total, heads, device=device, dtype=torch.bfloat16)
    d_aq = torch.randn(1, total, heads, chunk_size, device=device, dtype=torch.float32)
    d_ak = torch.randn_like(d_aq)
    dq = torch.randn(1, total, heads, dim, device=device, dtype=torch.float32)
    dk = torch.randn_like(dq)
    db = torch.randn(1, total, heads, device=device, dtype=torch.float32)
    dg = torch.randn_like(dq)
    return q, k, g, beta, d_aq, d_ak, dq, dk, db, dg, cu_seqlens, chunk_indices


def _assert_matches_fla(inputs) -> None:
    reference = fla_bwd_intra(*inputs, chunk_size=64, safe_gate=True)
    actual = kda_bwd_intra_mma(*inputs, chunk_size=64)
    torch.cuda.synchronize()
    errors = tuple(_relative_rmse(got, ref) for got, ref in zip(actual, reference))
    assert all(error < limit for error, limit in zip(errors, _LIMITS)), (errors, _LIMITS)
    assert tuple(value.dtype for value in actual) == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    )
    assert all(torch.isfinite(value).all() for value in actual)


@pytest.mark.kda_fast
@pytest.mark.parametrize("lengths", [[64], [65], [64, 127, 129]])
def test_kda_bwd_intra_mma_matches_fla(lengths: list[int]):
    _require_supported_device()
    _assert_matches_fla(_make_inputs(lengths))


@pytest.mark.kda_fast
def test_kda_bwd_intra_mma_is_deterministic():
    _require_supported_device()
    inputs = _make_inputs([64, 65], heads=2)
    expected = tuple(value.clone() for value in kda_bwd_intra_mma(*inputs, chunk_size=64))
    for _ in range(20):
        actual = kda_bwd_intra_mma(*inputs, chunk_size=64)
        assert all(torch.equal(got, ref) for got, ref in zip(actual, expected))


@pytest.mark.kda_fast
def test_kda_bwd_intra_dispatch_matches_direct_kernel():
    _require_supported_device()
    inputs = _make_inputs([64, 65], heads=2)
    direct = kda_bwd_intra_mma(*inputs, chunk_size=64)
    dispatched = cula_bwd_intra(*inputs, chunk_size=64, safe_gate=True)
    torch.cuda.synchronize()
    assert all(torch.equal(got, ref) for got, ref in zip(dispatched, direct))


@pytest.mark.kda_fast
def test_kda_bwd_intra_dispatch_handles_dense_batches():
    _require_supported_device()
    flat_inputs = _make_inputs([70, 70], heads=2)
    reference = fla_bwd_intra(*flat_inputs, chunk_size=64, safe_gate=True)
    batched = tuple(value.reshape(2, 70, *value.shape[2:]) for value in flat_inputs[:10])
    actual = cula_bwd_intra(*batched, chunk_size=64, safe_gate=True)
    torch.cuda.synchronize()
    errors = tuple(_relative_rmse(got.reshape_as(ref), ref) for got, ref in zip(actual, reference))
    assert all(error < limit for error, limit in zip(errors, _LIMITS)), errors


def test_kda_bwd_intra_float_beta_falls_back_to_triton(monkeypatch: pytest.MonkeyPatch):
    _require_supported_device()
    inputs = list(_make_inputs([64], heads=1))
    inputs[3] = inputs[3].float()
    sentinel = object()

    def fake_triton(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(chunk_intra_module, "_chunk_kda_bwd_intra_triton", fake_triton)
    assert cula_bwd_intra(*inputs, chunk_size=64, safe_gate=True) is sentinel


def test_kda_bwd_intra_support_predicate_rejects_cpu_inputs():
    tensors = _make_inputs([64], heads=1)[:10]
    cpu_tensors = tuple(tensor.cpu() for tensor in tensors)
    assert not _is_mma_bwd_intra_supported(*cpu_tensors, chunk_size=64, safe_gate=True)


@pytest.mark.kda_fast
def test_kda_bwd_intra_mma_rejects_empty_chunk_indices():
    _require_supported_device()
    inputs = list(_make_inputs([64], heads=1))
    inputs[-1] = torch.empty((0, 2), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="chunk_indices must contain at least one chunk"):
        kda_bwd_intra_mma(*inputs, chunk_size=64)
