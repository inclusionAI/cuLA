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

import pytest
import torch
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_bwd_intra
from fla.ops.utils import prepare_chunk_indices

from cula.kda.chunk_intra import chunk_kda_bwd_intra as cula_bwd_intra
from cula.ops.kda_bwd_intra_mma import kda_bwd_intra_mma


def _rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )


def _make_inputs(
    beta_dtype: torch.dtype,
    lengths: list[int] | None = None,
):
    torch.manual_seed(42)
    if lengths is None:
        lengths = [1, 7, 17, 63, 64, 65, 129]
    total = sum(lengths)
    heads, dim, chunk_size = 8, 128, 64
    device = torch.device("cuda")
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    chunk_indices = prepare_chunk_indices(
        cu_seqlens.to(torch.long), chunk_size
    ).to(torch.int32)

    q = torch.randn(
        1, total, heads, dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    g = torch.randn(
        1, total, heads, dim, device=device, dtype=torch.float32
    ) / 10
    beta = torch.randn(
        1, total, heads, device=device, dtype=beta_dtype
    )
    d_aq = torch.randn(
        1, total, heads, chunk_size, device=device, dtype=torch.float32
    )
    d_ak = torch.randn_like(d_aq)
    dq = torch.randn(
        1, total, heads, dim, device=device, dtype=torch.float32
    )
    dk = torch.randn_like(dq)
    db = torch.randn(1, total, heads, device=device)
    dg = torch.randn_like(dq)
    return (
        q,
        k,
        g,
        beta,
        d_aq,
        d_ak,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        chunk_size,
    )


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_kda_bwd_intra_mma_varlen_accuracy(beta_dtype: torch.dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip(f"mma.sync kernel does not support SM{major}{minor}")

    inputs = _make_inputs(beta_dtype)
    *kernel_inputs, chunk_size = inputs
    reference = fla_bwd_intra(
        *kernel_inputs,
        chunk_size=chunk_size,
        safe_gate=True,
    )
    q, k, _, _, _, _, _, _, db, dg, _, _ = kernel_inputs
    outputs = (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(db),
        torch.empty_like(dg),
    )
    actual = kda_bwd_intra_mma(
        *kernel_inputs,
        *outputs,
        chunk_size,
    )
    torch.cuda.synchronize()

    limits = (2.0e-4, 3.0e-4, 3.0e-5, 3.0e-6)
    errors = tuple(_rmse(got, ref) for got, ref in zip(actual, reference))
    assert all(error < limit for error, limit in zip(errors, limits)), (
        f"mma.sync CuTeDSL RMSE {errors} exceeds limits {limits}"
    )
    assert all(torch.isfinite(value).all() for value in actual)
    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == torch.bfloat16
    assert actual[2].dtype == torch.float32
    assert actual[3].dtype == torch.float32


def test_kda_bwd_intra_mma_cache_distinguishes_sequence_layouts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip(f"mma.sync kernel does not support SM{major}{minor}")

    for lengths in ([128], [64, 64]):
        inputs = _make_inputs(torch.bfloat16, lengths)
        *kernel_inputs, chunk_size = inputs
        reference = fla_bwd_intra(
            *kernel_inputs,
            chunk_size=chunk_size,
            safe_gate=True,
        )
        q, k, _, _, _, _, _, _, db, dg, _, _ = kernel_inputs
        actual = kda_bwd_intra_mma(
            *kernel_inputs,
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(db),
            torch.empty_like(dg),
            chunk_size,
        )
        torch.cuda.synchronize()
        errors = tuple(
            _rmse(got, ref) for got, ref in zip(actual, reference)
        )
        limits = (2.0e-4, 3.0e-4, 3.0e-5, 3.0e-6)
        assert all(
            error < limit for error, limit in zip(errors, limits)
        ), (lengths, errors)


def test_sm90_cutedsl_dispatch_uses_portable_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 dispatch test")

    inputs = _make_inputs(torch.bfloat16)
    *kernel_inputs, chunk_size = inputs
    monkeypatch.setenv("CULA_KDA_BWD_INTRA_BACKEND", "cutedsl")

    direct_outputs = (
        torch.empty_like(kernel_inputs[0]),
        torch.empty_like(kernel_inputs[1]),
        torch.empty_like(kernel_inputs[8]),
        torch.empty_like(kernel_inputs[9]),
    )
    direct = kda_bwd_intra_mma(
        *kernel_inputs,
        *direct_outputs,
        chunk_size,
    )
    dispatched = cula_bwd_intra(
        *kernel_inputs,
        chunk_size=chunk_size,
        safe_gate=True,
    )
    torch.cuda.synchronize()

    for expected, actual in zip(direct, dispatched):
        assert torch.equal(expected, actual)


def test_sm90_cutedsl_dispatch_flattens_uniform_batches(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 dispatch test")

    inputs = _make_inputs(torch.bfloat16, [70, 70])
    *flat_inputs, chunk_size = inputs
    reference = fla_bwd_intra(
        *flat_inputs,
        chunk_size=chunk_size,
        safe_gate=True,
    )
    batched_inputs = [
        value.reshape(2, 70, *value.shape[2:])
        for value in flat_inputs[:10]
    ]
    batched_inputs.extend(flat_inputs[10:])
    monkeypatch.setenv("CULA_KDA_BWD_INTRA_BACKEND", "cutedsl")

    actual = cula_bwd_intra(
        *batched_inputs,
        chunk_size=chunk_size,
        safe_gate=True,
    )
    torch.cuda.synchronize()

    errors = tuple(
        _rmse(got.reshape_as(ref), ref)
        for got, ref in zip(actual, reference)
    )
    limits = (2.0e-4, 3.0e-4, 3.0e-5, 3.0e-6)
    assert all(
        error < limit for error, limit in zip(errors, limits)
    ), errors
