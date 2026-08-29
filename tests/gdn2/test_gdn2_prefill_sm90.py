# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from cula.gdn2 import (
    chunk_gdn2,
    get_sm90_gdn2_backend,
    get_sm90_gdn2_backend_identity,
    is_sm90_gdn2_available,
)
from cula.ops.gdn2.sm90.config import (
    HEAD_SIZE,
    MAX_SEQUENCES,
    SM90_BACKEND_ID,
    SUPPORTED_G_MIN,
    SUPPORTED_Q_HEADS,
    SUPPORTED_V_HEADS,
    VALUE_SIZE,
)
from cula.ops.gdn2.sm90.prefill import _compiled

from .reference import tokenwise_gdn2_reference

_OUTPUT_RTOL = 0.01
_OUTPUT_ATOL = 0.01
_STATE_RTOL = 0.001
_STATE_ATOL = 0.005


@dataclass(frozen=True)
class _Case:
    case_id: str
    lengths: tuple[int, ...]
    value_heads: int
    initial_state: bool
    output_final_state: bool


_CASES = (
    _Case("mha-single-token", (1,), 16, False, False),
    _Case("mha-tail-and-init", (65, 1), 16, True, True),
    _Case("mha-initial-no-final", (65, 63), 16, True, False),
    _Case("mha-production-t1024", (1024,), 16, True, True),
    _Case("mha-t64-short-baseline", (64,), 16, True, True),
    _Case("mha-t65-v64-boundary", (65,), 16, True, True),
    _Case("mha-max-sequences", (1,) * 32, 16, False, True),
    _Case("gva2-packed-tails", (1, 63, 65, 2), 32, False, True),
    _Case("gva4-init", (4,), 64, True, True),
)


def _is_supported_sm90() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


requires_sm90 = pytest.mark.skipif(
    not _is_supported_sm90(),
    reason="requires compute capability 9.0 and nvidia-cutlass-dsl installed",
)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert bool(torch.isfinite(tensor).all()), f"{name} contains NaN or Inf"


def _make_inputs(case: _Case) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(20260727)
    total_tokens = sum(case.lengths)
    q_shape = (total_tokens, SUPPORTED_Q_HEADS, HEAD_SIZE)
    v_shape = (total_tokens, case.value_heads, VALUE_SIZE)

    def _bf16_normal(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).cuda()

    q = _bf16_normal(q_shape, 0.03)
    k = _bf16_normal(q_shape, 0.01)
    v = _bf16_normal(v_shape, 0.1)
    g = (-torch.rand(q_shape, generator=generator, dtype=torch.float32) * 0.05).cuda()
    b = torch.rand(q_shape, generator=generator).to(torch.bfloat16).cuda()
    w = torch.rand(v_shape, generator=generator).to(torch.bfloat16).cuda()
    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    initial_state = None
    if case.initial_state:
        initial_state = (
            torch.randn(
                (
                    len(case.lengths),
                    case.value_heads,
                    VALUE_SIZE,
                    HEAD_SIZE,
                ),
                generator=generator,
            )
            * 0.005
        ).cuda()
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": b,
        "w": w,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
    }


def test_product_identity_contract() -> None:
    assert get_sm90_gdn2_backend() == "dsl"
    assert get_sm90_gdn2_backend_identity() == SM90_BACKEND_ID
    assert SM90_BACKEND_ID == "sm90a_cutedsl_gdn2_prefill_v1"
    assert MAX_SEQUENCES == 32
    assert SUPPORTED_Q_HEADS == 16
    assert SUPPORTED_V_HEADS == (16, 32, 64)


def test_gdn2_has_no_gdn_namespace_dependency() -> None:
    source_root = Path(__file__).resolve().parents[2] / "cula"
    imported_modules: set[str] = set()
    for path in sorted((source_root / "gdn2").rglob("*.py")) + sorted(
        (source_root / "ops" / "gdn2").rglob("*.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)

    forbidden = sorted(
        module
        for module in imported_modules
        if module == "cula.gdn"
        or module.startswith("cula.gdn.")
        or module == "cula.ops.gdn"
        or module.startswith("cula.ops.gdn.")
    )
    assert forbidden == []


@requires_sm90
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
@torch.inference_mode()
def test_product_matches_tokenwise_reference(case: _Case) -> None:
    inputs = _make_inputs(case)
    read_only_before = {name: tensor.detach().clone() for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
    scale = HEAD_SIZE**-0.5
    expected_output, expected_state = tokenwise_gdn2_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        cu_seqlens=inputs["cu_seqlens"],
        initial_state=inputs["initial_state"],
        output_final_state=case.output_final_state,
        scale=scale,
    )

    output_storage = torch.full(
        (sum(case.lengths) + 2, case.value_heads, VALUE_SIZE),
        -123.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = output_storage[1:-1]
    state_storage = None
    output_state = None
    if case.output_final_state:
        state_storage = torch.full(
            (
                len(case.lengths) + 2,
                case.value_heads,
                VALUE_SIZE,
                HEAD_SIZE,
            ),
            -987654.0,
            dtype=torch.float32,
            device="cuda",
        )
        output_state = state_storage[1:-1]

    actual = chunk_gdn2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        initial_state=inputs["initial_state"],
        output_final_state=case.output_final_state,
        cu_seqlens=inputs["cu_seqlens"],
        scale=scale,
        output=output,
        output_state=output_state,
    )
    torch.cuda.synchronize()

    if case.output_final_state:
        actual_output, actual_state = actual
        assert actual_state is output_state
    else:
        actual_output = actual
        actual_state = None
    assert actual_output is output
    _assert_finite("reference output", expected_output)
    _assert_finite("product output", actual_output)
    torch.testing.assert_close(
        actual_output,
        expected_output.to(dtype=actual_output.dtype),
        rtol=_OUTPUT_RTOL,
        atol=_OUTPUT_ATOL,
    )
    if expected_state is not None:
        assert actual_state is not None
        _assert_finite("reference final state", expected_state)
        _assert_finite("product final state", actual_state)
        torch.testing.assert_close(
            actual_state,
            expected_state,
            rtol=_STATE_RTOL,
            atol=_STATE_ATOL,
        )

    first_output = actual_output.detach().clone()
    first_state = None if actual_state is None else actual_state.detach().clone()
    repeated = chunk_gdn2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        initial_state=inputs["initial_state"],
        output_final_state=case.output_final_state,
        cu_seqlens=inputs["cu_seqlens"],
        scale=scale,
        output=output,
        output_state=output_state,
    )
    torch.cuda.synchronize()
    if case.output_final_state:
        repeated_output, repeated_state = repeated
        assert repeated_state is output_state
        assert first_state is not None
        assert torch.equal(repeated_state, first_state)
    else:
        repeated_output = repeated
    assert repeated_output is output
    assert torch.equal(repeated_output, first_output)
    _assert_finite("repeated product output", repeated_output)

    assert bool((output_storage[0] == -123.0).all())
    assert bool((output_storage[-1] == -123.0).all())
    if state_storage is not None:
        assert bool((state_storage[0] == -987654.0).all())
        assert bool((state_storage[-1] == -987654.0).all())
    for name, before in read_only_before.items():
        torch.testing.assert_close(inputs[name], before, rtol=0, atol=0)


@requires_sm90
@torch.inference_mode()
def test_unsupported_metadata_fails_before_compile() -> None:
    def _attempt(*, q_heads: int, value_heads: int, sequences: int) -> None:
        q = torch.zeros(sequences, q_heads, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros_like(q)
        v = torch.zeros(
            sequences,
            value_heads,
            VALUE_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        g = torch.zeros_like(q, dtype=torch.float32)
        b = torch.zeros_like(q)
        w = torch.zeros_like(v)
        cu_seqlens = torch.arange(sequences + 1, dtype=torch.int64, device="cuda")
        chunk_gdn2(q, k, v, g, b, w, cu_seqlens=cu_seqlens)

    compile_count = len(_compiled)
    with pytest.raises(NotImplementedError, match="Hq=16"):
        _attempt(q_heads=8, value_heads=16, sequences=1)
    with pytest.raises(NotImplementedError, match="Hv in"):
        _attempt(q_heads=16, value_heads=48, sequences=1)
    with pytest.raises(NotImplementedError, match="1 <= N <= 32"):
        _attempt(q_heads=16, value_heads=16, sequences=33)
    assert len(_compiled) == compile_count


@requires_sm90
def test_availability_on_current_device() -> None:
    assert is_sm90_gdn2_available()
    assert is_sm90_gdn2_available(torch.cuda.current_device())
    assert not is_sm90_gdn2_available("cpu")


def _expected_compile_key(
    *,
    num_sequences: int,
    total_tokens: int,
    value_heads: int,
    has_initial_state: bool,
    store_final_state: bool,
) -> tuple[int, int, bool, bool, bool, bool]:
    """Mirror the documented compile-cache key derivation.

    Keep in sync with ``_compile`` in ``cula.ops.gdn2.sm90.prefill`` and the
    "State modes and dynamic compilation" section of
    ``docs/gdn2_sm90_pipeline.md``; this test is the contract regression
    guard for both.
    """

    use_n1_hv16_v64 = (
        num_sequences == 1 and value_heads == 16 and has_initial_state and store_final_state and total_tokens > 64
    )
    retain_final_tail = store_final_state and not (num_sequences == 1 and total_tokens <= 64)
    return (
        torch.cuda.current_device(),
        value_heads,
        has_initial_state,
        store_final_state,
        use_n1_hv16_v64,
        retain_final_tail,
    )


@requires_sm90
@torch.inference_mode()
def test_compile_cache_boundaries() -> None:
    """The cache key follows the documented route boundaries exactly.

    Moving across ``N=1,T<=64`` / ``N=1,T>64`` / ``N>1`` compiles one new
    specialization each for the final-state mode, while ``T``/``N`` stay
    dynamic within a route and while ``output_final_state=False`` collapses
    every shape onto one specialization.
    """

    from cula.ops.gdn2.sm90 import prefill as sm90_prefill

    def _run(case: _Case) -> tuple[int, int, bool, bool, bool, bool]:
        inputs = _make_inputs(case)
        before_keys = set(sm90_prefill._compiled)
        expected_key = _expected_compile_key(
            num_sequences=len(case.lengths),
            total_tokens=sum(case.lengths),
            value_heads=case.value_heads,
            has_initial_state=case.initial_state,
            store_final_state=case.output_final_state,
        )
        chunk_gdn2(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["b"],
            inputs["w"],
            initial_state=inputs["initial_state"],
            output_final_state=case.output_final_state,
            cu_seqlens=inputs["cu_seqlens"],
        )
        torch.cuda.synchronize()
        after_keys = set(sm90_prefill._compiled)
        assert expected_key in after_keys
        expected_growth = 0 if expected_key in before_keys else 1
        assert len(after_keys) == len(before_keys) + expected_growth
        return expected_key

    def _case(lengths: tuple[int, ...], final_state: bool) -> _Case:
        return _Case(
            case_id=f"cache-{len(lengths)}seq-{sum(lengths)}tok-{final_state}",
            lengths=lengths,
            value_heads=16,
            initial_state=True,
            output_final_state=final_state,
        )

    # Final-state mode: three distinct routes across the documented
    # boundaries...
    key_short = _run(_case((64,), True))
    key_n1_long = _run(_case((65,), True))
    key_packed = _run(_case((40, 25), True))
    assert len({key_short, key_n1_long, key_packed}) == 3

    # ...and T/N stay dynamic inside each route: repeats and different
    # shapes on the same route map to the same key (asserted inside _run
    # via expected_growth == 0).
    assert _run(_case((32,), True)) == key_short
    assert _run(_case((1024,), True)) == key_n1_long
    assert _run(_case((30, 20, 14), True)) == key_packed

    # No-final-state mode: every boundary collapses onto one key.
    key_no_final = _run(_case((64,), False))
    assert _run(_case((65,), False)) == key_no_final
    assert _run(_case((40, 25), False)) == key_no_final


def test_cutlass_dsl_version_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The availability gate enforces the one documented version range."""

    from cula.gdn2 import prefill as gdn2_prefill

    def _probe(version: str | None) -> str | None:
        monkeypatch.setattr(
            gdn2_prefill,
            "_installed_cutlass_dsl_version",
            lambda: version,
        )
        gdn2_prefill._supported_cutlass_dsl_version.cache_clear()
        try:
            return gdn2_prefill._supported_cutlass_dsl_version()
        finally:
            gdn2_prefill._supported_cutlass_dsl_version.cache_clear()

    # Endpoints are exercised on H20; interior versions follow the vendor's
    # release ordering. Local/post releases of a supported version stay
    # supported; pre-releases and unparseable strings do not.
    supported = ("4.5.1", "4.6.0", "4.6.2", "4.5.1+cu13", "4.5.1.post1")
    unsupported = (
        None,
        "4.4.2",
        "4.5.0",
        "4.7.0",
        "5.0.0",
        "4.5",
        "4.6.0rc1",
        "not-a-version",
    )
    for version in supported:
        assert _probe(version) == version, version
    for version in unsupported:
        assert _probe(version) is None, version


@dataclass(frozen=True)
class _DecayCase:
    case_id: str
    lengths: tuple[int, ...]
    decay: float | None  # None -> mixed uniform in [SUPPORTED_G_MIN, 0]
    gate_mode: str  # "random" | "zeros" | "ones"


_DECAY_CASES = (
    _DecayCase("uniform-g1", (150,), -1.0, "random"),
    _DecayCase("uniform-g2", (129,), -2.0, "random"),
    _DecayCase("uniform-g5-bound", (150,), SUPPORTED_G_MIN, "random"),
    _DecayCase("mixed-strong-decay", (65, 40), None, "random"),
    _DecayCase("gate-endpoint-zeros", (100,), -1.0, "zeros"),
    _DecayCase("gate-endpoint-ones", (100,), -1.0, "ones"),
)


def _make_decay_inputs(case: _DecayCase) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(20260814)
    total_tokens = sum(case.lengths)
    q_shape = (total_tokens, SUPPORTED_Q_HEADS, HEAD_SIZE)
    v_shape = (total_tokens, 16, VALUE_SIZE)

    def _bf16_normal(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).cuda()

    q = _bf16_normal(q_shape, 0.03)
    k = _bf16_normal(q_shape, 0.01)
    v = _bf16_normal(v_shape, 0.1)
    if case.decay is None:
        g = (torch.rand(q_shape, generator=generator, dtype=torch.float32) * SUPPORTED_G_MIN).cuda()
    else:
        g = torch.full(q_shape, case.decay, dtype=torch.float32).cuda()
    if case.gate_mode == "zeros":
        b = torch.zeros(q_shape, dtype=torch.bfloat16).cuda()
    elif case.gate_mode == "ones":
        b = torch.ones(q_shape, dtype=torch.bfloat16).cuda()
    else:
        b = torch.rand(q_shape, generator=generator).to(torch.bfloat16).cuda()
    w = torch.rand(v_shape, generator=generator).to(torch.bfloat16).cuda()
    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    initial_state = (
        torch.randn(
            (len(case.lengths), 16, VALUE_SIZE, HEAD_SIZE),
            generator=generator,
        )
        * 0.005
    ).cuda()
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": b,
        "w": w,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
    }


@requires_sm90
@pytest.mark.parametrize("case", _DECAY_CASES, ids=lambda case: case.case_id)
@torch.inference_mode()
def test_adversarial_decay_matches_tokenwise_reference(case: _DecayCase) -> None:
    """Strong in-contract decays stay finite and match the exact recurrence.

    The released chunk-start factorization overflowed FP32 for uniform
    ``g <= -1.5`` (64-token channel prefixes beyond ~88.7 nats) and poisoned
    the rest of the sequence with NaN. These cases pin the blockwise-rebased
    factorization across the documented ``[-5, 0]`` contract, the erase-gate
    endpoints, and the ``g = -5`` boundary itself.
    """

    inputs = _make_decay_inputs(case)
    scale = HEAD_SIZE**-0.5
    expected_output, expected_state = tokenwise_gdn2_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        cu_seqlens=inputs["cu_seqlens"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        scale=scale,
    )
    actual_output, actual_state = chunk_gdn2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        scale=scale,
        validate_inputs=True,
    )
    torch.cuda.synchronize()
    _assert_finite("reference output", expected_output)
    _assert_finite("product output", actual_output)
    _assert_finite("reference final state", expected_state)
    _assert_finite("product final state", actual_state)
    torch.testing.assert_close(
        actual_output,
        expected_output.to(dtype=actual_output.dtype),
        rtol=_OUTPUT_RTOL,
        atol=_OUTPUT_ATOL,
    )
    torch.testing.assert_close(
        actual_state,
        expected_state,
        rtol=_STATE_RTOL,
        atol=_STATE_ATOL,
    )


@requires_sm90
@torch.inference_mode()
def test_decay_below_bound_rejected() -> None:
    """validate_inputs enforces the documented elementwise g >= -5 bound."""

    case = _DecayCase("reject", (65,), SUPPORTED_G_MIN, "random")
    inputs = _make_decay_inputs(case)
    inputs["g"][3, 5, 7] = SUPPORTED_G_MIN - 0.5
    with pytest.raises(ValueError, match="elementwise >="):
        chunk_gdn2(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["b"],
            inputs["w"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            cu_seqlens=inputs["cu_seqlens"],
            validate_inputs=True,
        )
