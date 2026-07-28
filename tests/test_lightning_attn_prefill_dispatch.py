# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from setuptools import find_packages

from cula.ops.lightning import prefill as dispatch

REPO_ROOT = Path(__file__).resolve().parents[1]
SM90_PATH = REPO_ROOT / "cula/ops/lightning/prefill_sm90.py"
DISPATCH_PATH = REPO_ROOT / "cula/ops/lightning/prefill.py"
PUBLIC_INIT_PATH = REPO_ROOT / "cula/lightning/__init__.py"


def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_sm90_dispatch_imports_only_the_selected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    imports: list[str] = []
    identity_calls: list[dict] = []

    def identity(**kwargs):
        identity_calls.append(kwargs)
        return "sm90-exact"

    sm90 = SimpleNamespace(get_sm90_lightning_attn_prefill_backend_identity=identity)

    monkeypatch.setattr(dispatch, "_device_capability", lambda _: (9, 0))

    def import_module(name: str):
        imports.append(name)
        if name != "cula.ops.lightning.prefill_sm90":
            raise AssertionError(f"unexpected fallback import: {name}")
        return sm90

    monkeypatch.setattr(dispatch.importlib, "import_module", import_module)
    backend, module = dispatch._backend_module(torch.device("cuda:0"))

    assert backend == "sm90"
    assert module is sm90
    assert imports == ["cula.ops.lightning.prefill_sm90"]
    assert dispatch.get_lightning_attn_prefill_backend_identity(torch.device("cuda:0")) == "sm90-exact"
    assert identity_calls == [{"varlen": False, "persistent": False}]


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_blackwell_dispatch_remains_the_existing_sm100_module(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
) -> None:
    sm100 = object()
    imports: list[str] = []
    monkeypatch.setattr(dispatch, "_device_capability", lambda _: capability)

    def import_module(name: str):
        imports.append(name)
        assert name == "cula.ops.lightning.prefill_sm100"
        return sm100

    monkeypatch.setattr(dispatch.importlib, "import_module", import_module)
    backend, module = dispatch._backend_module(torch.device("cuda:0"))

    assert (backend, module) == ("sm100", sm100)
    assert imports == ["cula.ops.lightning.prefill_sm100"]


def test_backend_identity_reports_the_selected_execution_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dispatch, "_backend_module", lambda _: ("sm100", object()))
    device = torch.device("cuda:0")

    assert dispatch.get_lightning_attn_prefill_backend_identity(device) == dispatch.SM100_FIXED_BACKEND_IDENTITY
    assert (
        dispatch.get_lightning_attn_prefill_backend_identity(device, varlen=True)
        == dispatch.SM100_VARLEN_PERSISTENT_BACKEND_IDENTITY
    )
    assert (
        dispatch.get_lightning_attn_prefill_backend_identity(
            device,
            varlen=True,
            persistent=False,
        )
        == dispatch.SM100_VARLEN_NONPERSISTENT_BACKEND_IDENTITY
    )


def test_sm90_backend_identity_resolves_default_and_explicit_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def identity(**kwargs):
        calls.append(kwargs)
        return "sm90-exact"

    module = SimpleNamespace(get_sm90_lightning_attn_prefill_backend_identity=identity)
    monkeypatch.setattr(dispatch, "_backend_module", lambda _: ("sm90", module))
    device = torch.device("cuda:0")

    for persistent in (None, True, False):
        assert (
            dispatch.get_lightning_attn_prefill_backend_identity(
                device,
                varlen=True,
                persistent=persistent,
            )
            == "sm90-exact"
        )

    assert calls == [
        {"varlen": True, "persistent": False},
        {"varlen": True, "persistent": True},
        {"varlen": True, "persistent": False},
    ]


@pytest.mark.parametrize(
    ("backend", "persistent", "expected"),
    [
        ("sm90", None, False),
        ("sm100", None, True),
        ("sm90", True, True),
        ("sm100", False, False),
    ],
)
def test_packed_dispatch_resolves_architecture_policy(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    persistent: bool | None,
    expected: bool,
) -> None:
    calls: list[dict] = []

    def varlen(*args, **kwargs):
        calls.append(kwargs)
        return "output", "state"

    module = SimpleNamespace(lightning_attn_fwd_varlen=varlen)
    monkeypatch.setattr(dispatch, "_backend_module", lambda _: (backend, module))
    tensors = (object(),) * 5

    assert dispatch.lightning_attn_fwd_varlen(*tensors, persistent=persistent) == ("output", "state")
    assert calls == [
        {
            "scale": 1.0,
            "state_pool": None,
            "initial_state_indices": None,
            "chunk_size": 64,
            "persistent": expected,
        }
    ]


@pytest.mark.parametrize("persistent", [0, 1, "persistent", object()])
def test_packed_dispatch_rejects_invalid_scheduler_policy(
    monkeypatch: pytest.MonkeyPatch,
    persistent: object,
) -> None:
    module = SimpleNamespace(lightning_attn_fwd_varlen=lambda *_args, **_kwargs: pytest.fail("unexpected launch"))
    monkeypatch.setattr(dispatch, "_backend_module", lambda _: ("sm90", module))

    with pytest.raises(TypeError, match="persistent must be boolean or None"):
        dispatch.lightning_attn_fwd_varlen(*(object(),) * 5, persistent=persistent)


def test_unsupported_device_is_a_hard_error_before_backend_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda name: pytest.fail(f"unsupported dispatch imported {name}"),
    )

    with pytest.raises(RuntimeError, match="supports SM90, SM100, and SM103"):
        dispatch._backend_module(torch.device("cuda:0"))
    with pytest.raises(ValueError, match="requires a CUDA device"):
        dispatch._backend_module(torch.device("cpu"))


def test_public_functions_delegate_with_all_semantic_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def fixed(*args, **kwargs):
        calls.append(("fixed", args, kwargs))
        return "fixed-output", None

    def varlen(*args, **kwargs):
        calls.append(("varlen", args, kwargs))
        return "varlen-output", "state-pool"

    module = SimpleNamespace(lightning_attn_fwd=fixed, lightning_attn_fwd_varlen=varlen)
    monkeypatch.setattr(dispatch, "_backend_module", lambda _: ("sm90", module))
    q, k, v, decay, cu, state, indices = (object() for _ in range(7))

    assert dispatch.lightning_attn_fwd(
        q,
        k,
        v,
        decay,
        scale=0.75,
        initial_state=state,
        output_final_state=True,
        chunk_size=64,
    ) == ("fixed-output", None)
    assert dispatch.lightning_attn_fwd_varlen(
        q,
        k,
        v,
        decay,
        cu,
        scale=0.5,
        state_pool=state,
        initial_state_indices=indices,
        chunk_size=64,
        persistent=False,
    ) == ("varlen-output", "state-pool")

    assert calls[0][2] == {
        "scale": 0.75,
        "initial_state": state,
        "output_final_state": True,
        "chunk_size": 64,
    }
    assert calls[1][2] == {
        "scale": 0.5,
        "state_pool": state,
        "initial_state_indices": indices,
        "chunk_size": 64,
        "persistent": False,
    }


def test_sm90_public_wrapper_has_direct_tvm_ffi_and_exact_cache_keys() -> None:
    source, tree = _parse(SM90_PATH)
    compile(source, str(SM90_PATH), "exec")
    fixed_compile = ast.unparse(_function(tree, "_compile_fixed_variant"))
    packed_compile = ast.unparse(_function(tree, "_compile_varlen_variant"))
    fixed_cache = ast.unparse(_function(tree, "_get_compiled_fixed_variant"))
    packed_cache = ast.unparse(_function(tree, "_get_compiled_varlen_variant"))

    assert fixed_compile.count("cute.EnableTVMFFI(True)") == 1
    assert packed_compile.count("cute.EnableTVMFFI(True)") == 1
    assert fixed_compile.count("make_fake_stream(use_tvm_ffi_env_stream=True)") == 1
    assert packed_compile.count("make_fake_stream(use_tvm_ffi_env_stream=True)") == 1
    assert "scale" not in fixed_cache
    assert "scale" not in packed_cache
    assert (
        "key = (batch_size, sequence_length, qk_heads, value_heads, decay_heads, has_initial_state, output_final_state)"
        in fixed_cache
    )
    assert (
        "key = (total_length, qk_heads, value_heads, decay_heads, num_sequences, state_pool_size, persistent, persistent_ctas)"
    ) in packed_cache


def test_sm90_public_metadata_validation_precedes_compile_and_launch() -> None:
    _, tree = _parse(SM90_PATH)
    fixed = ast.unparse(_function(tree, "lightning_attn_fwd"))
    packed = ast.unparse(_function(tree, "lightning_attn_fwd_varlen"))
    packed_validation = ast.unparse(_function(tree, "_validate_varlen_inputs"))

    assert fixed.index("_require_chunk_size(chunk_size)") < fixed.index("_get_compiled_fixed_variant")
    assert fixed.index("_validate_fixed_inputs") < fixed.index("_get_compiled_fixed_variant")
    assert "if not isinstance(output_final_state, bool)" in fixed
    assert packed.index("_require_chunk_size(chunk_size)") < packed.index("_get_compiled_varlen_variant")
    assert packed.index("_validate_varlen_inputs") < packed.index("_get_compiled_varlen_variant")
    assert "if not isinstance(persistent, bool)" in packed
    assert "torch.arange(N, dtype=torch.int32, device=Q.device)" in packed
    assert "initial_state_indices is None and state_pool.shape[0] < N" in packed_validation
    assert "state_pool must contain at least N slots" in packed_validation


def test_public_default_path_has_no_dispatcher_visible_host_sync() -> None:
    dispatch_source, _ = _parse(DISPATCH_PATH)
    sm90_source, _ = _parse(SM90_PATH)
    combined = dispatch_source + "\n" + sm90_source

    for forbidden in (
        ".item(",
        ".cpu(",
        "torch.cuda.synchronize",
        "from_dlpack",
        "current_stream",
    ):
        assert forbidden not in combined
    assert "compiled(\n        Q," in sm90_source
    assert "torch.cuda.get_device_capability(device)" in dispatch_source


def test_public_import_and_package_discovery_cover_sm90_without_eager_kernel_import() -> None:
    public_source = PUBLIC_INIT_PATH.read_text(encoding="utf-8")
    packages = find_packages(include=["cula", "cula.*"])

    assert "from cula.ops.lightning.prefill import" in public_source
    assert "get_lightning_attn_prefill_backend_identity" in public_source
    assert "from cula.ops.lightning.prefill_sm100 import (" not in public_source
    assert "cula.ops.lightning.sm90" in packages
    assert dispatch.lightning_attn_fwd.__module__ == "cula.ops.lightning.prefill"
