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

import sys
import types

import pytest

from cula.ops import _mlir_compat
from cula.ops._mlir_compat import _parse_version
from cula.ops._nvvm_compat import adapt_keywords, bind_mma_ops


def test_adapt_keywords_preserves_46_binding():
    def mma(mma_kind, d, a, b, *, write_disable_mask=None):
        return mma_kind, d, a, b, write_disable_mask

    adapted = adapt_keywords(mma, {"mma_kind": "kind", "d": "matrix_d", "write_disable_mask": "disable_output_lane"})
    assert adapted is mma
    assert adapted(mma_kind="tf32", d=1, a=2, b=3, write_disable_mask=4) == ("tf32", 1, 2, 3, 4)


def test_adapt_keywords_translates_48_binding():
    def mma(kind, matrix_d, matrix_a, matrix_b, *, disable_output_lane=None):
        return kind, matrix_d, matrix_a, matrix_b, disable_output_lane

    adapted = adapt_keywords(
        mma,
        {
            "mma_kind": "kind",
            "d": "matrix_d",
            "a": "matrix_a",
            "b": "matrix_b",
            "write_disable_mask": "disable_output_lane",
        },
    )
    assert adapted(mma_kind="tf32", d=1, a=2, b=3, write_disable_mask=4) == ("tf32", 1, 2, 3, 4)


def test_adapt_keywords_rejects_duplicate_old_and_new_names():
    def mma(kind):
        return kind

    adapted = adapt_keywords(mma, {"mma_kind": "kind"})
    with pytest.raises(TypeError, match="Both 'mma_kind' and 'kind'"):
        adapted(mma_kind="tf32", kind="f16")


def test_bind_mma_ops_adapts_regular_and_weight_stationary_calls():
    def mma(kind, cta_group, matrix_d, matrix_a, matrix_b, idesc, enable_input_d, *, disable_output_lane=None):
        return kind, matrix_d, matrix_a, matrix_b, disable_output_lane

    def mma_ws(kind, matrix_d, matrix_a, matrix_b, idesc, enable_input_d, *, zero_col_mask=None):
        return kind, matrix_d, matrix_a, matrix_b, zero_col_mask

    regular, weight_stationary = bind_mma_ops(types.SimpleNamespace(tcgen05_mma=mma, tcgen05_mma_ws=mma_ws))
    assert regular(
        mma_kind="tf32",
        cta_group="one",
        d=1,
        a=2,
        b=3,
        idesc=4,
        enable_input_d=True,
        write_disable_mask=5,
    ) == ("tf32", 1, 2, 3, 5)
    assert weight_stationary(
        mma_kind="f16",
        d=1,
        a=2,
        b=3,
        idesc=4,
        enable_input_d=True,
        col_b_zero_mask=5,
    ) == ("f16", 1, 2, 3, 5)


# ---------------------------------------------------------------------------
# MLIR compat gateway (_mlir_compat)
# ---------------------------------------------------------------------------


def _touch_gateway(name):
    """Trigger the module-level ``__getattr__`` for a private binding."""
    return getattr(_mlir_compat, name)


_FAKE_DIALECTS = {
    "arith": (("constant",),),
    "cute": (),
    "ir": (("Type", "parse"), ("VectorType", "get"), ("IntegerType", "get_signless")),
    "llvm": (("inline_asm",), ("extractvalue",)),
    "nvvm": (),
    "vector": (("bitcast",), ("extractelement",), ("extract_strided_slice",)),
}


def _install_fake_cutlass(version, *, with_mlir=True, missing_dialect=None, broken_canary=None):
    """Install a fake ``cutlass`` package into ``sys.modules`` for fault injection.

    :param version: value for ``cutlass.__version__``
    :param with_mlir: when False, the fake exposes no ``_mlir`` subpackage
    :param missing_dialect: dialect that does not exist in the fake package
    :param broken_canary: dialect whose first canary entry point is absent
    """
    installed = ["cutlass"]
    cutlass = types.ModuleType("cutlass")
    cutlass.__version__ = version
    cutlass.__path__ = []
    if with_mlir:
        _mlir = types.ModuleType("cutlass._mlir")
        cutlass._mlir = _mlir
        installed.append("cutlass._mlir")
        dialects = types.ModuleType("cutlass._mlir.dialects")
        _mlir.dialects = dialects
        installed.append("cutlass._mlir.dialects")
        for name, canaries in _FAKE_DIALECTS.items():
            if name == missing_dialect:
                continue
            module = types.ModuleType(f"cutlass._mlir.dialects.{name}")
            module.non_public = types.SimpleNamespace()
            _install_chains(module, canaries, broken=name == broken_canary)
            setattr(dialects, name, module)
            installed.append(f"cutlass._mlir.dialects.{name}")
        ir = types.ModuleType("cutlass._mlir.ir")
        _install_chains(ir, _FAKE_DIALECTS["ir"], broken=False)
        _mlir.ir = ir
        installed.append("cutlass._mlir.ir")

    previous = {name: sys.modules.get(name) for name in installed}
    for name in installed:
        sys.modules[name] = _module_by_name(cutlass, name)
    _mlir_compat._CACHE.clear()
    return lambda: _restore_modules(previous)


def _install_chains(module, chains, *, broken):
    """Attach ``(parent, leaf)`` canary chains to *module* as fake callables."""
    chains = chains if not broken else ()
    for chain in chains:
        owner = module
        for index, part in enumerate(chain):
            is_leaf = index == len(chain) - 1
            if not hasattr(owner, part):
                if is_leaf:
                    setattr(owner, part, (lambda *args, _op=part, **kwargs: f"op:{_op}"))
                else:
                    setattr(owner, part, types.SimpleNamespace())
            owner = getattr(owner, part)


def _module_by_name(root, dotted):
    module = root
    for part in dotted.split(".")[1:]:
        module = getattr(module, part)
    return module


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    _mlir_compat._CACHE.clear()


def test_parse_version_accepts_release_and_dev_suffixes():
    assert _parse_version("4.5.3") == (4, 5, 3)
    assert _parse_version("4.6.0.dev0") == (4, 6, 0)


def test_parse_version_normalizes_missing_components():
    assert _parse_version("4") == (4, 0, 0)
    assert _parse_version("4.5") == (4, 5, 0)
    assert _parse_version("4.7") == (4, 7, 0)


def test_parse_version_rejects_garbage():
    with pytest.raises(RuntimeError, match="cannot parse"):
        _parse_version("not-a-version")


def test_version_contract_rejects_out_of_range():
    restore = _install_fake_cutlass("9.9.9")
    try:
        with pytest.raises(RuntimeError, match="outside the range validated by cuLA"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_version_contract_rejects_two_component_upper_bound():
    # "4.9" must normalize to (4, 9, 0) and not slip under the < 4.9 cap.
    restore = _install_fake_cutlass("4.9")
    try:
        with pytest.raises(RuntimeError, match="outside the range validated by cuLA"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_version_contract_rejects_below_minimum():
    restore = _install_fake_cutlass("4.6.1")
    try:
        with pytest.raises(RuntimeError, match="outside the range validated by cuLA"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_missing_cutlass_reports_install_hint():
    restore = _install_fake_cutlass("4.6.2", with_mlir=False)
    sys.modules["cutlass"] = None  # simulate the package being absent
    try:
        with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_missing_dialect_raises_with_dialect_name():
    restore = _install_fake_cutlass("4.8.1", missing_dialect="llvm")
    try:
        with pytest.raises(RuntimeError, match="no longer exposes 'llvm'"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_missing_canary_raises_naming_the_entry_point():
    restore = _install_fake_cutlass("4.8.1", broken_canary="llvm")
    try:
        with pytest.raises(RuntimeError, match="missing the canary entry point inline_asm"):
            _touch_gateway("llvm")
    finally:
        restore()


def test_vector_dispatch_uses_extractelement_when_extract_absent():
    restore = _install_fake_cutlass("4.6.2")
    try:
        seen = {}
        vector_dialect = _mlir_compat._load("vector")

        def fake_extractelement(vec, *, position, loc=None, ip=None):
            seen["position"] = position
            return "op:extractelement"

        vector_dialect.extractelement = fake_extractelement
        _mlir_compat._CACHE.clear()
        # the helper must build the i32 index constant itself, from the plain
        # Python index (the fake's arith.constant returns "op:constant")
        assert _mlir_compat.vector_extract_element("vec", 3) == "op:extractelement"
        assert seen["position"] == "op:constant"
    finally:
        restore()


def test_vector_dispatch_uses_extract_when_available():
    restore = _install_fake_cutlass("4.8.1")
    try:
        vector_dialect = _mlir_compat._load("vector")
        # simulate the 4.6 rename: drop extractelement, add extract
        delattr(vector_dialect, "extractelement")
        seen = {}

        def fake_extract(source, dynamic_position, static_position, *, loc=None, ip=None):
            seen["args"] = (source, dynamic_position, static_position)
            return "op:extract"

        vector_dialect.extract = fake_extract
        _mlir_compat._CACHE.clear()
        assert _mlir_compat.vector_extract_element("vec", 3) == "op:extract"
        # static-position form: no dynamic index operands, position in the
        # static-position array (matches the real 4.6.2 binding)
        assert seen["args"] == ("vec", [], [3])
    finally:
        restore()


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _touch_gateway("does_not_exist")


def test_dialect_bindings_are_cached():
    restore = _install_fake_cutlass("4.8.1")
    try:
        assert _mlir_compat.llvm is _mlir_compat.llvm
    finally:
        restore()


def test_real_cutlass_bindings_load():
    """Smoke-test the gateway against the installed CuTeDSL wheel."""
    try:
        import cutlass  # noqa: F401
    except ImportError:
        pytest.skip("nvidia-cutlass-dsl is not installed")

    for dialect in ("arith", "cute", "ir", "llvm", "nvvm", "vector"):
        _mlir_compat._CACHE.clear()
        assert getattr(_mlir_compat, dialect) is not None
