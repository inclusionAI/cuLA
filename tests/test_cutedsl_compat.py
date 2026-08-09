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

from cula.ops._cutedsl_compat import Tcgen05LdStApi, detect_tcgen05_ldst_api


def _legacy_ld(res, shape, num, tmem_addr, *, pack=None, half_split_offset=None):
    pass


def _legacy_st(shape, num, tmem_addr, r, *, unpack=None, half_split_offset=None):
    pass


def _inferred_ld(res, shape, tmem_addr, *, pack=None, offset=None):
    pass


def _inferred_st(shape, tmem_addr, val, *, unpack=None, offset=None):
    pass


def test_detect_tcgen05_ldst_legacy_api():
    assert detect_tcgen05_ldst_api(_legacy_ld, _legacy_st) == Tcgen05LdStApi(
        ld_has_num=True,
        st_has_num=True,
        st_value_keyword="r",
    )


def test_detect_tcgen05_ldst_inferred_api():
    assert detect_tcgen05_ldst_api(_inferred_ld, _inferred_st) == Tcgen05LdStApi(
        ld_has_num=False,
        st_has_num=False,
        st_value_keyword="val",
    )


def test_detect_tcgen05_ldst_mixed_api():
    assert detect_tcgen05_ldst_api(_legacy_ld, _inferred_st) == Tcgen05LdStApi(
        ld_has_num=True,
        st_has_num=False,
        st_value_keyword="val",
    )


def test_detect_tcgen05_ldst_rejects_unknown_store_value_keyword():
    def unsupported_st(shape, tmem_addr, value):
        pass

    with pytest.raises(RuntimeError, match="expected exactly one value keyword"):
        detect_tcgen05_ldst_api(_inferred_ld, unsupported_st)


# ---------------------------------------------------------------------------
# MLIR compat gateway (_mlir_compat)
# ---------------------------------------------------------------------------

import sys
import types

from cula.ops import _mlir_compat
from cula.ops._mlir_compat import _parse_version

_FAKE_DIALECTS = {
    "arith": (("constant",),),
    "cute": (),
    "ir": (("Type", "parse"), ("VectorType", "get")),
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
            setattr(module, "non_public", types.SimpleNamespace())
            chains = canaries if name != broken_canary else ()
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
            setattr(dialects, name, module)
            installed.append(f"cutlass._mlir.dialects.{name}")

    previous = {name: sys.modules.get(name) for name in installed}
    for name in installed:
        sys.modules[name] = _module_by_name(cutlass, name)
    _mlir_compat._CACHE.clear()
    return lambda: _restore_modules(previous)


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


def test_parse_version_rejects_garbage():
    with pytest.raises(RuntimeError, match="cannot parse"):
        _parse_version("not-a-version")


def test_version_contract_rejects_out_of_range():
    restore = _install_fake_cutlass("9.9.9")
    try:
        with pytest.raises(RuntimeError, match="outside the range validated by cuLA"):
            _mlir_compat.llvm
    finally:
        restore()


def test_missing_cutlass_reports_install_hint():
    restore = _install_fake_cutlass("4.5.3", with_mlir=False)
    sys.modules["cutlass"] = None  # simulate the package being absent
    try:
        with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
            _mlir_compat.llvm
    finally:
        restore()


def test_missing_dialect_raises_with_dialect_name():
    restore = _install_fake_cutlass("4.5.3", missing_dialect="llvm")
    try:
        with pytest.raises(RuntimeError, match="no longer exposes 'llvm'"):
            _mlir_compat.llvm
    finally:
        restore()


def test_missing_canary_raises_naming_the_entry_point():
    restore = _install_fake_cutlass("4.5.3", broken_canary="llvm")
    try:
        with pytest.raises(RuntimeError, match="missing the canary entry point inline_asm"):
            _mlir_compat.llvm
    finally:
        restore()


def test_vector_dispatch_uses_extractelement_when_extract_absent():
    restore = _install_fake_cutlass("4.5.3")
    try:
        assert _mlir_compat.vector_extract_element("vec", "pos") == "op:extractelement"
    finally:
        restore()


def test_vector_dispatch_uses_extract_when_available():
    restore = _install_fake_cutlass("4.6.0")
    try:
        vector_dialect = _mlir_compat._load("vector")
        # simulate the 4.6 rename: drop extractelement, add extract
        delattr(vector_dialect, "extractelement")
        setattr(vector_dialect, "extract", lambda *args, **kwargs: "op:extract")
        _mlir_compat._CACHE.clear()
        assert _mlir_compat.vector_extract_element("vec", "pos") == "op:extract"
    finally:
        restore()


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _mlir_compat.does_not_exist


def test_dialect_bindings_are_cached():
    restore = _install_fake_cutlass("4.5.3")
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
