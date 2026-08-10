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

"""Single-point gateway for CuTeDSL's private MLIR/NVVM bindings.

cuLA kernels are written against CuTeDSL's code-generation API. Most of that API
is public (``cutlass.cutlass_dsl``, ``cutlass.cute``, ...); a small part is not:
the generated MLIR dialect bindings under ``cutlass._mlir``. CuTeDSL ships them
as implementation detail and provides no stability contract across patch
releases -- the ``tcgen05_ld/st`` breakage between CutDSL 4.5.2 and 4.5.3 was one
such incident.

This module is the ONLY place in cuLA that may import from ``cutlass._mlir``.
Kernel modules bind their dialect aliases from here::

    from cula.ops._mlir_compat import arith as _arith
    from cula.ops._mlir_compat import ir
    from cula.ops._mlir_compat import llvm as _llvm
    from cula.ops._mlir_compat import vector as _vector

Design goals:

- Lazy: nothing is imported until a kernel actually needs a binding, so plain
  imports of cuLA never touch ``cutlass._mlir``.
- Explicity: an out-of-contract CuTeDSL version, or a missing/renamed binding,
  fails fast at first use with an actionable ``RuntimeError`` instead of
  surfacing mid-JIT as a confusing compile error (or silently emitting a
  different kernel).
- Zero dependencies: version parsing is done with a small regex so this module
  stays usable in any environment cuLA can be installed into.

The version contract mirrors ``pyproject.toml`` (including its ``!=4.5.0``
exclusion) and the canary probes make a broken binding fail fast with an
actionable message instead of surfacing mid-JIT.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Final

# Version contract for nvidia-cutlass-dsl. Kept in sync with
# ``pyproject.toml``; when CuTeDSL is bumped, extend ``_SUPPORTED_MIN`` /
# ``_SUPPORTED_MAX`` only after the new release has been validated against the
# canaries below (and ideally against the SM90/SM100 kernel test suites).
_SUPPORTED_MIN: Final[tuple[int, ...]] = (4, 4, 2)
_SUPPORTED_MAX: Final[tuple[int, ...]] = (4, 7, 0)
_EXCLUDED_VERSIONS: Final[frozenset[tuple[int, ...]]] = frozenset({(4, 5, 0)})

# dialect name -> (package, attribute) inside ``cutlass``.
_PRIVATE_TABLE: Final[dict[str, tuple[str, str]]] = {
    "arith": ("cutlass._mlir", "dialects.arith"),
    "cute": ("cutlass._mlir", "dialects.cute"),
    "ir": ("cutlass._mlir", "ir"),
    "llvm": ("cutlass._mlir", "dialects.llvm"),
    "nvvm": ("cutlass._mlir", "dialects.nvvm"),
    "vector": ("cutlass._mlir", "dialects.vector"),
}

# Canary entry points: attribute paths that must be present on each dialect
# binding for cuLA's kernel code to be emitted correctly. These are the names
# the migrated consumers actually call; extend the list when new usages land.
_CANARIES: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "arith": (("constant",),),
    "cute": (),
    "ir": (("Type", "parse"), ("VectorType", "get")),
    "llvm": (("inline_asm",), ("extractvalue",)),
    "nvvm": (),
    "vector": (("bitcast",), ("extract_strided_slice",)),
}

# Entry points whose name changed across CutDSL versions: for each group (a
# tuple of variant paths), at least one variant must exist. For example
# ``vector.extractelement`` was replaced by ``vector.extract`` in the 4.6
# line; cuLA helpers dispatch on whichever one exists.
_ANY_OF_CANARIES: Final[dict[str, tuple[tuple[tuple[str, ...], ...], ...]]] = {
    "vector": ((("extract",), ("extractelement",)),),
}

_VERSION_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[a-z0-9._+-]*)?$")

_INCIDENT_NOTE: Final[str] = (
    "cuLA depends on CuTeDSL's private MLIR bindings (`cutlass._mlir`), which are "
    "implementation detail and were broken by a CuTeDSL patch release once "
    "already (`tcgen05_ld/st`, 4.5.2 -> 4.5.3). To avoid silent kernel "
    "miscompiles, cuLA refuses bindings outside its validated contract."
)

_CACHE: Final[dict[str, Any]] = {}


def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse ``X.Y.Z[.devN...]`` into a comparable ``(major, minor, patch)`` tuple.

    Missing components normalize to zero so that ``4.5`` and ``4.7`` compare
    against the contract exactly like ``4.5.0`` and ``4.7.0`` (a two-component
    ``4.7`` must not slip under the ``< 4.7.0`` upper bound).
    """
    match = _VERSION_RE.match(version.strip())
    if match is None:
        raise RuntimeError(
            f"cuLA cannot parse the installed CuTeDSL version {version!r}; "
            f"refusing to use its private MLIR bindings. {_INCIDENT_NOTE}"
        )
    major, minor, patch = (int(part) if part is not None else 0 for part in match.groups())
    return (major, minor, patch)


def _installed_version() -> tuple[int, int, int]:
    try:
        import cutlass  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "cuLA requires the nvidia-cutlass-dsl package; install it with `pip install 'nvidia-cutlass-dsl>=4.4.2,<4.7'`."
        ) from None
    version = getattr(cutlass, "__version__", None)
    if not isinstance(version, str):
        raise RuntimeError(
            f"CuTeDSL is installed but exposes no `__version__` (got {version!r}); "
            f"refusing to use its private MLIR bindings. {_INCIDENT_NOTE}"
        )
    return _parse_version(version)


def _check_contract(version: tuple[int, ...]) -> None:
    if version in _EXCLUDED_VERSIONS:
        raise RuntimeError(
            f"Installed CuTeDSL version {'.'.join(map(str, version))} is explicitly "
            f"excluded by cuLA (see pyproject.toml). {_INCIDENT_NOTE}"
        )
    if not (_SUPPORTED_MIN <= version < _SUPPORTED_MAX):
        raise RuntimeError(
            f"Installed CuTeDSL version {'.'.join(map(str, version))} is outside the "
            f"range validated by cuLA ({'.'.join(map(str, _SUPPORTED_MIN))} to "
            f"{'.'.join(map(str, _SUPPORTED_MAX))}, exclusive). "
            f"{_INCIDENT_NOTE} To proceed, pin the validated range in "
            f"pyproject.toml and re-validate the canaries in this module."
        )


def _has_attribute_path(module: Any, path: tuple[str, ...]) -> bool:
    owner = module
    for part in path:
        owner = getattr(owner, part, None)
        if owner is None:
            return False
    return True


def _load(dialect: str) -> Any:
    if dialect in _CACHE:
        return _CACHE[dialect]

    package, attribute = _PRIVATE_TABLE[dialect]
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to import CuTeDSL's private {dialect!r} bindings ({package}): {exc}. {_INCIDENT_NOTE}"
        ) from exc
    if attribute:
        for part in attribute.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
    if module is None:
        raise RuntimeError(f"CuTeDSL no longer exposes {dialect!r} bindings ({package}.{attribute}). {_INCIDENT_NOTE}")

    for canary in _CANARIES[dialect]:
        owner: Any = module
        for part in canary:
            owner = getattr(owner, part, None)
            if owner is None:
                raise RuntimeError(
                    f"CuTeDSL dialect {dialect!r} is missing the canary entry point "
                    f"{'.'.join(canary)} used by cuLA kernels. {_INCIDENT_NOTE}"
                )
    for group in _ANY_OF_CANARIES.get(dialect, ()):
        if not any(_has_attribute_path(module, variant) for variant in group):
            raise RuntimeError(
                f"CuTeDSL dialect {dialect!r} exposes none of the entry points "
                f"{' / '.join('.'.join(variant) for variant in group)} expected by "
                f"cuLA kernels. {_INCIDENT_NOTE}"
            )

    _CACHE[dialect] = module
    return module


def __getattr__(name: str) -> Any:
    """Lazy, contract-checked access to private dialect bindings."""
    if name not in _PRIVATE_TABLE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    version = _installed_version()
    _check_contract(version)
    return _load(name)


def cutlass_dsl_version() -> str | None:
    """Installed CuTeDSL version string, or None when not installed."""
    try:
        import cutlass  # noqa: PLC0415
    except ImportError:
        return None
    version = getattr(cutlass, "__version__", None)
    return version if isinstance(version, str) else None


def vector_extract_element(vec, position, *, loc=None, ip=None):
    """Extract one element of ``vec`` at ``position``, across CutDSL versions.

    CutDSL renamed ``vector.extractelement`` to ``vector.extract`` in the 4.6
    line. ``position`` is the Python element index; each branch builds the
    operand shape its binding actually expects:

    - ``extractelement`` (4.5 line; also present in early 4.6): takes the
      index as a single i32 operand, so the constant is constructed here.
    - ``extract`` (4.6+): takes a sequence of index-typed dynamic operands
      plus a static-position array, so a constant position is ``extract(vec,
      [], [position], ...)``.

    Preferring ``extractelement`` when both exist keeps the pre-4.6 code path
    byte-identical to what cuLA shipped before the gateway.
    """
    vector_dialect = _load("vector")
    if _has_attribute_path(vector_dialect, ("extractelement",)):
        i32_ty = _load("ir").IntegerType.get_signless(32)
        index = _load("arith").constant(i32_ty, position, loc=loc, ip=ip)
        return vector_dialect.extractelement(vec, position=index, loc=loc, ip=ip)
    return vector_dialect.extract(vec, [], [position], loc=loc, ip=ip)
