# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared low-level helpers for the SM90 FlashKDA kernels."""

import re
from importlib.metadata import PackageNotFoundError, version as package_version

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T


def _parse_cutedsl_version(raw_version: str) -> tuple[int, int, int]:
    """Return the numeric CuTeDSL version from a release or dev version string."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", raw_version)
    if match is None:
        raise RuntimeError(f"Unable to parse the installed CuTeDSL version: {raw_version!r}")
    return tuple(int(component or 0) for component in match.groups())


def _installed_cutedsl_version() -> tuple[int, int, int]:
    """Read the CuTeDSL version at runtime without relying on a private symbol."""
    raw_version = getattr(cutlass, "__version__", None)
    if raw_version is None:
        try:
            raw_version = package_version("nvidia-cutlass-dsl")
        except PackageNotFoundError as exc:
            raise RuntimeError("nvidia-cutlass-dsl is required by the SM90 FlashKDA backend") from exc
    return _parse_cutedsl_version(raw_version)


# CuTeDSL 4.6.0 added the missing elect_one inside cute.copy for async bulk
# atoms. Older releases need an explicit elect_one, while nesting one around
# cute.copy is incorrect in 4.6+. Keep this decision as a compile-time
# constant after detecting the installed runtime version, so the same source
# supports both API behaviours.
_CUTEDSL_VERSION = _installed_cutedsl_version()
_CUTE_COPY_AUTO_ELECTS_BULK = _CUTEDSL_VERSION >= (4, 6, 0)


def copy_async_bulk(atom, src, dst, **kwargs) -> None:
    """Issue a CuTeDSL async bulk copy across supported elect_one APIs."""
    if cutlass.const_expr(_CUTE_COPY_AUTO_ELECTS_BULK):
        cute.copy(atom, src, dst, **kwargs)
    else:
        with cute.arch.elect_one():
            cute.copy(atom, src, dst, **kwargs)


def _stream_key(device: torch.device) -> tuple[str, int]:
    return str(device), int(torch.cuda.current_stream(device).cuda_stream)


@cutlass.dsl_user_op
def movm_t_b16(src_u32: Int32, *, loc=None, ip=None) -> Int32:
    """``movmatrix.sync.aligned.m8n8.trans.b16`` -- register-file 8x8 b16 transpose."""
    result = _llvm.inline_asm(
        _T.i32(),
        [Int32(src_u32).ir_value(loc=loc, ip=ip)],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cutlass.dsl_user_op
def add_f16x2_u32(a_u32: Int32, b_u32: Int32, *, loc=None, ip=None) -> Int32:
    """Packed ``add.f16x2`` on two u32 registers."""
    result = _llvm.inline_asm(
        _T.i32(),
        [
            Int32(a_u32).ir_value(loc=loc, ip=ip),
            Int32(b_u32).ir_value(loc=loc, ip=ip),
        ],
        "add.f16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)
