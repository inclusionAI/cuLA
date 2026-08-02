# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared low-level helpers for the SM90 FlashKDA kernels."""

import cutlass
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T


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
