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

# TODO: add tcgen05.cp for S2T, st.global (same as store_256b in C++) for direct R2G

"""Inline-PTX wrappers for ``tcgen05.ld`` / ``tcgen05.st`` (SM100 Blackwell).

Provides low-level, CuteDSL-compatible helpers that move data between
Tensor Memory (TMEM) and registers via the ``tcgen05.ld.sync.aligned``
and ``tcgen05.st.sync.aligned`` PTX instructions with the ``.32x32b``
shape qualifier.

PTX reference
-------------
    tcgen05.ld.sync.aligned.32x32b.xN.b32  {r0, ..., rN-1}, [taddr];
    tcgen05.st.sync.aligned.32x32b.xN.b32  [taddr], {r0, ..., rN-1};

where ``N ∈ {1, 2, 4, 8, 16, 32, 64, 128}`` and each ``r`` is a 32-bit
register.  ``taddr`` encodes both the TMEM column index (bits [15:0])
and the lane index (bits [31:16]).

See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld

Usage inside a ``@cute.kernel`` or ``@cute.jit`` function::

    from cula.ops.tmem_copy import tcgen05_ld_32x32b, tcgen05_st_32x32b

    # Load 4 × 32-bit values from TMEM into a list of 4 CuteDSL scalars
    vals = tcgen05_ld_32x32b(4, taddr)

    # Store 4 × 32-bit values from registers into TMEM
    tcgen05_st_32x32b(4, taddr, vals)
"""

__all__ = [
    "tcgen05_ld_32x32b",
    "tcgen05_st_32x32b",
    "store_256b",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir as _ir_mod
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Int32


def _to_ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuteDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ---------------------------------------------------------------------------
# tcgen05.ld.sync.aligned.32x32b.xN.b32
# ---------------------------------------------------------------------------

def _build_ld_asm(num: int) -> str:
    """Return the inline-asm string for ``tcgen05.ld.sync.aligned.32x32b.xN.b32``."""
    regs = ", ".join(f"${i}" for i in range(num))
    return f"tcgen05.ld.sync.aligned.32x32b.x{num}.b32 {{{regs}}}, [${num}];"


def _build_ld_constraints(num: int) -> str:
    """Return the inline-asm constraint string for the ld instruction."""
    out = ",".join(["=f"] * num)
    return f"{out},r"


@cute.jit
def tcgen05_ld_32x32b(num: int, taddr: int):
    """Load *num* × 32-bit values from TMEM → registers (FP32).

    ``num`` must be a **compile-time constant** in {1, 2, 4, 8, 16, 32, 64, 128}.
    Returns a Python ``list`` of CuteDSL ``Float32`` scalars (length *num*).

    Parameters
    ----------
    num : int
        Number of 32-bit registers to load.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    """
    asm_str = _build_ld_asm(num)
    constraints = _build_ld_constraints(num)

    @dsl_user_op
    def _do(addr_val, *, loc=None, ip=None):
        f32_ty = _ir_mod.F32Type.get()
        res_ty = _ir_mod.Type.parse(
            f"!llvm.struct<({', '.join(['f32'] * num)})>"
        )
        result = llvm.inline_asm(
            res_ty,
            [_to_ir(addr_val, loc, ip)],
            asm_str,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return [
            llvm.extractvalue(f32_ty, result, [i], loc=loc, ip=ip)
            for i in range(num)
        ]

    return _do(Int32(taddr))


# ---------------------------------------------------------------------------
# tcgen05.st.sync.aligned.32x32b.xN.b32
# ---------------------------------------------------------------------------

def _build_st_asm(num: int) -> str:
    """Return the inline-asm string for ``tcgen05.st.sync.aligned.32x32b.xN.b32``."""
    regs = ", ".join(f"${i + 1}" for i in range(num))
    return f"tcgen05.st.sync.aligned.32x32b.x{num}.b32 [${0}], {{{regs}}};"


def _build_st_constraints(num: int) -> str:
    """Return the inline-asm constraint string for the st instruction."""
    ins = ",".join(["f"] * num)
    return f"r,{ins}"


@cute.jit
def tcgen05_st_32x32b(num: int, taddr: int, values):
    """Store *num* × 32-bit values from registers → TMEM.

    ``num`` must be a **compile-time constant** in {1, 2, 4, 8, 16, 32, 64, 128}.

    Parameters
    ----------
    num : int
        Number of 32-bit registers to store.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    values : list
        A sequence of *num* CuteDSL scalar values (e.g. ``Float32``).
    """
    asm_str = _build_st_asm(num)
    constraints = _build_st_constraints(num)

    @dsl_user_op
    def _do(addr_val, *data_vals, loc=None, ip=None):
        operands = [_to_ir(addr_val, loc, ip)] + [
            _to_ir(v, loc, ip) for v in data_vals
        ]
        llvm.inline_asm(
            _ir_mod.Type.parse("!llvm.void"),
            operands,
            asm_str,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), *values)


# ---------------------------------------------------------------------------
# st.global.L1::no_allocate.v8.f32  (256-bit direct R2G store)
# ---------------------------------------------------------------------------

_STORE_256B_ASM = (
    "st.global.L1::no_allocate.v8.f32 [$0], "
    "{$1, $2, $3, $4, $5, $6, $7, $8};"
)
_STORE_256B_CONSTRAINTS = "l,f,f,f,f,f,f,f,f"


@cute.jit
def store_256b(gmem_ptr, values):
    """Store 256 bits (8 × FP32) to global memory, bypassing L1 allocation.

    Issues ``st.global.L1::no_allocate.v8.f32``.

    Parameters
    ----------
    gmem_ptr : pointer
        Global-memory destination address (must be 32-byte aligned).
    values : list
        Exactly 8 CuteDSL ``Float32`` scalars to store.
    """

    @dsl_user_op
    def _do(addr, s0, s1, s2, s3, s4, s5, s6, s7, *, loc=None, ip=None):
        operands = [
            _to_ir(addr, loc, ip),
            _to_ir(s0, loc, ip), _to_ir(s1, loc, ip),
            _to_ir(s2, loc, ip), _to_ir(s3, loc, ip),
            _to_ir(s4, loc, ip), _to_ir(s5, loc, ip),
            _to_ir(s6, loc, ip), _to_ir(s7, loc, ip),
        ]
        llvm.inline_asm(
            _ir_mod.Type.parse("!llvm.void"),
            operands,
            _STORE_256B_ASM,
            _STORE_256B_CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(gmem_ptr, values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7])
