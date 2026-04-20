"""CuteDSL UMMA extension wrappers for SM100 (Blackwell) ``tcgen05.mma``.

CuteDSL's high-level ``cute.gemm()`` / ``make_tiled_mma()`` API does not
expose all ``tcgen05.mma`` instruction variants.  This module provides
low-level wrappers for the two categories currently needed:

1. **Masked MMA** – SS and TS forms with the 128-bit ``disable-output-lane``
   mask operand (``{m0, m1, m2, m3}``).  Implemented via the native
   ``nvvm.tcgen05_mma`` MLIR op with its ``write_disable_mask`` parameter
   (``vector<4xi32>``).

2. **Weight-stationary (WS) MMA** – ``tcgen05.mma.ws`` SS / TS forms for
   both ``kind::tf32`` and ``kind::f16``.  Implemented via
   ``llvm.inline_asm``.

----------------------------------------------------------------------
PTX instruction forms
----------------------------------------------------------------------
SS (SMEM A, SMEM B):
    tcgen05.mma.cta_group::1.kind::tf32  [tmem_c], desc_a, desc_b,
                                          desc_val, {m0,m1,m2,m3}, p;

TS (TMEM A, SMEM B):
    tcgen05.mma.cta_group::1.kind::tf32  [tmem_c], [tmem_a], desc_b,
                                          desc_val, {m0,m1,m2,m3}, p;

WS_SS (weight-stationary, SMEM A, SMEM B):
    tcgen05.mma.ws.cta_group::1.kind::tf32  [tmem_c], desc_a, desc_b,
                                             desc_val, p;
    tcgen05.mma.ws.cta_group::1.kind::f16   [tmem_c], desc_a, desc_b,
                                             desc_val, p;

WS_TS (weight-stationary, TMEM A, SMEM B):
    tcgen05.mma.ws.cta_group::1.kind::tf32  [tmem_c], [tmem_a], desc_b,
                                             desc_val, p;
    tcgen05.mma.ws.cta_group::1.kind::f16   [tmem_c], [tmem_a], desc_b,
                                             desc_val, p;

----------------------------------------------------------------------
Disable-output-lane mask layout (4 × uint32 = 128 bits)
----------------------------------------------------------------------
Each uint32 covers 32 M-dimension rows (8 rows × 4 elements per group).
  0x00000000  → group is ACTIVE    (output written)
  0xFFFFFFFF  → group is DISABLED  (output suppressed)

Predefined SS mask constants (SMEM A variants):
  SS_NO_MASK  = (0, 0, 0, 0)                       all rows active
  SS_MASK0    = (0, 0xFF…, 0, 0xFF…)               odd groups disabled
  SS_MASK1    = (0xFF…, 0, 0xFF…, 0)               even groups disabled
  SS_MASK2    = (0xFF…, 0xFF…, 0, 0xFF…)           group 2 only active
  SS_MASK3    = (0xFF…, 0xFF…, 0xFF…, 0)           group 3 only active

Predefined TS mask constants (TMEM A variants):
  TS_NO_MASK  = (0, 0, 0, 0)                       all rows active
  TS_MASK0    = (0, 0xFF…, 0xFF…, 0xFF…)           group 0 only active
  TS_MASK1    = (0xFF…, 0, 0xFF…, 0xFF…)           group 1 only active
  TS_MASK2    = (0xFF…, 0xFF…, 0, 0xFF…)           group 2 only active
  TS_MASK3    = (0xFF…, 0xFF…, 0xFF…, 0)           group 3 only active
  TS_MASK02   = (0, 0xFF…, 0, 0xFF…)               groups 0,2 only active
  TS_MASK13   = (0xFF…, 0, 0xFF…, 0)               groups 1,3 only active

Public API (all decorated with @cute.jit)
----------------------------------------------------------------------
Descriptor helpers (call inside @cute.jit):
    Tcgen05SmemDescriptor          — 64-bit SMEM descriptor object
    initialize_tcgen05_descriptor  — fill descriptor bitfields

Low-level primitives (pass mask words explicitly):
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out,
                  mask0, mask1, mask2, mask3)
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out,
                  mask0, mask1, mask2, mask3)
    tcgen05mma_ws_ss_tf32(desc_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ts_tf32(tmem_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ts_f16(tmem_a, desc_b, tmem_c, desc_val, scale_out)

Named convenience wrappers (pre-set masks, pass only MMA operands):
    tcgen05mma_ss_no_mask / tcgen05mma_ss_mask0 / …mask1 / …mask2 / …mask3
    tcgen05mma_ts_no_mask / tcgen05mma_ts_mask0 / …mask1 / …mask2 / …mask3
    tcgen05mma_ts_mask02  / tcgen05mma_ts_mask13
"""

__all__ = [
    # descriptor helpers
    "Tcgen05SmemDescriptor",
    "initialize_tcgen05_descriptor",
    # low-level primitives
    "tcgen05mma_ss",
    "tcgen05mma_ts",
    "tcgen05mma_ws_ss_tf32",
    "tcgen05mma_ws_ts_tf32",
    "tcgen05mma_ws_ss_f16",
    "tcgen05mma_ws_ts_f16",
    # SS named wrappers
    "tcgen05mma_ss_no_mask",
    "tcgen05mma_ss_mask0",
    "tcgen05mma_ss_mask1",
    "tcgen05mma_ss_mask2",
    "tcgen05mma_ss_mask3",
    # TS named wrappers
    "tcgen05mma_ts_no_mask",
    "tcgen05mma_ts_mask0",
    "tcgen05mma_ts_mask1",
    "tcgen05mma_ts_mask2",
    "tcgen05mma_ts_mask3",
    "tcgen05mma_ts_mask02",
    "tcgen05mma_ts_mask13",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import dsl_user_op

# ---------------------------------------------------------------------------
# Mask constants (4 × uint32).  0 = ACTIVE, 0xFFFFFFFF = DISABLED.
# ---------------------------------------------------------------------------
_ALL_ACTIVE = 0x00000000
_ALL_OFF = 0xFFFFFFFF

# SS masks (SMEM A, SMEM B)
SS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
SS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {0,F,0,F}
SS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)  # {F,0,F,0}
SS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {F,F,0,F}
SS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)  # {F,F,F,0}

# TS masks (TMEM A, SMEM B)
TS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
TS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_OFF, _ALL_OFF)  # {0,F,F,F}
TS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_OFF)  # {F,0,F,F}
TS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {F,F,0,F}
TS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)  # {F,F,F,0}
TS_MASK02 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {0,F,0,F}
TS_MASK13 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)  # {F,0,F,0}


# ---------------------------------------------------------------------------
# Tcgen05SmemDescriptor — 64-bit SMEM descriptor stored as 2×Int32
# ---------------------------------------------------------------------------


class Tcgen05SmemDescriptor:
    """64-bit shared-memory descriptor for tcgen05 MMA (Blackwell / SM100).

    The descriptor encodes SMEM base address, leading/stride byte offsets,
    swizzle mode, and other fields required by the ``tcgen05.mma`` PTX
    instruction to locate a matrix tile in shared memory.

    64-bit layout (PTX ISA Table 40)::

      Bit 63                                                      Bit 0
      ┌──────────┬────────┬─────┬──────────┬────┬──────────┬──────┬──────────────┐
      │ 63    61 │ 60  53 │  52 │ 51    49 │ 48 │ 45    32 │31 30 │ 29   16│15 14│ 13     0│
      │layout_typ│ reservd│l_abs│base_offst│ 46 │   SBO    │ rsvd │  LBO   │rsvd │start_adr│
      │  (3 bit) │ (8 bit)│(1b) │  (3 bit) │=0b001│(14 bit)│(2 b) │(14 bit)│(2b) │(14 bit) │
      └──────────┴────────┴─────┴──────────┴────┴──────────┴──────┴────────┴─────┴─────────┘

    Field descriptions:

    - **start_address** [bits 0-13]: SMEM base pointer, encoded as
      ``smem_ptr >> 4`` (16-byte aligned). The hardware reconstructs the
      full address as ``encoded_value << 4``.

    - **LBO** (Leading Byte Offset) [bits 16-29]: distance in bytes between
      consecutive elements along the leading dimension, encoded as
      ``lbo_bytes >> 4``.  When ``lbo_mode=1`` this is an absolute byte
      address rather than a relative offset.

    - **SBO** (Stride Byte Offset) [bits 32-45]: distance in bytes between
      consecutive elements along the stride dimension, encoded as
      ``sbo_bytes >> 4``.

    - **version** [bits 46-48]: fixed constant ``0b001`` (= 1).

    - **base_offset** [bits 49-51]: 3-bit alignment correction when the
      SMEM tile does not start at a natural swizzle-pattern boundary
      (1024B for 128B swizzle, 512B for 64B, 256B for 32B).
      Computed as ``(start_addr >> 7) & 0x7``.  Usually 0.

    - **lbo_mode** (leading_abs) [bit 52]: 0 → LBO is a relative byte
      offset; 1 → LBO is an absolute byte address.

    - **layout_type** (swizzle_mode) [bits 61-63]:
        - 0 = SWIZZLE_NONE
        - 1 = SWIZZLE_128B_BASE32B  (128-byte pattern, 32-byte atom)
        - 2 = SWIZZLE_128B          (128-byte pattern)
        - 4 = SWIZZLE_64B           (64-byte pattern)
        - 6 = SWIZZLE_32B           (32-byte pattern)

    Storage: two Int32 registers (desc[0] = low 32 bits, desc[1] = high 32
    bits), recast to a single Int64 for the PTX ``l``-constraint operand.

    Usage inside a @cute.jit kernel::

        desc = Tcgen05SmemDescriptor()
        initialize_tcgen05_descriptor(desc, smem_ptr, lbo, sbo, 0, True, swizzle)
    """

    def __init__(self, desc_64: cute.Int64 = None):
        # desc[0]: low  32 bits → start_address[0:14] | LBO[16:30]
        # desc[1]: high 32 bits → SBO[0:14] | version[14:16] | base_offset[17:20]
        #                         | lbo_mode[20] | layout_type[29:32]
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        # Alias the 2×i32 as 1×i64 for PTX "l" constraint (64-bit operand)
        self.desc_i64 = cute.make_tensor(cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,))
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, byte_offset):
        """Return a new descriptor offset by ``byte_offset`` bytes.

        Only the start_address field (bits 0-13 of desc[0]) is modified.
        Since it is stored in 16-byte units, we add ``byte_offset >> 4``.
        All other fields (LBO, SBO, swizzle, etc.) are copied unchanged.
        """
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        res[0] = self.desc[0] + (byte_offset >> 4)  # adjust start_address
        res[1] = self.desc[1]  # high word unchanged
        return Tcgen05SmemDescriptor(res_i64[0])


# ---------------------------------------------------------------------------
# initialize_tcgen05_descriptor
# ---------------------------------------------------------------------------


def initialize_tcgen05_descriptor(
    desc,
    start_address,
    leading_byte_offset,
    stride_byte_offset,
    base_offset,
    leading_abs,
    swizzle_mode,
):
    """Pack SMEM descriptor bitfields into *desc* (a Tcgen05SmemDescriptor).

    Constructs the 64-bit descriptor in two 32-bit halves (desc[0] and desc[1]).
    All address/offset fields must be pre-divided by 16 (``>> 4``) before
    passing, because the hardware stores them in 16-byte granularity.

    Low 32 bits — desc[0]::

      ┌────────────────┬──────┬──────────────────┐
      │ bits 29…16     │15…14 │ bits 13…0        │
      │ LBO (14 bits)  │ rsvd │ start_addr >> 4   │
      └────────────────┴──────┴──────────────────┘

      - [0:14)   start_address >> 4  — SMEM tile base pointer in 16B units.
      - [14:16)  reserved (0).
      - [16:30)  leading_byte_offset — LBO in 16B units (caller passes >> 4).

    High 32 bits — desc[1]::

      ┌────────┬────────┬─────┬──────────┬────────┬──────────────────┐
      │ 31…29  │ 28…21  │  20 │ 19…17    │ 16…14  │ bits 13…0        │
      │ layout │  rsvd  │l_abs│base_off  │version │ SBO (14 bits)    │
      │ (3 bit)│ (8 bit)│(1b) │  (3 bit) │=0b001  │                  │
      └────────┴────────┴─────┴──────────┴────────┴──────────────────┘

      - [0:14)   stride_byte_offset — SBO in 16B units (caller passes >> 4).
      - [14:16)  version = 1 (fixed constant 0b001, only bit 14 set).
      - [17:20)  base_offset & 0x7 — swizzle alignment correction.
                 Typically 0.  Non-zero when the tile doesn't start at
                 the natural swizzle boundary (1024B/512B/256B).
      - [20:21)  lbo_mode — 0 = LBO is relative offset, 1 = absolute address.
      - [29:32)  layout_type (swizzle_mode & 0x7):
                   0 = SWIZZLE_NONE
                   1 = SWIZZLE_128B_BASE32B  (Swizzle<2,5,2>)
                   2 = SWIZZLE_128B          (Swizzle<3,4,3>)
                   4 = SWIZZLE_64B           (Swizzle<2,4,3>)
                   6 = SWIZZLE_32B           (Swizzle<1,4,3>)

    Args:
        desc:                 Tcgen05SmemDescriptor to fill.
        start_address:        CuTeDSL Pointer to the SMEM tile start.
        leading_byte_offset:  Leading-dimension byte offset, already >> 4.
        stride_byte_offset:   Stride  byte offset, already >> 4.
        base_offset:          Swizzle alignment correction (raw int, bits 17-19).
        leading_abs:          Bool — True → LBO is absolute address.
        swizzle_mode:         Swizzle layout_type integer (bits 29-31).
    """
    # Encode start_address: take SMEM pointer, shift right by 4 to get 16B units
    ptr_val = start_address.toint() >> 4

    # --- Low 32 bits (desc[0]) ---
    # bits [0:14)  = start_address >> 4
    # bits [16:30) = leading_byte_offset (already in 16B units)
    desc.desc[0] = cutlass.Int32(ptr_val) | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16)

    # --- High 32 bits (desc[1]) ---
    # bits [0:14)  = stride_byte_offset (already in 16B units)
    # bit  [14]    = version = 1  (fixed)
    # bits [17:20) = base_offset & 0x7  (swizzle alignment correction)
    # bit  [20]    = lbo_mode  (0=relative, 1=absolute)
    # bits [29:32) = layout_type  (swizzle mode)
    desc.desc[1] = (
        cutlass.Int32(stride_byte_offset)
        | cutlass.Int32(1 << 14)  # version = 1
        | cutlass.Int32(cutlass.Int32(base_offset & 0x7) << 17)
        | cutlass.Int32(cutlass.Int32(int(leading_abs)) << 20)
        | cutlass.Int32(cutlass.Int32(swizzle_mode & 0x7) << 29)
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuTeDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ===========================================================================
# Low-level primitives
# ===========================================================================

# ---------------------------------------------------------------------------
# tcgen05mma_ss  —  SMEM A, SMEM B (non-warp-specialised)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ss(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with SMEM operands.

    ``mask{0-3}`` are the four uint32 words of the 128-bit
    ``disable-output-lane`` mask (0=active, 0xFFFFFFFF=disabled).

    Caller must ensure single-thread execution (e.g. via ``elect_one``);
    no internal ``elect.sync`` is performed.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate into C, 0 → overwrite C (clear accumulators).
        mask0-3:   Four uint32 words of the disable-output-lane mask.
    """

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _ir(da_val, loc, ip)  # i64 SMEM descriptor
        db_ir = _ir(db_val, loc, ip)  # i64 SMEM descriptor
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _ir(m0_val, loc, ip)
        m1_ir = _ir(m1_val, loc, ip)
        m2_ir = _ir(m2_val, loc, ip)
        m3_ir = _ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ts  —  TMEM A, SMEM B (non-warp-specialised)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ts(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with TMEM A operand.

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    Caller must ensure single-thread execution (e.g. via ``elect_one``).

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate into C, 0 → overwrite C.
        mask0-3:   Four uint32 words of the disable-output-lane mask.
    """

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _ir(c_val, loc, ip)
        a_ir = _ir(a_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        b_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _ir(m0_val, loc, ip)
        m1_ir = _ir(m1_val, loc, ip)
        m2_ir = _ir(m2_val, loc, ip)
        m3_ir = _ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=a_ptr,
            b=b_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ss_tf32  —  weight-stationary, SMEM A, SMEM B, kind::tf32
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ss_tf32(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` (weight-stationary form).

    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
    """
    asm_str = "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\ntcgen05.mma.ws.cta_group::1.kind::tf32 [$0], $1, $2, $3, p;\n}"

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                _ir(c_val, loc, ip),
                _ir(da_val, loc, ip),
                _ir(db_val, loc, ip),
                _ir(dv_val, loc, ip),
                _ir(sc_val, loc, ip),
            ],
            asm_str,
            "r,l,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ss_f16  —  weight-stationary, SMEM A, SMEM B, kind::f16
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ss_f16(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` (weight-stationary form).

    Same as the tf32 variant but uses ``.kind::f16`` for half-precision
    input types (f16 / bf16).  K dimension is 16 instead of 8.

    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
    """
    asm_str = "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\ntcgen05.mma.ws.cta_group::1.kind::f16 [$0], $1, $2, $3, p;\n}"

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                _ir(c_val, loc, ip),
                _ir(da_val, loc, ip),
                _ir(db_val, loc, ip),
                _ir(dv_val, loc, ip),
                _ir(sc_val, loc, ip),
            ],
            asm_str,
            "r,l,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ts_tf32  —  weight-stationary, TMEM A, SMEM B, kind::tf32
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ts_tf32(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` with TMEM A (weight-stationary).

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
    """
    asm_str = "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\ntcgen05.mma.ws.cta_group::1.kind::tf32 [$0], [$1], $2, $3, p;\n}"

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                _ir(c_val, loc, ip),
                _ir(a_val, loc, ip),
                _ir(db_val, loc, ip),
                _ir(dv_val, loc, ip),
                _ir(sc_val, loc, ip),
            ],
            asm_str,
            "r,r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ts_f16  —  weight-stationary, TMEM A, SMEM B, kind::f16
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ts_f16(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` with TMEM A (weight-stationary).

    Same as the tf32 variant but uses ``.kind::f16`` for half-precision
    input types (f16 / bf16).  K dimension is 16 instead of 8.

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
    """
    asm_str = "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\ntcgen05.mma.ws.cta_group::1.kind::f16 [$0], [$1], $2, $3, p;\n}"

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                _ir(c_val, loc, ip),
                _ir(a_val, loc, ip),
                _ir(db_val, loc, ip),
                _ir(dv_val, loc, ip),
                _ir(sc_val, loc, ip),
            ],
            asm_str,
            "r,r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ===========================================================================
# Named convenience wrappers
# ===========================================================================
# These call the low-level primitives with pre-set mask constants so callers
# do not need to repeat the literal values.  Signature: same as the base
# function but without the mask0-3 args.

# ---------------------------------------------------------------------------
# SS named wrappers  (SMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ss_no_mask(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_NO_MASK[0], SS_NO_MASK[1], SS_NO_MASK[2], SS_NO_MASK[3])


@cute.jit
def tcgen05mma_ss_mask0(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK0[0], SS_MASK0[1], SS_MASK0[2], SS_MASK0[3])


@cute.jit
def tcgen05mma_ss_mask1(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK1[0], SS_MASK1[1], SS_MASK1[2], SS_MASK1[3])


@cute.jit
def tcgen05mma_ss_mask2(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK2[0], SS_MASK2[1], SS_MASK2[2], SS_MASK2[3])


@cute.jit
def tcgen05mma_ss_mask3(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK3[0], SS_MASK3[1], SS_MASK3[2], SS_MASK3[3])


# ---------------------------------------------------------------------------
# TS named wrappers  (TMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ts_no_mask(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_NO_MASK[0], TS_NO_MASK[1], TS_NO_MASK[2], TS_NO_MASK[3])


@cute.jit
def tcgen05mma_ts_mask0(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0xF…, 0xF…} — group 0 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK0[0], TS_MASK0[1], TS_MASK0[2], TS_MASK0[3])


@cute.jit
def tcgen05mma_ts_mask1(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0xF…} — group 1 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK1[0], TS_MASK1[1], TS_MASK1[2], TS_MASK1[3])


@cute.jit
def tcgen05mma_ts_mask2(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK2[0], TS_MASK2[1], TS_MASK2[2], TS_MASK2[3])


@cute.jit
def tcgen05mma_ts_mask3(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK3[0], TS_MASK3[1], TS_MASK3[2], TS_MASK3[3])


@cute.jit
def tcgen05mma_ts_mask02(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled).

    Used in the KDA intra-chunk backward kernel for the QK/KG phase where
    only even row-groups of the M tile contribute to the triangular region.
    """
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK02[0], TS_MASK02[1], TS_MASK02[2], TS_MASK02[3])


@cute.jit
def tcgen05mma_ts_mask13(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled).

    Used in the KDA intra-chunk backward kernel for the QK/KG phase where
    only odd row-groups of the M tile contribute to the triangular region.
    """
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK13[0], TS_MASK13[1], TS_MASK13[2], TS_MASK13[3])
