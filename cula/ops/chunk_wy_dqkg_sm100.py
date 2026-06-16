import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.arch import (
    elect_one,
    mbarrier_arrive,
    mbarrier_arrive_and_expect_tx,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
)
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import (
    make_umma_smem_desc,
    smem_descriptor_to_int,
)
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64
from fla.ops.utils import prepare_chunk_indices

from cula.ops.intrinsics_sm100 import (
    reinterpret_cast,
    store_256b,
    subvec,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld_32x32b,
    tcgen05_st_32x32b,
    umma_arrive,
)
from cula.ops.ptx_umma_ext import (
    Tcgen05SmemDescriptor,
    tcgen05mma_ws_ss_f16,
)
from cula.utils import USE_FAST_MATH, assert_blackwell, prepare_uniform_cu_seqlens

PRINT_DEBUG = False

LN2 = 0.6931471805599453
RCP_LN2 = 1.4426950408889634

COMPILE_OPTIONS = "--enable-tvm-ffi"

# Mapping from torch dtype to cutlass dtype (for beta_dtype conversion)
_torch_to_cutlass_dtype = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


def _exclusive_cumsum(a: list[int]):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r


# ── TMEM column offset constants (cta_group::1, M=64, .ws Layout E) ──
TMEM_DA_ACC_OFF = 0  # [0,32)   32 cols  dA fp32 acc; Phase 3: [0,16) overwritten by dA_bf16
TMEM_DQ_ACC_OFF = 32  # [32,96)  64 cols  dq fp32 acc; Phase 3: step2/step3 result [32,64)
TMEM_DK_ACC_OFF = 96  # [96,160) 64 cols  dk fp32 acc
TMEM_DW_ACC_OFF = 160  # [160,224] 64 cols dw fp32 acc
TMEM_FLEX_OFF = 224  # [224,256) 32 cols  dvb time-shared
TMEM_A_BF16_OFF = 256  # [256,272) 16 cols  A_bf16 TS opA (persistent) (not used currently)
TMEM_DKGB_ACC_OFF = 272  # [272,336) 64 cols, dkgb fp32 acc
TMEM_DA2_ACC_OFF = 336  # [336,368) 32 cols  dA fp32 acc, used for dA=dA@A and dA=A@dA
TMEM_DQ_SCALED_OFF = 368  # [368,432) 64 cols  dq_scaled (stored for dg)
TMEM_TOTAL = 512

# Instruction descriptor for M=64, N=64, BF16, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22], TransposeB at [16],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_K_MN = (4 << 24) | (8 << 17) | (1 << 16) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22], TransposeB at [16],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_K_MN = (4 << 24) | (16 << 17) | (1 << 16) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_K_K = (4 << 24) | (16 << 17) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense, TransposeA=1, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_MN_MN = (4 << 24) | (16 << 17) | (1 << 16) | (1 << 15) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=64, BF16, dense, TransposeA=1, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_MN_MN = (4 << 24) | (8 << 17) | (1 << 16) | (1 << 15) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=64, BF16, dense
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_K_K = (4 << 24) | (8 << 17) | (1 << 10) | (1 << 7) | (1 << 4)

ELEM_BYTES_BF16 = BFloat16.width // 8


@cute.jit
def smem_load_bf16x8_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 8 consecutive bfloat16 from SMEM with Swizzle<3,4,3> layout.
    raw_ptr: BFloat16 SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    Logical layout: [BT=64, BV=128] K-major, with the BV=128 dim split into
    two halves of 64 elements (high half offset by 4096 elements).
    Swizzle<3,4,3> on bf16: phys_elem = elem ^ ((row & 7) << 3) within a half.
    Returns an 8-element rmem fragment (bf16).
    """
    half = col_base >> Int32(6)
    k_inner = col_base & Int32(63)
    swizzled = k_inner ^ ((row & Int32(7)) << Int32(3))
    elem_off = half * Int32(4096) + row * Int32(64) + swizzled
    aligned_ptr = cute.make_ptr(
        BFloat16,
        (raw_ptr + elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(aligned_ptr, cute.make_layout((8,), stride=(1,)))
    rmem_t = cute.make_fragment_like(smem_t)
    cute.autovec_copy(smem_t, rmem_t)
    return rmem_t


@cute.jit
def smem_store_bf16x8_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32, data: cute.Tensor):
    """
    Store 8 consecutive bfloat16 to SMEM with Swizzle<3,4,3> layout.
    raw_ptr: BFloat16 SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    data: 8-element rmem fragment (bf16) to store.

    NOTE: For the K-major→MN-major dv re-swizzle, source layout
    `(BT,BV) K-major Swizzle<3,4,3>` and destination layout
    `(BV,BT) MN-major Swizzle<3,4,3>` produce **identical** physical
    addresses for the same (row=t, col=v). So this helper uses the same
    address formula as the load helper, and the caller passes (row=t, col=v)
    for both load (src K-maj) and store (dst MN-maj), implicitly transposing.
    """
    half = col_base >> Int32(6)
    k_inner = col_base & Int32(63)
    swizzled = k_inner ^ ((row & Int32(7)) << Int32(3))
    elem_off = half * Int32(4096) + row * Int32(64) + swizzled
    smem_ptr = cute.make_ptr(
        BFloat16,
        (raw_ptr + elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((8,), stride=(1,)))
    cute.autovec_copy(data, smem_t)


@cute.jit
def smem_load_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive float32 from SMEM with K_SW128 layout.
    Logical layout: [BT=64, BK=128] ROW_MAJOR, tiled over a Float32 K_SW128 atom.
    The atom provides a 32-element row stride. The 128-element column is broken
    into 4 blocks of 32 elements.
    PyCutlass tiles this such that outer blocks stride by 2048 elements:
      elem_idx = row * 32 + (col_base % 32) + (col_base / 32) * 2048

    The TMA hardware performs a 128B Swizzle on physical byte addresses:
      byte_idx = elem_idx * 4
      swizzled_byte = byte_idx ^ (((byte_idx >> 7) & 7) << 4)
    Dividing by 4 yields the element XOR offset:
      elem_xor = ((elem_idx >> 5) & 7) << 2
    Because (elem_idx >> 5) simplifies to 'row + (col_outer * 64)',
    the XOR offset simplifies exactly to ((row & 7) << 2).
    This only affects the inner 32-element column block.
    """
    c_inner = col_base & Int32(31)
    c_outer = col_base >> Int32(5)
    swizzled_inner = c_inner ^ ((row & Int32(7)) << Int32(2))

    elem_offset = row * Int32(32) + swizzled_inner + c_outer * Int32(2048)

    aligned_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_store_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32, data: cute.Tensor):
    """
    Store 4 consecutive float32 to SMEM with K_SW128 layout.
    Inverse of smem_load_f32x4_sw128 — same address formula, write path.
    raw_ptr: Float32 SMEM base pointer (raw buffer start)
    row: row index in [0, BT)
    col_base: 4-aligned column index (multiples of 4)
    data: 4-element rmem fragment (f32) to store.
    """
    c_inner = col_base & Int32(31)
    c_outer = col_base >> Int32(5)
    swizzled_inner = c_inner ^ ((row & Int32(7)) << Int32(2))
    elem_offset = row * Int32(32) + swizzled_inner + c_outer * Int32(2048)
    smem_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((4,), stride=(1,)))
    cute.autovec_copy(data, smem_t)


@cute.jit
def mma_ws_ss_m64n128_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N128_K_MN, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n128_k_k_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N128_K_K, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n128_mn_mn_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N128_MN_MN, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n64_k_k_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N64_K_K, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n64_mn_mn_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N64_MN_MN, scale)
            scale = 1


class ChunkKdaBwdWyDqkgFused:
    """
    CuTe DSL kernel for chunk_kda_bwd_kernel_wy_dqkg_fused.

    Computes backward gradients dq, dk, dv2, dg, db, dA for the KDA
    chunkwise delta-rule backward pass.

    Architecture: 1 CudaCore WG + 1 MMA warp + TMA/Aux warps.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        g_dtype: type[cutlass.Numeric] = cutlass.Float32,
        beta_dtype: type[cutlass.Numeric] = cutlass.Float32,
        scale: float = 1.0,
        min_occupancy: int = 1,
        use_fast_math: bool = True,
    ):
        assert chunk_size == 64, "chunk_size must be 64"
        assert head_dim_k == 128 and head_dim_v == 128, (
            f"head_dim_k and head_dim_v must both be 128, got head_dim_k={head_dim_k}, head_dim_v={head_dim_v}"
        )
        assert_blackwell()

        self.use_fast_math = use_fast_math
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.g_dtype = g_dtype
        self.beta_dtype = beta_dtype
        self.scale = scale

        # Tile sizes
        self.BT = chunk_size  # 64
        self.BK = 128  # K tiling for V-loop GEMM (single K tile)
        self.BV = 64  # V tiling for V-loop GEMM (single V tile)

        # Warp layout: WG0/WG1 (8 CudaCore warps) + WG2 (MMA/Load/Aux/Store)
        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)  # WG0: CudaCore + Store
        self.cuda2_warp_ids = (4, 5, 6, 7)  # WG1: CudaCore + Store
        self.mma_warp_id = 8  # WG2: MMA dispatch
        self.load_warp_id = 9  # WG2: TMA Load
        self.aux_warp_ids = (10, 11)  # WG2: Aux/Load/Store Aux
        self.threads_per_cta = self.threads_per_warp * 12  # 384 threads (3 WGs)

        self.num_regs_cuda = 208
        self.num_regs_others = 88
        self.min_occupancy = min_occupancy

        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Number of K/V tiles
        self.num_k_tiles = (head_dim_k + self.BK - 1) // self.BK  # 128/128 = 1
        self.num_v_tiles = (head_dim_v + self.BV - 1) // self.BV  # 128/64 = 2

        # ── Pipeline stages ──
        # V-loop TMA: 2-stage double buffer
        self.vloop_stage = 2
        self.kloop_stage = 1
        self.a_stage = 2
        self.mma_stage = 1

        # ── MMA tiler shapes ──
        # V-loop GEMMs: [BT, BV] × [BV, BK] → [BT, BK]
        # dq = do @ h :       (BT, BK, BV)  — M=BT, N=BK, K=BV
        # dk = v_new @ dh :   (BT, BK, BV)
        # dw = dv @ h :       (BT, BK, BV)
        self.vloop_gemm_tiler = (self.BT, self.BK, self.BV)

        # V-loop i_k==0 GEMMs: [BT, BV] × [BV, BT] → [BT, BT]
        # dA = dv @ v^T :     (BT, BT, BV)
        self.dA_vloop_tiler = (self.BT, self.BT, self.BV)

        # V-loop i_k==0: A @ dv : [BT, BT] × [BT, BV] → [BT, BV]
        self.dvb_tiler = (self.BT, self.BV, self.BT)

        # K-loop GEMMs:
        # dA += dw @ kg^T :  [BT, BK] × [BK, BT] → [BT, BT]  →  (BT, BT, BK)
        self.kloop_dA_tiler = (self.BT, self.BT, self.BK)
        # dkgb = A @ dw :    [BT, BT] × [BT, BK] → [BT, BK]  →  (BT, BK, BT)
        self.kloop_dkgb_tiler = (self.BT, self.BK, self.BT)

        # dA-post GEMMs:
        # dA @ A :  [BT, BT] × [BT, BT] → [BT, BT]  →  (BT, BT, BT)
        # A @ dA :  same
        self.dApost_tiler = (self.BT, self.BT, self.BT)

        # Named barriers
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.cuda_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * 8,
        )
        self.buffer_align_bytes = 1024

        # Persistent scheduling
        self.persistent = True
        hardware_info = cutlass.utils.HardwareInfo()
        self.num_sm = hardware_info.get_device_multiprocessor_count()

    def _compute_grid(self, B, T, HV, total_nt=None):
        """Compute grid dimensions for persistent kernel launch.

        Grid: (min(num_sm * min_occupancy, total_tiles), 1, 1)
        Each CTA handles multiple tiles via stride-by-gridDim.x loop.
        """
        assert total_nt is not None
        total_tiles = total_nt * HV
        grid_x = cutlass.min(Int32(self.num_sm * self.min_occupancy), total_tiles)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def __call__(
        self,
        # ── Inputs ──
        q_in: cute.Tensor,  # [B, T, H, K] bf16
        k_in: cute.Tensor,  # [B, T, H, K] bf16
        v_in: cute.Tensor,  # [B, T, HV, V] bf16
        v_new_in: cute.Tensor,  # [B, T, HV, V] bf16
        g_in: cute.Tensor,  # [B, T, HV, K] fp32
        beta_in: cute.Tensor,  # [B, T, HV]   fp32
        A_in: cute.Tensor,  # [B, T, HV, BT] bf16
        h_in: cute.Tensor,  # [B, NT, HV, K, V] bf16
        do_in: cute.Tensor,  # [B, T, HV, V] bf16
        dh_in: cute.Tensor,  # [B, NT, HV, K, V] bf16
        dv_in: cute.Tensor,  # [B, T, HV, V] bf16
        # ── Outputs ──
        dq_in: cute.Tensor,  # [B, T, HV, K] fp32
        dk_in: cute.Tensor,  # [B, T, HV, K] fp32
        dv2_in: cute.Tensor,  # [B, T, HV, V] bf16
        dg_in: cute.Tensor,  # [B, T, HV, K] fp32
        db_in: cute.Tensor,  # [B, T, HV]    fp32
        dA_in: cute.Tensor,  # [B, T, HV, BT] fp32
        # ── Metadata ──
        cu_seqlens_in: cute.Tensor,  # [N+1] int32
        chunk_indices_in: cute.Tensor,  # [NT, 2] int32
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32, Int32],  # (B, T, H, HV, K, V)
        total_nt: Int32,
        stream,
    ):
        # ── Extract pointers ──
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        v_ptr = v_in.iterator
        v_new_ptr = v_new_in.iterator
        g_ptr = g_in.iterator
        beta_ptr = beta_in.iterator
        A_ptr = A_in.iterator
        h_ptr = h_in.iterator
        do_ptr = do_in.iterator
        dh_ptr = dh_in.iterator
        dv_ptr = dv_in.iterator
        dq_ptr = dq_in.iterator
        dk_ptr = dk_in.iterator
        dv2_ptr = dv2_in.iterator
        dg_ptr = dg_in.iterator
        db_ptr = db_in.iterator
        dA_ptr = dA_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator
        chunk_indices_ptr = chunk_indices_in.iterator

        B, T, H, HV, K, V = problem_size
        BT = self.BT

        data_B = Int32(1)
        NT = total_nt

        # ===================== GMEM layouts =====================
        # Token-indexed tensors: (T, dim, (H, data_B))
        # q, k: (T, K, (H, data_B)) bf16
        qk_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        q = cute.make_tensor(q_ptr, qk_layout)
        k = cute.make_tensor(k_ptr, qk_layout)

        # v, v_new, do, dv, dv2: (T, V, (HV, data_B)) bf16
        tv_layout = cute.make_layout(
            (T, V, (HV, data_B)),
            stride=(HV * V, 1, (V, T * HV * V)),
        )
        v = cute.make_tensor(v_ptr, tv_layout)
        v_new = cute.make_tensor(v_new_ptr, tv_layout)
        do = cute.make_tensor(do_ptr, tv_layout)
        dv = cute.make_tensor(dv_ptr, tv_layout)
        dv2 = cute.make_tensor(dv2_ptr, tv_layout)

        # g: (T, K, (HV, data_B)) fp32
        g_layout = cute.make_layout(
            (T, K, (HV, data_B)),
            stride=(HV * K, 1, (K, T * HV * K)),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        # beta: (T, (HV, data_B)) fp32
        beta_layout = cute.make_layout(
            (T, (HV, data_B)),
            stride=(HV, (1, T * HV)),
        )
        beta = cute.make_tensor(beta_ptr, beta_layout)

        # A: (T, BT, (HV, data_B)) bf16
        # NOTE: for A as operand A, A is loaded as transposed view to do MMA
        a_t_layout = cute.make_layout(
            (BT, T, (HV, data_B)),
            stride=(1, HV * BT, (BT, T * HV * BT)),
        )
        A_T = cute.make_tensor(A_ptr, a_t_layout)

        # dq, dk: (T, K, (HV, data_B)) fp32
        dqk_layout = cute.make_layout(
            (T, K, (HV, data_B)),
            stride=(HV * K, 1, (K, T * HV * K)),
        )
        dq = cute.make_tensor(dq_ptr, dqk_layout)
        dk = cute.make_tensor(dk_ptr, dqk_layout)

        # dg: (T, K, (HV, data_B)) fp32
        dg = cute.make_tensor(dg_ptr, dqk_layout)

        # db: (T, (HV, data_B)) fp32
        db = cute.make_tensor(db_ptr, beta_layout)

        # dA: (T, BT, (HV, data_B)) fp32
        dA_layout = cute.make_layout(
            (T, BT, (HV, data_B)),
            stride=(HV * BT, 1, (BT, T * HV * BT)),
        )
        dA_out = cute.make_tensor(dA_ptr, dA_layout)

        h_nt_total = NT

        # h row-major: (K, V, (h_nt_total, HV)) as operand B
        h_layout = cute.make_layout(
            (K, V, (h_nt_total, HV)),
            stride=(V, 1, (HV * K * V, K * V)),
        )
        h = cute.make_tensor(h_ptr, h_layout)
        dh = cute.make_tensor(dh_ptr, h_layout)

        # ===================== MMA setup (4 objects) =====================
        # All use tcgen05.mma.ws (Layout E, M=64, cta_group::1).
        # 1. vloop_tiled_mma: SS K,K (64,128) — dq, dk, dw
        #    dq += do @ h, dk += vnew @ dh, dw += dv @ h
        vloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,  # A: K-major
            tcgen05.OperandMajorMode.K,  # B: K-major
            self.acc_dtype,
            self.cta_group,
            self.vloop_gemm_tiler[:2],  # (64, 128)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 2. dA_vloop_tiled_mma: SS K,K (64,64) — dA vloop + kpost_dA
        #    dA += dv @ v^T, dA += dw @ kg^T
        dA_vloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dA_vloop_tiler[:2],  # (64, 64)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 3. dvb_tiled_mma: SS MN,MN (64,64) — dvb + dkgb
        #    dvb = A @ dv, dkgb = A @ dw
        dvb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dvb_tiler[:2],  # (64, 64)
        )

        # dkgb_tiled_mma: SS MN,MN (64,128) - dkgb
        dkgb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dkgb_tiler[:2],  # (64, 128)
        )

        # dA_kloop_tiled_mma: SS K,K (64, 64)
        # dA += dw @ kg^T
        dA_kloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dA_tiler[:2],  # (64, 64)
        )

        # dA2post_tiled_mma: SS K,K (64,64)
        # dA = dA @ A
        dA2post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
            # tcgen05.OperandSource.SMEM,  # SS mode
        )

        # dA3post_tiled_mma: SS MN,MN (64,64)
        # dA = A @ dA
        dA3post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
            # tcgen05.OperandSource.SMEM,  # SS mode
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # SS opA layout: do/vnew/dv [BT,BV]=[64,64] K-major
        vloop_opA_smem = sm100_utils.make_smem_layout_a(
            vloop_tiled_mma,
            self.vloop_gemm_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opB layout: h/dh [BK,BV]=[128,64] K-major
        vloop_opB_smem = sm100_utils.make_smem_layout_b(
            vloop_tiled_mma,
            self.vloop_gemm_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opB layout: v [BV,BT]=[128,64] K-major (dA vloop)
        v_opB_smem = sm100_utils.make_smem_layout_b(
            dA_vloop_tiled_mma,
            self.dA_vloop_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opA layout: A MN-major [BT,BT]=[64,64]
        A_mn_opA_smem = sm100_utils.make_smem_layout_a(
            dvb_tiled_mma,
            self.dvb_tiler,
            self.io_dtype,
            self.a_stage,
        )

        # opB: dv MN-major [BV,BT]=[64,64]
        dv_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dvb_tiled_mma,
            self.dvb_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # opA: dw K-major [BT,BK]=[64,128]
        dw_k_opA_smem = sm100_utils.make_smem_layout_a(
            dA_vloop_tiled_mma,
            self.kloop_dA_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opB: dw MN-major [BK,BT]
        dw_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dkgb_tiled_mma,
            self.kloop_dkgb_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opB: kg^T K-major [BT, BK]
        kg_k_opB_smem = sm100_utils.make_smem_layout_b(
            dA_kloop_tiled_mma,
            self.kloop_dA_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opA: dA K-major [BT,BT]
        dA_k_opA_smem = sm100_utils.make_smem_layout_a(
            dA2post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.mma_stage,
        )

        # opB: A K-major [BT,BT]
        A_k_opB_smem = sm100_utils.make_smem_layout_b(
            dA2post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.a_stage,
        )

        # opB: dA MN-major [BT,BT]
        dA_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dA3post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.mma_stage,
        )

        # --- Epilogue (non-MMA) layouts ---
        g_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        k_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        q_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            1,
        )

        dg_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        # ===================== Cluster layout =====================
        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (vloop_tiled_mma.thr_id.shape,),
        )

        # ===================== TMA descriptors =====================
        # Strip stage dimension for TMA atom creation (expects 3 modes, not 4)
        vloop_opA_smem_no_stage = cute.select(vloop_opA_smem, mode=[0, 1, 2])
        vloop_opB_smem_no_stage = cute.select(vloop_opB_smem, mode=[0, 1, 2])
        v_opB_smem_no_stage = cute.select(v_opB_smem, mode=[0, 1, 2])
        A_mn_opA_smem_no_stage = cute.select(A_mn_opA_smem, mode=[0, 1, 2])

        tma_atom_dv, tma_tensor_dv = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dv,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_A, tma_tensor_A = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            A_T,
            A_mn_opA_smem_no_stage,
            self.dvb_tiler,
            dvb_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_h, tma_tensor_h = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h,
            vloop_opB_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_dh, tma_tensor_dh = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dh,
            vloop_opB_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            do,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_vnew, tma_tensor_vnew = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v_new,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_opB_smem_no_stage,
            self.dA_vloop_tiler,
            dA_vloop_tiled_mma,
            cluster_layout.shape,
        )

        g_epi_smem_no_stage = cute.select(g_epi_smem_layout, mode=[0, 1])
        tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g,
            g_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        k_epi_smem_no_stage = cute.select(k_epi_smem_layout, mode=[0, 1])
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            k,
            k_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        q_epi_smem_no_stage = cute.select(q_epi_smem_layout, mode=[0, 1])
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            q,
            q_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        dg_epi_smem_no_stage = cute.select(dg_epi_smem_layout, mode=[0, 1])
        tma_atom_dg, tma_tensor_dg = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dg,
            dg_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        # ===================== TMA byte counts =====================
        self.tma_bytes_A = cute.size_in_bytes(self.io_dtype, A_mn_opA_smem_no_stage)
        self.tma_bytes_dv = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_h = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        self.tma_bytes_dh = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        self.tma_bytes_do = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_vnew = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_g = cute.size_in_bytes(self.g_dtype, g_epi_smem_no_stage)
        self.tma_bytes_v = cute.size_in_bytes(self.io_dtype, v_opB_smem_no_stage)
        self.tma_bytes_k = cute.size_in_bytes(self.io_dtype, k_epi_smem_no_stage)
        self.tma_bytes_q = cute.size_in_bytes(self.io_dtype, q_epi_smem_no_stage)

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            # ======= mbarrier =======
            bar_load_A: cute.struct.MemRange[Int64, self.a_stage * 2]
            bar_load_dv: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_mma_dvb: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_load_beta: cute.struct.MemRange[Int64, 1 * 2]
            bar_tma_h: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_h: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_tma_dh: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_dh: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_tma_v: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_v: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_load_do: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_g: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_load_vnew: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_q: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_load_k: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_mma_dq: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dw: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dk: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dkgb: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA2: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA3: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_done_vloop: cute.struct.MemRange[Int64, self.mma_stage]
            bar_prologue_dw: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_prologue_kg: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_prologue_dA2: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_prologue_dA3: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_store_dg: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            # TMEM holding buffer
            tmem_holding_buf: Int32
            # A, stage=2, [BT,BT], 16KB
            buf_A: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(A_mn_opA_smem)],
                self.buffer_align_bytes,
            ]
            # k, stage=1, [BT,BK], 16KB
            buf_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # g, stage=1, [BT,BK], 32KB
            buf_g: cute.struct.Align[
                cute.struct.MemRange[self.g_dtype, cute.cosize(g_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # q, stage=1, [BT,BK], 16KB
            buf_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # V-loop buffers, stage=2
            # h, dh, [BK,BV] 32KB*2
            buf_h: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opB_smem)],
                self.buffer_align_bytes,
            ]
            buf_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opB_smem)],
                self.buffer_align_bytes,
            ]
            # do, dv, v_new, v, [BT,BV] 16KB*4
            buf_do: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_dv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_vnew: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_v: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_opB_smem)],
                self.buffer_align_bytes,
            ]

            # dw, stage=1, [BT,BK] 16KB
            buf_dw: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dw_k_opA_smem)],
                self.buffer_align_bytes,
            ]
            # Scalars
            s_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT],
                128,
            ]
            # 2 slots per row, one per warpgroup, for deterministic db reduction
            # (avoids cross-wg fp32 atomicAdd on shared memory).
            s_db: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT * 2],
                128,
            ]
            s_gn: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK],
                128,
            ]
            s_dgk: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK],
                128,
            ]

        self.shared_storage = SharedStorage

        # ===================== cu_seqlens / chunk_indices tensors =====================
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((B + 1,)))
        chunk_indices = cute.make_tensor(chunk_indices_ptr, cute.make_layout((total_nt, 2), stride=(2, 1)))

        # ===================== Grid =====================
        grid = self._compute_grid(B, T, HV, total_nt=total_nt)

        # ===================== Launch kernel =====================
        self.kernel(
            # MMA objects (4)
            vloop_tiled_mma,
            dA_vloop_tiled_mma,
            dvb_tiled_mma,
            dA_kloop_tiled_mma,
            dA2post_tiled_mma,
            dA3post_tiled_mma,
            # TMA atoms
            tma_atom_dv,
            tma_tensor_dv,
            tma_atom_A,
            tma_tensor_A,
            tma_atom_h,
            tma_tensor_h,
            tma_atom_dh,
            tma_tensor_dh,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_g,
            tma_tensor_g,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_vnew,
            tma_tensor_vnew,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_dg,
            tma_tensor_dg,
            # SMEM layouts
            vloop_opA_smem,
            vloop_opB_smem,
            v_opB_smem,
            A_mn_opA_smem,
            dv_mn_opB_smem,
            dw_k_opA_smem,
            dw_mn_opB_smem,
            kg_k_opB_smem,
            A_k_opB_smem,
            dA_k_opA_smem,
            dA_mn_opB_smem,
            g_epi_smem_layout,
            k_epi_smem_layout,
            q_epi_smem_layout,
            # GMEM tensors
            q,
            k,
            g,
            beta,
            dq,
            dk,
            dv2,
            dg,
            db,
            dA_out,
            # Metadata
            cu_seqlens,
            chunk_indices,
            problem_size,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        # MMA objects (4)
        vloop_tiled_mma: cute.TiledMma,
        dA_vloop_tiled_mma: cute.TiledMma,
        dvb_tiled_mma: cute.TiledMma,
        dA_kloop_tiled_mma: cute.TiledMma,
        dA2post_tiled_mma: cute.TiledMma,
        dA3post_tiled_mma: cute.TiledMma,
        # TMA atoms + tensors
        tma_atom_dv: cute.CopyAtom,
        tma_tensor_dv: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_tensor_A: cute.Tensor,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
        tma_atom_dh: cute.CopyAtom,
        tma_tensor_dh: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_g: cute.CopyAtom,
        tma_tensor_g: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_vnew: cute.CopyAtom,
        tma_tensor_vnew: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_dg: cute.CopyAtom,
        tma_tensor_dg: cute.Tensor,
        # SMEM layouts
        vloop_opA_smem: cute.ComposedLayout,
        vloop_opB_smem: cute.ComposedLayout,
        v_opB_smem: cute.ComposedLayout,
        A_mn_opA_smem: cute.ComposedLayout,
        dv_mn_opB_smem: cute.ComposedLayout,
        dw_k_opA_smem: cute.ComposedLayout,
        dw_mn_opB_smem: cute.ComposedLayout,
        kg_k_opB_smem: cute.ComposedLayout,
        A_k_opB_smem: cute.ComposedLayout,
        dA_k_opA_smem: cute.ComposedLayout,
        dA_mn_opB_smem: cute.ComposedLayout,
        g_epi_smem_layout: cute.ComposedLayout,
        k_epi_smem_layout: cute.ComposedLayout,
        q_epi_smem_layout: cute.ComposedLayout,
        # GMEM tensors
        q_gmem: cute.Tensor,
        k_gmem: cute.Tensor,
        g_gmem: cute.Tensor,
        beta_gmem: cute.Tensor,
        dq_gmem: cute.Tensor,
        dk_gmem: cute.Tensor,
        dv2_gmem: cute.Tensor,
        dg_gmem: cute.Tensor,
        db_gmem: cute.Tensor,
        dA_gmem: cute.Tensor,
        # Metadata
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32, Int32],  # (B, T, H, HV, K, V)
    ):
        B, T, H, HV, K, V = problem_size
        BT = self.BT

        # ===================== Persistent work decode =====================
        # Grid: (min(num_sm * occ, total_tiles), 1, 1) — persistent
        block_idx_x = cute.arch.block_idx()[0]
        grid_dim_x = cute.arch.grid_dim()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = thread_idx % 32

        total_work_units = chunk_indices.layout.shape[0] * HV
        num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x

        num_cuda_warps_total = len(self.cuda_warp_ids) + len(self.cuda2_warp_ids)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_dv)
            cpasync.prefetch_descriptor(tma_atom_h)
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_g)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_vnew)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_dg)

        # ===================== SMEM allocation =====================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Barrier Initialization
        bar_mma_done_vloop_ptr = storage.bar_mma_done_vloop.data_ptr()
        # NOTE: for h, dh and v, consumer contains both MMA and CUDA Core, so we use original mbarrier declaration instead of pipeline utils
        bar_tma_h_ptr = storage.bar_tma_h.data_ptr()
        bar_mma_cuda_h_ptr = storage.bar_mma_cuda_h.data_ptr()
        bar_tma_dh_ptr = storage.bar_tma_dh.data_ptr()
        bar_mma_cuda_dh_ptr = storage.bar_mma_cuda_dh.data_ptr()
        bar_tma_v_ptr = storage.bar_tma_v.data_ptr()
        bar_mma_cuda_v_ptr = storage.bar_mma_cuda_v.data_ptr()
        if warp_idx == 0:
            with elect_one():
                for i in cutlass.range(self.mma_stage, unroll_full=True):
                    mbarrier_init(bar_mma_done_vloop_ptr + i, 1)
                for i in cutlass.range(self.vloop_stage, unroll_full=True):
                    mbarrier_init(bar_tma_h_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_h_ptr + i, num_cuda_warps_total * 32 + 1)
                    mbarrier_init(bar_tma_dh_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_dh_ptr + i, num_cuda_warps_total * 32 + 1)
                    mbarrier_init(bar_tma_v_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_v_ptr + i, num_cuda_warps_total * 32 + 1)
                mbarrier_init_fence()

        # ====== Pipeline Definition ======
        pipeline_load_A = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_A.data_ptr(),
            num_stages=self.a_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_A,
        )
        pipeline_load_dv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_dv.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_dv,
        )
        pipeline_load_do = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_do.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_do,
        )
        pipeline_load_vnew = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_vnew.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_vnew,
        )
        pipeline_load_g = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_g.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total + len(self.aux_warp_ids)),
            tx_count=self.tma_bytes_g,
        )
        pipeline_load_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_k.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total),
            tx_count=self.tma_bytes_k,
        )
        pipeline_load_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_q.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total),
            tx_count=self.tma_bytes_q,
        )
        pipeline_mma_dvb = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dvb.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dq = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dq.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dk = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dk.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dw = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dw.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA2 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA2.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA3 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA3.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_prologue_dw = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dw.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_kg = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_kg.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_dA2 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dA2.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_dA3 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dA3.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_mma_dkgb = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dkgb.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_load_beta = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_load_beta.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(len(self.aux_warp_ids) * 32),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_store_dg = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_store_dg.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len(self.aux_warp_ids) * 32),
        )

        # ===================== TMEM allocation =====================
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        vloop_opA_smem_no_stage = cute.select(vloop_opA_smem, mode=[0, 1, 2])
        vloop_opB_smem_no_stage = cute.select(vloop_opB_smem, mode=[0, 1, 2])
        A_mn_opA_smem_no_stage = cute.select(A_mn_opA_smem, mode=[0, 1, 2])
        v_opB_smem_no_stage = cute.select(v_opB_smem, mode=[0, 1, 2])

        sA = storage.buf_A.get_tensor(A_mn_opA_smem.outer, swizzle=A_mn_opA_smem.inner)
        sDv = storage.buf_dv.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sH = storage.buf_h.get_tensor(vloop_opB_smem.outer, swizzle=vloop_opB_smem.inner)
        sDh = storage.buf_dh.get_tensor(vloop_opB_smem.outer, swizzle=vloop_opB_smem.inner)
        sDo = storage.buf_do.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sVnew = storage.buf_vnew.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sV = storage.buf_v.get_tensor(v_opB_smem.outer, swizzle=v_opB_smem.inner)

        sDv_ptr_base = storage.buf_dv.data_ptr().toint()
        vloop_opA_bytes_per_stage = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        sDo_ptr_base = storage.buf_do.data_ptr().toint()
        sVnew_ptr_base = storage.buf_vnew.data_ptr().toint()
        sV_ptr_base = storage.buf_v.data_ptr().toint()
        v_opB_bytes_per_stage = cute.size_in_bytes(self.io_dtype, v_opB_smem_no_stage)
        sH_ptr_base = storage.buf_h.data_ptr().toint()
        sDh_ptr_base = storage.buf_dh.data_ptr().toint()
        vloop_opB_bytes_per_stage = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        sA_ptr_base = storage.buf_A.data_ptr().toint()
        A_bytes_per_stage = cute.size_in_bytes(self.io_dtype, A_mn_opA_smem_no_stage)

        # NOTE: make_umma_smem_desc requires the iterator to carry the swizzle
        # (and ≥16B alignment). When constructing a tensor over a ComposedLayout
        # via make_ptr+make_tensor, the swizzle ends up composed on the layout
        # rather than the iterator, which breaks make_umma_smem_desc. Use
        # recast_ptr to move the swizzle onto the iterator and pair it with the
        # underlying (non-swizzle) outer layout.
        sDv_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dv.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dv_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dv_mn_opB_smem.outer,
        )
        sDw_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dw.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dw_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dw_mn_opB_smem.outer,
        )
        sDw_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dw.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dw_k_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            dw_k_opA_smem.outer,
        )
        sDv_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dv.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sV_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_v.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=v_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            v_opB_smem.outer,
        )
        sA_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_A.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=A_mn_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            A_mn_opA_smem.outer,
        )
        sDo_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_do.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sVnew_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_vnew.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sH_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_h.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=vloop_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opB_smem.outer,
        )
        sDh_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_dh.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=vloop_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opB_smem.outer,
        )
        sKG_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_k.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=kg_k_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            kg_k_opB_smem.outer,
        )
        sA_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_A.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=A_k_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            A_k_opB_smem.outer,
        )
        sDA_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.io_dtype, storage.buf_q.data_ptr().toint(), cute.AddressSpace.smem, assumed_align=128),
                swizzle_=dA_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dA_mn_opB_smem.outer,
        )
        sDA_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_q.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dA_k_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            dA_k_opA_smem.outer,
        )
        sG_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.g_dtype,
                    storage.buf_g.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=g_epi_smem_layout.inner,
                dtype=self.g_dtype,
            ),
            g_epi_smem_layout.outer,
        )
        sG_raw_ptr = cute.make_ptr(self.g_dtype, storage.buf_g.data_ptr().toint(), cute.AddressSpace.smem)
        sK_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_k.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=k_epi_smem_layout.inner,
                dtype=self.io_dtype,
            ),
            k_epi_smem_layout.outer,
        )
        sK_raw_ptr = cute.make_ptr(self.io_dtype, storage.buf_k.data_ptr().toint(), cute.AddressSpace.smem)
        sDw_raw_ptr = cute.make_ptr(self.io_dtype, storage.buf_dw.data_ptr().toint(), cute.AddressSpace.smem)
        sQ_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_q.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=q_epi_smem_layout.inner,
                dtype=self.io_dtype,
            ),
            q_epi_smem_layout.outer,
        )
        sQ_raw_ptr = cute.make_ptr(self.io_dtype, storage.buf_q.data_ptr().toint(), cute.AddressSpace.smem)

        # Scalar SMEM buffers (plain layouts, no swizzle)
        sBeta = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_beta.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT,), stride=(1,)),
        )
        # sDb layout: (BT, 2). Inner dim = wg_idx slot. Stride (1, BT) so each
        # wg's column is contiguous (better for the reduce in Phase 3).
        sDb = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_db.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT, 2), stride=(1, self.BT)),
        )
        sDgk = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_dgk.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BK,), stride=(1,)),
        )
        sGn = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_gn.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BK,), stride=(1,)),
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        tmem.allocate(TMEM_TOTAL)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ===================== Warp dispatch =====================
        # CUDA Core loop body
        if warp_idx in self.cuda_warp_ids or warp_idx in self.cuda2_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)

            load_beta_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_g_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            mma_dvb_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            mma_dq_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            mma_dw_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            mma_dk_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            load_k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            prologue_dw_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)
            prologue_kg_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)
            mma_dgkb_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            load_q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            mma_dA_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            mma_dA2_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            mma_dA3_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            prologue_dA2_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            prologue_dA3_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            store_dg_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)

            wg_idx = tidx // 128
            local_tidx = tidx % 128
            warp_id = local_tidx // 32
            warp_row_tile = warp_id % 2
            warp_col_tile = warp_id // 2
            row = warp_row_tile * 32 + lane_idx  # BT1
            bk_num_cols = self.BK // 2
            bv_num_cols = self.BV // 2
            bk_num_cols_per_wg = bk_num_cols // 2
            bv_num_cols_per_wg = bv_num_cols // 2
            bt_num_cols_per_wg = self.BT // 4
            # ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e
            bv_col_base = warp_col_tile * (self.BV // 2) + wg_idx * bv_num_cols_per_wg
            bk_col_base = warp_col_tile * (self.BK // 2) + wg_idx * bk_num_cols_per_wg
            bt_col_base = warp_col_tile * (self.BT // 2) + wg_idx * bt_num_cols_per_wg
            # 8 fp32 store each time for store_256b
            num_stores_f32 = bk_num_cols_per_wg // 8

            vloop_stage_idx = 0
            vloop_phase = 0
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index
                # Decode chunk_indices
                batch_idx = chunk_indices[(i_t, 0)]
                tile_idx = chunk_indices[(i_t, 1)]
                tok_offset = cu_seqlens[(batch_idx,)]
                seq_len = cu_seqlens[(batch_idx + 1,)] - tok_offset
                sub_seq_len = min(self.BT, seq_len - tile_idx * self.BT)

                # NOTE: must sync before next wu_iter's `sDgk[local_tidx] = 0`
                # init, otherwise WG0 of next iter may overwrite sDgk while
                # WG1 of this iter (row == sub_seq_len - 1 lane) is still
                # reading sDgk[col] above. This was the source of the
                # non-deterministic dg accuracy bug.
                self.cuda_wg_sync_barrier.arrive_and_wait()
                # fill db, dgk to 0. Each wg zeroes its own sDb column.
                if local_tidx < self.BT:
                    sDb[local_tidx, 0] = Float32(0.0)
                    sDb[local_tidx, 1] = Float32(0.0)
                if local_tidx < self.BK:
                    sDgk[local_tidx] = Float32(0.0)
                self.cuda_wg_sync_barrier.arrive_and_wait()

                pipeline_load_beta.consumer_wait(load_beta_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                beta_val = sBeta[(row,)]
                db_val = Float32(0.0)
                for v_iter in cutlass.range(self.num_v_tiles):
                    # dgk += sum(h * dh, axis=0)
                    mbarrier_wait(bar_tma_h_ptr + vloop_stage_idx, vloop_phase)
                    mbarrier_wait(bar_tma_dh_ptr + vloop_stage_idx, vloop_phase)

                    sH_raw_ptr = cute.make_ptr(
                        self.io_dtype, sH_ptr_base + vloop_stage_idx * vloop_opB_bytes_per_stage, cute.AddressSpace.smem
                    )
                    sDh_raw_ptr = cute.make_ptr(
                        self.io_dtype, sDh_ptr_base + vloop_stage_idx * vloop_opB_bytes_per_stage, cute.AddressSpace.smem
                    )
                    # each thread in one WG processes one row
                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if wg_idx == 0:
                        for i in cutlass.range_constexpr(self.BV // 8):
                            col = i * 8
                            h_vals = smem_load_bf16x8_sw128(sH_raw_ptr, local_tidx, col)
                            dh_vals = smem_load_bf16x8_sw128(sDh_raw_ptr, local_tidx, col)
                            h_dh_vals = cute.make_rmem_tensor((8,), Float32)
                            h_dh_vals.store(h_vals.load().to(Float32) * dh_vals.load().to(Float32))
                            for j in cutlass.range_constexpr(8):
                                sDgk[(local_tidx,)] += h_dh_vals[j]

                    mbarrier_arrive(bar_mma_cuda_h_ptr + vloop_stage_idx)
                    mbarrier_arrive(bar_mma_cuda_dh_ptr + vloop_stage_idx)

                    pipeline_mma_dvb.consumer_wait(mma_dvb_consumer_state)
                    tcgen05_fence_after()
                    dvb_i32 = tcgen05_ld_32x32b(bv_num_cols_per_wg, TMEM_FLEX_OFF + wg_idx * bv_num_cols_per_wg)
                    tcgen05_fence_before()
                    cute.arch.fence_view_async_tmem_load()

                    pipeline_mma_dvb.consumer_release(mma_dvb_consumer_state)
                    mma_dvb_consumer_state.advance()

                    dvb_f32 = reinterpret_cast(dvb_i32, Int32, bv_num_cols_per_wg, Float32)
                    dvb_f32_val = TensorSSA(dvb_f32, (bv_num_cols_per_wg,), Float32)

                    # db += sum(dvb * v, axis=1)
                    mbarrier_wait(bar_tma_v_ptr + vloop_stage_idx, vloop_phase)
                    rV_bf16 = cute.make_rmem_tensor((bv_num_cols_per_wg,), self.io_dtype)
                    sV_raw_ptr_cur = cute.make_ptr(
                        self.io_dtype, sV_ptr_base + vloop_stage_idx * v_opB_bytes_per_stage, cute.AddressSpace.smem
                    )
                    if row < sub_seq_len:
                        for i in cutlass.range_constexpr(bv_num_cols_per_wg // 8):
                            col_base = bv_col_base + i * 8
                            vals = smem_load_bf16x8_sw128(sV_raw_ptr_cur, row, col_base)
                            rV_bf16[i * 8 + 0] = vals[0]
                            rV_bf16[i * 8 + 1] = vals[1]
                            rV_bf16[i * 8 + 2] = vals[2]
                            rV_bf16[i * 8 + 3] = vals[3]
                            rV_bf16[i * 8 + 4] = vals[4]
                            rV_bf16[i * 8 + 5] = vals[5]
                            rV_bf16[i * 8 + 6] = vals[6]
                            rV_bf16[i * 8 + 7] = vals[7]
                    else:
                        rV_bf16.fill(BFloat16(0.0))
                    rV_fp32 = cute.make_rmem_tensor((bv_num_cols_per_wg,), Float32)
                    rV_fp32.store(rV_bf16.load().to(Float32))
                    rV_fp32.store(rV_fp32.load() * dvb_f32_val)
                    if row < sub_seq_len:
                        for i in cutlass.range_constexpr(bv_num_cols_per_wg):
                            db_val += rV_fp32[i]

                    mbarrier_arrive(bar_mma_cuda_v_ptr + vloop_stage_idx)

                    # ── dv2 epilogue: dv2 = dvb * beta, cast to bf16, store to gmem ──
                    dvb_f32_rmem = cute.make_rmem_tensor((bv_num_cols_per_wg,), Float32)
                    dvb_f32_rmem.store(dvb_f32_val * beta_val)

                    dvb_bf16_rmem = cute.make_rmem_tensor((bv_num_cols_per_wg,), self.io_dtype)
                    dvb_bf16_rmem.store(dvb_f32_rmem.load().to(self.io_dtype))

                    # bf16 vector → i32 vector for store_256b (8 i32 = 16 bf16 = 32 bytes per store).
                    dvb_bf16_val = dvb_bf16_rmem.load()
                    dvb_i32_vec = reinterpret_cast(dvb_bf16_val, self.io_dtype, bv_num_cols_per_wg, Int32)
                    # bv_num_cols bf16 = bv_num_cols // 16 stores of 256b each.
                    num_stores_per_row = bv_num_cols_per_wg // 16  # = 4 for BV=128

                    base_addr = (
                        dv2_gmem.iterator
                        + (tok_offset + tile_idx * self.BT + row) * HV * V
                        + i_hv * V
                        + v_iter * self.BV
                        + bv_col_base
                    ).toint()
                    if row < sub_seq_len:
                        for s in cutlass.range_constexpr(num_stores_per_row):
                            chunk = subvec(dvb_i32_vec, s * 8, 8)
                            store_256b(base_addr + s * 32, chunk)

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                # gk_exp = exp2(g)
                pipeline_load_g.consumer_wait(load_g_consumer_state)
                # write to gn
                sGn[local_tidx] = sG_raw[(sub_seq_len - 1, local_tidx, 0)]

                # row-major load, match TMEM layout
                rG = cute.make_rmem_tensor((self.BK // 4,), self.g_dtype)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(self.BK // 4 // 4):
                        col_base = bk_col_base + i * 4
                        vals = smem_load_f32x4_sw128(sG_raw_ptr, row, col_base)
                        rG[i * 4 + 0] = vals[0]
                        rG[i * 4 + 1] = vals[1]
                        rG[i * 4 + 2] = vals[2]
                        rG[i * 4 + 3] = vals[3]
                else:
                    rG.fill(Float32(0.0))
                rG_val = rG.load()
                rG_exp_val = cute.exp2(rG_val, fastmath=self.use_fast_math)

                # wait for dq, dq=dq*gk_exp*scale, GMEM store
                pipeline_mma_dq.consumer_wait(mma_dq_consumer_state)
                tcgen05_fence_after()
                dq_i32 = tcgen05_ld_32x32b(bk_num_cols_per_wg, TMEM_DQ_ACC_OFF + wg_idx * bk_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dq.consumer_release(mma_dq_consumer_state)
                mma_dq_consumer_state.advance()

                dq_f32 = reinterpret_cast(dq_i32, Int32, bk_num_cols_per_wg, Float32)
                dq_f32_val = TensorSSA(dq_f32, (bk_num_cols_per_wg,), Float32)

                rDq = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rDq.store(dq_f32_val * rG_exp_val * Float32(self.scale))

                dq_f32_val_store = rDq.load()
                dq_i32_vec = reinterpret_cast(dq_f32_val_store, Float32, bk_num_cols_per_wg, Int32)
                # store to TMEM first to reduce register usage
                tcgen05_st_32x32b(bk_num_cols_per_wg, TMEM_DQ_SCALED_OFF + wg_idx * bk_num_cols_per_wg, dq_i32_vec)
                cute.arch.fence_view_async_tmem_store()
                dq_base_addr = (
                    dq_gmem.iterator + (tok_offset + tile_idx * self.BT + row) * HV * K + i_hv * K + bk_col_base
                ).toint()
                if row < sub_seq_len:
                    for s in cutlass.range_constexpr(num_stores_f32):
                        chunk = subvec(dq_i32_vec, s * 8, 8)
                        store_256b(dq_base_addr + s * 32, chunk)

                # wait for dw
                pipeline_mma_dw.consumer_wait(mma_dw_consumer_state)
                tcgen05_fence_after()
                dw_i32 = tcgen05_ld_32x32b(bk_num_cols_per_wg, TMEM_DW_ACC_OFF + wg_idx * bk_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dw.consumer_release(mma_dw_consumer_state)
                mma_dw_consumer_state.advance()

                # dw = -dw, convert to bf16, write to smem
                dw_f32 = reinterpret_cast(dw_i32, Int32, bk_num_cols_per_wg, Float32)
                dw_f32_val = TensorSSA(dw_f32, (bk_num_cols_per_wg,), Float32)

                dw_bf16_rmem = cute.make_rmem_tensor((bk_num_cols_per_wg,), BFloat16)
                if row < sub_seq_len:
                    dw_bf16_rmem.store((-dw_f32_val).to(BFloat16))
                else:
                    dw_bf16_rmem.fill(BFloat16(0.0))

                pipeline_prologue_dw.producer_acquire(prologue_dw_producer_state)
                # store bf16x8 each time
                dw_smem_num_stores = bk_num_cols_per_wg // 8
                for i in cutlass.range_constexpr(dw_smem_num_stores):
                    col_base = bk_col_base + i * 8
                    chunk = cute.local_tile(dw_bf16_rmem, (8,), (i,))
                    smem_store_bf16x8_sw128(sDw_raw_ptr, row, col_base, chunk)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dw.producer_commit(prologue_dw_producer_state)
                prologue_dw_producer_state.advance()

                pipeline_load_k.consumer_wait(load_k_consumer_state)
                # compute kg = k * gk_exp
                rK = cute.make_rmem_tensor((self.BK // 4,), self.io_dtype)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(self.BK // 4 // 8):
                        col_base = bk_col_base + i * 8
                        vals = smem_load_bf16x8_sw128(sK_raw_ptr, row, col_base)
                        rK[i * 8 + 0] = vals[0]
                        rK[i * 8 + 1] = vals[1]
                        rK[i * 8 + 2] = vals[2]
                        rK[i * 8 + 3] = vals[3]
                        rK[i * 8 + 4] = vals[4]
                        rK[i * 8 + 5] = vals[5]
                        rK[i * 8 + 6] = vals[6]
                        rK[i * 8 + 7] = vals[7]
                else:
                    rK.fill(BFloat16(0.0))
                rK_fp32 = cute.make_rmem_tensor((self.BK // 4,), Float32)
                rK_fp32.store(rK.load().to(Float32))
                rK_fp32_val = rK_fp32.load()
                rKG_val = rK_fp32_val * rG_exp_val

                # write kg to K smem,
                # notify dA += dw @ kg^T
                rKG_bf16 = cute.make_rmem_tensor((self.BK // 4,), BFloat16)
                rKG_bf16.store(rKG_val.to(BFloat16))

                pipeline_prologue_kg.producer_acquire(prologue_kg_producer_state)
                for i in cutlass.range_constexpr(self.BK // 4 // 8):
                    col_base = bk_col_base + i * 8
                    chunk_kg = cute.local_tile(rKG_bf16, (8,), (i,))
                    smem_store_bf16x8_sw128(sK_raw_ptr, row, col_base, chunk_kg)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_kg.producer_commit(prologue_kg_producer_state)
                prologue_kg_producer_state.advance()

                # wait for dkgb
                pipeline_mma_dkgb.consumer_wait(mma_dgkb_consumer_state)
                tcgen05_fence_after()
                dkgb_i32 = tcgen05_ld_32x32b(bk_num_cols_per_wg, TMEM_DKGB_ACC_OFF + wg_idx * bk_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dkgb.consumer_release(mma_dgkb_consumer_state)
                mma_dgkb_consumer_state.advance()

                # db += sum(dkgb * kg, axis=1)
                dkgb_f32 = reinterpret_cast(dkgb_i32, Int32, bk_num_cols_per_wg, Float32)
                dkgb_f32_val = TensorSSA(dkgb_f32, (bk_num_cols_per_wg,), Float32)
                rKgb_kg = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rKgb_kg.store(dkgb_f32_val * rKG_val)

                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                        db_val += rKgb_kg[i]

                # Deterministic db reduction without atomicAdd.
                # 4 partitions per row come from 4 warps (warp_row_tile in {0,1},
                # warp_col_tile in {0,1}) x 2 wgs. Reduce in a fixed order so
                # the result is bitwise reproducible across launches:
                #   Phase 1: warp_col_tile==0 writes its db_val into
                #            sDb[row, wg_idx]   (single writer per slot)
                #   Phase 2: warp_col_tile==1 RMW-adds its db_val into the
                #            same slot          (still single writer per slot)
                #   Phase 3: WG0 sums the 2 wg-slots in fixed order and stores
                #            to GMEM.
                # No race, no atomic, no fp ordering nondeterminism.
                if warp_col_tile == 0 and row < sub_seq_len:
                    sDb[row, wg_idx] = db_val
                self.cuda_wg_sync_barrier.arrive_and_wait()
                if warp_col_tile == 1 and row < sub_seq_len:
                    sDb[row, wg_idx] = sDb[row, wg_idx] + db_val
                self.cuda_wg_sync_barrier.arrive_and_wait()
                # store db to GMEM (WG0 only). Sum order is fixed (slot 0 + slot 1).
                if wg_idx == 0 and local_tidx < sub_seq_len:
                    db_sum = sDb[(local_tidx, 0)] + sDb[(local_tidx, 1)]
                    db_gmem[(tok_offset + tile_idx * self.BT + local_tidx, (i_hv, Int32(0)))] = db_sum

                # dk = dk * exp2(gn[None, :] - g)
                pipeline_mma_dk.consumer_wait(mma_dk_consumer_state)
                tcgen05_fence_after()
                dk_i32 = tcgen05_ld_32x32b(bk_num_cols_per_wg, TMEM_DK_ACC_OFF + wg_idx * bk_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dk.consumer_release(mma_dk_consumer_state)
                mma_dk_consumer_state.advance()

                dk_f32 = reinterpret_cast(dk_i32, Int32, bk_num_cols_per_wg, Float32)
                dk_f32_val = TensorSSA(dk_f32, (bk_num_cols_per_wg,), Float32)

                rDk = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                        exp_g_gn = cute.exp2(sGn[(bk_col_base + i,)] - rG_val[i], fastmath=self.use_fast_math)
                        rDk[i] = dk_f32_val[i] * exp_g_gn
                else:
                    rDk.fill(Float32(0.0))

                # kdk = k * dk
                rKdk = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rKdk.store(rK_fp32.load() * rDk.load())

                # gb = gk_exp * beta[:, None]
                rGb = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rGb.store(rG_exp_val * beta_val)

                # dk = dk + dkgb * gb
                rDk.store(rDk.load() + dkgb_f32_val * rGb.load())
                rDk_val = rDk.load()
                dk_i32_vec = reinterpret_cast(rDk_val, Float32, bk_num_cols_per_wg, Int32)
                # GMEM store dk
                # 8 fp32 store each time for store_256b
                dk_base_addr = (
                    dk_gmem.iterator + (tok_offset + tile_idx * self.BT + row) * HV * K + i_hv * K + bk_col_base
                ).toint()
                if row < sub_seq_len:
                    for s in cutlass.range_constexpr(num_stores_f32):
                        chunk_dk = subvec(dk_i32_vec, s * 8, 8)
                        store_256b(dk_base_addr + s * 32, chunk_dk)

                # dgk += sum(kdk, axis=0)
                # write kdk to G SMEM then do BT-dim reduce
                for i in cutlass.range_constexpr(self.BK // 4 // 4):
                    col_base = bk_col_base + i * 4
                    chunk_kdk = cute.local_tile(rKdk, (4,), (i,))
                    smem_store_f32x4_sw128(sG_raw_ptr, row, col_base, chunk_kdk)
                self.cuda_wg_sync_barrier.arrive_and_wait()

                # dgk *= exp2(gn)
                if wg_idx == 0:
                    sDgk[(local_tidx,)] *= cute.exp2(sGn[(local_tidx,)], fastmath=self.use_fast_math)

                self.cuda_wg_sync_barrier.arrive_and_wait()
                if wg_idx == 0:
                    sum = Float32(0.0)
                    for r in cutlass.range(self.BT, unroll_full=True):
                        sum += sG_raw[(r, local_tidx, 0)]
                    sDgk[(local_tidx,)] += sum

                # dg1 = kg * dkgb * beta[:, None], can reuse kg RMEM
                rDg = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rDg.store(rKG_val * dkgb_f32_val * beta_val)

                pipeline_load_q.consumer_wait(load_q_consumer_state)
                # dg2 = q * dq - kdk + dg1
                rQ = cute.make_rmem_tensor((bk_num_cols_per_wg,), self.io_dtype)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(self.BK // 4 // 8):
                        col_base = bk_col_base + i * 8
                        vals = smem_load_bf16x8_sw128(sQ_raw_ptr, row, col_base)
                        rQ[i * 8 + 0] = vals[0]
                        rQ[i * 8 + 1] = vals[1]
                        rQ[i * 8 + 2] = vals[2]
                        rQ[i * 8 + 3] = vals[3]
                        rQ[i * 8 + 4] = vals[4]
                        rQ[i * 8 + 5] = vals[5]
                        rQ[i * 8 + 6] = vals[6]
                        rQ[i * 8 + 7] = vals[7]
                else:
                    rQ.fill(BFloat16(0.0))
                dq_scaled_i32 = tcgen05_ld_32x32b(bk_num_cols_per_wg, TMEM_DQ_SCALED_OFF + wg_idx * bk_num_cols_per_wg)
                cute.arch.fence_view_async_tmem_load()
                dq_scaled_f32 = reinterpret_cast(dq_scaled_i32, Int32, bk_num_cols_per_wg, Float32)
                dq_scaled_f32_val = TensorSSA(dq_scaled_f32, (bk_num_cols_per_wg,), Float32)
                rDg.store(rQ.load().to(Float32) * dq_scaled_f32_val + rDg.load() - rKdk.load())

                self.cuda_wg_sync_barrier.arrive_and_wait()
                # dg = dg2 + m_last * dgk, GMEM store dg
                if row == sub_seq_len - 1:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                        col = bk_col_base + i
                        rDg[i] += sDgk[(col,)]

                # Stage dg to SMEM first. A dedicated store warp later does
                # SMEM -> RMEM -> GMEM with store_256b, keeping GMEM store
                # address/vector live ranges out of the high-register CC path.
                pipeline_store_dg.producer_acquire(store_dg_producer_state)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg // 4):
                        col_base = bk_col_base + i * 4
                        chunk_dg = cute.local_tile(rDg, (4,), (i,))
                        smem_store_f32x4_sw128(sG_raw_ptr, row, col_base, chunk_dg)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_store_dg.producer_commit(store_dg_producer_state)
                store_dg_producer_state.advance()

                pipeline_load_g.consumer_release(load_g_consumer_state)
                load_g_consumer_state.advance()

                pipeline_mma_dA.consumer_wait(mma_dA_consumer_state)
                tcgen05_fence_after()
                dA_i32 = tcgen05_ld_32x32b(bt_num_cols_per_wg, TMEM_DA_ACC_OFF + wg_idx * bt_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dA.consumer_release(mma_dA_consumer_state)
                mma_dA_consumer_state.advance()
                # NOTE: only release k smem after dA finished, because kg reuses k smem in dA += dw @ kg^T
                pipeline_load_k.consumer_release(load_k_consumer_state)
                load_k_consumer_state.advance()

                # dA = dA * beta[None, :], apply strict lower-triangular mask.
                # Triton reference multiplies by the column beta (`b_beta[None, :]`)
                # and keeps only `row > col`.
                dA_f32 = reinterpret_cast(dA_i32, Int32, bt_num_cols_per_wg, Float32)
                dA_f32_val = TensorSSA(dA_f32, (bt_num_cols_per_wg,), Float32)
                rDA = cute.make_rmem_tensor((bt_num_cols_per_wg,), BFloat16)
                for i in cutlass.range_constexpr(bt_num_cols_per_wg):
                    col = bt_col_base + i
                    beta_col = sBeta[(col,)]
                    dA_scaled = (dA_f32_val[i] * beta_col).to(BFloat16)
                    if col < row:
                        rDA[i] = dA_scaled
                    else:
                        rDA[i] = BFloat16(0.0)
                if row >= sub_seq_len:
                    rDA.fill(BFloat16(0.0))

                pipeline_prologue_dA2.producer_acquire(prologue_dA2_producer_state)

                for i in cutlass.range_constexpr(bt_num_cols_per_wg // 8):
                    col_base = bt_col_base + i * 8
                    chunk_dA = cute.local_tile(rDA, (8,), (i,))
                    smem_store_bf16x8_sw128(sQ_raw_ptr, row, col_base, chunk_dA)
                # notify dA2 = dA @ A
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dA2.producer_commit(prologue_dA2_producer_state)
                prologue_dA2_producer_state.advance()

                pipeline_load_beta.consumer_release(load_beta_consumer_state)
                load_beta_consumer_state.advance()

                # wait for dA2
                pipeline_mma_dA2.consumer_wait(mma_dA2_consumer_state)
                tcgen05_fence_after()
                dA2_i32 = tcgen05_ld_32x32b(bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_prologue_dA3.producer_acquire(prologue_dA3_producer_state)
                # write dA2 to smem notify dA2 = A @ dA2
                dA2_f32 = reinterpret_cast(dA2_i32, Int32, bt_num_cols_per_wg, Float32)
                dA2_f32_val = TensorSSA(dA2_f32, (bt_num_cols_per_wg,), Float32)
                rDA2 = cute.make_rmem_tensor((bt_num_cols_per_wg,), BFloat16)
                if row < sub_seq_len:
                    rDA2.store(dA2_f32_val.to(BFloat16))
                else:
                    rDA2.fill(BFloat16(0.0))
                for i in cutlass.range_constexpr(bt_num_cols_per_wg // 8):
                    col_base = bt_col_base + i * 8
                    chunk_dA2 = cute.local_tile(rDA2, (8,), (i,))
                    smem_store_bf16x8_sw128(sQ_raw_ptr, row, col_base, chunk_dA2)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dA3.producer_commit(prologue_dA3_producer_state)
                prologue_dA3_producer_state.advance()

                # wait for dA2
                pipeline_mma_dA3.consumer_wait(mma_dA3_consumer_state)
                tcgen05_fence_after()
                dA3_i32 = tcgen05_ld_32x32b(bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg)
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                # release mma dA2 after dA3 is finished, protect DA2 TMEM
                pipeline_mma_dA2.consumer_release(mma_dA2_consumer_state)
                mma_dA2_consumer_state.advance()
                pipeline_mma_dA3.consumer_release(mma_dA3_consumer_state)
                mma_dA3_consumer_state.advance()
                # NOTE: release smem Q because we reuse to store bf16 dA
                pipeline_load_q.consumer_release(load_q_consumer_state)
                load_q_consumer_state.advance()

                # dA = -dA, apply strict lower-triangular mask
                dA3_f32 = reinterpret_cast(dA3_i32, Int32, bt_num_cols_per_wg, Float32)
                dA3_f32_val = TensorSSA(dA3_f32, (bt_num_cols_per_wg,), Float32)
                rDA3 = cute.make_rmem_tensor((bt_num_cols_per_wg,), Float32)
                rDA3.store(-dA3_f32_val)
                for i in cutlass.range_constexpr(bt_num_cols_per_wg):
                    col = bt_col_base + i
                    if col >= row:
                        rDA3[i] = Float32(0.0)
                rDA3_val = rDA3.load()
                dA3_i32_vec = reinterpret_cast(rDA3_val, Float32, bt_num_cols_per_wg, Int32)
                # GMEM store dA
                num_stores_dA = bt_num_cols_per_wg // 8
                dA_base_addr = (
                    dA_gmem.iterator + (tok_offset + tile_idx * self.BT + row) * HV * BT + i_hv * BT + bt_col_base
                ).toint()
                if row < sub_seq_len:
                    for s in cutlass.range_constexpr(num_stores_dA):
                        chunk_dA_store = subvec(dA3_i32_vec, s * 8, 8)
                        store_256b(dA_base_addr + s * 32, chunk_dA_store)

        # Load loop body
        elif warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            load_A_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.a_stage)
            load_dv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
            load_do_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
            load_vnew_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
            load_g_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)
            load_k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)
            load_q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kloop_stage)

            vloop_stage_idx = 0
            vloop_phase = 1  # init as 1 for producer
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index

                # Decode chunk_indices
                batch_idx = chunk_indices[(i_t, 0)]
                tile_idx = chunk_indices[(i_t, 1)]
                tok_offset = cu_seqlens[(batch_idx,)]
                seq_len = cu_seqlens[(batch_idx + 1,)] - tok_offset
                sub_seq_len = min(self.BT, seq_len - tile_idx * self.BT)

                # Load A
                tma_A_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_A)
                tAsA, tAgA = self._tma_partition_A(
                    tma_atom_A,
                    tma_A_v,
                    sA,
                    self.dvb_tiler,  # [BT, BV, BT]
                    dvb_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                pipeline_load_A.producer_acquire(load_A_producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, 0, tile_idx)],
                    tAsA[(None, load_A_producer_state.index)],
                    tma_bar_ptr=pipeline_load_A.producer_get_barrier(load_A_producer_state),
                )
                load_A_producer_state.advance()

                # V-loop
                for v_iter in cutlass.range(self.num_v_tiles):
                    tma_h_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), tma_tensor_h)
                    tHsH, tHgH = self._tma_partition_B(
                        tma_atom_h,
                        tma_h_v,
                        sH,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        i_hv,
                        i_t,
                    )
                    mbarrier_wait(bar_mma_cuda_h_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(bar_tma_h_ptr + vloop_stage_idx, self.tma_bytes_h)
                    cute.copy(
                        tma_atom_h,
                        tHgH[(None, 0, 0)],
                        tHsH[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_h_ptr + vloop_stage_idx,
                    )

                    tma_dh_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), tma_tensor_dh)
                    tDHsDH, tDHgDH = self._tma_partition_B(
                        tma_atom_dh,
                        tma_dh_v,
                        sDh,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        i_hv,
                        i_t,
                    )
                    mbarrier_wait(bar_mma_cuda_dh_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(bar_tma_dh_ptr + vloop_stage_idx, self.tma_bytes_dh)
                    cute.copy(
                        tma_atom_dh,
                        tDHgDH[(None, 0, 0)],
                        tDHsDH[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_dh_ptr + vloop_stage_idx,
                    )

                    tma_do_v = cute.domain_offset((tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_do)
                    tDOsDo, tDOgDo = self._tma_partition_A(
                        tma_atom_do,
                        tma_do_v,
                        sDo,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_do.producer_acquire(load_do_producer_state)
                    cute.copy(
                        tma_atom_do,
                        tDOgDo[(None, tile_idx, 0)],
                        tDOsDo[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_do.producer_get_barrier(load_do_producer_state),
                    )
                    load_do_producer_state.advance()

                    tma_dv_v = cute.domain_offset((tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_dv)
                    tDVsDv, tDVgDV = self._tma_partition_A(
                        tma_atom_dv,
                        tma_dv_v,
                        sDv,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_dv.producer_acquire(load_dv_producer_state)
                    cute.copy(
                        tma_atom_dv,
                        tDVgDV[(None, tile_idx, 0)],
                        tDVsDv[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_dv.producer_get_barrier(load_dv_producer_state),
                    )
                    load_dv_producer_state.advance()

                    tma_v_v = cute.domain_offset((tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_v)
                    tVsV, tVgV = self._tma_partition_B(
                        tma_atom_v,
                        tma_v_v,
                        sV,
                        self.dA_vloop_tiler,  # [BT, BT, BV]
                        dA_vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    mbarrier_wait(bar_mma_cuda_v_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(bar_tma_v_ptr + vloop_stage_idx, self.tma_bytes_v)
                    cute.copy(
                        tma_atom_v,
                        tVgV[(None, tile_idx, 0)],
                        tVsV[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_v_ptr + vloop_stage_idx,
                    )

                    # load v_new
                    tma_vnew_v = cute.domain_offset((tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_vnew)
                    tVnewsVnew, tVnewgVnew = self._tma_partition_A(
                        tma_atom_vnew,
                        tma_vnew_v,
                        sVnew,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_vnew.producer_acquire(load_vnew_producer_state)
                    cute.copy(
                        tma_atom_vnew,
                        tVnewgVnew[(None, tile_idx, 0)],
                        tVnewsVnew[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_vnew.producer_get_barrier(load_vnew_producer_state),
                    )
                    load_vnew_producer_state.advance()

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                # Load g
                tma_g_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_g)
                tGsG, tGgG = self._epilog_partition_varlen(
                    tma_atom_g,
                    tma_g_v[None, None, (i_hv, Int32(0))],
                    (self.BT, self.BK),
                    sG_raw,
                )
                pipeline_load_g.producer_acquire(load_g_producer_state)
                cute.copy(
                    tma_atom_g,
                    tGgG[(None, tile_idx, 0)],
                    tGsG[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_g.producer_get_barrier(load_g_producer_state),
                )
                load_g_producer_state.advance()

                # Load k
                tma_k_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
                tKsK, tKgK = self._epilog_partition_varlen(
                    tma_atom_k,
                    tma_k_v[None, None, (i_h, Int32(0))],
                    (self.BT, self.BK),
                    sK_raw,
                )
                pipeline_load_k.producer_acquire(load_k_producer_state)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, tile_idx, 0)],
                    tKsK[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_k.producer_get_barrier(load_k_producer_state),
                )
                load_k_producer_state.advance()

                tma_q_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_q)
                tQsQ, tQgQ = self._epilog_partition_varlen(
                    tma_atom_q,
                    tma_q_v[None, None, (i_h, Int32(0))],
                    (self.BT, self.BK),
                    sQ_raw,
                )
                pipeline_load_q.producer_acquire(load_q_producer_state)
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, tile_idx, 0)],
                    tQsQ[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_q.producer_get_barrier(load_q_producer_state),
                )
                load_q_producer_state.advance()

        # MMA loop body
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            load_A_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.a_stage)
            load_dv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            mma_dvb_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            load_do_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            load_vnew_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            mma_dq_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            mma_dk_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            mma_dw_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            prologue_dw_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            prologue_kg_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            mma_dgkb_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            mma_dA_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            mma_dA2_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            mma_dA3_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_stage)
            prologue_dA2_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)
            prologue_dA3_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_stage)

            vloop_stage_idx = 0
            a_stage_idx = 0
            mma_vloop_phase = 0
            vloop_phase = 0
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index (unused in MMA warp)
                i_h = i_hv // G  # q/k head index (unused in MMA warp)

                # Decode chunk_indices
                batch_idx = chunk_indices[(i_t, 0)]
                tile_idx = chunk_indices[(i_t, 1)]
                tok_offset = cu_seqlens[(batch_idx,)]
                seq_len = cu_seqlens[(batch_idx + 1,)] - tok_offset
                sub_seq_len = min(self.BT, seq_len - tile_idx * self.BT)

                zeros8 = cute.make_rmem_tensor((8,), dtype=self.io_dtype)
                zeros8.fill(BFloat16(0.0))

                pipeline_load_A.consumer_wait(load_A_consumer_state)
                sA_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    sA_ptr_base + a_stage_idx * A_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if sub_seq_len < self.BT:
                    for i in cutlass.range_constexpr(self.BT // 32):
                        row = i * 32 + lane_idx
                        if row >= sub_seq_len:
                            for col in cutlass.range_constexpr(self.BT // 8):
                                # A tile is MN_SW128 in shared memory; use raw swizzled
                                # address stores to avoid layout-coordinate ambiguity.
                                smem_store_bf16x8_sw128(sA_raw_ptr, row, col * 8, zeros8)
                    # Make generic-proxy SMEM stores visible to UMMA async-proxy readers.
                    cute.arch.fence_proxy("async.shared", space="cta")

                for v_iter in cutlass.range(self.num_v_tiles):
                    is_accum = False if v_iter == 0 else True
                    mbarrier_wait(bar_tma_h_ptr + vloop_stage_idx, vloop_phase)
                    pipeline_load_do.consumer_wait(load_do_consumer_state)
                    sDo_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sDo_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    if sub_seq_len < self.BT:
                        for i in cutlass.range_constexpr(self.BT // 32):
                            row = i * 32 + lane_idx
                            if row >= sub_seq_len:
                                for col in cutlass.range_constexpr(self.BV // 8):
                                    # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                    smem_store_bf16x8_sw128(sDo_raw_ptr, row, col * 8, zeros8)
                        cute.arch.fence_proxy("async.shared", space="cta")

                    if v_iter == 0:
                        pipeline_mma_dq.producer_acquire(mma_dq_producer_state)

                    # dq+=do@h
                    sDo_k_cur = sDo_k[(None, None, None, vloop_stage_idx)]
                    sH_k_cur = sH_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDo_k_cur.iterator, sDo_k_cur.layout, "k"))
                    desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k"))
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_m64n128_k_k_call(
                        vloop_opA_smem, desc_a_base, vloop_opB_smem, desc_b_base, TMEM_DQ_ACC_OFF, self.BV, is_accum
                    )

                    pipeline_load_do.consumer_release(load_do_consumer_state)
                    load_do_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dq.producer_commit(mma_dq_producer_state)
                        mma_dq_producer_state.advance()

                    pipeline_load_dv.consumer_wait(load_dv_consumer_state)
                    sDv_raw = cute.make_ptr(
                        self.io_dtype,
                        sDv_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    if sub_seq_len < self.BT:
                        for i in cutlass.range_constexpr(self.BT // 32):
                            row = i * 32 + lane_idx
                            if row >= sub_seq_len:
                                for col in cutlass.range_constexpr(self.BV // 8):
                                    # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                    smem_store_bf16x8_sw128(sDv_raw, row, col * 8, zeros8)
                        cute.arch.fence_proxy("async.shared", space="cta")

                    # if lane_idx == 0:
                    #     cute.printf("V_iter", v_iter)
                    #     cute.print_tensor(sDv[None, None, None, vloop_stage_idx])
                    pipeline_mma_dvb.producer_acquire(mma_dvb_producer_state)
                    sDv_mn_cur = sDv_mn[(None, None, None, vloop_stage_idx)]
                    sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn"))
                    desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDv_mn_cur.iterator, sDv_mn_cur.layout, "mn"))
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_m64n64_mn_mn_call(
                        A_mn_opA_smem, desc_a_base, dv_mn_opB_smem, desc_b_base, TMEM_FLEX_OFF, self.BT
                    )

                    pipeline_mma_dvb.producer_commit(mma_dvb_producer_state)
                    mma_dvb_producer_state.advance()

                    # dw += dv @ h
                    if v_iter == 0:
                        pipeline_mma_dw.producer_acquire(mma_dw_producer_state)

                    sDv_k_cur = sDv_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k"))
                    desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k"))
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_m64n128_k_k_call(
                        vloop_opA_smem, desc_a_base, vloop_opB_smem, desc_b_base, TMEM_DW_ACC_OFF, self.BV, is_accum
                    )

                    # dA += dv @ v^T
                    mbarrier_wait(bar_tma_v_ptr + vloop_stage_idx, vloop_phase)
                    sV_raw = cute.make_ptr(
                        self.io_dtype, sV_ptr_base + vloop_stage_idx * v_opB_bytes_per_stage, cute.AddressSpace.smem
                    )
                    if sub_seq_len < self.BT:
                        for i in cutlass.range_constexpr(self.BT // 32):
                            row = i * 32 + lane_idx
                            if row >= sub_seq_len:
                                for col in cutlass.range_constexpr(self.BV // 8):
                                    # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                    smem_store_bf16x8_sw128(sV_raw, row, col * 8, zeros8)
                        cute.arch.fence_proxy("async.shared", space="cta")

                    if v_iter == 0:
                        pipeline_mma_dA.producer_acquire(mma_dA_producer_state)

                    sV_k_cur = sV_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k"))
                    desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sV_k_cur.iterator, sV_k_cur.layout, "k"))
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_m64n64_k_k_call(
                        vloop_opA_smem, desc_a_base, v_opB_smem, desc_b_base, TMEM_DA_ACC_OFF, self.BV, is_accum
                    )

                    # dv pipeline calls tcgen05.commit for dv@h and dv@v^T
                    pipeline_load_dv.consumer_release(load_dv_consumer_state)
                    load_dv_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dw.producer_commit(mma_dw_producer_state)
                        mma_dw_producer_state.advance()

                    umma_arrive(bar_mma_cuda_h_ptr + vloop_stage_idx)
                    umma_arrive(bar_mma_cuda_v_ptr + vloop_stage_idx)

                    # dk += v_new @ dh
                    pipeline_load_vnew.consumer_wait(load_vnew_consumer_state)
                    sDvnew_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sVnew_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    if sub_seq_len < self.BT:
                        for i in cutlass.range_constexpr(self.BT // 32):
                            row = i * 32 + lane_idx
                            if row >= sub_seq_len:
                                for col in cutlass.range_constexpr(self.BV // 8):
                                    # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                    smem_store_bf16x8_sw128(sDvnew_raw_ptr, row, col * 8, zeros8)
                        cute.arch.fence_proxy("async.shared", space="cta")

                    mbarrier_wait(bar_tma_dh_ptr + vloop_stage_idx, vloop_phase)
                    if v_iter == 0:
                        pipeline_mma_dk.producer_acquire(mma_dk_producer_state)

                    sVnew_k_cur = sVnew_k[(None, None, None, vloop_stage_idx)]
                    sDh_k_cur = sDh_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sVnew_k_cur.iterator, sVnew_k_cur.layout, "k"))
                    desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDh_k_cur.iterator, sDh_k_cur.layout, "k"))
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_m64n128_k_k_call(
                        vloop_opA_smem, desc_a_base, vloop_opB_smem, desc_b_base, TMEM_DK_ACC_OFF, self.BV, is_accum
                    )

                    # vnew pipeline calls tcgen05.commit
                    pipeline_load_vnew.consumer_release(load_vnew_consumer_state)
                    load_vnew_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dk.producer_commit(mma_dk_producer_state)
                        mma_dk_producer_state.advance()

                    umma_arrive(bar_mma_cuda_dh_ptr + vloop_stage_idx)

                    # add tcgen05.commit and mbar.wait to make sure dq/dk/dw MMA finished
                    umma_arrive(bar_mma_done_vloop_ptr + 0)
                    mbarrier_wait(bar_mma_done_vloop_ptr + 0, mma_vloop_phase)
                    mma_vloop_phase ^= 1

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                pipeline_prologue_dw.consumer_wait(prologue_dw_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")
                # dkgb = A @ dw
                pipeline_mma_dkgb.producer_acquire(mma_dgkb_producer_state)
                sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                sDw_mn_cur = sDw_mn[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn"))
                desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDw_mn_cur.iterator, sDw_mn_cur.layout, "mn"))
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_m64n128_mn_mn_call(
                    A_mn_opA_smem, desc_a_base, dw_mn_opB_smem, desc_b_base, TMEM_DKGB_ACC_OFF, self.BT
                )

                pipeline_mma_dkgb.producer_commit(mma_dgkb_producer_state)
                mma_dgkb_producer_state.advance()

                pipeline_prologue_kg.consumer_wait(prologue_kg_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")
                # dA += dw @ kg^T
                sDw_k_cur = sDw_k[(None, None, None, 0)]
                sKG_k_cur = sKG_k[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDw_k_cur.iterator, sDw_k_cur.layout, "k"))
                desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sKG_k_cur.iterator, sKG_k_cur.layout, "k"))
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_m64n64_k_k_call(
                    dw_k_opA_smem, desc_a_base, kg_k_opB_smem, desc_b_base, TMEM_DA_ACC_OFF, self.BK, True
                )

                pipeline_mma_dA.producer_commit(mma_dA_producer_state)
                mma_dA_producer_state.advance()
                pipeline_prologue_kg.consumer_release(prologue_kg_consumer_state)
                prologue_kg_consumer_state.advance()

                pipeline_prologue_dw.consumer_release(prologue_dw_consumer_state)
                prologue_dw_consumer_state.advance()

                # dA2 = dA @ A
                pipeline_mma_dA2.producer_acquire(mma_dA2_producer_state)
                pipeline_prologue_dA2.consumer_wait(prologue_dA2_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                sDA_k_cur = sDA_k[(None, None, None, 0)]
                sA_k_cur = sA_k[(None, None, None, a_stage_idx)]
                desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDA_k_cur.iterator, sDA_k_cur.layout, "k"))
                desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sA_k_cur.iterator, sA_k_cur.layout, "k"))
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_m64n64_k_k_call(dA_k_opA_smem, desc_a_base, A_k_opB_smem, desc_b_base, TMEM_DA2_ACC_OFF, self.BT)

                pipeline_mma_dA2.producer_commit(mma_dA2_producer_state)
                mma_dA2_producer_state.advance()
                pipeline_prologue_dA2.consumer_release(prologue_dA2_consumer_state)
                prologue_dA2_consumer_state.advance()

                # dA3 = A @ dA2
                pipeline_mma_dA3.producer_acquire(mma_dA3_producer_state)
                pipeline_prologue_dA3.consumer_wait(prologue_dA3_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                sDA_mn_cur = sDA_mn[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn"))
                desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(sDA_mn_cur.iterator, sDA_mn_cur.layout, "mn"))
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_m64n64_mn_mn_call(A_mn_opA_smem, desc_a_base, dA_mn_opB_smem, desc_b_base, TMEM_DA2_ACC_OFF, self.BT)

                pipeline_mma_dA3.producer_commit(mma_dA3_producer_state)
                mma_dA3_producer_state.advance()
                pipeline_prologue_dA3.consumer_release(prologue_dA3_consumer_state)
                prologue_dA3_consumer_state.advance()

                pipeline_load_A.consumer_release(load_A_consumer_state)
                load_A_consumer_state.advance()

                a_stage_idx = (a_stage_idx + 1) % self.a_stage

        # Load aux loop body
        elif warp_idx in self.aux_warp_ids:
            cute.arch.setmaxregister_decrease(self.num_regs_others)
            tidx = thread_idx - (self.threads_per_cta - 64)

            load_beta_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            load_g_store_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)
            store_dg_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kloop_stage)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index (unused in aux warp)

                # Decode chunk_indices
                batch_idx = chunk_indices[(i_t, 0)]
                tile_idx = chunk_indices[(i_t, 1)]
                tok_offset = cu_seqlens[(batch_idx,)]
                seq_len = cu_seqlens[(batch_idx + 1,)] - tok_offset
                sub_seq_len = min(self.BT, seq_len - tile_idx * self.BT)

                pipeline_load_beta.producer_acquire(load_beta_producer_state)
                beta_f32 = Float32(0.0)
                if tidx < sub_seq_len:
                    beta_f32 = Float32(beta_gmem[(tok_offset + tile_idx * self.BT + tidx, (i_hv, Int32(0)))])
                sBeta[(tidx,)] = beta_f32

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_load_beta.producer_commit(load_beta_producer_state)
                load_beta_producer_state.advance()

                pipeline_load_g.consumer_wait(load_g_store_consumer_state)
                pipeline_store_dg.consumer_wait(store_dg_consumer_state)

                tma_dg_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_dg)
                tDGsDG, tDGgDG = self._epilog_partition_varlen(
                    tma_atom_dg,
                    tma_dg_v[None, None, (i_hv, Int32(0))],
                    (self.BT, self.BK),
                    sG_raw,
                )
                if sub_seq_len < self.BT:
                    # Tail chunk, direct store
                    store_lane_row = tidx >> Int32(4)  # 0..3
                    store_col_base = (tidx & Int32(15)) * Int32(8)  # 0,8,...,120
                    for row_quad in cutlass.range_constexpr(self.BT // 4):
                        store_row = row_quad * 4 + store_lane_row
                        if store_row < sub_seq_len:
                            vals0 = smem_load_f32x4_sw128(sG_raw_ptr, store_row, store_col_base)
                            vals1 = smem_load_f32x4_sw128(sG_raw_ptr, store_row, store_col_base + Int32(4))
                            dg_store_rmem = cute.make_rmem_tensor((8,), Float32)
                            dg_store_rmem[0] = vals0[0]
                            dg_store_rmem[1] = vals0[1]
                            dg_store_rmem[2] = vals0[2]
                            dg_store_rmem[3] = vals0[3]
                            dg_store_rmem[4] = vals1[0]
                            dg_store_rmem[5] = vals1[1]
                            dg_store_rmem[6] = vals1[2]
                            dg_store_rmem[7] = vals1[3]
                            dg_store_i32_vec = reinterpret_cast(dg_store_rmem.load(), Float32, 8, Int32)
                            dg_base_addr = (
                                dg_gmem.iterator
                                + (tok_offset + tile_idx * self.BT + store_row) * HV * K
                                + i_hv * K
                                + store_col_base
                            ).toint()
                            store_256b(dg_base_addr, dg_store_i32_vec)
                else:
                    # Non-tail chunk, TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.copy(
                        tma_atom_dg,
                        tDGsDG[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                        tDGgDG[(None, tile_idx, 0)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                pipeline_store_dg.consumer_release(store_dg_consumer_state)
                store_dg_consumer_state.advance()
                pipeline_load_g.consumer_release(load_g_store_consumer_state)
                load_g_store_consumer_state.advance()

        # ===================== TMEM cleanup =====================
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr, TMEM_TOTAL)

    @cute.jit
    def _tma_partition_A(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition a TMA tensor as MMA A-operand (M,K dims).

        ``tma_tensor`` should already have domain_offset applied for varlen.

        For tile_shape = (BT, BK, BV) = (M, N, K):
          coord = (None, 0, None) — slices out the N-tile axis (mode 1) at 0,
          leaving mode 0 (M=BT) and mode 2 (K=BV) free for TMA to iterate.

        Returns (tXsX, tXgX) — SMEM partition and GMEM coordinate partition.
        """
        coord = (None, 0, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_A(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition a TMA tensor as MMA B-operand (N,K dims).

        Mirrors the identical helper in recompute_wu.py / fwd_o.py.
        ``tma_tensor`` should already have domain_offset applied for varlen.

        For tile_shape = (BT, BK, BV) = (M, N, K):
          coord = (0, None, None) — slices out the M-tile axis (mode 0) at 0,
          leaving mode 1 (N=BK) and mode 2 (K=BV) free for TMA to iterate.

        Returns (tXsX, tXgX) — SMEM partition and GMEM coordinate partition.
        """
        coord = (0, None, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition_varlen(self, atom, gC_2d, epi_tile, sC):
        """Partition for varlen epilog TMA load (2D tensor with domain_offset).

        Uses local_tile instead of flat_divide to correctly preserve TMA basis
        stride coordinates through domain_offset.  Matches Flash Attention's
        pattern: slice mode2 → domain_offset(2D) → local_tile → tma_partition.

        Uses (None, None) to keep all tile-count modes, producing the same
        rank as _epilog_partition (flat_divide) so copy indexing is unchanged.
        """
        gC_tiled = cute.local_tile(gC_2d, epi_tile, (None, None))
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_tiled, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
        )
        return bSG_sC, bSG_gC


# =====================================================================
# Compilation & Cache
# =====================================================================

_bwd_wy_kernel_cache: dict = {}


def _compile_bwd_wy_variant(H, HV, K, V, scale, chunk_size, beta_dtype, use_fast_math):
    """Compile one ChunkKdaBwdWyDqkgFused kernel variant.

    Uses make_fake_compact_tensor and make_fake_stream for compilation with
    TVM-FFI. At runtime, torch tensors are passed directly (zero-copy).
    Uses sym_int() for dynamic B, T, NT dimensions.
    """
    kernel_obj = ChunkKdaBwdWyDqkgFused(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        scale=scale,
        beta_dtype=beta_dtype,
        use_fast_math=use_fast_math,
    )

    sym_b = cute.sym_int()  # T (non-varlen) or T_total (varlen)
    sym_nt = cute.sym_int()  # NT_total
    sym_cu = cute.sym_int()  # cu_seqlens size
    sym_ci = cute.sym_int()  # chunk_indices rows

    BT = chunk_size

    # only support varlen for real-world use cases
    # varlen: data tensors are [1, T_total, H, ...]
    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    v_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    vnew_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    g_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    beta_fake = make_fake_compact_tensor(beta_dtype, (1, sym_b, HV), stride_order=(2, 1, 0), assumed_align=128)
    A_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, BT), stride_order=(3, 2, 1, 0), assumed_align=128)
    do_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dv_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, V), stride_order=(3, 2, 1, 0), assumed_align=128)

    dq_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dk_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dv2_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, HV, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dg_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    db_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV), stride_order=(2, 1, 0), assumed_align=128)
    dA_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, HV, BT), stride_order=(3, 2, 1, 0), assumed_align=128)

    h_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_nt, HV, K, V), stride_order=(4, 3, 2, 1, 0), assumed_align=128)
    dh_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (1, sym_nt, HV, K, V), stride_order=(4, 3, 2, 1, 0), assumed_align=128
    )

    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    ci_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ci, 2), stride_order=(1, 0), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        # Inputs
        q_fake,
        k_fake,
        v_fake,
        vnew_fake,
        g_fake,
        beta_fake,
        A_fake,
        h_fake,
        do_fake,
        dh_fake,
        dv_fake,
        # Outputs
        dq_fake,
        dk_fake,
        dv2_fake,
        dg_fake,
        db_fake,
        dA_fake,
        # Metadata
        cu_fake,
        ci_fake,
        (Int32(1), Int32(1), Int32(H), Int32(HV), Int32(K), Int32(V)),
        Int32(1),  # total_nt dummy
        stream_fake,
        options=COMPILE_OPTIONS,
    )
    return compiled_fn


def _get_compiled_bwd_wy(H, HV, K, V, scale, chunk_size, beta_dtype):
    """Get a compiled ChunkKdaBwdWyDqkgFused kernel with on-demand (lazy) compilation.

    Cache key: (H, HV, K, V, scale, chunk_size, beta_dtype, USE_FAST_MATH)
    """
    key = (H, HV, K, V, scale, chunk_size, beta_dtype, USE_FAST_MATH)
    if key not in _bwd_wy_kernel_cache:
        _bwd_wy_kernel_cache[key] = _compile_bwd_wy_variant(
            H,
            HV,
            K,
            V,
            scale,
            chunk_size,
            _torch_to_cutlass_dtype[beta_dtype],
            USE_FAST_MATH,
        )
    return _bwd_wy_kernel_cache[key]


# =====================================================================
# Python API (FLA-compatible)
# =====================================================================

_bwd_wy_dummy_cu_seqlens = None
_bwd_wy_dummy_chunk_indices = None


def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ChunkKdaBwdWyDqkgFused — FLA-compatible Python API.

    Computes backward gradients dq, dk, dv2, db, dg, dA for the KDA
    chunkwise delta-rule backward pass using the CuTe DSL Blackwell kernel.

    Returns:
        (dq, dk, dv2, db, dg, dA) matching FLA's chunk_kda_bwd_wy_dqkg_fused output order.
    """
    B, T, H, K = q.shape
    V = v.shape[3]
    HV = v.shape[2]
    BT = chunk_size
    beta_dtype = beta.dtype
    device = q.device

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(B, T, device, torch.int32)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if scale is None:
        scale = K**-0.5

    assert cu_seqlens is not None and chunk_indices is not None
    # Ensure cu_seqlens is int32
    assert cu_seqlens.dtype == torch.int32, "cu_seqlens must be int32"
    T_total = B * T
    num_seqs = cu_seqlens.shape[0] - 1
    total_nt_val = chunk_indices.shape[0]
    ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(HV), Int32(K), Int32(V))

    # Allocate output tensors
    dq = torch.empty(1, T_total, HV, K, dtype=torch.float32, device=device)
    dk = torch.empty(1, T_total, HV, K, dtype=torch.float32, device=device)
    dv2 = torch.empty(1, T_total, HV, V, dtype=torch.bfloat16, device=device)
    dg = torch.empty(1, T_total, HV, K, dtype=torch.float32, device=device)
    db = torch.empty(1, T_total, HV, dtype=torch.float32, device=device)
    dA = torch.empty(1, T_total, HV, BT, dtype=torch.float32, device=device)

    compiled_fn = _get_compiled_bwd_wy(
        H,
        HV,
        K,
        V,
        scale,
        chunk_size,
        beta_dtype,
    )

    if B != 1:
        q = q.reshape(1, T_total, H, K)
        k = k.reshape(1, T_total, H, K)
        v = v.reshape(1, T_total, HV, V)
        v_new = v_new.reshape(1, T_total, HV, V)
        g = g.reshape(1, T_total, HV, K)
        beta = beta.reshape(1, T_total, HV)
        A = A.reshape(1, T_total, HV, BT)
        h = h.reshape(1, total_nt_val, HV, K, V)
        do = do.reshape(1, T_total, HV, V)
        dh = dh.reshape(1, total_nt_val, HV, K, V)
        dv = dv.reshape(1, T_total, HV, V)

    # TVM-FFI call
    compiled_fn(
        # Inputs
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        do,
        dh,
        dv,
        # Outputs
        dq,
        dk,
        dv2,
        dg,
        db,
        dA,
        # Metadata
        cu_seqlens,
        chunk_indices,
        ps,
        Int32(total_nt_val),
    )

    # rearrange back
    if B != 1:
        dq = dq.reshape(B, T, HV, K)
        dk = dk.reshape(B, T, HV, K)
        dv2 = dv2.reshape(B, T, HV, V)
        dg = dg.reshape(B, T, HV, K)
        db = db.reshape(B, T, HV)
        dA = dA.reshape(B, T, HV, BT)

    return dq, dk, dv2, db, dg, dA


# =====================================================================
# Main (test entry point)
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Chunk KDA BWD WY DqKG Fused kernel test")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--H", type=int, default=1)
    parser.add_argument("--HV", type=int, default=None, help="Number of value heads (default: H, i.e. no GVA)")
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    if args.scale is None:
        args.scale = args.K**-0.5
    B, T, H, K, V = args.B, args.T, args.H, args.K, args.V
    HV = args.HV if args.HV is not None else H
    BT = args.chunk_size
    seq_lens = [63, 63, 63]
    seq_lens = [64]
    total_len = sum(seq_lens)
    T = total_len
    scale = args.scale
    NT = (T + BT - 1) // BT
    dtype, device = torch.bfloat16, "cuda"
    cu_seqlens = torch.tensor(_exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    print(f"Config: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, BT={BT}, scale={scale:.4f}")
    print(f"  Chunks per seq: {NT}, Total chunks: {B * NT}")
    print(f"  BK={64}, BV={64}, NK={K // 64}, NV={V // 64}")

    # Generate test data (q/k use H heads; all others use HV heads)
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    v_new = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, K, dtype=torch.float32, device=device) * 0.1
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    A = torch.randn(B, T, HV, BT, dtype=dtype, device=device) * 0.1
    h = torch.randn(B, NT, HV, K, V, dtype=dtype, device=device) * 0.01
    do_t = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    dh = torch.randn(B, NT, HV, K, V, dtype=dtype, device=device) * 0.01
    dv = torch.randn(B, T, HV, V, dtype=dtype, device=device)

    print("\n=== Compilation Test ===")
    try:
        dq, dk, dv2, db, dg, dA = chunk_kda_bwd_wy_dqkg_fused(
            q=q,
            k=k,
            v=v,
            v_new=v_new,
            g=g,
            beta=beta,
            A=A,
            h=h,
            do=do_t,
            dh=dh,
            dv=dv,
            cu_seqlens=cu_seqlens,
            scale=scale,
            chunk_size=BT,
        )
        torch.cuda.synchronize()
        print(f"  dq shape: {dq.shape}, dtype: {dq.dtype}")
        print(f"  dk shape: {dk.shape}, dtype: {dk.dtype}")
        print(f"  dv2 shape: {dv2.shape}, dtype: {dv2.dtype}")
        print(f"  dg shape: {dg.shape}, dtype: {dg.dtype}")
        print(f"  db shape: {db.shape}, dtype: {db.dtype}")
        print(f"  dA shape: {dA.shape}, dtype: {dA.dtype}")
    except Exception as e:
        import traceback

        print(f"  ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
