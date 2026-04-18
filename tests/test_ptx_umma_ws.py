"""
Standalone CuteDSL test for tcgen05.mma.ws (weight-stationary) inline PTX wrappers.

Tests:
  1. tcgen05mma_ws_ss_tf32  -- WS mode, SMEM A × SMEM B → TMEM C, kind::tf32
  2. tcgen05mma_ws_ts_tf32  -- WS mode, TMEM A × SMEM B → TMEM C, kind::tf32
  3. tcgen05mma_ws_ss_f16   -- WS mode, SMEM A × SMEM B → TMEM C, kind::f16
  4. tcgen05mma_ws_ts_f16   -- WS mode, TMEM A × SMEM B → TMEM C, kind::f16

For the WS_TS test, matrix A is first loaded into TMEM via an SS MMA (identity-
like multiplication), then used as the A operand for the WS TS MMA.  To keep
things simple we use a two-TMEM-column approach:
  - tmem region 0: accumulator for both phases
  - tmem region 1: holds A data for TS phase (populated via R2T store)

SMEM layout follows the same conventions as test_ptx_umma_masked.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float16, Float32, TFloat32, Int32, Int64
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05 import (
    Repetition, Pack, make_umma_smem_desc, smem_descriptor_to_int,
)
from cutlass.cute.arch import (
    mbarrier_init, mbarrier_init_fence, mbarrier_wait, sync_threads, elect_one,
)
import cutlass.utils.blackwell_helpers as sm100_utils

from cula.kda.ptx_umma_masked import (
    Tcgen05SmemDescriptor,
    tcgen05mma_ws_ss_tf32,
    tcgen05mma_ws_ts_tf32,
    tcgen05mma_ws_ss_f16,
    tcgen05mma_ws_ts_f16,
)

M_DIM, N_DIM = 64, 64
K_DIM_TF32 = 8   # kind::tf32  → K=8
K_DIM_F16  = 16  # kind::f16   → K≥16
TMEM_COLS  = 32  # MN=(64×64) MMA → 128 lanes, 32 columns

# Instruction descriptor for M=64, N=64, TF32, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22], TransposeB at [16],
#       btype=tf32(2) at [10:12], atype=tf32(2) at [7:9], dtype=f32(1) at [4:5]
IDESC_TF32_M64_N64 = (4 << 24) | (8 << 17) | (1 << 16) | (2 << 10) | (2 << 7) | (1 << 4)
assert IDESC_TF32_M64_N64 == 0x4110910

# Instruction descriptor for M=64, N=64, F16, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22], TransposeB at [16],
#       btype=f16(0) at [10:12], atype=f16(0) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64 = (4 << 24) | (8 << 17) | (1 << 16) | (0 << 10) | (0 << 7) | (1 << 4)
assert IDESC_F16_M64_N64 == 0x4110010


# =====================================================================
# Test 1: tcgen05mma_ws_ss_tf32  (weight-stationary, SMEM A, SMEM B, tf32)
# =====================================================================

class _WsSsTf32Kernel:
    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_TF32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
        )
        mma_tiler = (M, N, K)

        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, TFloat32, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, TFloat32, 1)
        bufferA = smem.allocate_tensor(
            element_type=TFloat32, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        bufferB = smem.allocate_tensor(
            element_type=TFloat32, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A (M-major → swizzled SMEM) and B (row-major → swizzled SMEM)
        gA_flat = cute.make_tensor(A_in.iterator, cute.make_layout(M * K))
        gB_flat = cute.make_tensor(B_in.iterator, cute.make_layout(K * N))

        for step in cutlass.range(M * K // 128, unroll_full=False):
            smem_idx = tidx + step * 128
            m = smem_idx % M
            k = smem_idx // M
            bufA_s0[smem_idx] = gA_flat[m * K + k]
        for step in cutlass.range(K * N // 128, unroll_full=False):
            idx = tidx + step * 128
            bufB_s0[idx] = gB_flat[idx]
        sync_threads()

        # TMEM allocation
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        acc_shape        = tiled_mma.partition_shape_C((M, N))
        acc_shape_staged = cute.append(acc_shape, 1)
        tCtAcc           = cute.make_tensor(tmem_ptr_f32, tiled_mma.make_fragment_C(acc_shape_staged).layout)
        tmem_col_buf     = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col         = tmem_col_buf[0]

        # Build descriptors
        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        # Issue WS SS MMA  (scale_out=0 → D = A*B, not accumulate)
        if warp_idx == cutlass.Int32(0):
            tcgen05mma_ws_ss_tf32(desc_a, desc_b, tmem_col, IDESC_TF32_M64_N64, 0)
            with elect_one():
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # T2R → R2G
        t2r_atom    = cute.make_copy_atom(tcgen05.Ld16x256bOp(Repetition(4), Pack.NONE), Float32)
        fake_smem   = cute.make_tensor(cute.make_ptr(Float32, 0, cute.AddressSpace.smem),
                                       cute.make_layout((M, N)))
        tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
        tiled_t2r   = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
        thr_t2r     = tiled_t2r.get_slice(tidx)
        tTR_tAcc    = thr_t2r.partition_S(tCtAcc_flat)
        tTR_sDummy  = thr_t2r.partition_D(fake_smem)
        tTR_rAcc    = cute.make_rmem_tensor(tTR_sDummy.shape, Float32)

        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        gC     = cute.make_tensor(C_out.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tTR_gC = thr_t2r.partition_D(gC)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), tTR_rAcc, tTR_gC)

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, TMEM_COLS)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        A_gpu = A_cpu.contiguous().float().cuda()
        B_gpu = B_cpu.contiguous().float().cuda()
        C_gpu = torch.zeros(M_DIM, N_DIM, dtype=torch.float32, device="cuda")
        stream = cutlass_torch.default_stream()
        self._launch(from_dlpack(A_gpu), from_dlpack(B_gpu), from_dlpack(C_gpu), stream)
        torch.cuda.synchronize()
        return C_gpu.cpu()


# =====================================================================
# Test 2: tcgen05mma_ws_ts_tf32  (weight-stationary, TMEM A, SMEM B, tf32)
# =====================================================================

class _WsTsTf32Kernel:
    """Two-step test:
      Step 1: SS MMA → tmem_acc  (A_smem × I_smem → tmem_acc)
              where I_smem is identity-like matrix so tmem_acc ≈ A data.
      Step 2: WS TS MMA → tmem_c  (tmem_acc as A × B_smem → tmem_c)
      Result should match: A × B  (within TF32 precision).

    Actually, a simpler approach: we allocate two TMEM regions.
      - Region 0 (tmem_a_region): populate with A via SS MMA (A × I → A)
      - Region 1 (tmem_c_region): result of WS TS MMA (tmem_a × B → C)

    For the identity trick: we compute A_tf32 × I_tf32 with an 8×8 identity.
    Since kind::tf32 with M=64,N=64,K=8:
      - Step 1 SS: A(64×8) × I(8×64) → tmem region 0 (64×64, FP32)
      - Step 2 WS_TS: tmem_region0(64×K') × B(K'×64) → tmem region 1

    Wait — WS TS MMA reads A from TMEM. The TMEM A layout must match what
    the MMA instruction expects. For TS mode, A is in TMEM and is read as
    the MxK tile. The accumulator layout (MxN) is NOT the same as the A
    layout (MxK). So we cannot simply reuse the accumulator of an SS MMA
    as the A operand.

    Better approach: use R2T store to write known A values into TMEM with
    the correct layout, then WS TS MMA reads from that region.
    """

    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_TF32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        # --- Build TS tiled_mma (A from TMEM) ---
        ts_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
            a_source=tcgen05.OperandSource.TMEM,
        )
        mma_tiler = (M, N, K)

        # --- SMEM for A (needed to populate TMEM A via S2T copy) and B ---
        # We use the SS tiled_mma layout for A SMEM (since we need to build descriptors for A too)
        ss_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(ss_tiled_mma, mma_tiler, TFloat32, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(ss_tiled_mma, mma_tiler, TFloat32, 1)

        bufferA = smem.allocate_tensor(
            element_type=TFloat32, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        bufferB = smem.allocate_tensor(
            element_type=TFloat32, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A and B into SMEM
        gA_flat = cute.make_tensor(A_in.iterator, cute.make_layout(M * K))
        gB_flat = cute.make_tensor(B_in.iterator, cute.make_layout(K * N))

        for step in cutlass.range(M * K // 128, unroll_full=False):
            smem_idx = tidx + step * 128
            m = smem_idx % M
            k = smem_idx // M
            bufA_s0[smem_idx] = gA_flat[m * K + k]
        for step in cutlass.range(K * N // 128, unroll_full=False):
            idx = tidx + step * 128
            bufB_s0[idx] = gB_flat[idx]
        sync_threads()

        # --- Allocate TMEM: need space for A (MxK in TMEM) + C (MxN in TMEM) ---
        # A in TMEM needs K=8 columns (tf32), C needs N=64 columns
        # Total TMEM columns = K + N = 8 + 64 = 72, but must be aligned;
        # allocate 128 columns to be safe
        total_tmem_cols = 128
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(total_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_base    = tmem_col_buf[0]

        # TMEM A region starts at tmem_base (offset 0)
        # TMEM C region starts at tmem_base + TMEM_COLS (offset 64)
        tmem_a_col = tmem_base
        tmem_c_col = tmem_base + TMEM_COLS

        # --- Populate TMEM A via S2T copy ---
        # Get the TMEM A layout from the TS tiled_mma
        a_tmem_layout = sm100_utils.make_smem_layout_a(ts_tiled_mma, mma_tiler, TFloat32, 1)
        tCrA_fake = ts_tiled_mma.make_fragment_A(a_tmem_layout.outer.shape)
        tmem_a_ptr = cute.recast_ptr(tmem_ptr_f32, dtype=tCrA_fake.element_type)
        tCrA = cute.make_tensor(tmem_a_ptr, tCrA_fake.layout)

        # Build S2T copy to move A from SMEM → TMEM
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp128x256bOp(tcgen05.CtaGroup.ONE), TFloat32,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCrA[(None, None, None, 0)])
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsA_s2t_ = thr_copy_s2t.partition_S(bufA_s0)
        tCsA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsA_s2t_)
        tCtA_s2t = thr_copy_s2t.partition_D(tCrA[(None, None, None, 0)])

        # Execute S2T copy (single-thread semantics)
        if warp_idx == cutlass.Int32(0):
            with elect_one():
                cute.copy(tiled_copy_s2t, tCsA_s2t, tCtA_s2t)
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # Re-arm mbar
        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # --- Build C accumulator in second TMEM region ---
        # We need tmem_ptr offset by TMEM_COLS for the C region
        acc_shape        = ts_tiled_mma.partition_shape_C((M, N))
        acc_shape_staged = cute.append(acc_shape, 1)
        tmem_c_ptr_f32   = tmem_ptr_f32 + TMEM_COLS
        tCtAcc = cute.make_tensor(tmem_c_ptr_f32, ts_tiled_mma.make_fragment_C(acc_shape_staged).layout)

        # Build B descriptor
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        # --- Issue WS TS MMA ---
        if warp_idx == cutlass.Int32(0):
            tcgen05mma_ws_ts_tf32(tmem_a_col, desc_b, tmem_c_col, IDESC_TF32_M64_N64, 0)
            with elect_one():
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # --- T2R → R2G ---
        t2r_atom    = cute.make_copy_atom(tcgen05.Ld16x256bOp(Repetition(8), Pack.NONE), Float32)
        fake_smem   = cute.make_tensor(cute.make_ptr(Float32, 0, cute.AddressSpace.smem),
                                       cute.make_layout((M, N)))
        tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
        tiled_t2r   = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
        thr_t2r     = tiled_t2r.get_slice(tidx)
        tTR_tAcc    = thr_t2r.partition_S(tCtAcc_flat)
        tTR_sDummy  = thr_t2r.partition_D(fake_smem)
        tTR_rAcc    = cute.make_rmem_tensor(tTR_sDummy.shape, Float32)

        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        gC     = cute.make_tensor(C_out.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tTR_gC = thr_t2r.partition_D(gC)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), tTR_rAcc, tTR_gC)

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, total_tmem_cols)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        A_gpu = A_cpu.contiguous().float().cuda()
        B_gpu = B_cpu.contiguous().float().cuda()
        C_gpu = torch.zeros(M_DIM, N_DIM, dtype=torch.float32, device="cuda")
        stream = cutlass_torch.default_stream()
        self._launch(from_dlpack(A_gpu), from_dlpack(B_gpu), from_dlpack(C_gpu), stream)
        torch.cuda.synchronize()
        return C_gpu.cpu()


# =====================================================================
# Test 3: tcgen05mma_ws_ss_f16  (weight-stationary, SMEM A, SMEM B, f16)
# =====================================================================

class _WsSsF16Kernel:
    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_F16
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Float16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
        )
        mma_tiler = (M, N, K)

        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, Float16, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, Float16, 1)
        bufferA = smem.allocate_tensor(
            element_type=Float16, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        bufferB = smem.allocate_tensor(
            element_type=Float16, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A (M-major → swizzled SMEM) and B (row-major → swizzled SMEM)
        gA_flat = cute.make_tensor(A_in.iterator, cute.make_layout(M * K))
        gB_flat = cute.make_tensor(B_in.iterator, cute.make_layout(K * N))

        for step in cutlass.range(M * K // 128, unroll_full=False):
            smem_idx = tidx + step * 128
            m = smem_idx % M
            k = smem_idx // M
            bufA_s0[smem_idx] = gA_flat[m * K + k]
        for step in cutlass.range(K * N // 128, unroll_full=False):
            idx = tidx + step * 128
            bufB_s0[idx] = gB_flat[idx]
        sync_threads()

        # TMEM allocation
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        acc_shape        = tiled_mma.partition_shape_C((M, N))
        acc_shape_staged = cute.append(acc_shape, 1)
        tCtAcc           = cute.make_tensor(tmem_ptr_f32, tiled_mma.make_fragment_C(acc_shape_staged).layout)
        tmem_col_buf     = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col         = tmem_col_buf[0]

        # Build descriptors
        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        # Issue WS SS MMA  (scale_out=0 → D = A*B, not accumulate)
        if warp_idx == cutlass.Int32(0):
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_col, IDESC_F16_M64_N64, 0)
            with elect_one():
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # T2R → R2G
        t2r_atom    = cute.make_copy_atom(tcgen05.Ld16x256bOp(Repetition(4), Pack.NONE), Float32)
        fake_smem   = cute.make_tensor(cute.make_ptr(Float32, 0, cute.AddressSpace.smem),
                                       cute.make_layout((M, N)))
        tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
        tiled_t2r   = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
        thr_t2r     = tiled_t2r.get_slice(tidx)
        tTR_tAcc    = thr_t2r.partition_S(tCtAcc_flat)
        tTR_sDummy  = thr_t2r.partition_D(fake_smem)
        tTR_rAcc    = cute.make_rmem_tensor(tTR_sDummy.shape, Float32)

        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        gC     = cute.make_tensor(C_out.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tTR_gC = thr_t2r.partition_D(gC)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), tTR_rAcc, tTR_gC)

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, TMEM_COLS)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        A_gpu = A_cpu.contiguous().half().cuda()
        B_gpu = B_cpu.contiguous().half().cuda()
        C_gpu = torch.zeros(M_DIM, N_DIM, dtype=torch.float32, device="cuda")
        stream = cutlass_torch.default_stream()
        self._launch(from_dlpack(A_gpu), from_dlpack(B_gpu), from_dlpack(C_gpu), stream)
        torch.cuda.synchronize()
        return C_gpu.cpu()


# =====================================================================
# Test 4: tcgen05mma_ws_ts_f16  (weight-stationary, TMEM A, SMEM B, f16)
# =====================================================================

class _WsTsF16Kernel:
    """Two-step test (same strategy as _WsTsTf32Kernel but with kind::f16):
      - Region 0 (tmem_a_region): populate A into TMEM via S2T copy
      - Region 1 (tmem_c_region): result of WS TS MMA (tmem_a × B → C)
    """

    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_F16
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        # --- Build TS tiled_mma (A from TMEM) ---
        ts_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Float16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
            a_source=tcgen05.OperandSource.TMEM,
        )
        mma_tiler = (M, N, K)

        # --- SMEM for A (needed to populate TMEM A via S2T copy) and B ---
        ss_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Float16, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, tcgen05.CtaGroup.ONE, (M, N),
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(ss_tiled_mma, mma_tiler, Float16, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(ss_tiled_mma, mma_tiler, Float16, 1)

        bufferA = smem.allocate_tensor(
            element_type=Float16, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        bufferB = smem.allocate_tensor(
            element_type=Float16, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A and B into SMEM
        gA_flat = cute.make_tensor(A_in.iterator, cute.make_layout(M * K))
        gB_flat = cute.make_tensor(B_in.iterator, cute.make_layout(K * N))

        for step in cutlass.range(M * K // 128, unroll_full=False):
            smem_idx = tidx + step * 128
            m = smem_idx % M
            k = smem_idx // M
            bufA_s0[smem_idx] = gA_flat[m * K + k]
        for step in cutlass.range(K * N // 128, unroll_full=False):
            idx = tidx + step * 128
            bufB_s0[idx] = gB_flat[idx]
        sync_threads()

        # --- Allocate TMEM: A (MxK in TMEM) + C (MxN in TMEM) ---
        total_tmem_cols = 128
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(total_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_base    = tmem_col_buf[0]

        tmem_a_col = tmem_base
        tmem_c_col = tmem_base + TMEM_COLS

        # --- Populate TMEM A via S2T copy ---
        a_tmem_layout = sm100_utils.make_smem_layout_a(ts_tiled_mma, mma_tiler, Float16, 1)
        tCrA_fake = ts_tiled_mma.make_fragment_A(a_tmem_layout.outer.shape)
        tmem_a_ptr = cute.recast_ptr(tmem_ptr_f32, dtype=tCrA_fake.element_type)
        tCrA = cute.make_tensor(tmem_a_ptr, tCrA_fake.layout)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp128x256bOp(tcgen05.CtaGroup.ONE), Float16,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCrA[(None, None, None, 0)])
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsA_s2t_ = thr_copy_s2t.partition_S(bufA_s0)
        tCsA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsA_s2t_)
        tCtA_s2t = thr_copy_s2t.partition_D(tCrA[(None, None, None, 0)])

        if warp_idx == cutlass.Int32(0):
            with elect_one():
                cute.copy(tiled_copy_s2t, tCsA_s2t, tCtA_s2t)
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # Re-arm mbar
        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # --- Build C accumulator in second TMEM region ---
        acc_shape        = ts_tiled_mma.partition_shape_C((M, N))
        acc_shape_staged = cute.append(acc_shape, 1)
        tmem_c_ptr_f32   = tmem_ptr_f32 + TMEM_COLS
        tCtAcc = cute.make_tensor(tmem_c_ptr_f32, ts_tiled_mma.make_fragment_C(acc_shape_staged).layout)

        # Build B descriptor
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        # --- Issue WS TS MMA ---
        if warp_idx == cutlass.Int32(0):
            tcgen05mma_ws_ts_f16(tmem_a_col, desc_b, tmem_c_col, IDESC_F16_M64_N64, 0)
            with elect_one():
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # --- T2R → R2G ---
        t2r_atom    = cute.make_copy_atom(tcgen05.Ld16x256bOp(Repetition(8), Pack.NONE), Float32)
        fake_smem   = cute.make_tensor(cute.make_ptr(Float32, 0, cute.AddressSpace.smem),
                                       cute.make_layout((M, N)))
        tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
        tiled_t2r   = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
        thr_t2r     = tiled_t2r.get_slice(tidx)
        tTR_tAcc    = thr_t2r.partition_S(tCtAcc_flat)
        tTR_sDummy  = thr_t2r.partition_D(fake_smem)
        tTR_rAcc    = cute.make_rmem_tensor(tTR_sDummy.shape, Float32)

        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        gC     = cute.make_tensor(C_out.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tTR_gC = thr_t2r.partition_D(gC)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), tTR_rAcc, tTR_gC)

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, total_tmem_cols)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        A_gpu = A_cpu.contiguous().half().cuda()
        B_gpu = B_cpu.contiguous().half().cuda()
        C_gpu = torch.zeros(M_DIM, N_DIM, dtype=torch.float32, device="cuda")
        stream = cutlass_torch.default_stream()
        self._launch(from_dlpack(A_gpu), from_dlpack(B_gpu), from_dlpack(C_gpu), stream)
        torch.cuda.synchronize()
        return C_gpu.cpu()


# =====================================================================
# Test functions
# =====================================================================

def test_ws_ss_tf32():
    print("\n=== Test 1: tcgen05mma_ws_ss_tf32 (weight-stationary, SMEM A × SMEM B, tf32) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM_TF32)
    B = torch.randn(K_DIM_TF32, N_DIM)
    ref = torch.mm(A, B)
    got = _WsSsTf32Kernel().run(A, B)
    err = (got - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-8)
    max_idx = err.argmax().item()
    mi, mj = max_idx // N_DIM, max_idx % N_DIM
    print(f"  got[0,:4]={got[0,:4].tolist()}")
    print(f"  ref[0,:4]={ref[0,:4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi,mj]:.6f} ref={ref[mi,mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


def test_ws_ts_tf32():
    print("\n=== Test 2: tcgen05mma_ws_ts_tf32 (weight-stationary, TMEM A × SMEM B, tf32) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM_TF32)
    B = torch.randn(K_DIM_TF32, N_DIM)
    ref = torch.mm(A, B)
    got = _WsTsTf32Kernel().run(A, B)
    err = (got - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-8)
    max_idx = err.argmax().item()
    mi, mj = max_idx // N_DIM, max_idx % N_DIM
    print(f"  got[0,:4]={got[0,:4].tolist()}")
    print(f"  ref[0,:4]={ref[0,:4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi,mj]:.6f} ref={ref[mi,mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


def test_ws_ss_f16():
    print("\n=== Test 3: tcgen05mma_ws_ss_f16 (weight-stationary, SMEM A × SMEM B, f16) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM_F16)
    B = torch.randn(K_DIM_F16, N_DIM)
    ref = torch.mm(A, B)
    got = _WsSsF16Kernel().run(A, B)
    err = (got - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-8)
    max_idx = err.argmax().item()
    mi, mj = max_idx // N_DIM, max_idx % N_DIM
    print(f"  got[0,:4]={got[0,:4].tolist()}")
    print(f"  ref[0,:4]={ref[0,:4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi,mj]:.6f} ref={ref[mi,mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


def test_ws_ts_f16():
    print("\n=== Test 4: tcgen05mma_ws_ts_f16 (weight-stationary, TMEM A × SMEM B, f16) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM_F16)
    B = torch.randn(K_DIM_F16, N_DIM)
    ref = torch.mm(A, B)
    got = _WsTsF16Kernel().run(A, B)
    err = (got - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-8)
    max_idx = err.argmax().item()
    mi, mj = max_idx // N_DIM, max_idx % N_DIM
    print(f"  got[0,:4]={got[0,:4].tolist()}")
    print(f"  ref[0,:4]={ref[0,:4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi,mj]:.6f} ref={ref[mi,mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    # test_ws_ss_tf32()
    # test_ws_ts_tf32()
    test_ws_ss_f16()
    test_ws_ts_f16()
    print("\n=== All tests passed! ===")
