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

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.arch import (
    elect_one,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
    sync_threads,
)
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05 import (
    make_umma_smem_desc,
    smem_descriptor_to_int,
)
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import Float16, Float32, Int32, Int64, TFloat32

from cula.ops.intrinsics_sm100 import (
    store_256b,
    subvec,
    tcgen05_cp_128x256b,
    reinterpret_cast,
    tcgen05_ld_32x32b,
)
from cula.ops.ptx_umma_ext import (
    CollectorBBuffer,
    CollectorOp,
    Tcgen05SmemDescriptor,
    tcgen05mma_ws_ss_f16,
    tcgen05mma_ws_ss_tf32,
)

M_DIM, N_DIM = 64, 64
# TODO: support arbitrary K
K_DIM_TF32 = 8  # kind::tf32  → K>=8, tile size
A_K_STEP_BYTES_TF32 = M_DIM * 8 * 4 # smem offset for each K-atom in operand A
B_K_STEP_BYTES_TF32 = N_DIM * 8 * 4 # smem offset for each K-atom in operand B
K_DIM_F16 = 128  # default after sweep
# NOTE: per-K-atom byte offsets are derived from the SMEM layout at runtime
# (see _WsSsF16Kernel) so K_DIM_F16 can be any multiple of 16. The layout's
# k_iter mode becomes hierarchical at K≥128 (e.g. (4, K/64):(16, 4096) for A),
# which the layout-based offset computation handles transparently.

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
        ACC_NUM_COLS = N // 2
        NUM_COLS = ACC_NUM_COLS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr = smem.allocate(Int64, byte_alignment=8)

        # --- SMEM layouts via sm100_utils (handles swizzle correctly for TF32) ---
        # NOTE: we use non-ws mode TiledMMA for creating smem layout in a easy way,
        # because smem layouts of ws mode and non-ws mode are the same
        non_ws_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M, N),
        )
        mma_tiler = (M, N, K)
        a_smem_layout = sm100_utils.make_smem_layout_a(non_ws_tiled_mma, mma_tiler, TFloat32, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(non_ws_tiled_mma, mma_tiler, TFloat32, 1)
        bufferA = smem.allocate_tensor(
            element_type=TFloat32,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        bufferB = smem.allocate_tensor(
            element_type=TFloat32,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A (row-major input → K-major swizzled SMEM) and B
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

        # --- TMEM allocation ---
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr,
            barrier_for_retrieve=alloc_bar,
            allocator_warp_id=0,
        )
        tmem.allocate(NUM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col = tmem_col_buf[0]

        # Build SMEM descriptors (rank-2 vec_mode layout required)
        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)

        # Issue WS SS MMA  (scale_out=0 → D = A*B, not accumulate)
        if warp_idx == cutlass.Int32(0):
            with elect_one():
                for ks in cutlass.range_constexpr(K // 8):
                    scale = 0 if ks == 0 else 1
                    desc_a = desc_a_base + (ks * A_K_STEP_BYTES_TF32)
                    desc_b = desc_b_base + (ks * B_K_STEP_BYTES_TF32)
                    tcgen05mma_ws_ss_tf32(desc_a, desc_b, tmem_col, IDESC_TF32_M64_N64, scale)
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # T2R → R2G: tcgen05_ld directly into store_256b (type-agnostic, like C++ reinterpret_cast)
        vec_i32 = tcgen05_ld_32x32b(ACC_NUM_COLS, tmem_col)
        cute.arch.fence_view_async_tmem_load()
        
        # 1. reinterpret_cast to f32 (zero-cost bitcast)
        # vec_f32 = reinterpret_cast(vec_i32, Int32, ACC_NUM_COLS, Float32)

        # 2. TensorSSA wrap → .to(BFloat16) (real CUDA core CVT)
        # regs = TensorSSA(vec_f32, (ACC_NUM_COLS,), Float32)

        # Debug print: thread 0, first 4 register values
        # if tidx == cutlass.Int32(0):
        #     cute.printf("[T2R] tid=0, regs[0..3] = %f, %f, %f, %f",
        #                 regs[0], regs[1], regs[2], regs[3])

        # R2G via store_256b (4 × 256-bit stores per thread)
        # Layout E (column-major warp order):
        #   warp0->(M0,N0), warp1->(M1,N0), warp2->(M0,N1), warp3->(M1,N1)
        lane_idx = tidx % 32
        row = (warp_idx % 2) * 32 + lane_idx
        col_base = (warp_idx // 2) * 32
        base_addr = (C_out.iterator + row * N + col_base).toint()
        for chunk in cutlass.range_constexpr(ACC_NUM_COLS // 8):
            store_256b(base_addr + chunk * 32, subvec(vec_i32, chunk * 8, 8))

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, NUM_COLS)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        assert K_DIM_TF32 == 8, "TODO: support larger K-dimension"
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

# TODO
class _WsTsTf32Kernel:
    """Two-step test:
    - Region 0 (tmem_a_region): populate A into TMEM via S2T copy
    - Region 1 (tmem_c_region): result of WS TS MMA (tmem_a × B → C)
    """

    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        pass

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        raise NotImplementedError(
            "TODO: implement this test following the pattern of _WsSsTf32Kernel, but with the two-step approach described in the docstring"
        )


# =====================================================================
# Test 3: tcgen05mma_ws_ss_f16  (weight-stationary, SMEM A, SMEM B, f16)
# =====================================================================


class _WsSsF16Kernel:
    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_F16
        ACC_NUM_COLS = N // 2
        NUM_COLS = ACC_NUM_COLS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr = smem.allocate(Int64, byte_alignment=8)

        # Create MMA SMEM Layouts
        # NOTE: we use non-ws mode TiledMMA for creating smem layout in a easy way,
        # because smem layouts of ws mode and non-ws mode are the same
        mma_tiler = (M, N, K)
        non_ws_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Float16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M, N),
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(non_ws_tiled_mma, mma_tiler, Float16, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(non_ws_tiled_mma, mma_tiler, Float16, 1)
        bufferA = smem.allocate_tensor(
            element_type=Float16,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        bufferB = smem.allocate_tensor(
            element_type=Float16,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

        # Load A (row-major input → K-major swizzled SMEM)
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

        # --- TMEM allocation ---
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr,
            barrier_for_retrieve=alloc_bar,
            allocator_warp_id=0,
        )
        tmem.allocate(NUM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col = tmem_col_buf[0]

        # Build SMEM descriptors (rank-2 vec_mode layout required)
        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)

        # Per-K-atom byte offsets are derived from the (unswizzled) outer layout
        # so we transparently handle every K size:
        #   K∈{16,32,64}        → A k_iter is single-mode, uniform stride
        #   K≥128               → A k_iter is hierarchical e.g. (4,K/64):(16,4096)
        #   B is always uniform stride=1024 elem
        # Coord ((0,0), 0, ks, 0) into outer layout gives the linear elem offset
        # of the ks-th MMA-K atom; * sizeof(elem) → byte offset to add to desc.
        ELEM_BYTES_F16 = Float16.width // 8
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer

        # Issue WS SS MMA  (scale_out=0 → D = A*B, not accumulate)
        if warp_idx == cutlass.Int32(0):
            with elect_one():
                for ks in cutlass.range_constexpr(K // 16):
                    scale = 0 if ks == 0 else 1
                    a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_F16
                    b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_F16
                    desc_a = desc_a_base + a_off
                    desc_b = desc_b_base + b_off
                    tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_col, IDESC_F16_M64_N64, scale)
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        # T2R
        # Layout E (M=64, ws mode): 128 lanes, 32 columns
        # ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e
        # .32x32b.x32 loads all 32 columns → 32 FP32 regs per thread
        # Layout: warp0->(M0,N0), warp1->(M0,N1), warp2->(M1,N0), warp3->(M1,N1)
        # for 64x64 Acc, each warp process 32x32, with 128 lanes in TMEM all used

        vec_i32 = tcgen05_ld_32x32b(ACC_NUM_COLS, tmem_col)
        cute.arch.fence_view_async_tmem_load()

        # =======DEBUG========
        # # 1. reinterpret_cast to f32 (zero-cost bitcast)
        # vec_f32 = reinterpret_cast(vec_i32, Int32, ACC_NUM_COLS, Float32)

        # # 2. TensorSSA wrap → .to(BFloat16) (real CUDA core CVT)
        # regs = TensorSSA(vec_f32, (ACC_NUM_COLS,), Float32)

        # # Debug print: thread 0, first 4 register values
        # if tidx == cutlass.Int32(0):
        #     cute.printf("[T2R] tid=0, regs[0..3] = %f, %f, %f, %f",
        #                 regs[0], regs[1], regs[2], regs[3])

        # R2G via store_256b (4 × 256-bit stores per thread)
        # Layout E (column-major warp order):
        #   warp0->(M0,N0), warp1->(M1,N0), warp2->(M0,N1), warp3->(M1,N1)
        # in each warp, each thread process one row, T0->[0, 0:31], T1->[1, 0:31], ..., T31->[31, 0:31]
        lane_idx = tidx % 32
        row = (warp_idx % 2) * M // 2 + lane_idx  # M0 or M1
        col_base = (warp_idx // 2) * ACC_NUM_COLS  # N0 or N1
        # 32 regs = 4 chunks of 8 × 32-bit each (256 bits)
        base_addr = (C_out.iterator + row * N + col_base).toint()
        for chunk in cutlass.range_constexpr(ACC_NUM_COLS // 8):
            store_256b(base_addr + chunk * 32, subvec(vec_i32, chunk * 8, 8))

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, NUM_COLS)

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
        pass

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        raise NotImplementedError(
            "TODO: implement this test following the pattern of _WsTsTf32Kernel, but with F16 data and using tcgen05mma_ws_ts_f16 instruction"
        )


# =====================================================================
# Test 5: tcgen05mma_ws_ss_tf32 with explicit collector_b_buffer/collector_op
# =====================================================================


class _WsSsTf32CollectorKernel:
    """Same as _WsSsTf32Kernel but passes collector_b_buffer=B0, collector_op=DISCARD."""

    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, 8 # default K with 8
        ACC_NUM_COLS = N // 2
        NUM_COLS = ACC_NUM_COLS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr = smem.allocate(Int64, byte_alignment=8)

        non_ws_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M, N),
        )
        mma_tiler = (M, N, K)
        a_smem_layout = sm100_utils.make_smem_layout_a(non_ws_tiled_mma, mma_tiler, TFloat32, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(non_ws_tiled_mma, mma_tiler, TFloat32, 1)
        bufferA = smem.allocate_tensor(
            element_type=TFloat32, layout=a_smem_layout.outer, byte_alignment=128, swizzle=a_smem_layout.inner
        )
        bufferB = smem.allocate_tensor(
            element_type=TFloat32, layout=b_smem_layout.outer, byte_alignment=128, swizzle=b_smem_layout.inner
        )
        bufA_s0 = bufferA[(None, None, None, 0)]
        bufB_s0 = bufferB[(None, None, None, 0)]

        if tidx == cutlass.Int32(0):
            mbarrier_init(mbar_ptr, 1)
        mbarrier_init_fence()

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

        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0)
        tmem.allocate(NUM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)
        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col = tmem_col_buf[0]

        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        if warp_idx == cutlass.Int32(0):
            tcgen05mma_ws_ss_tf32(
                desc_a,
                desc_b,
                tmem_col,
                IDESC_TF32_M64_N64,
                0,
                collector_b_buffer=CollectorBBuffer.B0,
                collector_op=CollectorOp.DISCARD,
            )
            with elect_one():
                tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
        mbarrier_wait(mbar_ptr, 0)
        sync_threads()

        vec_i32 = tcgen05_ld_32x32b(NUM_COLS, tmem_col)
        cute.arch.fence_view_async_tmem_load()
        lane_idx = tidx % 32
        row = (warp_idx % 2) * M // 2 + lane_idx
        col_base = (warp_idx // 2) * ACC_NUM_COLS
        base_addr = (C_out.iterator + row * N + col_base).toint()
        for chunk in cutlass.range_constexpr(ACC_NUM_COLS // 8):
            store_256b(base_addr + chunk * 32, subvec(vec_i32, chunk * 8, 8))

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, NUM_COLS)

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
    print(f"  got[0,:4]={got[0, :4].tolist()}")
    print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi, mj]:.6f} ref={ref[mi, mj]:.6f}")
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
    print(f"  got[0,:4]={got[0, :4].tolist()}")
    print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi, mj]:.6f} ref={ref[mi, mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


def test_ws_ss_tf32_collector():
    """Explicit collector_b_buffer=B0, collector_op=DISCARD should match default."""
    print("\n=== Test 5: tcgen05mma_ws_ss_tf32 + collector (B0::DISCARD) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM_TF32)
    B = torch.randn(K_DIM_TF32, N_DIM)
    ref = torch.mm(A, B)
    got = _WsSsTf32CollectorKernel().run(A, B)
    err = (got - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-8)
    max_idx = err.argmax().item()
    mi, mj = max_idx // N_DIM, max_idx % N_DIM
    print(f"  got[0,:4]={got[0, :4].tolist()}")
    print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    print(f"  max_rel_err={rel:.4f}  at ({mi},{mj}): got={got[mi, mj]:.6f} ref={ref[mi, mj]:.6f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_ws_ss_tf32()
    test_ws_ss_f16()
    test_ws_ss_tf32_collector()
    print("\n=== All tests passed! ===")
