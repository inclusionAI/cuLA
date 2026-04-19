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

from cula.ops.ptx_umma_ext import (
    Tcgen05SmemDescriptor,
    tcgen05mma_ws_ss_tf32,
    tcgen05mma_ws_ss_f16,
)
from cula.ops.intrinsics_sm100 import tcgen05_ld_32x32b, tcgen05_st_32x32b, store_256b

M_DIM, N_DIM = 64, 64
K_DIM_TF32 = 8   # kind::tf32  → K=8
K_DIM_F16  = 16  # kind::f16   → K≥16

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
        ACC_NUM_COLS = 32
        NUM_COLS = ACC_NUM_COLS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        # --- SMEM layouts via sm100_utils (handles swizzle correctly for TF32) ---
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
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(NUM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf     = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col         = tmem_col_buf[0]

        # Build SMEM descriptors (rank-2 vec_mode layout required)
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

        # T2R via tcgen05_ld_32x32b
        regs = tcgen05_ld_32x32b(ACC_NUM_COLS, tmem_col)
        cute.arch.fence_view_async_tmem_load()

        # R2G via store_256b (4 × 256-bit stores per thread)
        # Layout E (column-major warp order):
        #   warp0->(M0,N0), warp1->(M1,N0), warp2->(M0,N1), warp3->(M1,N1)
        lane_idx = tidx % 32
        row      = (warp_idx %  2) * 32 + lane_idx
        col_base = (warp_idx // 2) * 32
        base_addr = (C_out.iterator + row * N + col_base).toint()
        for chunk in cutlass.range_constexpr(ACC_NUM_COLS // 8):
            store_256b(base_addr + chunk * 32, regs[chunk * 8 : chunk * 8 + 8])

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
        ACC_NUM_COLS = 32
        OPA_NUM_COLS = 4 # for (M,K)=(64,8), each warp processes 4 columns
        NUM_COLS = OPA_NUM_COLS + ACC_NUM_COLS
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

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        raise NotImplementedError("TODO: implement this test following the pattern of _WsSsTf32Kernel, but with the two-step approach described in the docstring")


# =====================================================================
# Test 3: tcgen05mma_ws_ss_f16  (weight-stationary, SMEM A, SMEM B, f16)
# =====================================================================

class _WsSsF16Kernel:
    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        M, N, K = M_DIM, N_DIM, K_DIM_F16
        ACC_NUM_COLS = 32
        NUM_COLS = ACC_NUM_COLS
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        num_stages = 1

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)

        # --- Manual SMEM layouts (no sm100_utils dependency since no tcgen05.mma.ws support) ---
        # A: K-major, F16, (M, K)=(64, 16)
        # K=16 F16 = 32 bytes → Swizzle S<1,4,3> (SW32)
        # Nested first mode ((M, K), ...) gives rank-2 vec_mode for make_umma_smem_desc
        sw_a = cute.make_swizzle(1, 4, 3)
        outer_a = cute.make_layout(
            ((M, K), 1, 1, num_stages),
            stride=((K, 1), M * K, M * K, M * K),
        )
        bufferA = smem.allocate_tensor(
            element_type=Float16, layout=outer_a,
            byte_alignment=128, swizzle=sw_a,
        )

        # B: MN-major, F16, (N, K)=(64, 16)
        # N=64 F16 = 128 bytes → Swizzle S<3,4,3> (SW128)
        sw_b = cute.make_swizzle(3, 4, 3)
        outer_b = cute.make_layout(
            ((N, K), 1, 1, num_stages),
            stride=((1, N), N * K, N * K, N * K),
        )
        bufferB = smem.allocate_tensor(
            element_type=Float16, layout=outer_b,
            byte_alignment=128, swizzle=sw_b,
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
            tmem_hold_ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
        )
        tmem.allocate(NUM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        tmem_col_buf     = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col         = tmem_col_buf[0]

        # Build SMEM descriptors (rank-2 vec_mode layout required)
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

        # T2R
        # Layout E (M=64, ws mode): 128 lanes, 32 columns
        # ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e
        # .32x32b.x32 loads all 32 columns → 32 FP32 regs per thread
        # Layout: warp0->(M0,N0), warp1->(M0,N1), warp2->(M1,N0), warp3->(M1,N1)
        # for 64x64 Acc, each warp process 32x32, with 128 lanes in TMEM all used

        regs = tcgen05_ld_32x32b(ACC_NUM_COLS, tmem_col)
        cute.arch.fence_view_async_tmem_load()

        # Debug print: thread 0, first 4 register values
        # if tidx == cutlass.Int32(0):
        #     cute.printf("[T2R] tid=0, regs[0..3] = %f, %f, %f, %f",
        #                 regs[0], regs[1], regs[2], regs[3])

        # R2G via store_256b (4 × 256-bit stores per thread)
        # Layout E (column-major warp order):
        #   warp0->(M0,N0), warp1->(M1,N0), warp2->(M0,N1), warp3->(M1,N1)
        # in each warp, each thread process one row, T0->[0, 0:31], T1->[1, 0:31], ..., T31->[31, 0:31]
        lane_idx = tidx % 32
        row      = (warp_idx %  2) * 32 + lane_idx   # M0 or M1
        col_base = (warp_idx // 2) * 32               # N0 or N1
        # 32 regs = 4 chunks of 8 FP32 each (256 bits)
        # store_256b needs a raw i64 address → use iterator.toint() + byte offset
        base_addr = (C_out.iterator + row * N + col_base).toint()
        for chunk in cutlass.range_constexpr(ACC_NUM_COLS // 8):
            # byte offset: chunk * 8 elements * 4 bytes/element
            store_256b(base_addr + chunk * 32, regs[chunk * 8 : chunk * 8 + 8])

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
        M, N, K = M_DIM, N_DIM, K_DIM_F16
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem          = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr      = smem.allocate(Int64, byte_alignment=8)


    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        raise NotImplementedError("TODO: implement this test following the pattern of _WsTsTf32Kernel, but with F16 data and using tcgen05mma_ws_ts_f16 instruction")


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

# TODO
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

# TODO
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
    test_ws_ss_tf32()
    test_ws_ss_f16()
    print("\n=== All tests passed! ===")
