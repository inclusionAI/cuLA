"""
Standalone CuteDSL test for ptx_umma_masked.py inline PTX MMA wrappers.

Tests:
  1. tcgen05mma_ss_no_mask  -- M=64, N=64, K=8, TF32, all rows active → matches torch.mm
  2. tcgen05mma_ss_mask0    -- groups 0,2 active (rows 0-15, 32-47), groups 1,3 disabled
  3. tcgen05mma_ss_mask1    -- groups 1,3 active (rows 16-31, 48-63), groups 0,2 disabled

SMEM layout:
  A: swizzled K-major (Swizzle<1,4,3>, SWIZZLE_32B), descriptor LBO=1 SBO=16 layout=6
  B: swizzled MN-major (Swizzle<2,5,2>, SWIZZLE_128B_BASE32B), LBO=64 SBO=32 layout=1
  Data is loaded with M-major mapping for A, direct row-major for B.

All descriptor values are computed via make_umma_smem_desc / smem_descriptor_to_int
(proven correct in test_umma_ptx_jit.py). Wrapped in Tcgen05SmemDescriptor for API
compatibility with ptx_umma_masked.py convenience wrappers.
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
    Pack,
    Repetition,
    make_umma_smem_desc,
    smem_descriptor_to_int,
)
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64, TFloat32

from cula.ops.ptx_umma_ext import (
    Tcgen05SmemDescriptor,
    tcgen05mma_ss_mask0,
    tcgen05mma_ss_mask1,
    tcgen05mma_ss_no_mask,
)

M_DIM, N_DIM, K_DIM = 64, 64, 8
TMEM_COLS = 64

IDESC_M64_N64 = (4 << 24) | (8 << 17) | (1 << 16) | (2 << 10) | (2 << 7) | (1 << 4)
assert IDESC_M64_N64 == 0x4110910


class _Kernel:
    def __init__(self, mask_mode: str = "none"):
        self.mask_mode = mask_mode

    @cute.kernel
    def kernel(self, A_in: cute.Tensor, B_in: cute.Tensor, C_out: cute.Tensor):
        """
        For mask_mode == "none": single-phase MMA, all rows written.
        For mask_mode == "mask0"/"mask1": two-phase MMA:
            Phase 1 - full no-mask MMA with A_in used as A_zero (passed by caller as zeros),
                      scale_out=0 → zeroes TMEM for all rows.
            Phase 2 - masked MMA using B column from B_in (same B, new A from A_in second half).

        To keep the interface simple, for mask tests A_in is [A_zero (64×8) || A_real (64×8)]
        concatenated to shape (128, 8). Phase 1 loads rows [0:64], phase 2 loads rows [64:128].
        """
        M, N, K = M_DIM, N_DIM, K_DIM
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        smem = utils.SmemAllocator()
        tmem_hold_ptr = smem.allocate(Int32)
        mbar_ptr = smem.allocate(Int64, byte_alignment=8)

        # Build tiled_mma to get correct SMEM layout
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            TFloat32,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (M, N),
        )
        mma_tiler = (M, N, K)

        # Allocate swizzled SMEM
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, TFloat32, 1)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, TFloat32, 1)
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

        # gA_all: flat view of input A tensor (either M*K or 2M*K elements)
        # For no_mask: caller passes (M,K) → total = M*K
        # For mask tests: caller passes (2M,K) → total = 2M*K; row_offset selects which half
        if cutlass.const_expr(self.mask_mode != "none"):
            gA_all = cute.make_tensor(A_in.iterator, cute.make_layout(2 * M_DIM * K_DIM))
        else:
            gA_all = cute.make_tensor(A_in.iterator, cute.make_layout(M_DIM * K_DIM))

        # Load B once (shared by both phases)
        gB_flat = cute.make_tensor(B_in.iterator, cute.make_layout(K * N))
        for step in cutlass.range(K * N // 128, unroll_full=False):
            idx = tidx + step * 128
            bufB_s0[idx] = gB_flat[idx]
        sync_threads()

        # TMEM allocation
        alloc_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        tmem = utils.TmemAllocator(
            tmem_hold_ptr,
            barrier_for_retrieve=alloc_bar,
            allocator_warp_id=0,
        )
        tmem.allocate(TMEM_COLS)
        tmem.wait_for_alloc()
        tmem_ptr_f32 = tmem.retrieve_ptr(Float32)

        acc_shape = tiled_mma.partition_shape_C((M, N))
        acc_shape_staged = cute.append(acc_shape, 1)
        tCtAcc = cute.make_tensor(tmem_ptr_f32, tiled_mma.make_fragment_C(acc_shape_staged).layout)
        tmem_col_buf = cute.make_tensor(tmem_hold_ptr, cute.make_layout(1))
        tmem_col = tmem_col_buf[0]

        # Build descriptors
        desc_a_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufA_s0.iterator, bufA_s0.layout, "k"))
        desc_b_i64 = smem_descriptor_to_int(make_umma_smem_desc(bufB_s0.iterator, bufB_s0.layout, "mn"))
        desc_a = Tcgen05SmemDescriptor(desc_a_i64)
        desc_b = Tcgen05SmemDescriptor(desc_b_i64)

        if cutlass.const_expr(self.mask_mode != "none"):
            # Phase 1: Load A_zero (first M rows of gA_all = all zeros), no_mask MMA → zero TMEM
            for step in cutlass.range(M_DIM * K_DIM // 128, unroll_full=False):
                smem_idx = tidx + step * 128
                m = smem_idx % M_DIM
                k = smem_idx // M_DIM
                bufA_s0[smem_idx] = gA_all[m * K_DIM + k]  # row_offset=0
            sync_threads()
            if warp_idx == cutlass.Int32(0):
                tcgen05mma_ss_no_mask(desc_a, desc_b, tmem_col, IDESC_M64_N64, 0)
                with elect_one():
                    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
            mbarrier_wait(mbar_ptr, 0)
            sync_threads()
            # Re-arm mbar for second MMA
            if tidx == cutlass.Int32(0):
                mbarrier_init(mbar_ptr, 1)
            mbarrier_init_fence()

            # Phase 2: Load A_real (rows M..2M of gA_all = real data), masked MMA
            for step in cutlass.range(M_DIM * K_DIM // 128, unroll_full=False):
                smem_idx = tidx + step * 128
                m = smem_idx % M_DIM
                k = smem_idx // M_DIM
                bufA_s0[smem_idx] = gA_all[(M_DIM + m) * K_DIM + k]  # row_offset=M
            sync_threads()
            if warp_idx == cutlass.Int32(0):
                if cutlass.const_expr(self.mask_mode == "mask0"):
                    tcgen05mma_ss_mask0(desc_a, desc_b, tmem_col, IDESC_M64_N64, 0)
                elif cutlass.const_expr(self.mask_mode == "mask1"):
                    tcgen05mma_ss_mask1(desc_a, desc_b, tmem_col, IDESC_M64_N64, 0)
                with elect_one():
                    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
            mbarrier_wait(mbar_ptr, 0)
            sync_threads()
        else:
            # Simple single-phase no_mask
            for step in cutlass.range(M_DIM * K_DIM // 128, unroll_full=False):
                smem_idx = tidx + step * 128
                m = smem_idx % M_DIM
                k = smem_idx // M_DIM
                bufA_s0[smem_idx] = gA_all[m * K_DIM + k]
            sync_threads()
            if warp_idx == cutlass.Int32(0):
                tcgen05mma_ss_no_mask(desc_a, desc_b, tmem_col, IDESC_M64_N64, 0)
                with elect_one():
                    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)
            mbarrier_wait(mbar_ptr, 0)
            sync_threads()

        # T2R: TMEM → RMEM
        t2r_atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(Repetition(8), Pack.NONE), Float32)
        fake_smem = cute.make_tensor(cute.make_ptr(Float32, 0, cute.AddressSpace.smem), cute.make_layout((M, N)))
        tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tCtAcc_flat)
        tTR_sDummy = thr_t2r.partition_D(fake_smem)
        tTR_rAcc = cute.make_rmem_tensor(tTR_sDummy.shape, Float32)

        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        # R2G: RMEM → GMEM (row-major)
        gC = cute.make_tensor(C_out.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tTR_gC = thr_t2r.partition_D(gC)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), tTR_rAcc, tTR_gC)

        sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr_f32, TMEM_COLS)

    @cute.jit
    def _launch(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream):
        self.kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    def run(self, A_cpu, B_cpu):
        """
        For no_mask: A_cpu is (M, K).
        For mask tests: A_cpu is (2M, K) where [0:M] = zeros, [M:2M] = real A.
        """
        A_gpu = A_cpu.contiguous().float().cuda()
        B_gpu = B_cpu.contiguous().float().cuda()
        C_gpu = torch.zeros(M_DIM, N_DIM, dtype=torch.float32, device="cuda")
        stream = cutlass_torch.default_stream()
        self._launch(from_dlpack(A_gpu), from_dlpack(B_gpu), from_dlpack(C_gpu), stream)
        torch.cuda.synchronize()
        return C_gpu.cpu()


def test_ss_no_mask():
    print("\n=== Test 1: tcgen05mma_ss_no_mask (all rows active) ===")
    torch.manual_seed(42)
    A = torch.randn(M_DIM, K_DIM)
    B = torch.randn(K_DIM, N_DIM)
    ref = torch.mm(A, B)
    got = _Kernel("none").run(A, B)
    rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
    print(f"  got[0,:4]={got[0, :4].tolist()}")
    print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    print(f"  max_rel_err={rel:.4f}")
    assert rel < 0.02, f"FAIL: rel={rel:.4f}"
    print("  PASSED")


def _run_masked(mask_mode, A_real, B):
    """
    Two-phase: phase1 = no_mask with A_zero (rows 0..M-1 of combined A = zeros),
               phase2 = masked with A_real (rows M..2M-1 of combined A).
    Combined A is shape (2M, K): [zeros || A_real].
    """
    A_combined = torch.cat([torch.zeros_like(A_real), A_real], dim=0)
    return _Kernel(mask_mode).run(A_combined, B)


def test_ss_mask0():
    print("\n=== Test 2: tcgen05mma_ss_mask0 ===")
    torch.manual_seed(0)
    A = torch.randn(M_DIM, K_DIM)
    B = torch.randn(K_DIM, N_DIM)
    ref = torch.mm(A, B)  # expected for active rows
    got = _run_masked("mask0", A, B)

    # SS_MASK0 = (0, 0xFF..., 0, 0xFF...) → mask words 1,3 disable rows 16-31 and 48-63
    # Active: rows 0-15 and 32-47, Disabled: rows 16-31 and 48-63
    active_rows = list(range(0, 16)) + list(range(32, 48))
    masked_rows = list(range(16, 32)) + list(range(48, 64))

    rel_active = (got[active_rows] - ref[active_rows]).abs().max().item() / (ref[active_rows].abs().max().item() + 1e-8)
    zero_max = got[masked_rows].abs().max().item()

    print(f"  active rows 0-15,32-47: max_rel_err={rel_active:.4f}  (expect <0.02)")
    print(f"  masked rows 16-31,48-63: max_abs={zero_max:.4f}  (expect 0.0)")
    assert rel_active < 0.02, f"FAIL active rows: {rel_active:.4f}"
    assert zero_max == 0.0, f"FAIL masked rows not zero: {zero_max}"
    print("  PASSED")


def test_ss_mask1():
    print("\n=== Test 3: tcgen05mma_ss_mask1 ===")
    torch.manual_seed(7)
    A = torch.randn(M_DIM, K_DIM)
    B = torch.randn(K_DIM, N_DIM)
    ref = torch.mm(A, B)
    got = _run_masked("mask1", A, B)

    # SS_MASK1 = (0xFF..., 0, 0xFF..., 0) → mask words 0,2 disable rows 0-15 and 32-47
    # Active: rows 16-31 and 48-63, Disabled: rows 0-15 and 32-47
    active_rows = list(range(16, 32)) + list(range(48, 64))
    masked_rows = list(range(0, 16)) + list(range(32, 48))

    rel_active = (got[active_rows] - ref[active_rows]).abs().max().item() / (ref[active_rows].abs().max().item() + 1e-8)
    zero_max = got[masked_rows].abs().max().item()

    print(f"  active rows 16-31,48-63: max_rel_err={rel_active:.4f}  (expect <0.02)")
    print(f"  masked rows 0-15,32-47: max_abs={zero_max:.4f}  (expect 0.0)")
    assert rel_active < 0.02, f"FAIL active rows: {rel_active:.4f}"
    assert zero_max == 0.0, f"FAIL masked rows not zero: {zero_max}"
    print("  PASSED")


if __name__ == "__main__":
    test_ss_no_mask()
    test_ss_mask0()
    test_ss_mask1()
    print("\n=== All tests passed! ===")
