# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk Gated Delta Rule Forward H Kernel V2 - No GMEM Roundtrip (Register-Carry)

Optimized version eliminating GMEM roundtrip:
- h_state: carried in CUDA registers across chunks (no R2T needed)
- v_new: computed in registers → R2S to sVnew (SMEM) → KV MMA A operand

Both MMAs share M=BV and use TMEM×SMEM operand mode:
- WH MMA: state(BV,BK) @ W(BT,BK) → WH_acc(BV,BT)
- KV MMA: v_new^T(BV,BT) @ K^T(BK,BT) → update(BV,BK)  [ACCUMULATE=False]

After KV MMA:  h_new = G * h + update  (in registers)

BV must be a multiple of 64 (tcgen05.mma.ws M-mode constraint for bf16).
"""

import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
import torch.nn.functional as F
import triton
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T as _T

from cula.utils import USE_FAST_MATH, assert_blackwell, prepare_chunk_indices, prepare_lens, tensor_cache


# in FLA, cumsum returns int64 tensor by default
@tensor_cache
def prepare_chunk_offsets_i32(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1).int()


@cutlass.dsl_user_op
def _atomic_add_global_i32(ptr_i64, addend_i32, *, loc=None, ip=None):
    """Atomic add on global memory int32. Returns old value."""
    result = _llvm.inline_asm(
        _T.i32(),
        [ptr_i64, addend_i32],
        "atom.global.add.s32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


PRINT_DEBUG = False

LN2 = 0.6931471805599453
INV_LN2 = 1.4426950408889634


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkDeltaRuleFwdH:
    """
    V2: No GMEM roundtrip. Both MMAs share M=BV=64, TMEM×SMEM operand mode.
    h carried in CUDA registers; KV MMA only computes update term.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        is_varlen: bool = False,
        persistent: bool = True,
        use_fast_math: bool = True,
    ):
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
        self.is_varlen = is_varlen

        self.BT = chunk_size  # 64
        self.BK = head_dim_k  # 128
        self.BV = 64  # V tiling fixed at 64

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        # Register allocation (occ=1 only):
        # - 232 regs for CUDA warps (both varlen and non-varlen)
        #   Extra headroom reduces register spilling and improves ILP
        self.min_occupancy = 1
        self.persistent = persistent if is_varlen else False  # only meaningful for varlen
        self.num_regs_cuda = 232
        self.num_regs_others = 40
        self.threads_per_cta = self.threads_per_warp * 8

        # WH MMA tiler: (M=BV, N=BT=64, K=BK=128), A from TMEM, B from SMEM
        self.wh_mma_tiler = (self.BV, self.BT, self.BK)
        # KV MMA tiler: (M=BV, N=BK=128, K=BT=64), A from TMEM, B from SMEM
        self.kv_mma_tiler = (self.BV, self.BK, self.BT)

        # Pipeline stage assignment (hardcoded optimal for BV=64, occ=1, 228KB SMEM):
        self.k_stage = 3
        self.w_stage = 3
        self.u_stage = 3
        self.h_out_stage = 2
        self.vnew_store_stage = 2
        self.acc_stage = 1
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        self.gk_stage = 3

        # TMA h0 load: bytes per BK×BV fp32 tile
        self.tma_h0_bytes = self.BK * self.BV * 4  # 128×64×4 = 32,768 bytes

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        # Barrier for CUDA warp-group sync during cooperative gk_scale precomputation
        self.gk_precompute_bar = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.cuda_warp_ids),  # 128
        )
        # No CTA-wide barrier needed for WU scheduling:
        # Load warp elect_one arrives on a lightweight mbarrier (count=1),
        # other warps just wait on mbarrier phase (no arrive needed).
        self.buffer_align_bytes = 1024

    @staticmethod
    def _plan_tmem_offsets(tiled_mma_wh, tile_wh, tiled_mma_kv, tile_kv, state_tmem_layout, vnew_tmem_layout, acc_stages):
        SM100_TMEM_CAPACITY_COLS = 512
        # WH acc: (BV, BT) FP32
        wh_shape = tiled_mma_wh.partition_shape_C(tile_wh[:2])
        wh_fake = tiled_mma_wh.make_fragment_C(cute.append(wh_shape, acc_stages))
        num_wh = tcgen05.find_tmem_tensor_col_offset(wh_fake)
        # State TMEM A operand for WH MMA: (BV, BK) BF16
        tCrState_fake = tiled_mma_wh.make_fragment_A(state_tmem_layout.outer.shape)
        num_state = tcgen05.find_tmem_tensor_col_offset(tCrState_fake)
        # v_new TMEM A operand for KV MMA: (BV, BT) BF16
        tCrVnew_fake = tiled_mma_kv.make_fragment_A(vnew_tmem_layout.outer.shape)
        num_vnew = tcgen05.find_tmem_tensor_col_offset(tCrVnew_fake)
        # KV acc: (BV, BK) FP32
        kv_shape = tiled_mma_kv.partition_shape_C(tile_kv[:2])
        kv_fake = tiled_mma_kv.make_fragment_C(cute.append(kv_shape, 1))
        num_kv = tcgen05.find_tmem_tensor_col_offset(kv_fake)

        wh_off = 0
        state_off = wh_off + num_wh
        vnew_off = state_off + num_state
        kv_off = vnew_off + num_vnew
        total_tmp = kv_off + num_kv
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        if cutlass.const_expr(PRINT_DEBUG):
            print(
                f"  TMEM: WH={num_wh}@{wh_off}, State={num_state}@{state_off}, Vnew={num_vnew}@{vnew_off}, KV={num_kv}@{kv_off}, total={total}"
            )
        return wh_off, state_off, vnew_off, kv_off, total

    def _compute_grid(self, B, H, V):
        num_v_tiles = (V + self.BV - 1) // self.BV
        if self.is_varlen:
            if self.persistent:
                import torch

                sm_count = torch.cuda.get_device_properties(0).multi_processor_count
                return (sm_count, 1, 1)
            else:
                # Non-persistent: one CTA per work unit, free HW scheduling
                total_work_units = num_v_tiles * H * B
                return (total_work_units, 1, 1)
        return (num_v_tiles, H, B)

    @cute.jit
    def __call__(
        self,
        k_in: cute.Tensor,  # [B, T, H, K] or [T_total, H, K]
        w_in: cute.Tensor,  # [B, T, H, K] or [T_total, H, K]
        u_in: cute.Tensor,  # [B, T, H, V] or [T_total, H, V]
        g_in: cute.Tensor,  # [B, T, H] or [T_total, H] (fp32, unused currently)
        gk_in: cute.Tensor,  # [B, T, H, K] or [T_total, H, K] (fp32)
        h_out_in: cute.Tensor,  # [B, NT, H, K, V] or [NT_total, H, K, V]
        v_new_in: cute.Tensor,  # [B, T, H, V] or [T_total, H, V]
        h0_in: cute.Tensor,  # [B, H, K, V] (fp32)
        ht_in: cute.Tensor,  # [B, H, K, V]
        cu_seqlens_in: cute.Tensor,  # [N+1] int32
        chunk_offsets_in: cute.Tensor,  # [N+1] int32
        workspace_in: cute.Tensor,  # workspace buffer
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
        use_g: Int32,
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
        stream,
    ):
        # Extract pointers from tensor args (TVM-FFI compatible)
        k_ptr = k_in.iterator
        w_ptr = w_in.iterator
        u_ptr = u_in.iterator
        gk_ptr = gk_in.iterator
        h_out_ptr = h_out_in.iterator
        v_new_ptr = v_new_in.iterator
        h0_ptr = h0_in.iterator
        ht_ptr = ht_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator
        chunk_offsets_ptr = chunk_offsets_in.iterator
        workspace_ptr = workspace_in.iterator

        B, T, H, K, V = problem_size

        # For varlen: B=num_seqs, T=total_tokens, data tensors use data_B=1.
        # For non-varlen: data_B=B, NT=ceil(T/BT).
        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
            NT = total_nt  # total number of chunks across all sequences
        else:
            data_B = B
            NT = (T + self.BT - 1) // self.BT

        # ===================== GMEM layouts =====================
        # Data tensors use data_B for batch dimension (1 for varlen, B for non-varlen)
        kt_layout = cute.make_layout((K, T, (H, data_B)), stride=(1, H * K, (K, T * H * K)))
        kt = cute.make_tensor(k_ptr, kt_layout)

        w_layout = cute.make_layout((T, K, (H, data_B)), stride=(H * K, 1, (K, T * H * K)))
        w = cute.make_tensor(w_ptr, w_layout)

        u_layout = cute.make_layout((T, V, (H, data_B)), stride=(H * V, 1, (V, T * H * V)))
        u = cute.make_tensor(u_ptr, u_layout)

        v_new = cute.make_tensor(v_new_ptr, u_layout)

        # h_out: for varlen, NT=total_chunks and data_B=1; for non-varlen, NT=per-seq chunks and data_B=B
        h_out_T_layout = cute.make_layout(
            (V, K, (NT, H, data_B)),
            stride=(1, V, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out_T = cute.make_tensor(h_out_ptr, h_out_T_layout)

        # h0/ht always use B=num_seqs (same for both varlen and non-varlen)
        h0_layout = cute.make_layout((K, V, (H, B)), stride=(V, 1, (K * V, H * K * V)))
        h0 = cute.make_tensor(h0_ptr, h0_layout)

        ht_T_layout = cute.make_layout((V, K, (H, B)), stride=(1, V, (K * V, H * K * V)))
        ht_T = cute.make_tensor(ht_ptr, ht_T_layout)

        # gk K-first view for TMA: (K, T, (H, data_B)) with K contiguous
        gk_K_layout = cute.make_layout((K, T, (H, data_B)), stride=(1, H * K, (K, T * H * K)))
        gk_K = cute.make_tensor(gk_ptr, gk_K_layout)

        # Transposed U view: (V, T, (H, data_B)) to match WH acc shape (M=BV, N=BT)
        u_T_layout = cute.make_layout((V, T, (H, data_B)), stride=(1, H * V, (V, T * H * V)))
        u_T = cute.make_tensor(u_ptr, u_T_layout)

        self.k_dtype = kt.element_type
        self.w_dtype = w.element_type
        self.u_dtype = u.element_type

        # ===================== MMA setup =====================
        # WH MMA: A=state(TMEM, K-major), B=W(SMEM, K-major)
        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,  # A: state, K-major (required for TMEM source)
            tcgen05.OperandMajorMode.K,  # B: W, K-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,  # A operand from TMEM (zero-copy)
        )

        # KV MMA: A=v_new^T(TMEM, K-major required), B=K^T(SMEM, MN-major)
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,  # A: v_new, K-major (required for TMEM source)
            tcgen05.OperandMajorMode.MN,  # B: K^T, MN-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,  # A operand from TMEM (zero-copy)
        )

        # v_new TMEM layout for KV MMA A operand
        vnew_tmem_layout = sm100_utils.make_smem_layout_a(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            1,
        )
        # State TMEM layout for WH MMA A operand
        state_tmem_layout = sm100_utils.make_smem_layout_a(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            1,
        )

        # ===================== TMEM offsets =====================
        (self.tmem_wh_off, self.tmem_state_off, self.tmem_vnew_off, self.tmem_kv_off, self.tmem_total) = (
            self._plan_tmem_offsets(
                wh_tiled_mma,
                self.wh_mma_tiler,
                kv_tiled_mma,
                self.kv_mma_tiler,
                state_tmem_layout,
                vnew_tmem_layout,
                self.acc_stage,
            )
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # W as B operand of WH MMA
        w_smem_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            self.w_stage,
        )
        # K^T as B operand of KV MMA
        kt_smem_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            self.k_stage,
        )
        # State A operand now from TMEM (no SMEM layout needed)
        # v_new A operand now from TMEM (no SMEM layout needed for MMA path)
        # h_out epilogue for TMA store
        # COL_MAJOR for (BV, BK): BV contiguous → matches V stride 1 in h_out_T GMEM
        h_out_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BK),
            self.h_out_stage,
        )
        # U SMEM for TMA load — COL_MAJOR (BV, BT), BV contiguous matches u_T GMEM
        u_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.u_stage,
        )
        # v_new store SMEM — COL_MAJOR (BV, BT) for TMA S2G
        vnew_store_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.vnew_store_stage,
        )

        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        # ===================== TMA descriptors =====================
        w_smem = cute.select(w_smem_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            w,
            w_smem,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout.shape,
        )

        kt_smem = cute.select(kt_smem_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem,
            self.kv_mma_tiler,
            kv_tiled_mma,
            cluster_layout.shape,
        )

        h_epi_smem = cute.select(h_out_epi_staged, mode=[0, 1])
        tma_atom_h_out, tma_tensor_h_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            h_out_T,
            h_epi_smem,
            (self.BV, self.BK),
        )

        # TMA descriptor for U load (G2S) — non-MMA operand
        u_smem = cute.select(u_epi_staged, mode=[0, 1])
        tma_atom_u, tma_tensor_u = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            u_T,
            u_smem,
            (self.BV, self.BT),
        )

        # v_new transposed GMEM view: (V, T, (H, data_B)) for TMA store
        v_new_T_layout = cute.make_layout(
            (V, T, (H, data_B)),
            stride=(1, H * V, (V, T * H * V)),
        )
        v_new_T = cute.make_tensor(v_new_ptr, v_new_T_layout)

        # cu_seqlens and chunk_offsets tensors for varlen
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((B + 1,)))
        chunk_offsets = cute.make_tensor(chunk_offsets_ptr, cute.make_layout((B + 1,)))

        # TMA descriptor for v_new store (S2G) — used only for non-varlen
        vnew_store_smem = cute.select(vnew_store_epi_staged, mode=[0, 1])
        tma_atom_vnew_st, tma_tensor_vnew_st = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            v_new_T,
            vnew_store_smem,
            (self.BV, self.BT),
        )

        # Direct GMEM write TiledCopy for v_new — used for varlen mode
        # (avoids TMA store tail fixup descriptor corruption bug)
        # sVnew_store is COL_MAJOR (BV, BT): BV contiguous (mode 0)
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.io_dtype.width  # 8 for bf16
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # Thread layout for 32 store-warp threads over (BV=64, BT=64) tile:
        # dim0=8 thread-groups × 8 values = 64, dim1=4 threads, 16 BT-repeats
        vnew_thr_dim0 = self.BV // async_copy_elems  # 8
        vnew_thr_dim1 = self.threads_per_warp // vnew_thr_dim0  # 4
        assert self.BT % vnew_thr_dim1 == 0
        vnew_thr_layout = cute.make_ordered_layout(
            (vnew_thr_dim0, vnew_thr_dim1),
            order=(0, 1),
        )  # (8, 4), BV-groups faster → coalesced GMEM writes
        vnew_val_layout = cute.make_layout((async_copy_elems, 1))  # (8, 1)
        gmem_tiled_copy_vnew = cute.make_tiled_copy_tv(
            atom_universal_copy,
            vnew_thr_layout,
            vnew_val_layout,
        )

        # TMA descriptor for gk load (G2S) — 2D tile (BK, 1) along K dimension
        gk_smem_2d = cute.make_layout((self.BK, 1))
        tma_atom_gk, tma_tensor_gk = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            gk_K,
            gk_smem_2d,
            (self.BK, 1),
        )

        # TMA descriptor for h0 load (G2S) — 2D tile (BK, BV) of Float32
        # No swizzle — h0 is read with scalar indexing in CUDA warps
        # Explicit row-major strides (BV, 1) so TMA write order matches sH0 read order
        h0_smem_layout = cute.make_layout(
            (self.BK, self.BV),
            stride=(self.BV, 1),
        )
        tma_atom_h0, tma_tensor_h0 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            h0,
            h0_smem_layout,
            (self.BK, self.BV),
        )

        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem)
        self.tma_kt_bytes = cute.size_in_bytes(self.io_dtype, kt_smem)
        self.tma_u_bytes = cute.size_in_bytes(self.io_dtype, u_smem)
        self.tma_gk_bytes = self.BK * 4  # BK Float32 elements

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            load_w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_kt_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_u_mbar: cute.struct.MemRange[Int64, self.u_stage * 2]  # Load→CUDA: sU ready
            load_gk_mbar: cute.struct.MemRange[Int64, self.gk_stage * 2]  # Load→CUDA: sGK ready
            state_tmem_mbar: cute.struct.MemRange[Int64, 1 * 2]  # CUDA→MMA: state TMEM ready
            wh_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: WH done
            vnew_smem_mbar: cute.struct.MemRange[Int64, 1 * 2]  # CUDA→MMA: sVnew ready
            kv_done_mbar: cute.struct.MemRange[Int64, 1 * 2]  # MMA→CUDA: KV done
            h_out_mbar: cute.struct.MemRange[Int64, self.h_out_stage * 2]  # CUDA→Store: sH_epi ready
            vnew_store_mbar: cute.struct.MemRange[Int64, self.vnew_store_stage * 2]  # CUDA→Store: sVnew_store ready
            h0_load_mbar: cute.struct.MemRange[Int64, 1 * 2]  # Load→CUDA: h0 ready
            tmem_holding_buf: Int32
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kt_smem_staged)],
                self.buffer_align_bytes,
            ]
            # sState removed: state now goes through TMEM, not SMEM
            # sVnew removed: v_new now goes through TMEM, not SMEM
            sH_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_out_epi_staged)],
                self.buffer_align_bytes,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(u_epi_staged)],
                self.buffer_align_bytes,
            ]
            sVnew_store: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vnew_store_epi_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.gk_stage],
                128,
            ]
            sH0: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.BV],
                self.buffer_align_bytes,
            ]
            # Double-buffered work index for dynamic scheduling
            sWorkIdx: cute.struct.MemRange[Int32, 2]
            # Double-buffered scheduling mbarriers (count=1 each, Load warp elect_one arrives)
            sched_mbar: cute.struct.MemRange[Int64, 2]
            # Double-buffered consumed mbarriers (count=4, one arrive per consumer warp group)
            sched_consumed_mbar: cute.struct.MemRange[Int64, 2]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(B, H, V)

        self.kernel(
            wh_tiled_mma,
            kv_tiled_mma,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_h_out,
            tma_tensor_h_out,
            tma_atom_u,
            tma_tensor_u,
            tma_atom_vnew_st,
            tma_tensor_vnew_st,
            tma_atom_gk,
            tma_tensor_gk,
            tma_atom_h0,
            tma_tensor_h0,
            gmem_tiled_copy_vnew,
            h0,
            ht_T,
            u,
            u_T,
            h_out_T,
            v_new,
            w_smem_staged,
            kt_smem_staged,
            state_tmem_layout,
            vnew_tmem_layout,
            h_out_epi_staged,
            u_epi_staged,
            vnew_store_epi_staged,
            cu_seqlens,
            chunk_offsets,
            workspace_ptr,
            problem_size,
            use_gk,
            use_initial_state,
            store_final_state,
            save_v_new,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        wh_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_h_out: cute.CopyAtom,
        tma_tensor_h_out: cute.Tensor,
        tma_atom_u: cute.CopyAtom,
        tma_tensor_u: cute.Tensor,
        tma_atom_vnew_st: cute.CopyAtom,
        tma_tensor_vnew_st: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        tma_atom_h0: cute.CopyAtom,
        tma_tensor_h0: cute.Tensor,
        gmem_tiled_copy_vnew: cute.TiledCopy,
        h0: cute.Tensor,
        ht_tensor: cute.Tensor,
        u_tensor: cute.Tensor,
        u_T_tensor: cute.Tensor,
        h_out_T_tensor: cute.Tensor,
        v_new_tensor: cute.Tensor,
        w_smem_staged: cute.ComposedLayout,
        kt_smem_staged: cute.ComposedLayout,
        state_tmem_layout: cute.ComposedLayout,
        vnew_tmem_layout: cute.ComposedLayout,
        h_out_epi_staged: cute.ComposedLayout,
        u_epi_staged: cute.ComposedLayout,
        vnew_store_epi_staged: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        workspace_iter: cute.Pointer,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_u)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_gk)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sGK_smem = storage.sGK.get_tensor(cute.make_layout((self.BK, self.gk_stage)))
        # 3D SMEM view for _epilog_partition in Load warp: (BK, 1, gk_stage)
        sGK_3d = storage.sGK.get_tensor(cute.make_layout((self.BK, 1, self.gk_stage), stride=(1, self.BK, self.BK)))

        # ===================== Pipelines =====================
        load_w_P, load_w_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar.data_ptr(),
        ).make_participants()

        load_kt_P, load_kt_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_kt_bytes,
            barrier_storage=storage.load_kt_mbar.data_ptr(),
        ).make_participants()

        state_smem_P, state_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.state_tmem_mbar.data_ptr(),
        ).make_participants()

        wh_done_P, wh_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.wh_done_mbar.data_ptr(),
        ).make_participants()

        vnew_smem_P, vnew_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.vnew_smem_mbar.data_ptr(),
        ).make_participants()

        kv_done_P, kv_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_done_mbar.data_ptr(),
        ).make_participants()

        h_out_P, h_out_C = pipeline.PipelineAsync.create(
            num_stages=self.h_out_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.h_out_mbar.data_ptr(),
        ).make_participants()

        load_u_P, load_u_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.u_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_u_bytes,
            barrier_storage=storage.load_u_mbar.data_ptr(),
        ).make_participants()

        vnew_store_P, vnew_store_C = pipeline.PipelineAsync.create(
            num_stages=self.vnew_store_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.vnew_store_mbar.data_ptr(),
        ).make_participants()

        load_gk_P, load_gk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_gk_bytes,
            barrier_storage=storage.load_gk_mbar.data_ptr(),
        ).make_participants()

        load_h0_P, load_h0_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_h0_bytes,
            barrier_storage=storage.h0_load_mbar.data_ptr(),
        ).make_participants()

        # ===================== Scheduling mbarrier init (persistent varlen) =====================
        if cutlass.const_expr(self.is_varlen and self.persistent):
            sched_mbar_base = storage.sched_mbar.data_ptr()
            sched_consumed_mbar_base = storage.sched_consumed_mbar.data_ptr()
            # Init 2 scheduling mbarriers with count=1: only Load warp elect_one arrives
            # Init 2 consumed mbarriers with count=7: one arrive per non-load warp
            # (MMA=1, CC=4, Store=1, Empty=1)
            if warp_idx == 0:
                cute.arch.mbarrier_init(sched_mbar_base, 1)
                cute.arch.mbarrier_init(sched_mbar_base + 1, 1)
                cute.arch.mbarrier_init(sched_consumed_mbar_base, 7)
                cute.arch.mbarrier_init(sched_consumed_mbar_base + 1, 7)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier(barrier_id=0, number_of_threads=self.threads_per_cta)

        # ===================== TMEM =====================
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ===================== SMEM views =====================
        sW = storage.sW.get_tensor(w_smem_staged.outer, swizzle=w_smem_staged.inner)
        sKt = storage.sKt.get_tensor(kt_smem_staged.outer, swizzle=kt_smem_staged.inner)
        sH_epi = storage.sH_epi.get_tensor(h_out_epi_staged.outer, swizzle=h_out_epi_staged.inner)
        sU_epi = storage.sU.get_tensor(u_epi_staged.outer, swizzle=u_epi_staged.inner)
        sVnew_store_epi = storage.sVnew_store.get_tensor(
            vnew_store_epi_staged.outer,
            swizzle=vnew_store_epi_staged.inner,
        )

        # ===================== MMA fragments =====================
        # h0 SMEM buffer: (BK, BV, 1) fp32, no swizzle — plain layout for scalar reads
        sH0 = storage.sH0.get_tensor(
            cute.make_layout(
                (self.BK, self.BV, 1),
                stride=(self.BV, 1, self.BK * self.BV),
            )
        )

        # GK SMEM 3D view
        sGK_smem = storage.sGK.get_tensor(
            cute.make_layout((self.BK, self.gk_stage)),
        )
        # WH MMA: A=state(TMEM), B=sW, acc=WH TMEM
        tCrState_fake = wh_tiled_mma.make_fragment_A(state_tmem_layout.outer.shape)
        tCrState = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_off, dtype=tCrState_fake.element_type),
            tCrState_fake.layout,
        )
        tCrW = wh_tiled_mma.make_fragment_B(sW)
        wh_shape = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(wh_shape, self.acc_stage))
        tCtAccWH = cute.make_tensor(tmem_ptr + self.tmem_wh_off, tCtAccWH_fake.layout)

        # KV MMA: A=v_new(TMEM), B=sKt, acc=KV TMEM
        # Create v_new TMEM A fragment (Mamba2-style: get layout from fake, bind TMEM ptr)
        tCrVnew_fake = kv_tiled_mma.make_fragment_A(vnew_tmem_layout.outer.shape)
        tCrVnew = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_vnew_off, dtype=tCrVnew_fake.element_type),
            tCrVnew_fake.layout,
        )
        tCrKt = kv_tiled_mma.make_fragment_B(sKt)
        kv_shape = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(kv_shape, 1))
        tCtAccKV = cute.make_tensor(tmem_ptr + self.tmem_kv_off, tCtAccKV_fake.layout)

        # ===================== Block indices =====================
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            # 1D grid work decode: persistent (grid=SM_count, multi-iter) or
            # non-persistent (grid=total_work_units, single iter per CTA)
            block_idx_x = cute.arch.block_idx()[0]
            grid_dim_x = cute.arch.grid_dim()[0]
            num_v_tiles = (V + self.BV - 1) // self.BV
            total_work_units = num_v_tiles * H * B
            if cutlass.const_expr(self.persistent):
                # Dynamic scheduling: while loop uses work_idx < total_work_units
                num_iters = Int32(0)  # not used, while loop controls iteration
            else:
                num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x
            # Pre-initialize variables reassigned inside persistent loop (CuTe DSL requirement)
            work_idx = Int32(0)
            v_tile_idx = Int32(0)
            hidx = Int32(0)
            bidx = Int32(0)
            tok_offset = Int32(0)
            seq_len = Int32(0)
            NT = Int32(0)
            data_bidx = Int32(0)
            chunk_off = Int32(0)
        else:
            (v_tile_idx, hidx, bidx) = cute.arch.block_idx()
            tok_offset = Int32(0)
            seq_len = T
            NT = (T + BT - 1) // BT
            data_bidx = bidx
            chunk_off = Int32(0)
            num_iters = Int32(1)

        # =========================================================================
        # DYNAMIC SCHEDULING: initial work_idx fetch (persistent varlen only)
        # Load warp elect_one does atomicAdd → sWorkIdx[buf] → fence → arrive mbar[buf].
        # Other warps wait on mbar[buf] at the correct phase, then read sWorkIdx[buf].
        # Double-buffered to avoid phase racing.
        # =========================================================================
        if cutlass.const_expr(self.is_varlen and self.persistent):
            sWorkIdx = storage.sWorkIdx.get_tensor(cute.make_layout((2,)))
            if warp_idx == self.load_warp_id:
                with cute.arch.elect_one():
                    first_work_idx = _atomic_add_global_i32(workspace_iter.toint().ir_value(), Int32(1).ir_value())
                    sWorkIdx[(0,)] = first_work_idx
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_arrive(sched_mbar_base)
            else:
                # All other warps: wait for Load warp's signal on mbar[0], phase=0
                cute.arch.mbarrier_wait(sched_mbar_base, Int32(0))
                # Signal consumed_mbar[0]: buf 0 has been consumed
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(sched_consumed_mbar_base)

        # =========================================================================
        # LOAD WARP
        # =========================================================================
        is_debug_cta = block_idx_x == 0 if cutlass.const_expr(self.is_varlen) else cute.arch.block_idx()[0] == 0

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h0)

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                # Load warp wrote sWorkIdx[0], read it back (sync_warp ensures visibility)
                cute.arch.sync_warp()
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)  # next buffer to write (0 was just written at init)
                should_continue = work_idx < total_work_units
                # Back-pressure: track whether consumers consumed the previous
                # signal on each scheduling buffer to prevent ABA phase issue
                first_wu = Int32(1)  # skip consumed wait on first iteration
                consumed_phase0 = Int32(0)
                consumed_phase1 = Int32(0)
            else:
                should_continue = wu_iter < num_iters

            while should_continue:
                # --- Work decode ---
                if cutlass.const_expr(self.is_varlen):
                    if cutlass.const_expr(not self.persistent):
                        work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT
                    data_bidx = Int32(0)
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[LD] wi=%d bidx=%d NT=%d h0=%d\n", work_idx, bidx, NT, use_initial_state)

                # Apply domain_offset for varlen TMA tensors (shift T dim by tok_offset)
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_w_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_w)
                    tma_tensor_kt_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_kt)
                    tma_tensor_u_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_u)
                    tma_tensor_gk_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_gk)
                else:
                    tma_tensor_w_v = tma_tensor_w
                    tma_tensor_kt_v = tma_tensor_kt
                    tma_tensor_u_v = tma_tensor_u
                    tma_tensor_gk_v = tma_tensor_gk

                tWsW, tWgW = self._tma_partition_B(
                    tma_atom_w,
                    tma_tensor_w_v,
                    sW,
                    self.wh_mma_tiler,
                    wh_tiled_mma,
                    data_bidx,
                    hidx,
                )
                tKsK, tKgK = self._tma_partition_B(
                    tma_atom_kt,
                    tma_tensor_kt_v,
                    sKt,
                    self.kv_mma_tiler,
                    kv_tiled_mma,
                    data_bidx,
                    hidx,
                )

                # U TMA load partition (non-MMA, epilog-style)
                gU_ld = tma_tensor_u_v[None, None, (hidx, data_bidx)]
                _, bSG_sU, bSG_gU = self._epilog_partition(
                    tma_atom_u,
                    gU_ld,
                    (self.BV, self.BT),
                    sU_epi,
                )

                # gk TMA load partition: gk_K shape (K, T, (H, data_B)), load (BK, 1) per timestep
                gGK_ld = tma_tensor_gk_v[None, None, (hidx, data_bidx)]  # (K, T)
                _, bSG_sGK, bSG_gGK = self._epilog_partition(
                    tma_atom_gk,
                    gGK_ld,
                    (self.BK, 1),
                    sGK_3d,
                )

                # h0 TMA load partition: h0 shape (K, V, (H, B)), load (BK, BV) fp32 tile
                # Note: h0 uses B (not data_B) and no domain_offset (not token-indexed)
                gH0_ld = tma_tensor_h0[None, None, (hidx, bidx)]  # (K, V)
                _, bSG_sH0_ld, bSG_gH0 = self._epilog_partition(
                    tma_atom_h0,
                    gH0_ld,
                    (self.BK, self.BV),
                    sH0,
                )

                # Issue TMA load for h0 before chunk loop (overlaps with chunk TMA loads)
                if use_initial_state:
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[LD] wi=%d h0_P.acq\n", work_idx)
                    h0_h = load_h0_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_h0,
                        src=bSG_gH0[(None, 0, v_tile_idx)],
                        dst=bSG_sH0_ld[None, h0_h.index],
                        tma_bar_ptr=h0_h.barrier,
                    )
                else:
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[LD] wi=%d skip_h0\n", work_idx)

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[LD] wi=%d c=%d w_P.acq\n", work_idx, chunk_idx)
                    w_h = load_w_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_w, src=tWgW[None, chunk_idx, 0], dst=tWsW[None, w_h.index], tma_bar_ptr=w_h.barrier
                    )

                    kt_h = load_kt_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_kt, src=tKgK[None, 0, chunk_idx], dst=tKsK[None, kt_h.index], tma_bar_ptr=kt_h.barrier
                    )

                    u_h = load_u_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_u,
                        src=bSG_gU[(None, v_tile_idx, chunk_idx)],
                        dst=bSG_sU[None, u_h.index],
                        tma_bar_ptr=u_h.barrier,
                    )

                    # TMA load gk for this chunk (BK Float32 values)
                    if use_gk:
                        # For tail chunk (T not divisible by BT), clamp to last valid position
                        gk_t_idx = chunk_idx * self.BT + self.BT - 1
                        remaining = seq_len - chunk_idx * self.BT
                        if remaining < self.BT:
                            gk_t_idx = seq_len - 1
                        gk_h = load_gk_P.acquire_and_advance()
                        cute.copy(
                            atom=tma_atom_gk,
                            src=bSG_gGK[(None, 0, gk_t_idx)],
                            dst=bSG_sGK[None, gk_h.index],
                            tma_bar_ptr=gk_h.barrier,
                        )

                # --- End-of-WU: Load warp fetches next work_idx and signals ---
                if cutlass.const_expr(PRINT_DEBUG):
                    if is_debug_cta:
                        with cute.arch.elect_one():
                            cute.printf("[LD] wi=%d end_wu\n", work_idx)
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # Back-pressure: wait for consumers to have consumed the
                    # previous signal on sched_mbar[sched_buf] before re-producing.
                    # Skip on first iteration (sched_buf=1 first use).
                    if first_wu == Int32(0):
                        if sched_buf == 0:
                            cute.arch.mbarrier_wait(sched_consumed_mbar_base, consumed_phase0)
                            consumed_phase0 = Int32(1) - consumed_phase0
                        else:
                            cute.arch.mbarrier_wait(sched_consumed_mbar_base + 1, consumed_phase1)
                            consumed_phase1 = Int32(1) - consumed_phase1
                    first_wu = Int32(0)
                    with cute.arch.elect_one():
                        next_idx = _atomic_add_global_i32(workspace_iter.toint().ir_value(), Int32(1).ir_value())
                        sWorkIdx[(sched_buf,)] = next_idx
                        cute.arch.fence_acq_rel_cta()
                        cute.arch.mbarrier_arrive(sched_mbar_base + sched_buf)
                    cute.arch.sync_warp()
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                else:
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters

        # =========================================================================
        # MMA WARP
        # =========================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)  # next buffer to wait on
                sched_phase0 = Int32(1)  # mbar[0]: init consumed phase=0, next=1
                sched_phase1 = Int32(0)  # mbar[1]: not yet used, next=0
                should_continue = work_idx < total_work_units
            else:
                should_continue = wu_iter < num_iters

            while should_continue:
                # --- Work decode (MMA only needs NT) ---
                if cutlass.const_expr(self.is_varlen):
                    if cutlass.const_expr(not self.persistent):
                        work_idx = block_idx_x + wu_iter * grid_dim_x
                    bidx_mma = (work_idx // num_v_tiles) // H
                    tok_off_mma = cu_seqlens[bidx_mma]
                    NT = (cu_seqlens[bidx_mma + 1] - tok_off_mma + BT - 1) // BT
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[MMA] wi=%d NT=%d\n", work_idx, NT)

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[MMA] wi=%d c=%d state_C.wait\n", work_idx, chunk_idx)
                    # --- WH MMA: state(SMEM) × W(SMEM) → acc_wh ---
                    state_h = state_smem_C.wait_and_advance()
                    w_h = load_w_C.wait_and_advance()

                    wh_h = wh_done_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(tCrW, mode=[2]), unroll_full=True):
                        wh_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            wh_tiled_mma,
                            tCtAccWH[None, None, None, wh_h.index],
                            tCrState[None, None, kp, state_h.index],
                            tCrW[None, None, kp, w_h.index],
                            tCtAccWH[None, None, None, wh_h.index],
                        )
                    wh_h.commit()
                    w_h.release()
                    state_h.release()

                    # --- KV MMA: v_new(TMEM) × K^T(SMEM) → update (ACCUMULATE=False always) ---
                    vnew_h = vnew_smem_C.wait_and_advance()
                    kt_h = load_kt_C.wait_and_advance()

                    kv_h = kv_done_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(tCrKt, mode=[2]), unroll_full=True):
                        # Always ACCUMULATE=False: we only compute the update term
                        kv_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            kv_tiled_mma,
                            tCtAccKV[None, None, None, 0],
                            tCrVnew[None, None, kp, vnew_h.index],
                            tCrKt[None, None, kp, kt_h.index],
                            tCtAccKV[None, None, None, 0],
                        )
                    kv_h.commit()
                    kt_h.release()
                    vnew_h.release()
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[MMA] wi=%d c=%d done\n", work_idx, chunk_idx)

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(PRINT_DEBUG):
                    if is_debug_cta:
                        with cute.arch.elect_one():
                            cute.printf("[MMA] wi=%d end_wu, wait_sched\n", work_idx)
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # Wait on mbar[sched_buf] at the right phase
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    # Signal consumed: Load warp can reuse this sched buffer
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(sched_consumed_mbar_base + sched_buf)
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                else:
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters

        # =========================================================================
        # CUDA CORE WARPS
        # =========================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            # ----- T2R setup for KV acc (BV, BK FP32) -----
            # Repetition determined by N(=BK) cols: BK=128 FP32 → Rep=16
            # Independent of M(=BV): same Rep for BV=32 and BV=64
            t2r_atom_kv = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccKV_flat = tCtAccKV[((None, None), 0, 0, None)]
            fake_sKV = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.kv_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_kv = tcgen05.make_tmem_copy(t2r_atom_kv, tCtAccKV_flat[(None, None, 0)])
            thr_t2r_kv = tiled_t2r_kv.get_slice(local_tidx)
            tTR_tKV = thr_t2r_kv.partition_S(tCtAccKV_flat)
            tTR_sKV = thr_t2r_kv.partition_D(fake_sKV)
            # h state in registers (persistent across chunks)
            tTR_rKV = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)

            # ----- T2R setup for WH acc (BV, BT FP32) -----
            # Repetition determined by N(=BT) cols: BT=64 FP32 → Rep=8
            t2r_atom_wh = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccWH_flat = tCtAccWH[((None, None), 0, 0, None)]
            fake_sWH = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.wh_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_wh = tcgen05.make_tmem_copy(t2r_atom_wh, tCtAccWH_flat[(None, None, 0)])
            thr_t2r_wh = tiled_t2r_wh.get_slice(local_tidx)
            tTR_tWH = thr_t2r_wh.partition_S(tCtAccWH_flat)
            tTR_sWH = thr_t2r_wh.partition_D(fake_sWH)

            # ----- R2T: h state regs → TMEM for WH MMA A operand -----
            # Repetition by N(=BK) col count: BK=128 BF16 → 64 cols → Rep=16
            copy_atom_r2t_state = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_state = tcgen05.make_tmem_copy(copy_atom_r2t_state, tCrState)
            thr_r2t_state = tiled_r2t_state.get_slice(local_tidx)
            r2t_state_shape = cute.slice_(thr_r2t_state.partition_S(tCrState).shape, (None, None, None, None, 0))
            tRT_tState = thr_r2t_state.partition_D(tCrState)

            # ----- R2S: KV T2R regs → sH_epi (COL_MAJOR, BV×BK) -----
            r2s_atom_h = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                tiled_t2r_kv,
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_atom_h, tiled_t2r_kv)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)

            # ----- R2S: WH T2R regs → sVnew_store_epi (COL_MAJOR, BV×BT) for TMA store -----
            r2s_atom_vnew = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                tiled_t2r_wh,
            )
            tiled_r2s_vnew = cute.make_tiled_copy_D(r2s_atom_vnew, tiled_t2r_wh)
            thr_r2s_vnew = tiled_r2s_vnew.get_slice(local_tidx)
            tRS_sVnew_store = thr_r2s_vnew.partition_D(sVnew_store_epi)

            # ----- R2T: v_new regs → TMEM for KV MMA A operand -----
            # Repetition by N(=BT) col count: BT=64 BF16 → 32 cols → Rep=8
            copy_atom_r2t_vnew = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_vnew = tcgen05.make_tmem_copy(copy_atom_r2t_vnew, tCrVnew)
            thr_r2t_vnew = tiled_r2t_vnew.get_slice(local_tidx)
            r2t_vnew_shape = cute.slice_(thr_r2t_vnew.partition_S(tCrVnew).shape, (None, None, None, None, 0))
            tRT_tVnew = thr_r2t_vnew.partition_D(tCrVnew)

            # ----- Identity tensor for WH tile (BV, BT) → v_new coords -----
            vnew_tile = cute.dice(self.wh_mma_tiler, (1, 1, None))  # (BV, BT)
            cM_vnew = cute.make_identity_tensor(vnew_tile)
            tTR_cM = thr_t2r_wh.partition_D(cM_vnew)

            # ----- Identity tensor for KV tile (BV, BK) → h coords -----
            h_tile = cute.dice(self.kv_mma_tiler, (1, 1, None))  # (BV, BK)
            cM_h = cute.make_identity_tensor(h_tile)
            tTR_cM_h = thr_t2r_kv.partition_D(cM_h)

            # ===== Persistent outer loop =====
            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue = work_idx < total_work_units
            else:
                should_continue = wu_iter < num_iters

            while should_continue:
                # --- Work decode ---
                if cutlass.const_expr(self.is_varlen):
                    if cutlass.const_expr(not self.persistent):
                        work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT

                # ===== Initialize h in registers =====
                if cutlass.const_expr(PRINT_DEBUG):
                    if is_debug_cta:
                        if local_tidx == 0:
                            cute.printf("[CC] wi=%d NT=%d h0=%d\n", work_idx, NT, use_initial_state)
                if use_initial_state:
                    # Wait for load warp's TMA h0 load into SMEM, then read from SMEM
                    h0_h = load_h0_C.wait_and_advance()
                    for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                        v_coord, k_coord = tTR_cM_h[ei]
                        tTR_rKV[ei] = sH0[(k_coord, v_coord, 0)]
                    h0_h.release()
                else:
                    for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                        tTR_rKV[ei] = Float32(0.0)

                # ===== Main loop (gk-only optimized pipeline) =====
                # Pipeline: Phase1(R2T+R2S+U_preload)→WH MMA→Phase2(v_new)→KV starts→gk_decay(overlaps KV)→KV wait→Phase4(h update)
                # gk decay moved AFTER KV MMA starts to overlap with KV MMA compute window
                # (R2T/R2S publish h BEFORE gk_decay, so gk can be deferred safely)

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            if local_tidx == 0:
                                cute.printf("[CC] wi=%d c=%d state_P.acq\n", work_idx, chunk_idx)
                    # ========================================
                    # Phase 1: Publish h for WH MMA + h_out store
                    # ========================================
                    # Declare per-phase register tensors at point of use
                    # to help compiler see non-overlapping lifetimes
                    tTR_rKV_bf16 = cute.make_rmem_tensor(tTR_rKV.shape, self.io_dtype)
                    tRT_rState = cute.make_rmem_tensor(r2t_state_shape, self.io_dtype)
                    h_vec = tTR_rKV.load()
                    h_vec_bf16 = h_vec.to(self.io_dtype)  # single FP32→BF16 conversion
                    tTR_rKV_bf16.store(h_vec_bf16)

                    # R2T h state → TMEM (triggers WH MMA — zero-copy A operand)
                    tRT_rState.store(h_vec_bf16)
                    state_h = state_smem_P.acquire_and_advance()
                    cute.copy(tiled_r2t_state, tRT_rState, tRT_tState[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    state_h.commit()  # WH MMA can start now!

                    # R2S to sH_epi (overlaps with WH MMA)
                    tRS_rH = tiled_r2s_h.retile(tTR_rKV_bf16)
                    h_handle = h_out_P.acquire_and_advance()
                    cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[(None, None, None, h_handle.index)])
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    h_handle.commit()

                    # Preload U from SMEM → registers (still overlapping WH MMA)
                    u_handle = load_u_C.wait_and_advance()
                    tTR_rU = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        v_coord, t_coord = tTR_cM[ei]
                        tTR_rU[ei] = sU_epi[(v_coord, t_coord, u_handle.index)].to(self.acc_dtype)
                    u_handle.release()

                    # ========================================
                    # Phase 2: v_new from WH result → triggers KV MMA
                    # ========================================
                    wh_h = wh_done_C.wait_and_advance()
                    tTR_rWH = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
                    cute.copy(tiled_t2r_wh, tTR_tWH[(None, None, None, wh_h.index)], tTR_rWH)
                    cute.arch.fence_view_async_tmem_load()
                    wh_h.release()

                    # v_new = u - WH (register-only, no SMEM reads on critical path)
                    for ei in cutlass.range_constexpr(cute.size(tTR_rWH)):
                        tTR_rWH[ei] = tTR_rU[ei] - tTR_rWH[ei]

                    # Zero v_new for positions beyond sequence boundary (varlen tail chunk)
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_idx * self.BT
                        if valid_len_chunk < self.BT:
                            for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                                v_coord, t_coord = tTR_cM[ei]
                                if t_coord >= valid_len_chunk:
                                    tTR_rWH[ei] = Float32(0.0)

                    # Prepare bf16 v_new for both R2T and R2S (single conversion)
                    vnew_vec_bf16 = tTR_rWH.load().to(self.io_dtype)
                    tTR_rVnew_bf16 = cute.make_rmem_tensor(tTR_rWH.shape, self.io_dtype)
                    tTR_rVnew_bf16.store(vnew_vec_bf16)

                    # R2T v_new → TMEM FIRST (triggers KV MMA — zero-copy A operand)
                    tRT_rVnew = cute.make_rmem_tensor(r2t_vnew_shape, self.io_dtype)
                    tRT_rVnew.store(vnew_vec_bf16)
                    vnew_h = vnew_smem_P.acquire_and_advance()
                    cute.copy(tiled_r2t_vnew, tRT_rVnew, tRT_tVnew[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    vnew_h.commit()  # KV MMA starts now!

                    # Save v_new to SMEM for TMA store (overlaps with KV MMA)
                    if save_v_new:
                        tRS_rVnew_st = tiled_r2s_vnew.retile(tTR_rVnew_bf16)
                        vnew_st_h = vnew_store_P.acquire_and_advance()
                        cute.copy(tiled_r2s_vnew, tRS_rVnew_st, tRS_sVnew_store[(None, None, None, vnew_st_h.index)])
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        vnew_st_h.commit()

                    # ========================================
                    # gk decay: overlapping with KV MMA
                    # ========================================
                    # h *= exp(gk) — cooperative precomputation
                    # R2T/R2S already published h BEFORE decay, so gk can be deferred here
                    # 128 CUDA threads cooperatively compute 128 gk_scale values (1 per K position)
                    # then each thread applies the precomputed scales (SMEM reads only)
                    if use_gk:
                        gk_h = load_gk_C.wait_and_advance()
                        # Step 1: Each CUDA thread (tidx 0-127) computes one exp2 and overwrites sGK in-place
                        # NOTE: gk values are already pre-scaled by RCP_LN2 (= 1/ln2) in
                        # chunk_local_cumsum preprocessing, so they are in base-2 form. We apply exp2()
                        # directly — do NOT multiply by INV_LN2 again (that would double-scale).
                        gk_raw = sGK_smem[(tidx, gk_h.index)]
                        sGK_smem[(tidx, gk_h.index)] = cute.exp2(gk_raw, fastmath=self.use_fast_math)
                        # Step 2: Sync all 4 CUDA warps so all 128 scales are visible
                        self.gk_precompute_bar.arrive_and_wait()
                        # Step 3: Apply precomputed scales (SMEM read only, no exp2)
                        for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                            v_coord, k_coord = tTR_cM_h[ei]
                            tTR_rKV[ei] = tTR_rKV[ei] * sGK_smem[(k_coord, gk_h.index)]
                        gk_h.release()

                    # ========================================
                    # Phase 4: KV update → h
                    # ========================================
                    kv_h = kv_done_C.wait_and_advance()
                    tTR_rUpdate = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)
                    cute.copy(tiled_t2r_kv, tTR_tKV[(None, None, None, 0)], tTR_rUpdate)
                    cute.arch.fence_view_async_tmem_load()
                    kv_h.release()

                    h_vec = tTR_rKV.load()
                    update_vec = tTR_rUpdate.load()
                    tTR_rKV.store(h_vec + update_vec)
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            if local_tidx == 0:
                                cute.printf("[CC] wi=%d c=%d done\n", work_idx, chunk_idx)

                # ===== After main loop: store final state ht (fp32 reg → fp32 GMEM) =====
                if store_final_state:
                    gHt = ht_tensor[None, None, (hidx, bidx)]  # (V, K)
                    for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                        v_coord, k_coord = tTR_cM_h[ei]
                        gHt[v_coord + v_tile_idx * self.BV, k_coord] = tTR_rKV[ei]

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(PRINT_DEBUG):
                    if is_debug_cta:
                        if local_tidx == 0:
                            cute.printf("[CC] wi=%d end_wu, wait_sched\n", work_idx)
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    # Signal consumed: CC has 4 warps, elect_one per warp → 4 arrives
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(sched_consumed_mbar_base + sched_buf)
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                else:
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters

        # =========================================================================
        # STORE WARP
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_out)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vnew_st)

            # For varlen: prepare direct GMEM write infrastructure for v_new
            # Store warp local thread index (0..31)
            store_local_tidx = tidx - self.store_warp_id * self.threads_per_warp

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue = work_idx < total_work_units
            else:
                should_continue = wu_iter < num_iters

            while should_continue:
                # --- Work decode ---
                if cutlass.const_expr(self.is_varlen):
                    if cutlass.const_expr(not self.persistent):
                        work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT
                    data_bidx = Int32(0)
                    chunk_off = chunk_offsets[bidx]
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[ST] wi=%d NT=%d\n", work_idx, NT)

                # Apply domain_offset for varlen store TMA tensors
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_h_out_v = cute.domain_offset((0, 0, (chunk_off, 0, 0)), tma_tensor_h_out)
                else:
                    tma_tensor_h_out_v = tma_tensor_h_out

                gH_st = tma_tensor_h_out_v[None, None, (None, hidx, data_bidx)]
                tma_h_st, bSG_sH, bSG_gH = self._epilog_partition(
                    tma_atom_h_out,
                    gH_st,
                    (self.BV, self.BK),
                    sH_epi,
                )

                # v_new store partition: TMA for both modes, CopyUniversal fallback for varlen tail
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_vnew_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_vnew_st)
                else:
                    tma_tensor_vnew_v = tma_tensor_vnew_st
                gVnew_st = tma_tensor_vnew_v[None, None, (hidx, data_bidx)]
                tma_vnew_st_local, bSG_sVnew_st, bSG_gVnew_st = self._epilog_partition(
                    tma_atom_vnew_st,
                    gVnew_st,
                    (self.BV, self.BT),
                    sVnew_store_epi,
                )

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    if cutlass.const_expr(PRINT_DEBUG):
                        if is_debug_cta:
                            with cute.arch.elect_one():
                                cute.printf("[ST] wi=%d c=%d hout_C.wait\n", work_idx, chunk_idx)
                    h_handle = h_out_C.wait_and_advance()

                    cute.copy(tma_h_st, bSG_sH[None, h_handle.index], bSG_gH[(None, v_tile_idx, 0, chunk_idx)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                    h_handle.release()

                    # v_new store
                    if save_v_new:
                        vnew_handle = vnew_store_C.wait_and_advance()
                        if cutlass.const_expr(self.is_varlen):
                            remaining = seq_len - chunk_idx * self.BT
                            if remaining >= self.BT:
                                # Non-tail chunk: full tile, safe to use TMA S2G
                                cute.copy(
                                    tma_vnew_st_local,
                                    bSG_sVnew_st[None, vnew_handle.index],
                                    bSG_gVnew_st[(None, v_tile_idx, chunk_idx)],
                                )
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                            else:
                                # Tail chunk: partial tile, use CopyUniversal with bounds check
                                # (TMA store would write past sequence boundary into next seq's data)

                                # Get SMEM stage view (BV, BT)
                                sVnew_stage = sVnew_store_epi[None, None, vnew_handle.index]

                                # Partition SMEM source with gmem_tiled_copy_vnew
                                gmem_thr_copy = gmem_tiled_copy_vnew.get_slice(store_local_tidx)
                                tOsVnew = gmem_thr_copy.partition_S(sVnew_stage)

                                # Identity tensor for coordinate tracking
                                cVnew = cute.make_identity_tensor((self.BV, self.BT))
                                tOcVnew = gmem_thr_copy.partition_S(cVnew)

                                # SMEM → REG (handles swizzle via autovec_copy)
                                tOrVnew = cute.make_fragment_like(tOsVnew, self.io_dtype)
                                cute.autovec_copy(tOsVnew, tOrVnew)

                                # Construct GMEM tile for this chunk
                                vnew_chunk_raw = (
                                    v_new_tensor.iterator
                                    + (tok_offset + chunk_idx * BT) * H * V
                                    + hidx * V
                                    + v_tile_idx * self.BV
                                )
                                vnew_chunk_ptr = cute.make_ptr(
                                    self.io_dtype,
                                    vnew_chunk_raw.toint(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                vnew_stride_t = cute.assume(
                                    H * V,
                                    divby=128 // self.io_dtype.width,
                                )
                                gVnew_chunk = cute.make_tensor(
                                    vnew_chunk_ptr,
                                    cute.make_layout(
                                        (self.BV, self.BT),
                                        stride=(1, vnew_stride_t),
                                    ),
                                )

                                # Partition GMEM destination
                                tOgVnew = gmem_thr_copy.partition_D(gVnew_chunk)

                                # REG → GMEM with per-BT-row bounds check
                                for rest_bt in cutlass.range_constexpr(cute.size(tOrVnew.shape[2])):
                                    bt_coord = tOcVnew[0, 0, rest_bt][1]
                                    if bt_coord < remaining:
                                        cute.copy(
                                            gmem_tiled_copy_vnew,
                                            tOrVnew[None, None, rest_bt],
                                            tOgVnew[None, None, rest_bt],
                                        )
                        else:
                            cute.copy(
                                tma_vnew_st_local,
                                bSG_sVnew_st[None, vnew_handle.index],
                                bSG_gVnew_st[(None, v_tile_idx, chunk_idx)],
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        vnew_handle.release()

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(PRINT_DEBUG):
                    if is_debug_cta:
                        with cute.arch.elect_one():
                            cute.printf("[ST] wi=%d end_wu, wait_sched\n", work_idx)
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    # Signal consumed: Store warp → 1 arrive
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(sched_consumed_mbar_base + sched_buf)
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                else:
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters

        # =========================================================================
        # EMPTY WARP
        # =========================================================================
        elif warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)
            # Dynamic scheduling: wait on double-buffered mbarriers for each WU
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                while work_idx < total_work_units:
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    # Signal consumed: Empty warp → 1 arrive
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(sched_consumed_mbar_base + sched_buf)
                    sched_buf = Int32(1) - sched_buf

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition B operand tensors for TMA copy."""
        coord = (0, None, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
        """Partition for epilogue TMA store."""
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
        )
        return atom, bSG_sC, bSG_gC


# ===================== Reference implementations =====================


def reference_chunk_delta_rule_fwd_h(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    """Reference implementation.  Gate values (g, gk) are assumed to be
    pre-scaled by RCP_LN2 (= 1/ln2), matching the convention used by
    chunk_local_cumsum in KDA.  The kernel applies exp2() directly."""
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    h_out = torch.zeros(B, NT, H, K, V, device=k.device, dtype=torch.bfloat16)
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    h_after = []
    for t in range(NT):
        s, e = t * BT, min((t + 1) * BT, T)
        h_out[:, t] = h.to(torch.bfloat16)
        wc = w[:, s:e].permute(0, 2, 1, 3).float()
        kc = k[:, s:e].permute(0, 2, 1, 3).float()
        uc = u[:, s:e].permute(0, 2, 1, 3).float()
        wh = torch.matmul(wc, h)
        vnc = uc - wh
        if g is not None:
            gc = g[:, s:e].permute(0, 2, 1).float()
            gl = gc[:, :, -1:].float()
            vnc = vnc * torch.exp2(gl - gc).unsqueeze(-1)
        v_new_out[:, s:e] = vnc.permute(0, 2, 1, 3).to(torch.bfloat16)
        if g is not None:
            gls = gc[:, :, -1].float()
            h = h * torch.exp2(gls).unsqueeze(-1).unsqueeze(-1)
        if gk is not None:
            gkc = gk[:, s:e].permute(0, 2, 1, 3).float()
            gkl = gkc[:, :, -1, :].float()
            h = h * torch.exp2(gkl).unsqueeze(-1)
        h = h + torch.matmul(kc.transpose(-2, -1), vnc)
        h_after.append(h[0, 0].to(torch.bfloat16).clone())
    return h_out, v_new_out, h_after


def reference_bf16_roundtrip(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    """Reference with bf16 roundtrip.  Gate values (g, gk) are assumed to be
    pre-scaled by RCP_LN2 (= 1/ln2).  Uses exp2() to match kernel."""
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    h_after = []
    for t in range(NT):
        s, e = t * BT, min((t + 1) * BT, T)
        wc = w[:, s:e].permute(0, 2, 1, 3).float()
        kc = k[:, s:e].permute(0, 2, 1, 3).float()
        uc = u[:, s:e].permute(0, 2, 1, 3).float()
        h_bf16 = h.to(torch.bfloat16).float()
        wh = torch.matmul(wc, h_bf16)
        vnc = uc - wh
        if g is not None:
            gc = g[:, s:e].permute(0, 2, 1).float()
            gl = gc[:, :, -1:].float()
            vnc = vnc * torch.exp2(gl - gc).unsqueeze(-1)
        v_new_out[:, s:e] = vnc.permute(0, 2, 1, 3).to(torch.bfloat16)
        if g is not None:
            gls = gc[:, :, -1].float()
            h = h * torch.exp2(gls).unsqueeze(-1).unsqueeze(-1)
        if gk is not None:
            gkc = gk[:, s:e].permute(0, 2, 1, 3).float()
            gkl = gkc[:, :, -1, :].float()
            h = h * torch.exp2(gkl).unsqueeze(-1)
        vn_bf16 = vnc.to(torch.bfloat16).float()
        h = h + torch.matmul(kc.transpose(-2, -1), vn_bf16)
        h_after.append(h[0, 0].to(torch.bfloat16).clone())
    return v_new_out, h_after


# ---------------------------------------------------------------------------
# Compile cache + TVM-FFI API
# ---------------------------------------------------------------------------

# Internal cache: maps (is_varlen, persistent, H, K, V, chunk_size) → compiled_fn
_delta_h_kernel_cache: dict = {}


def _compile_delta_h_variant(is_varlen, persistent, H, K, V, chunk_size, use_fast_math):
    """Compile one ChunkDeltaRuleFwdH kernel variant. Returns the compiled TVM-FFI callable.

    Uses make_fake_compact_tensor and make_fake_stream for compilation with
    TVM-FFI.  At runtime, torch tensors are passed directly (zero-copy).
    Uses sym_int() for dynamic B, T, NT dimensions so one compiled kernel
    handles all batch-size / sequence-length combinations.

    Note: use_g, use_gk, use_initial_state, store_final_state, save_v_new
    are runtime Int32 flags (NOT const_expr), so they don't need separate
    compilations — a single kernel handles all flag combinations.
    """
    kernel_obj = ChunkDeltaRuleFwdH(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        is_varlen=is_varlen,
        persistent=persistent,
        use_fast_math=use_fast_math,
    )

    sym_a = cute.sym_int()  # B (non-varlen) or T_total (varlen)
    sym_b = cute.sym_int()  # T (non-varlen) or unused
    sym_nt = cute.sym_int()  # NT or NT_total
    sym_n = cute.sym_int()  # metadata size (cu_seqlens, chunk_offsets)
    sym_ws = cute.sym_int()  # workspace size (separate from metadata)
    sym_ns = cute.sym_int()  # num_seqs (varlen h0/ht) or B (non-varlen, == sym_a)

    if is_varlen:
        # varlen: data tensors are [T_total, H, ...] (3D)
        k_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, H, K),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        w_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, H, K),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        u_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, H, V),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        g_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_a, H),
            stride_order=(1, 0),
            assumed_align=128,
        )
        gk_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_a, H, K),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        v_new_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, H, V),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        h_out_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_nt, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
    else:
        # non-varlen: data tensors are [B, T, H, ...] (4D)
        k_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        w_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        u_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        g_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_a, sym_b, H),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        gk_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        v_new_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        h_out_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_nt, H, K, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )

    # h0/ht use [B, H, K, V] (non-varlen) or [num_seqs, H, K, V] (varlen)
    # In varlen mode, num_seqs != T_total, so use a separate sym_ns
    h0_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_ns, H, K, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    ht_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_ns, H, K, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    cu_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n,),
        assumed_align=128,
    )
    co_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_n,),
        assumed_align=128,
    )
    ws_fake = make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_ws,),
        assumed_align=128,
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake,
        w_fake,
        u_fake,
        g_fake,
        gk_fake,
        h_out_fake,
        v_new_fake,
        h0_fake,
        ht_fake,
        cu_fake,
        co_fake,
        ws_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Int32(1),  # total_nt dummy
        Int32(0),  # use_g
        Int32(0),  # use_gk
        Int32(0),  # use_initial_state
        Int32(0),  # store_final_state
        Int32(0),  # save_v_new
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def _get_compiled_delta_h(is_varlen, persistent, H, K, V, chunk_size):
    """Get a compiled ChunkDeltaRuleFwdH kernel with on-demand (lazy) compilation.

    Each variant is compiled exactly once and cached.  Compilation is deferred
    until the variant is actually needed so that cute.compile is always
    immediately followed by execution — this avoids a CuTe DSL runtime issue
    where a subsequent cute.compile can invalidate previously compiled but
    not-yet-executed functions.

    Cache key: (is_varlen, persistent, H, K, V, chunk_size, USE_FAST_MATH)
    """
    key = (is_varlen, persistent, H, K, V, chunk_size, USE_FAST_MATH)
    if key not in _delta_h_kernel_cache:
        _delta_h_kernel_cache[key] = _compile_delta_h_variant(
            is_varlen,
            persistent,
            H,
            K,
            V,
            chunk_size,
            USE_FAST_MATH,
        )
    return _delta_h_kernel_cache[key]


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    persistent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    ChunkDeltaRuleFwdH forward pass — FLA-compatible API.

    Interface aligned with FLA's chunk_gated_delta_rule_fwd_h for fair benchmarking.
    Allocates output tensors internally and returns (h, v_new, final_state).

    Args:
        k:  key tensor           [B, T, H, K]  bf16
        w:  decay weight tensor  [B, T, H, K]  bf16
        u:  value tensor         [B, T, H, V]  bf16
        g:  scalar gate          [B, T, H]     fp32, or None
        gk: key gate             [B, T, H, K]  fp32, or None
        initial_state: h0        [N, H, K, V]  fp32, or None
        output_final_state: whether to return final_state
        chunk_size: chunk size (default 64)
        save_new_value: whether to return v_new
        cu_seqlens: cumulative sequence lengths [N+1] int64/int32 for varlen, or None
        chunk_indices: chunk index pairs [NT, 2] int32 (varlen only)
        persistent: whether to use persistent kernel (default True)

    Returns:
        (h, v_new, final_state) — same as FLA
        h:           [B, NT, H, K, V] bf16  (or [1, NT_total, H, K, V] for varlen)
        v_new:       [B, T, H, V] bf16      (or None if save_new_value=False)
        final_state: [N, H, K, V] fp32      (or None if output_final_state=False)
    """
    B, T, H, K_dim = k.shape
    V_dim = u.shape[3]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets_i32(cu_seqlens, BT)
    assert K_dim == 128, "current kernel only supports 128."
    total_nt = B * NT

    use_g_flag = 1 if g is not None else 0
    use_gk_flag = 1 if gk is not None else 0
    use_h0_flag = 1 if initial_state is not None else 0
    store_ht_flag = 1 if output_final_state else 0
    save_vnew_flag = 1 if save_new_value else 0

    if is_varlen:
        # Varlen mode: B must be 1 (FLA convention), cu_seqlens is [N+1]
        assert B == 1, "varlen mode requires B=1 (data packed in T dimension)"

        # Ensure cu_seqlens is int32 for the kernel
        cu_seqlens_i32 = cu_seqlens.int() if cu_seqlens.dtype != torch.int32 else cu_seqlens

        # View as 3D for kernel: [1, T, H, K] -> [T, H, K] (zero-copy)
        k_kern = k[0]
        w_kern = w[0]
        u_kern = u[0]
        # Use torch.empty for dummies the kernel won't read (flag-gated)
        g_kern = g[0] if g is not None else torch.empty(T, H, device=k.device, dtype=torch.float32)
        gk_kern = gk[0] if gk is not None else torch.empty(T, H, K_dim, device=k.device, dtype=torch.float32)

        # Allocate outputs (3D for kernel)
        h_out_kern = k.new_empty(total_nt, H, K_dim, V_dim)  # bf16
        v_new_kern = torch.empty_like(u_kern)  # always allocate; kernel checks save_v_new flag
        h0_kern = (
            initial_state
            if initial_state is not None
            else torch.empty(N, H, K_dim, V_dim, device=k.device, dtype=torch.float32)
        )
        # ht is purely an output (kernel writes all elements when store_final_state=1);
        # use empty instead of zeros to skip the zero-fill kernel launch.
        # NOTE: Ensure final output is zeros
        # vLLM will use padding for CUDA Graph
        ht_kern = torch.zeros(N, H, K_dim, V_dim, device=k.device, dtype=torch.float32)

        # Workspace: first 4 bytes used as atomic counter for dynamic scheduling
        workspace = torch.zeros(max(N * 128, 4), dtype=torch.uint8, device=k.device)

        ps = (Int32(N), Int32(T), Int32(H), Int32(K_dim), Int32(V_dim))

        compiled_fn = _get_compiled_delta_h(True, persistent, H, K_dim, V_dim, chunk_size)
        compiled_fn(
            k_kern,
            w_kern,
            u_kern,
            g_kern,
            gk_kern,
            h_out_kern,
            v_new_kern,
            h0_kern,
            ht_kern,
            cu_seqlens_i32,
            chunk_offsets,
            workspace,
            ps,
            Int32(total_nt),
            Int32(use_g_flag),
            Int32(use_gk_flag),
            Int32(use_h0_flag),
            Int32(store_ht_flag),
            Int32(save_vnew_flag),
        )

        # Wrap outputs to 4D to match FLA's return shapes:
        # FLA returns h as [1, NT_total, H, K, V]
        h = h_out_kern.unsqueeze(0)
        v_new = v_new_kern.unsqueeze(0) if save_new_value else None
        final_state = ht_kern if output_final_state else None
    else:
        # Non-varlen mode
        NT = (T + BT - 1) // BT
        N = B

        # Allocate outputs
        h = k.new_empty(B, NT, H, K_dim, V_dim)  # bf16
        v_new_out = torch.empty_like(u)  # always allocate; kernel checks save_v_new flag
        # Use torch.empty for dummies the kernel won't read (flag-gated)
        h0 = (
            initial_state
            if initial_state is not None
            else torch.empty(B, H, K_dim, V_dim, device=k.device, dtype=torch.float32)
        )
        # ht must share sym_ns (first dim) with h0, so always use B
        ht = k.new_zeros(B, H, K_dim, V_dim, dtype=torch.float32)

        # Dummy tensors for unused optional gate inputs (kernel checks flags)
        g_kern = g if g is not None else torch.empty(B, T, H, device=k.device, dtype=torch.float32)
        gk_kern = gk if gk is not None else torch.empty(B, T, H, K_dim, device=k.device, dtype=torch.float32)

        # Dummy cu_seqlens / chunk_offsets / workspace (kernel requires them)
        cu_dummy = torch.empty(2, dtype=torch.int32, device=k.device)
        co_dummy = torch.empty(2, dtype=torch.int32, device=k.device)
        ws_dummy = torch.empty(128, dtype=torch.uint8, device=k.device)

        ps = (Int32(B), Int32(T), Int32(H), Int32(K_dim), Int32(V_dim))

        compiled_fn = _get_compiled_delta_h(False, persistent, H, K_dim, V_dim, chunk_size)
        compiled_fn(
            k,
            w,
            u,
            g_kern,
            gk_kern,
            h,
            v_new_out,
            h0,
            ht,
            cu_dummy,
            co_dummy,
            ws_dummy,
            ps,
            Int32(NT),
            Int32(use_g_flag),
            Int32(use_gk_flag),
            Int32(use_h0_flag),
            Int32(store_ht_flag),
            Int32(save_vnew_flag),
        )

        v_new = v_new_out if save_new_value else None
        final_state = ht if output_final_state else None

    return h, v_new, final_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    B, T, H, K, V = args.batch_size, args.seq_len, args.num_heads, args.head_dim_k, args.head_dim_v
    BT = args.chunk_size
    NT = (T + BT - 1) // BT

    print(f"V2 Test: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}, NT={NT}")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    def run_kernel(k_t, w_t, u_t, g_t, gk_t, h0_t, use_g_val, use_gk_val, use_h0, store_ht, do_save_vnew=0):
        h_out, v_new, ht = chunk_gated_delta_rule_fwd_h(
            k=k_t,
            w=w_t,
            u=u_t,
            g=g_t if use_g_val else None,
            gk=gk_t if use_gk_val else None,
            initial_state=h0_t if use_h0 else None,
            output_final_state=bool(store_ht),
            chunk_size=BT,
            save_new_value=bool(do_save_vnew),
        )
        torch.cuda.synchronize()
        # Ensure consistent return shapes for backward compat with manual tests
        if h_out is None:
            h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
        if v_new is None:
            v_new = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        if ht is None:
            ht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
        return h_out, v_new, ht

    all_pass = True

    # ===== Test 1: No gating, no h0 =====
    print("\n" + "=" * 60)
    print("Test 1: No gating, no h0")
    g_z = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk_z = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0_z = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t1_pass = max_diff < 0.5
    print(f"  {'PASS' if t1_pass else 'FAIL'}")
    all_pass = all_pass and t1_pass

    # ===== Test 2: With gk + h0 =====
    print("\n" + "=" * 60)
    print("Test 2: With gk + h0")
    gk_val = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    gk_val = -torch.abs(gk_val)
    gk_val = gk_val.cumsum(dim=1)
    # Pre-scale by RCP_LN2 to match KDA convention (kernel does exp2 directly)
    gk_val = gk_val * INV_LN2
    h0_val = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_val, 0, 1, 1, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=h0_val, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t2_pass = max_diff < 0.5
    print(f"  {'PASS' if t2_pass else 'FAIL'}")
    all_pass = all_pass and t2_pass

    # ===== Test 3: With gk gating =====
    print("\n" + "=" * 60)
    print("Test 3: With gk gating")
    gk_val = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    gk_val = -torch.abs(gk_val)
    gk_val = gk_val.cumsum(dim=1)
    # Pre-scale by RCP_LN2 to match KDA convention (kernel does exp2 directly)
    gk_val = gk_val * INV_LN2

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_z, 0, 1, 0, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t3_pass = max_diff < 0.5
    print(f"  {'PASS' if t3_pass else 'FAIL'}")
    all_pass = all_pass and t3_pass

    # ===== Test 4: With h0 initial state =====
    print("\n" + "=" * 60)
    print("Test 4: With h0 initial state")
    h0_val = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_val, 0, 0, 1, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=h0_val, chunk_size=BT)

    # h_out[0] should be h0 (bf16 rounded)
    h0_bf16 = h0_val.to(torch.bfloat16)
    d0 = (h_out[0, 0, 0].float() - h0_bf16[0, 0].float()).abs().max().item()
    print(f"  h_out[0] vs h0 bf16: {d0:.6f}")

    max_diff = d0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t4_pass = max_diff < 0.5
    print(f"  {'PASS' if t4_pass else 'FAIL'}")
    all_pass = all_pass and t4_pass

    # ===== Test 5: With store_final_state (ht) =====
    print("\n" + "=" * 60)
    print("Test 5: store_final_state")

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 1)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    # ht should match the last h_ref (after all chunks)
    ht_ref = h_ref_bf16[-1]  # last chunk's state
    # ht layout: (B, H, K, V) but kernel writes in transposed (V, K) format
    # Compare ht[0, 0] with ht_ref
    d_ht = (ht[0, 0].float() - ht_ref.float()).abs().max().item()
    print(f"  ht vs ref: {d_ht:.6f}")
    t5_pass = d_ht < 0.5
    print(f"  {'PASS' if t5_pass else 'FAIL'}")
    all_pass = all_pass and t5_pass

    # ===== Test 6: gk + h0 + ht (all features) =====
    print("\n" + "=" * 60)
    print("Test 6: gk + h0 + ht (all features)")

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_val, 0, 1, 1, 1)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=h0_val, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    d_ht = (ht[0, 0].float() - h_ref_bf16[-1].float()).abs().max().item()
    max_diff = max(max_diff, d_ht)
    print(f"  max diff (h_out + ht): {max_diff:.6f}")
    t6_pass = max_diff < 0.5
    print(f"  {'PASS' if t6_pass else 'FAIL'}")
    all_pass = all_pass and t6_pass

    # ===== Test 7: Larger config =====
    print("\n" + "=" * 60)
    print("Test 7: B=2, T=512, H=4 (no gating)")
    B2, T2, H2 = 2, 512, 4
    NT2 = (T2 + BT - 1) // BT
    torch.manual_seed(123)
    k2 = torch.randn(B2, T2, H2, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(B2, T2, H2, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u2 = torch.randn(B2, T2, H2, V, device="cuda", dtype=torch.bfloat16) * 0.1

    h_out2, v_new2, ht2 = chunk_gated_delta_rule_fwd_h(
        k=k2,
        w=w2,
        u=u2,
        chunk_size=BT,
        save_new_value=False,
    )
    torch.cuda.synchronize()

    _, h_ref2 = reference_bf16_roundtrip(k2, w2, u2, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT2 - 1, len(h_ref2))):
        d = (h_out2[0, t + 1, 0].float() - h_ref2[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t7_pass = max_diff < 0.5
    print(f"  {'PASS' if t7_pass else 'FAIL'}")
    all_pass = all_pass and t7_pass

    # ===== Test 8: v_new output (no gating) =====
    print("\n" + "=" * 60)
    print("Test 8: v_new output (no gating)")

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 0, do_save_vnew=1)
    vnew_ref, _ = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    d_vnew = (v_new.float() - vnew_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vnew:.6f}")
    t8_pass = d_vnew < 0.5
    print(f"  {'PASS' if t8_pass else 'FAIL'}")
    all_pass = all_pass and t8_pass

    # ===== Test 9: v_new output (with gk gating) =====
    print("\n" + "=" * 60)
    print("Test 9: v_new output (with gk gating)")

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_z, 0, 1, 0, 0, do_save_vnew=1)
    vnew_ref, _ = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=None, chunk_size=BT)

    d_vnew = (v_new.float() - vnew_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vnew:.6f}")
    t9_pass = d_vnew < 0.5
    print(f"  {'PASS' if t9_pass else 'FAIL'}")
    all_pass = all_pass and t9_pass

    # ===== Summary =====
    print("\n" + "=" * 60)
    results = [t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass, t7_pass, t8_pass, t9_pass]
    names = [
        "No gate",
        "gk + h0",
        "gk gate",
        "h0 init",
        "ht store",
        "All features",
        "Larger config",
        "v_new (no gk)",
        "v_new (gk)",
    ]
    for i, (name, r) in enumerate(zip(names, results)):
        print(f"  Test {i + 1} ({name}): {'PASS' if r else 'FAIL'}")
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} tests passed")
    print("ALL PASS" if all_pass else "SOME FAILED")

    # ===== Benchmark =====
    print("\n" + "=" * 60)
    print("Benchmark: B=4, T=4096, H=64, K=128, V=128")
    Bb, Tb, Hb = 4, 4096, 64
    torch.manual_seed(999)
    kb = torch.randn(Bb, Tb, Hb, K, device="cuda", dtype=torch.bfloat16) * 0.1
    wb = torch.randn(Bb, Tb, Hb, K, device="cuda", dtype=torch.bfloat16) * 0.1
    ub = torch.randn(Bb, Tb, Hb, V, device="cuda", dtype=torch.bfloat16) * 0.1

    def run_bench():
        chunk_gated_delta_rule_fwd_h(
            k=kb,
            w=wb,
            u=ub,
            chunk_size=BT,
            save_new_value=False,
        )

    # Warmup (also triggers lazy compilation)
    for _ in range(3):
        run_bench()
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        run_bench()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / n_iter
    print(f"  V2 kernel: {elapsed_ms:.3f} ms")

    # FLA h-kernel reference (apples-to-apples: h-state recurrence only)
    try:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

        # Warmup
        for _ in range(3):
            fla_fwd_h(
                k=kb,
                w=wb,
                u=ub,
                g=None,
                gk=None,
                initial_state=None,
                output_final_state=False,
                chunk_size=BT,
                save_new_value=True,
            )
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_iter):
            fla_fwd_h(
                k=kb,
                w=wb,
                u=ub,
                g=None,
                gk=None,
                initial_state=None,
                output_final_state=False,
                chunk_size=BT,
                save_new_value=True,
            )
        end_event.record()
        torch.cuda.synchronize()
        fla_ms = start_event.elapsed_time(end_event) / n_iter
        print(f"  FLA h-kernel: {fla_ms:.3f} ms")
        print(f"  Speedup vs FLA h-kernel: {fla_ms / elapsed_ms:.2f}x")
    except Exception as e:
        print(f"  FLA not available: {e}")


if __name__ == "__main__":
    main()
