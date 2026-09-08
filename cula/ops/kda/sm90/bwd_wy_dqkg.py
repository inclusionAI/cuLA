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

"""
bwd_wy_dqkg — Hopper SM90 WGMMA implementation of KDA chunkwise
backward fused WY DqKG path.

Architecture: Hopper warp specialization (1 DMA WG + 1 MMA WG).
  DMA WG (warps 0-3): warp 0 = TMA G2S load, warp 1 = TMA S2G store
  MMA WG (warps 4-7): WGMMA + r2s epilogue (writes SMEM only)

Computes:
    dq, dk, dv, db, dg, dA in FLA-compatible output order.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32, Int64


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


@cute.jit
def smem_load_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive float32 from SMEM with 128B swizzle layout.
    Layout: tile_to_shape(epi_smem_atom, (BT, BK)) where BK=64.
    Atom row width = 32 elements. Two outer blocks for BK=64.
    Swizzle: 128B → elem_xor = ((row & 7) << 2).
    col_base must be 4-aligned.
    """
    c_inner = col_base & Int32(31)
    c_outer = col_base >> Int32(5)
    swizzled_inner = c_inner ^ ((row & Int32(7)) << Int32(2))
    elem_offset = row * Int32(32) + swizzled_inner + c_outer * Int32(2048)
    aligned_ptr = cute.make_ptr(
        cutlass.Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    return t.load()


@cute.jit
def gmem_load_f32x4(gmem_addr: Int64):
    """Load 4 contiguous fp32 values from 16-byte-aligned GMEM."""
    ptr = cute.make_ptr(
        cutlass.Float32,
        gmem_addr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    t = cute.make_tensor(ptr, cute.make_layout((4,), stride=(1,)))
    return t.load()


@cute.jit
def gmem_store_f32x4(gmem_addr: Int64, val):
    """Store 4 contiguous fp32 values to 16-byte-aligned GMEM."""
    ptr = cute.make_ptr(
        cutlass.Float32,
        gmem_addr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    t = cute.make_tensor(ptr, cute.make_layout((4,), stride=(1,)))
    t.store(val)


@cute.jit
def copy_partial_epi_tile(tiled_copy, thr_copy, tOs, tOc, tOr, gmem_tile, rows: Int32):
    """SMEM -> REG -> GMEM for a 64x32 epilogue tile with per-row mask."""
    tOg = thr_copy.partition_D(gmem_tile)
    for m1 in cutlass.range_constexpr(cute.size(tOs.shape[1])):
        row = tOc[(0, 0), m1, 0][0]
        if row < rows:
            cute.autovec_copy(tOs[(None, m1, None)], tOr[(None, m1, None)])
            cute.copy(tiled_copy, tOr[(None, m1, None)], tOg[(None, m1, None)])


@cute.jit
def copy_partial_epi_tile_gmem_f32(
    tiled_copy,
    thr_copy,
    tOs,
    tOc,
    tOr,
    gmem_iter: cute.Pointer,
    chunk_row_base: Int32,
    row_stride: Int32,
    head_idx: Int32,
    head_stride: Int32,
    col_base: Int32,
    rows: Int32,
):
    """Build a 64x32 fp32 GMEM tile view and copy only valid rows."""
    gmem_ptr = cute.make_ptr(
        cutlass.Float32,
        (gmem_iter + chunk_row_base * row_stride + head_idx * head_stride + col_base).toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    stride_t = cute.assume(row_stride, divby=4)
    gmem_tile = cute.make_tensor(
        gmem_ptr,
        cute.make_layout((64, 32), stride=(stride_t, 1)),
    )
    copy_partial_epi_tile(tiled_copy, thr_copy, tOs, tOc, tOr, gmem_tile, rows)


@cute.jit
def copy_partial_epi_tile_gmem_bf16(
    tiled_copy,
    thr_copy,
    tOs,
    tOc,
    tOr,
    gmem_iter: cute.Pointer,
    chunk_row_base: Int32,
    row_stride: Int32,
    head_idx: Int32,
    head_stride: Int32,
    col_base: Int32,
    rows: Int32,
):
    """Build a 64x32 bf16 GMEM tile view and copy only valid rows."""
    gmem_ptr = cute.make_ptr(
        cutlass.BFloat16,
        (gmem_iter + chunk_row_base * row_stride + head_idx * head_stride + col_base).toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    stride_t = cute.assume(row_stride, divby=8)
    gmem_tile = cute.make_tensor(
        gmem_ptr,
        cute.make_layout((64, 32), stride=(stride_t, 1)),
    )
    copy_partial_epi_tile(tiled_copy, thr_copy, tOs, tOc, tOr, gmem_tile, rows)


USE_FAST_MATH = True
DEBUG_PRINT = False

COMPILE_OPTIONS = "--enable-tvm-ffi --generate-line-info --ptxas-options '--verbose'"

BFloat16 = cutlass.BFloat16
Float32 = cutlass.Float32

_torch_to_cutlass_dtype = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@cute.jit
def smem_load_bf16x8_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """Load 8 consecutive bf16 from SMEM with K_SW128 (Swizzle<3,4,3>) layout.
    raw_ptr: bf16 SMEM base pointer for one stage
    row: row index in [0, BK=64)
    col_base: 8-aligned column index in [0, BV=64)
    """
    swizzled = col_base ^ ((row & Int32(7)) << Int32(3))
    elem_off = row * Int32(64) + swizzled
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


# ── Named barrier IDs ──
# barrier 0 is reserved by CUDA runtime (sync_threads), do not use.
BARRIER_DW_READY = 1  # dw stmatrix visible to all MMA warps (128 thr)
BARRIER_DG_COMPUTE = 2  # intra-MMA-WG sync for dgk/dg epilogue (128 thr)
BARRIER_DB_SYNC = 3  # sDb read/write synchronization (128 thr)


class ChunkKdaBwdWyDqkgFusedSM90:
    """Hopper SM90 WGMMA kernel for KDA chunkwise WY DqKG backward."""

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        scale: float = 1.0,
        min_occupancy: int = 2,
        use_fast_math: bool = True,
        bk: int = 32,
        bv: int = 64,
    ):
        assert chunk_size == 64, "chunk_size must be 64"
        assert head_dim_k == 128 and head_dim_v == 128
        assert bk in (32, 64), "bk must be 32 or 64"
        assert bv in (32, 64), "bv must be 32 or 64"
        assert head_dim_k % bk == 0
        assert head_dim_v % bv == 0

        self.use_fast_math = use_fast_math
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.scale = scale
        self.min_occupancy = min_occupancy

        self.BT = chunk_size  # 64
        # head_dim_k is always 128, accessed via self.head_dim_k
        self.BK = bk
        self.BV = bv
        self.num_v_tiles = (head_dim_v + self.BV - 1) // self.BV  # 2
        self.num_k_iters = self.head_dim_k // self.BK  # 2
        self.vloop_gemm_tiler = (self.BT, self.BK, self.BV)  # M=BT, N=BK, K=BV
        self.dv2_gemm_tiler = (self.BT, self.BV, self.BT)  # M=BT, N=BV, K=BT
        self.vloop_stage = max(2, self.num_v_tiles)

        self.threads_per_warp = 32
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = 1
        self.threads_per_cta = self.num_threads_per_warp_group * (self.num_dma_warp_groups + self.num_mma_warp_groups)

        self.load_register_requirement = 40
        if self.min_occupancy >= 2:
            self.mma_register_requirement = 200
        else:
            self.mma_register_requirement = 256

        self.persistent = True
        hardware_info = cutlass.utils.HardwareInfo()
        self.num_sm = hardware_info.get_device_multiprocessor_count()
        self.buffer_align_bytes = 1024

        # Epilogue tile: (BT, 32) — per k_iter writes BK/32 = 2 epi-tiles
        self.epi_tile = (self.BT, 32)
        self.epi_stage = 1
        self.num_epi_tiles = self.BK // self.epi_tile[1]  # 2
        self.num_dA_epi_tiles = self.BT // self.epi_tile[1]  # 2
        self.num_dv2_epi_tiles = self.BV // self.epi_tile[1]  # 2

    def _compute_grid(self, B: int, T: int, H: int, total_nt: Int32 | None = None):
        assert total_nt is not None
        total_tiles = total_nt * H
        grid_x = cutlass.min(Int32(self.num_sm * self.min_occupancy), total_tiles)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def __call__(
        self,
        do_in: cute.Tensor,  # [B, T, H, V] bf16
        h_in: cute.Tensor,  # [B, NT, H, K, V] bf16
        vnew_in: cute.Tensor,  # [B, T, H, V] bf16
        dh_in: cute.Tensor,  # [B, NT, H, K, V] bf16
        g_in: cute.Tensor,  # [B, T, H, K] fp32 — gating
        q_in: cute.Tensor,  # [B, T, H, K] bf16 — query (for dg)
        k_in: cute.Tensor,  # [B, T, H, K] bf16 — key (for kg)
        dq_in: cute.Tensor,  # [B, T, H, K] fp32
        dk_in: cute.Tensor,  # [B, T, H, K] fp32
        dg_in: cute.Tensor,  # [B, T, H, K] fp32
        dv_in: cute.Tensor,  # [B, T, H, V] bf16
        v_in: cute.Tensor,  # [B, T, H, V] bf16
        A_in: cute.Tensor,  # [B, T, H, BT] bf16 — intra-chunk attn
        dA_out: cute.Tensor,  # [B, T, H, BT] fp32
        dv2_out: cute.Tensor,  # [B, T, H, V] bf16
        db_out: cute.Tensor,  # [B, T, H] fp32 — gradient of beta
        beta_in: cute.Tensor,  # [B, T, H] fp32/bf16 — per-token decay scalar
        cu_seqlens_in: cute.Tensor,  # [N+1] int32
        chunk_indices_in: cute.Tensor,  # [NT, 2] int32
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
        stream,
    ):
        do_ptr = do_in.iterator
        h_ptr = h_in.iterator
        vnew_ptr = vnew_in.iterator
        dh_ptr = dh_in.iterator
        g_ptr = g_in.iterator
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        dv_ptr = dv_in.iterator
        v_ptr = v_in.iterator
        A_ptr = A_in.iterator
        dq_ptr = dq_in.iterator
        dk_ptr = dk_in.iterator
        dg_ptr = dg_in.iterator
        dA_ptr = dA_out.iterator
        dv2_ptr = dv2_out.iterator
        db_ptr = db_out.iterator
        beta_ptr = beta_in.iterator

        B, T, H, K, V = problem_size
        BT, BV = self.BT, self.BV
        data_B = Int32(1)
        NT = total_nt

        # ===================== GMEM layouts =====================
        tv_layout = cute.make_layout(
            (T, V, (H, data_B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        do = cute.make_tensor(do_ptr, tv_layout)
        vnew = cute.make_tensor(vnew_ptr, tv_layout)

        h_layout = cute.make_layout(
            (K, V, (NT, H)),
            stride=(V, 1, (H * K * V, K * V)),
        )
        h = cute.make_tensor(h_ptr, h_layout)
        dh = cute.make_tensor(dh_ptr, h_layout)

        # q layout: bf16 [B, T, H, K] — query for dg computation
        q_tk_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        q = cute.make_tensor(q_ptr, q_tk_layout)

        # k layout: bf16 [B, T, H, K] — key for kg computation
        k = cute.make_tensor(k_ptr, q_tk_layout)

        # g layout: fp32 [B, T, H, K] — same shape as dq
        g_tk_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        g = cute.make_tensor(g_ptr, g_tk_layout)

        dqk_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        dq = cute.make_tensor(dq_ptr, dqk_layout)
        dk = cute.make_tensor(dk_ptr, dqk_layout)
        dg = cute.make_tensor(dg_ptr, dqk_layout)

        dv = cute.make_tensor(dv_ptr, tv_layout)
        v = cute.make_tensor(v_ptr, tv_layout)

        # A^T: transposed GMEM view (BT, T) — first dim contiguous
        a_t_layout = cute.make_layout(
            (BT, T, (H, data_B)),
            stride=(1, H * BT, (BT, T * H * BT)),
        )
        A_attn = cute.make_tensor(A_ptr, a_t_layout)

        dA_layout = cute.make_layout(
            (T, BT, (H, data_B)),
            stride=(H * BT, 1, (BT, T * H * BT)),
        )
        dA = cute.make_tensor(dA_ptr, dA_layout)

        # dv2: bf16 [B, T, H, V] — same layout as do/vnew
        dv2_layout = cute.make_layout(
            (T, V, (H, data_B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        dv2 = cute.make_tensor(dv2_ptr, dv2_layout)

        # beta: fp32 [B, T, H] — per-token decay scalar (no K dim)
        beta_layout = cute.make_layout(
            (T, (H, data_B)),
            stride=(H, (1, T * H)),
        )
        beta_gmem = cute.make_tensor(beta_ptr, beta_layout)

        # db: fp32 [B, T, H] — gradient of beta (same layout as beta)
        db_gmem = cute.make_tensor(db_ptr, beta_layout)

        # ===================== TiledMMA =====================
        vloop_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            tiler_mn=(self.BT, self.BK),
        )
        # 64×16 MMA layout — only used to derive ldmatrix tiled_copy for
        # chunked q loading (reduces register pressure in dq*q elementwise)
        q16_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            tiler_mn=(self.BT, 16),
        )

        # dwkg GEMM: dw(BT,BK) @ kg(BT,BK)^T → (BT,BT)
        # A from registers (converted from vloop C layout), B from SMEM
        dwkg_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            (self.BT, self.BT),
            warpgroup.OperandSource.RMEM,
        )
        dkgb_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            (self.BT, self.BK),
        )
        # dA GEMM: dv(BT,BV) @ v^T(BV,BT) → (BT,BT), always m64n64
        dA_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            tiler_mn=(self.BT, self.BT),
        )
        # dA post GEMM: sA(MN-major) @ scratch(K-major) → (BT,BT), always m64n64
        dA_post1_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            (1, 1, 1),
            (self.BT, self.BT),
        )
        # dv2 GEMM: A(BT,BT) @ dv(BT,BV) → (BT,BV)
        # A from sA (COL_MAJOR → MN-major), B from sDv_col (BV,BT) COL_MAJOR MN-major
        # sDv_col is a COL_MAJOR read view of buf_dv (same pattern as sA_row)
        dv2_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            self.acc_dtype,
            (1, 1, 1),
            (self.BT, self.BV),
        )

        # ===================== SMEM layouts =====================
        tv_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.BV,
            ),
            self.io_dtype,
        )
        tv_smem_layout_staged = cute.tile_to_shape(
            tv_smem_atom,
            cute.append((self.BT, self.BV), self.vloop_stage),
            order=(0, 1, 2),
        )

        # COL_MAJOR read view of buf_dv for dv2 B-operand (same pattern as sA_row)
        dv_col_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                self.BV,
            ),
            self.io_dtype,
        )
        dv_col_smem_layout_staged = cute.tile_to_shape(
            dv_col_smem_atom,
            cute.append((self.BV, self.BT), self.vloop_stage),
            order=(0, 1, 2),
        )

        kv_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.BV,
            ),
            self.io_dtype,
        )
        kv_smem_layout_staged = cute.tile_to_shape(
            kv_smem_atom,
            cute.append((self.BK, self.BV), self.vloop_stage),
            order=(0, 1, 2),
        )

        epi_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.epi_tile[1],
            ),
            self.acc_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            epi_smem_atom,
            cute.append(self.epi_tile, self.epi_stage),
            order=(0, 1, 2),
        )

        epi_smem_atom_bf16 = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.epi_tile[1],
            ),
            self.io_dtype,
        )
        epi_smem_layout_staged_bf16 = cute.tile_to_shape(
            epi_smem_atom_bf16,
            cute.append(self.epi_tile, self.epi_stage),
            order=(0, 1, 2),
        )

        # ===================== TMA atoms =====================
        tv_smem_no_stage = cute.slice_(tv_smem_layout_staged, (None, None, 0))
        kv_smem_no_stage = cute.slice_(kv_smem_layout_staged, (None, None, 0))
        epi_smem_no_stage = cute.tile_to_shape(epi_smem_atom, self.epi_tile, order=(0, 1))
        epi_smem_no_stage_bf16 = cute.tile_to_shape(epi_smem_atom_bf16, self.epi_tile, order=(0, 1))

        tma_atom_do, tma_tensor_do = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            do,
            tv_smem_no_stage,
            (BT, BV),
        )
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            h,
            kv_smem_no_stage,
            (self.BK, BV),
        )
        tma_atom_vnew, tma_tensor_vnew = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            vnew,
            tv_smem_no_stage,
            (BT, BV),
        )
        tma_atom_dh, tma_tensor_dh = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            dh,
            kv_smem_no_stage,
            (self.BK, BV),
        )
        tma_atom_dq, tma_tensor_dq = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dq,
            epi_smem_no_stage,
            self.epi_tile,
        )
        tma_atom_dk, tma_tensor_dk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dk,
            epi_smem_no_stage,
            self.epi_tile,
        )
        tma_atom_dg, tma_tensor_dg = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dg,
            epi_smem_no_stage,
            self.epi_tile,
        )
        tma_atom_dg_reduce, tma_tensor_dg_reduce = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(reduction_kind=cute.ReductionOp.ADD),
            dg,
            epi_smem_no_stage,
            self.epi_tile,
        )
        tma_atom_dv, tma_tensor_dv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            dv,
            tv_smem_no_stage,
            (BT, BV),
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            v,
            tv_smem_no_stage,
            (BT, BV),
        )
        tma_atom_dv2, tma_tensor_dv2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dv2,
            epi_smem_no_stage_bf16,
            self.epi_tile,
        )
        tma_atom_dA, tma_tensor_dA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dA,
            epi_smem_no_stage,
            self.epi_tile,
        )

        # g TMA: tile = (BT, 32), SMEM covers (BT, BK) — loaded per k_iter
        self.g_tma_tile = (BT, 32)
        self.num_g_tma_tiles_per_k = self.BK // self.g_tma_tile[1]  # 2 = 64/32
        self.g_smem_layout = cute.tile_to_shape(
            epi_smem_atom,
            (BT, self.BK),
            order=(0, 1),
        )
        g_smem_layout = self.g_smem_layout

        # dg accumulator: (BT, BK) x f32, 2-stage — one stage per commit
        # (2 k_iters × 2 parts = 4 commits per wu_iter).
        self.num_dg_stages = 1
        self.dg_smem_layout = g_smem_layout  # single-stage layout for TMA/read
        self.dg_smem_layout_staged = cute.tile_to_shape(
            epi_smem_atom,
            cute.append((BT, self.BK), self.num_dg_stages),
            order=(0, 1, 2),
        )
        self.dg_smem_layout_write = cute.tile_to_shape(
            epi_smem_atom,
            (BT, 32, self.num_epi_tiles),
            order=(0, 1, 2),
        )
        self.dg_smem_layout_write_staged = cute.tile_to_shape(
            epi_smem_atom,
            cute.append((BT, 32, self.num_epi_tiles), self.num_dg_stages),
            order=(0, 1, 2, 3),
        )
        tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            g,
            epi_smem_no_stage,
            self.g_tma_tile,
        )

        # q TMA: bf16 (BT, BK), same tile (BT, 32) but bf16 swizzle atom
        # q SMEM layout: (BT, BK) bf16, no stage — loaded once per wu_iter
        q_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.epi_tile[1],
            ),
            self.io_dtype,
        )
        self.tk_smem_layout = cute.tile_to_shape(
            q_smem_atom,
            (BT, self.BK),
            order=(0, 1),
        )
        # q TMA tile: bf16 (BT, 32) — same shape as g TMA tile
        q_smem_tma_slice = cute.tile_to_shape(
            q_smem_atom,
            self.g_tma_tile,
            order=(0, 1),
        )
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q,
            q_smem_tma_slice,
            self.g_tma_tile,
        )

        # k TMA: bf16 (BT, BK), same layout/tile as q — loaded once per wu_iter
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k,
            q_smem_tma_slice,
            self.g_tma_tile,
        )

        # A SMEM: (BT, BT) COL_MAJOR — matches A^T GMEM (first dim contiguous)
        # MN-major MMA A-operand also expects first dim contiguous
        A_smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                BT,
            ),
            self.io_dtype,
        )
        self.A_smem_layout = cute.tile_to_shape(
            A_smem_atom,
            (BT, BT),
            order=(0, 1),
        )
        # ROW_MAJOR read view of same buf_A — sA_row[i,j] = sA[j,i]
        # Used as K-major B-operand for GEMM 2 in dA post-processing
        A_smem_atom_row = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                BT,
            ),
            self.io_dtype,
        )
        self.A_smem_layout_row = cute.tile_to_shape(
            A_smem_atom_row,
            (BT, BT),
            order=(0, 1),
        )
        tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            A_attn,
            self.A_smem_layout,
            (BT, BT),
        )

        # dw scratch: two views of the same buffer for stmatrix.trans write + WGMMA read
        # Write view: (BT, BK) — stmatrix.trans writes MMA C(BT,BK) M-major
        # Read view:  (BK, BT) — dkgb B-operand K-major (BT contiguous)
        dw_smem_atom_write = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                BT,
            ),
            self.io_dtype,
        )
        self.dw_smem_layout_write = cute.tile_to_shape(
            dw_smem_atom_write,
            (BT, self.BK),
            order=(0, 1),
        )
        dw_smem_atom_read = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                BT,
            ),
            self.io_dtype,
        )
        self.dw_smem_layout_read = cute.tile_to_shape(
            dw_smem_atom_read,
            (self.BK, BT),
            order=(0, 1),
        )
        # Wide views for dA post-processing M matrix (BT×BT)
        self.dw_smem_layout_write_wide = cute.tile_to_shape(
            dw_smem_atom_write,
            (BT, BT),
            order=(0, 1),
        )
        self.dw_smem_layout_read_wide = cute.tile_to_shape(
            dw_smem_atom_read,
            (BT, BT),
            order=(0, 1),
        )
        # buf_dw sized to max(narrow, wide) — wide needed when BK < BT
        # Keep narrow read view as dw_smem_layout for the sDw tensor creation
        self.dw_smem_layout = self.dw_smem_layout_read
        self.dw_buf_cosize = max(
            cute.cosize(self.dw_smem_layout_read),
            cute.cosize(self.dw_smem_layout_read_wide),
        )

        # kg scratch: (BT, BK) x bf16, no stage — holds kg = k * exp2(g)
        self.kg_smem_layout = cute.tile_to_shape(
            q_smem_atom,
            (BT, self.BK),
            order=(0, 1),
        )

        # ===================== TMA byte counts =====================
        self.tma_bytes_tv = cute.size_in_bytes(self.io_dtype, tv_smem_no_stage)
        self.tma_bytes_kv = cute.size_in_bytes(self.io_dtype, kv_smem_no_stage)
        # g: num_g_tma_tiles_per_k TMA copies arrive on single barrier per k_iter
        g_gmem_dtype = cutlass.Float32
        self.tma_bytes_g_single = cute.size_in_bytes(g_gmem_dtype, epi_smem_no_stage)
        self.tma_bytes_g = self.tma_bytes_g_single * self.num_g_tma_tiles_per_k
        # q/k: bf16 TMA tiles per k_iter (BK / 32 = 2 tiles per acquire)
        self.tma_bytes_q_single = cute.size_in_bytes(self.io_dtype, q_smem_tma_slice)
        self.num_q_tma_tiles_per_kiter = self.BK // self.g_tma_tile[1]  # 2
        self.tma_bytes_q = self.tma_bytes_q_single * self.num_q_tma_tiles_per_kiter
        self.tma_bytes_k = self.tma_bytes_q  # same layout as q
        # A: bf16 (BT, BT) = 8KB — one full TMA load per wu_iter
        self.tma_bytes_A = cute.size_in_bytes(self.io_dtype, self.A_smem_layout)

        # ===================== SharedStorage =====================
        # ===================== SMEM budget (BK=32, OCC=2) =====================
        # buf_epi:     (64, 32)   x f32,  1 stage,   8 KB  (also bf16 view for dv2)
        # buf_tv:      (64, 64)   x bf16, 2 stages, 16 KB  (shared: do/v/vnew)
        # buf_h:       (32, 64)   x bf16, 2 stages,  8 KB
        # buf_dh:      (32, 64)   x bf16, 2 stages,  8 KB
        # buf_dv:      (64, 64)   x bf16, 2 stages, 16 KB
        # buf_g:       (64, 32)   x f32,  1 stage,    8 KB
        # buf_q:       (64, 32)   x bf16, 1 stage,    4 KB
        # buf_k:       (64, 32)   x bf16, 1 stage,    4 KB
        # buf_A:       (64, 64)   x bf16, 1 stage,    8 KB
        # buf_dw:      (64, 64)   x bf16, 1 stage,    8 KB
        # buf_kg:      (64, 32)   x bf16, 1 stage,    4 KB
        # buf_dg:      (64, 32)   x f32,  1 stage,   8 KB
        # buf_kdk:     (64, 32)   x f32,  1 stage,    8 KB
        # buf_dgk_hdh: (128,)     x f32,  1 stage,  0.5 KB
        # buf_db:      (64,)      x f32,  1 stage, 0.25 KB
        # s_beta:      (64,)      x f32,  1 stage, 0.25 KB
        # + barriers + 1KB alignment padding per buffer
        # ─────────────────────────────────────────────────
        # BK=32 total: 113.66 KB (NCU measured), OCC=2 limit = 114 KB
        # BK=64 total: ~151 KB (OCC=1 only)
        @cute.struct
        class SharedStorage:
            bar_load_tv: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_h: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_dh: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_dv: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_g: cute.struct.MemRange[Int64, 1 * 2]
            bar_load_q: cute.struct.MemRange[Int64, 1 * 2]
            bar_load_k: cute.struct.MemRange[Int64, 1 * 2]
            bar_load_A: cute.struct.MemRange[Int64, 1 * 2]
            bar_epi_ready: cute.struct.MemRange[Int64, self.epi_stage * 2]
            bar_epi_done: cute.struct.MemRange[Int64, self.epi_stage * 2]
            bar_dg_ready: cute.struct.MemRange[Int64, self.num_dg_stages * 2]
            bar_dgk_hdh_ready: cute.struct.MemRange[Int64, self.num_k_iters * 2]
            buf_epi: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            buf_tv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(tv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            buf_h: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            buf_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            buf_dv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(tv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            buf_g: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(g_smem_layout)],
                self.buffer_align_bytes,
            ]
            buf_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.tk_smem_layout)],
                self.buffer_align_bytes,
            ]
            buf_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.tk_smem_layout)],
                self.buffer_align_bytes,
            ]
            buf_A: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.A_smem_layout)],
                self.buffer_align_bytes,
            ]
            buf_dw: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.dw_buf_cosize],
                self.buffer_align_bytes,
            ]
            buf_kg: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kg_smem_layout)],
                self.buffer_align_bytes,
            ]
            buf_dg: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(self.dg_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # kdk scratch: (BT, BK) f32 for dgk column reduction
            buf_kdk: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(g_smem_layout)],
                self.buffer_align_bytes,
            ]
            # dgk_hdh cache: BK fp32 entries (512 bytes), computed by warp 2.
            buf_dgk_hdh: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.head_dim_k],
                128,
            ]
            # db accumulator: BT fp32 entries (256 bytes)
            buf_db: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT],
                128,
            ]
            bar_load_beta: cute.struct.MemRange[Int64, 1 * 2]
            s_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT],
                128,
            ]

        self.shared_storage = SharedStorage

        cu_seqlens = cute.make_tensor(cu_seqlens_in.iterator, cute.make_layout((B + 1,)))
        chunk_indices = cute.make_tensor(
            chunk_indices_in.iterator,
            cute.make_layout((NT, 2), stride=(2, 1)),
        )

        grid = self._compute_grid(B, T, H, total_nt=NT)

        self._kernel(
            vloop_tiled_mma,
            q16_tiled_mma,
            dwkg_tiled_mma,
            dkgb_tiled_mma,
            dv2_tiled_mma,
            dA_tiled_mma,
            dA_post1_tiled_mma,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_h,
            tma_tensor_h,
            tma_atom_vnew,
            tma_tensor_vnew,
            tma_atom_dh,
            tma_tensor_dh,
            tma_atom_dq,
            tma_tensor_dq,
            tma_atom_dk,
            tma_tensor_dk,
            tma_atom_dg,
            tma_tensor_dg,
            tma_atom_dg_reduce,
            tma_tensor_dg_reduce,
            tma_atom_dv,
            tma_tensor_dv,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_dv2,
            tma_tensor_dv2,
            tma_atom_dA,
            tma_tensor_dA,
            tma_atom_A,
            tma_tensor_A,
            tma_atom_g,
            tma_tensor_g,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            g,
            beta_gmem,
            db_gmem,
            dq,
            dk,
            dg,
            dA,
            dv2,
            tv_smem_layout_staged,
            dv_col_smem_layout_staged,
            kv_smem_layout_staged,
            epi_smem_layout_staged,
            epi_smem_layout_staged_bf16,
            self.g_smem_layout,
            self.tk_smem_layout,
            self.A_smem_layout,
            self.A_smem_layout_row,
            self.dw_smem_layout,
            self.dw_smem_layout_write,
            self.dw_smem_layout_read_wide,
            self.dw_smem_layout_write_wide,
            self.kg_smem_layout,
            self.dg_smem_layout,
            self.dg_smem_layout_staged,
            self.dg_smem_layout_write,
            self.dg_smem_layout_write_staged,
            cu_seqlens,
            chunk_indices,
            problem_size,
            NT,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # C→A layout conversion
    # ---------------------------------------------------------------
    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        """Convert accumulator C layout to A operand layout for RMEM-sourced WGMMA.

        Handles nested strides like (1,2,(4,16)) from non-coalesced WGMMA atoms.
        """
        c_stride_0 = c.stride[0]
        if isinstance(c_stride_0[2], tuple):
            c_stride_0_flat = (c_stride_0[0], c_stride_0[1], c_stride_0[2][0])
            inner_base_stride = c_stride_0[2][0]
        else:
            c_stride_0_flat = c_stride_0
            inner_base_stride = c_stride_0[2]

        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c_stride_0_flat,
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * inner_base_stride),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, tiled_mma_target, negate=False):
        """Convert fp32 accumulator (C layout) → bf16 RMEM tensor (A layout).

        Follows FMHA's make_acc_into_op pattern:
        1. Compute A layout from C layout via convert_c_layout_to_a_layout
        2. Allocate RMEM tensor with A layout
        3. Write acc values (as bf16, optionally negated) through C-layout view
        """
        a_layout = self.convert_c_layout_to_a_layout(acc.layout, tiled_mma_target.tv_layout_A.shape[1])
        operand = cute.make_rmem_tensor(a_layout, cutlass.BFloat16)
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        for i in cutlass.range_constexpr(cute.size(acc)):
            val = acc[i]
            if negate:
                val = -val
            operand_as_acc[i] = cutlass.BFloat16(val)
        return operand

    # ---------------------------------------------------------------
    # Register-level row reduction helpers (from FMHA softmax pattern)
    # ---------------------------------------------------------------
    @staticmethod
    def _layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @staticmethod
    @cute.jit
    def _layout_acc_mn(tiled_mma, acc):
        separated = ChunkKdaBwdWyDqkgFusedSM90._layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )
        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    @staticmethod
    @cute.jit
    def _reduction_target_n(tiled_mma):
        separated = ChunkKdaBwdWyDqkgFusedSM90._layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    # ---------------------------------------------------------------
    # TMA partition helpers
    # ---------------------------------------------------------------
    @cute.jit
    def _tma_partition_A(
        self,
        tma_atom,
        tma_tensor,
        smem,
        tile_shape,
        tiled_mma,
        batch_idx,
        hidx,
    ):
        """Partition TMA tensor as MMA A-operand (M, K dims)."""
        coord = (None, 0, None)
        gX = cute.local_tile(
            tma_tensor,
            cute.slice_(tile_shape, coord),
            (None, None, (hidx, batch_idx)),
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_A(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 2),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _tma_partition_B(
        self,
        tma_atom,
        tma_tensor,
        smem,
        tile_shape,
        tiled_mma,
        batch_idx,
        hidx,
    ):
        """Partition TMA tensor as MMA B-operand (N, K dims)."""
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor,
            cute.slice_(tile_shape, coord),
            (None, None, (hidx, batch_idx)),
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 2),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    # ---------------------------------------------------------------
    # MMA copy factories
    # ---------------------------------------------------------------
    def _make_ldmatrix_copy_atom(self, transpose=False):
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=transpose,
                num_matrices=4,
            ),
            self.io_dtype,
        )

    def _make_stmatrix_copy_atom(self, elem_ty, transpose=False):
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                transpose=transpose,
                num_matrices=4,
            ),
            elem_ty,
        )

    def _make_r2s_tiled_copy(self, elem_ty_d, tiled_mma):
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=elem_ty_d,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = self._make_stmatrix_copy_atom(elem_ty_d)
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)

    def _make_stmatrix_r2s_tiled_copy(self, elem_ty_d, tiled_mma, transpose=False):
        copy_atom = self._make_stmatrix_copy_atom(elem_ty_d, transpose=transpose)
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom, tiled_mma)
        return cute.make_tiled_copy_S(copy_atom, tiled_copy_C_atom)

    def _make_ldmatrix_c_tiled_copy(self, tiled_mma, transpose=False):
        return cute.make_tiled_copy_C(
            self._make_ldmatrix_copy_atom(transpose=transpose),
            tiled_mma,
        )

    def _make_stmatrix_c_tiled_copy(self, elem_ty_d, tiled_mma, transpose=False):
        return cute.make_tiled_copy_C(
            self._make_stmatrix_copy_atom(elem_ty_d, transpose=transpose),
            tiled_mma,
        )

    def _make_ldmatrix_a_tiled_copy(self, tiled_mma, transpose=False):
        return cute.make_tiled_copy_A(
            self._make_ldmatrix_copy_atom(transpose=transpose),
            tiled_mma,
        )

    def _make_stmatrix_a_tiled_copy(self, elem_ty_d, tiled_mma, transpose=False):
        return cute.make_tiled_copy_A(
            self._make_stmatrix_copy_atom(elem_ty_d, transpose=transpose),
            tiled_mma,
        )

    # ---------------------------------------------------------------
    # Epilogue helper: r2s + signal store warp
    # ---------------------------------------------------------------
    @cute.jit
    def _write_epi_tile(
        self,
        epi_idx,
        tiled_copy_r2s_fp32,
        tRS_rAcc,
        tRS_sDq,
        size_tRS_rD,
        tRS_rD,
        pipeline_epi_ready,
        epi_ready_state,
    ):
        """Write one epi-tile from rmem to SMEM, then signal store warp."""
        pipeline_epi_ready.producer_acquire(epi_ready_state)

        # rmem chunk -> register staging
        for epi_v in cutlass.range_constexpr(size_tRS_rD):
            tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

        # r2s: register staging -> SMEM buffer
        epi_buffer = epi_idx % cute.size(tRS_sDq, mode=[3])
        cute.copy(
            tiled_copy_r2s_fp32,
            tRS_rD,
            tRS_sDq[(None, None, None, epi_buffer)],
        )

        # SMEM fence so store warp sees the writes
        cute.arch.fence_view_async_shared()

        # Signal store warp that epi-tile is ready
        pipeline_epi_ready.producer_commit(epi_ready_state)

    # ---------------------------------------------------------------
    # Kernel body
    # ---------------------------------------------------------------
    @cute.kernel
    def _kernel(
        self,
        vloop_tiled_mma: cute.TiledMma,
        q16_tiled_mma: cute.TiledMma,
        dwkg_tiled_mma: cute.TiledMma,
        dkgb_tiled_mma: cute.TiledMma,
        dv2_tiled_mma: cute.TiledMma,
        dA_tiled_mma: cute.TiledMma,
        dA_post1_tiled_mma: cute.TiledMma,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
        tma_atom_vnew: cute.CopyAtom,
        tma_tensor_vnew: cute.Tensor,
        tma_atom_dh: cute.CopyAtom,
        tma_tensor_dh: cute.Tensor,
        tma_atom_dq: cute.CopyAtom,
        tma_tensor_dq: cute.Tensor,
        tma_atom_dk: cute.CopyAtom,
        tma_tensor_dk: cute.Tensor,
        tma_atom_dg: cute.CopyAtom,
        tma_tensor_dg: cute.Tensor,
        tma_atom_dg_reduce: cute.CopyAtom,
        tma_tensor_dg_reduce: cute.Tensor,
        tma_atom_dv: cute.CopyAtom,
        tma_tensor_dv: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_dv2: cute.CopyAtom,
        tma_tensor_dv2: cute.Tensor,
        tma_atom_dA: cute.CopyAtom,
        tma_tensor_dA: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_tensor_A: cute.Tensor,
        tma_atom_g: cute.CopyAtom,
        tma_tensor_g: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        g_gmem: cute.Tensor,
        beta_gmem: cute.Tensor,
        db_gmem: cute.Tensor,
        dq_gmem: cute.Tensor,
        dk_gmem: cute.Tensor,
        dg_gmem: cute.Tensor,
        dA_gmem: cute.Tensor,
        dv2_gmem: cute.Tensor,
        tv_smem_layout_staged: cute.ComposedLayout,
        dv_col_smem_layout_staged: cute.ComposedLayout,
        kv_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged_bf16: cute.ComposedLayout,
        g_smem_layout: cute.ComposedLayout,
        tk_smem_layout: cute.ComposedLayout,
        A_smem_layout: cute.ComposedLayout,
        A_smem_layout_row: cute.ComposedLayout,  # ROW_MAJOR read view of buf_A
        dw_smem_layout: cute.ComposedLayout,  # read view (BK, BT)
        dw_smem_layout_write: cute.ComposedLayout,  # write view (BT, BK)
        dw_smem_layout_read_wide: cute.ComposedLayout,  # read view (BT, BT) for dA post
        dw_smem_layout_write_wide: cute.ComposedLayout,  # write view (BT, BT) for dA post
        kg_smem_layout: cute.ComposedLayout,
        dg_smem_layout: cute.ComposedLayout,
        dg_smem_layout_staged: cute.ComposedLayout,
        dg_smem_layout_write: cute.ComposedLayout,
        dg_smem_layout_write_staged: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        NT: Int32,
    ):
        B, T, H, K, V = problem_size
        BT, BV = self.BT, self.BV

        block_idx_x = cute.arch.block_idx()[0]
        grid_dim_x = cute.arch.grid_dim()[0]
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        total_work_units = NT * H
        num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x

        # Warp assignment: WG0 warps 0-3 = DMA, WG1 warps 4-7 = MMA
        load_warp_id = 0
        store_warp_id = 1

        # Prefetch TMA descriptors
        if warp_idx == load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_h)
            cpasync.prefetch_descriptor(tma_atom_vnew)
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_dv)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_g)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_A)
        if warp_idx == store_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dq)
            cpasync.prefetch_descriptor(tma_atom_dk)
            cpasync.prefetch_descriptor(tma_atom_dg)
            cpasync.prefetch_descriptor(tma_atom_dg_reduce)
            cpasync.prefetch_descriptor(tma_atom_dA)
            cpasync.prefetch_descriptor(tma_atom_dv2)

        # ===================== SMEM allocation =====================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ===================== Pipelines =====================
        # consumer_group size = num_mma_warps (not num_threads!).
        # PipelineTmaAsync.consumer_release only arrives from is_signalling_thread
        # (1 lane per warp for cluster_size=1), so the empty-barrier arrive_count
        # must equal the number of warps, not threads.  With arrive_count=128 the
        # barrier never flips when DMA needs to reuse a stage (exposed by 2-deep).
        num_mma_warps = self.num_threads_per_warp_group // self.threads_per_warp  # 4
        num_h_dh_consumers = num_mma_warps + 1  # 5: MMA WG (4 warps) + warp 2
        pipeline_load_tv = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_tv.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_tv,
        )
        pipeline_load_h = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_h.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_h_dh_consumers),
            tx_count=self.tma_bytes_kv,
        )
        pipeline_load_dh = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_dh.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_h_dh_consumers),
            tx_count=self.tma_bytes_kv,
        )
        pipeline_load_dv = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_dv.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_tv,
        )

        # Epilogue handshake: MMA WG (128 thr) -> store warp (1 thr)
        # Must use PipelineAsync (not PipelineTmaAsync) because these are
        # pure thread-to-thread notifications with no TMA involvement.
        # PipelineTmaAsync.producer_commit() is a noop (designed for TMA
        # hardware auto-arrive), so it would never signal the consumer.
        # g pipeline: 1-deep (full BT×BK loaded once per wu_iter, reused by both passes)
        pipeline_load_g = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_g.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_g,
        )
        # q pipeline: 1-deep (full BT×BK loaded once per wu_iter, same as g)
        pipeline_load_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_q.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_q,
        )
        # k pipeline: 1-deep (full BT×BK loaded once per wu_iter, same as q)
        pipeline_load_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_k.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_k,
        )
        # A pipeline: 1-deep (BT×BT loaded once per wu_iter)
        pipeline_load_A = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_A.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_mma_warps),
            tx_count=self.tma_bytes_A,
        )

        # beta pipeline: 1-deep, warp 2 (32 threads) -> MMA WG (128 threads)
        # PipelineAsync uses thread counts (not warp counts like PipelineTmaAsync)
        pipeline_load_beta = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_load_beta.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.num_threads_per_warp_group),
        )

        pipeline_epi_ready = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_epi_ready.data_ptr(),
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(self.num_threads_per_warp_group),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
        )
        pipeline_epi_done = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_epi_done.data_ptr(),
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.num_threads_per_warp_group),
        )

        # dg store pipeline: MMA WG (128 thr) → warp 3 (32 thr), single-stage
        pipeline_dg_ready = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_dg_ready.data_ptr(),
            num_stages=self.num_dg_stages,
            producer_group=make_thread_cooperative_group(self.num_threads_per_warp_group),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
        )
        # dgk_hdh pipeline: one slot per k_iter, warp 2 -> MMA WG.
        pipeline_dgk_hdh_ready = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_dgk_hdh_ready.data_ptr(),
            num_stages=self.num_k_iters,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.num_threads_per_warp_group),
        )

        # ===================== SMEM tensors =====================
        sDo = storage.buf_tv.get_tensor(tv_smem_layout_staged.outer, swizzle=tv_smem_layout_staged.inner)
        sH = storage.buf_h.get_tensor(kv_smem_layout_staged.outer, swizzle=kv_smem_layout_staged.inner)
        sVnew = storage.buf_tv.get_tensor(tv_smem_layout_staged.outer, swizzle=tv_smem_layout_staged.inner)
        sDh = storage.buf_dh.get_tensor(kv_smem_layout_staged.outer, swizzle=kv_smem_layout_staged.inner)
        sDv = storage.buf_dv.get_tensor(tv_smem_layout_staged.outer, swizzle=tv_smem_layout_staged.inner)
        sDv_col = storage.buf_dv.get_tensor(dv_col_smem_layout_staged.outer, swizzle=dv_col_smem_layout_staged.inner)
        sV = storage.buf_tv.get_tensor(tv_smem_layout_staged.outer, swizzle=tv_smem_layout_staged.inner)
        sEpi = storage.buf_epi.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)
        sEpi_bf16 = storage.buf_epi.get_tensor(
            epi_smem_layout_staged_bf16.outer,
            swizzle=epi_smem_layout_staged_bf16.inner,
            dtype=self.io_dtype,
        )
        sG = storage.buf_g.get_tensor(g_smem_layout.outer, swizzle=g_smem_layout.inner)
        sDg_staged = storage.buf_dg.get_tensor(dg_smem_layout_staged.outer, swizzle=dg_smem_layout_staged.inner)
        sDg_write_staged = storage.buf_dg.get_tensor(
            dg_smem_layout_write_staged.outer,
            swizzle=dg_smem_layout_write_staged.inner,
        )
        sQ = storage.buf_q.get_tensor(tk_smem_layout.outer, swizzle=tk_smem_layout.inner)
        sK = storage.buf_k.get_tensor(tk_smem_layout.outer, swizzle=tk_smem_layout.inner)
        sA = storage.buf_A.get_tensor(A_smem_layout.outer, swizzle=A_smem_layout.inner)
        sA_row = storage.buf_A.get_tensor(A_smem_layout_row.outer, swizzle=A_smem_layout_row.inner)
        sDw = storage.buf_dw.get_tensor(dw_smem_layout.outer, swizzle=dw_smem_layout.inner)
        # Write view (BT, BK) for stmatrix.trans — same physical buffer
        sDw_write = storage.buf_dw.get_tensor(dw_smem_layout_write.outer, swizzle=dw_smem_layout_write.inner)
        # Wide views (BT, BT) for dA post-processing M matrix
        sDw_read_wide = storage.buf_dw.get_tensor(dw_smem_layout_read_wide.outer, swizzle=dw_smem_layout_read_wide.inner)
        sDw_write_wide = storage.buf_dw.get_tensor(dw_smem_layout_write_wide.outer, swizzle=dw_smem_layout_write_wide.inner)
        sKg = storage.buf_kg.get_tensor(kg_smem_layout.outer, swizzle=kg_smem_layout.inner)
        sKdk_write = storage.buf_kdk.get_tensor(dg_smem_layout_write.outer, swizzle=dg_smem_layout_write.inner)
        sKdk_raw_ptr = cute.make_ptr(
            cutlass.Float32,
            storage.buf_kdk.data_ptr().toint(),
            cute.AddressSpace.smem,
        )
        sG_raw_ptr = cute.make_ptr(
            cutlass.Float32,
            storage.buf_g.data_ptr().toint(),
            cute.AddressSpace.smem,
        )
        sDg_raw_ptr = cute.make_ptr(
            cutlass.Float32,
            storage.buf_dg.data_ptr().toint(),
            cute.AddressSpace.smem,
        )
        sDgkHdh = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, storage.buf_dgk_hdh.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.head_dim_k,), stride=(1,)),
        )
        sDb = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, storage.buf_db.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((BT,), stride=(1,)),
        )
        sBeta = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, storage.s_beta.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((BT,), stride=(1,)),
        )
        sH_base = storage.buf_h.data_ptr().toint()
        sDh_base = storage.buf_dh.data_ptr().toint()
        kv_bytes_per_stage = self.tma_bytes_kv

        # ===================== Warp specialization =====================
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

        # ══════════════════════════════════════════════════════════════
        # DMA WARP GROUP (warps 0-3)
        # ══════════════════════════════════════════════════════════════
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            # ── Warp 0: TMA G2S load (do, h, vnew, dh, g) ──
            if warp_idx == load_warp_id:
                load_tv_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
                load_h_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
                load_dh_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
                load_dv_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.vloop_stage)
                load_g_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                load_q_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                load_k_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                load_A_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_t = work_idx // H
                    head_idx = work_idx % H
                    batch_idx = chunk_indices[(i_t, 0)]
                    tile_idx = chunk_indices[(i_t, 1)]
                    seq_tok_offset = cu_seqlens[(batch_idx,)]
                    BK = self.BK

                    # ── Load A^T (BT, T) via COL_MAJOR SMEM ──
                    gA = cute.local_tile(
                        cute.domain_offset(
                            (Int32(0), seq_tok_offset, (Int32(0), Int32(0))),
                            tma_tensor_A,
                        ),
                        (BT, BT),
                        (0, tile_idx, (head_idx, Int32(0))),
                    )
                    gA_for_tma = cute.zipped_divide(gA, (BT, BT))
                    sA_for_tma = cute.zipped_divide(sA, (BT, BT))
                    bSA_sA, bSA_gA = cpasync.tma_partition(
                        tma_atom_A,
                        0,
                        cute.make_layout(1),
                        sA_for_tma,
                        gA_for_tma,
                    )
                    pipeline_load_A.producer_acquire(load_A_ps)
                    cute.copy(
                        tma_atom_A,
                        bSA_gA[(None, (0, 0))],
                        bSA_sA[(None, 0)],
                        tma_bar_ptr=pipeline_load_A.producer_get_barrier(load_A_ps),
                    )
                    pipeline_load_A.producer_commit(load_A_ps)
                    load_A_ps.advance()

                    # ── dA V-loop: load dv + v (no k_iter dependency) ──
                    for v_iter in cutlass.range(self.num_v_tiles):
                        tma_dv_v = cute.domain_offset(
                            (seq_tok_offset, v_iter * BV, (Int32(0), Int32(0))),
                            tma_tensor_dv,
                        )
                        tDVsDv, tDVgDv = self._tma_partition_A(
                            tma_atom_dv,
                            tma_dv_v,
                            sDv,
                            self.vloop_gemm_tiler,
                            vloop_tiled_mma,
                            Int32(0),
                            head_idx,
                        )
                        pipeline_load_dv.producer_acquire(load_dv_ps)
                        dv_bar_ptr = pipeline_load_dv.producer_get_barrier(load_dv_ps)
                        cute.copy(
                            tma_atom_dv,
                            tDVgDv[(None, tile_idx, 0)],
                            tDVsDv[(None, load_dv_ps.index)],
                            tma_bar_ptr=dv_bar_ptr,
                        )
                        pipeline_load_dv.producer_commit(load_dv_ps)
                        load_dv_ps.advance()

                        tma_v_v = cute.domain_offset(
                            (seq_tok_offset, v_iter * BV, (Int32(0), Int32(0))),
                            tma_tensor_v,
                        )
                        tVsV, tVgV = self._tma_partition_A(
                            tma_atom_v,
                            tma_v_v,
                            sV,
                            self.vloop_gemm_tiler,
                            vloop_tiled_mma,
                            Int32(0),
                            head_idx,
                        )
                        pipeline_load_tv.producer_acquire(load_tv_ps)
                        cute.copy(
                            tma_atom_v,
                            tVgV[(None, tile_idx, 0)],
                            tVsV[(None, load_tv_ps.index)],
                            tma_bar_ptr=pipeline_load_tv.producer_get_barrier(load_tv_ps),
                        )
                        pipeline_load_tv.producer_commit(load_tv_ps)
                        load_tv_ps.advance()

                    # ── Unified k_iter loop: dq + dw + dk ──
                    for k_iter in cutlass.range(self.num_k_iters):
                        # ── Load g (BT, BK) fp32 per k_iter ──
                        gG = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_g,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gG_for_tma = cute.zipped_divide(gG, self.g_tma_tile)
                        sG_for_tma = cute.zipped_divide(sG, self.g_tma_tile)
                        bSG_sG, bSG_gG = cpasync.tma_partition(
                            tma_atom_g,
                            0,
                            cute.make_layout(1),
                            sG_for_tma,
                            gG_for_tma,
                        )
                        pipeline_load_g.producer_acquire(load_g_ps)
                        for k_sub in cutlass.range_constexpr(self.num_g_tma_tiles_per_k):
                            cute.copy(
                                tma_atom_g,
                                bSG_gG[(None, (0, k_sub))],
                                bSG_sG[(None, k_sub)],
                                tma_bar_ptr=pipeline_load_g.producer_get_barrier(load_g_ps),
                            )
                        pipeline_load_g.producer_commit(load_g_ps)
                        load_g_ps.advance()

                        # ── Load q (BT, BK) bf16 per k_iter ──
                        gQ = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_q,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gQ_for_tma = cute.zipped_divide(gQ, self.g_tma_tile)
                        sQ_for_tma = cute.zipped_divide(sQ, self.g_tma_tile)
                        bSQ_sQ, bSQ_gQ = cpasync.tma_partition(
                            tma_atom_q,
                            0,
                            cute.make_layout(1),
                            sQ_for_tma,
                            gQ_for_tma,
                        )
                        pipeline_load_q.producer_acquire(load_q_ps)
                        for k_sub in cutlass.range_constexpr(self.num_q_tma_tiles_per_kiter):
                            cute.copy(
                                tma_atom_q,
                                bSQ_gQ[(None, (0, k_sub))],
                                bSQ_sQ[(None, k_sub)],
                                tma_bar_ptr=pipeline_load_q.producer_get_barrier(load_q_ps),
                            )
                        pipeline_load_q.producer_commit(load_q_ps)
                        load_q_ps.advance()

                        # ── Load k (BT, BK) bf16 per k_iter ──
                        gK = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_k,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gK_for_tma = cute.zipped_divide(gK, self.g_tma_tile)
                        sK_for_tma = cute.zipped_divide(sK, self.g_tma_tile)
                        bSK_sK, bSK_gK = cpasync.tma_partition(
                            tma_atom_k,
                            0,
                            cute.make_layout(1),
                            sK_for_tma,
                            gK_for_tma,
                        )
                        pipeline_load_k.producer_acquire(load_k_ps)
                        for k_sub in cutlass.range_constexpr(self.num_q_tma_tiles_per_kiter):
                            cute.copy(
                                tma_atom_k,
                                bSK_gK[(None, (0, k_sub))],
                                bSK_sK[(None, k_sub)],
                                tma_bar_ptr=pipeline_load_k.producer_get_barrier(load_k_ps),
                            )
                        pipeline_load_k.producer_commit(load_k_ps)
                        load_k_ps.advance()

                        # dq+dw merged V-loop: load do + dv + h[k_iter]
                        # h is loaded ONCE and reused by both dq and dw compute loops
                        for v_iter in cutlass.range(self.num_v_tiles):
                            tma_do_v = cute.domain_offset(
                                (seq_tok_offset, v_iter * BV, (Int32(0), Int32(0))),
                                tma_tensor_do,
                            )
                            tDOsDo, tDOgDo = self._tma_partition_A(
                                tma_atom_do,
                                tma_do_v,
                                sDo,
                                self.vloop_gemm_tiler,
                                vloop_tiled_mma,
                                Int32(0),
                                head_idx,
                            )
                            pipeline_load_tv.producer_acquire(load_tv_ps)
                            cute.copy(
                                tma_atom_do,
                                tDOgDo[(None, tile_idx, 0)],
                                tDOsDo[(None, load_tv_ps.index)],
                                tma_bar_ptr=pipeline_load_tv.producer_get_barrier(load_tv_ps),
                            )
                            pipeline_load_tv.producer_commit(load_tv_ps)
                            load_tv_ps.advance()

                            tma_dv_v = cute.domain_offset(
                                (seq_tok_offset, v_iter * BV, (Int32(0), Int32(0))),
                                tma_tensor_dv,
                            )
                            tDVsDv, tDVgDv = self._tma_partition_A(
                                tma_atom_dv,
                                tma_dv_v,
                                sDv,
                                self.vloop_gemm_tiler,
                                vloop_tiled_mma,
                                Int32(0),
                                head_idx,
                            )
                            pipeline_load_dv.producer_acquire(load_dv_ps)
                            dv_bar_ptr = pipeline_load_dv.producer_get_barrier(load_dv_ps)
                            cute.copy(
                                tma_atom_dv,
                                tDVgDv[(None, tile_idx, 0)],
                                tDVsDv[(None, load_dv_ps.index)],
                                tma_bar_ptr=dv_bar_ptr,
                            )
                            pipeline_load_dv.producer_commit(load_dv_ps)
                            load_dv_ps.advance()

                            tma_h_v = cute.domain_offset(
                                (k_iter * BK, v_iter * BV, (0, 0)),
                                tma_tensor_h,
                            )
                            tHsH, tHgH = self._tma_partition_B(
                                tma_atom_h,
                                tma_h_v,
                                sH,
                                self.vloop_gemm_tiler,
                                vloop_tiled_mma,
                                head_idx,
                                i_t,
                            )
                            pipeline_load_h.producer_acquire(load_h_ps)
                            cute.copy(
                                tma_atom_h,
                                tHgH[(None, 0, 0)],
                                tHsH[(None, load_h_ps.index)],
                                tma_bar_ptr=pipeline_load_h.producer_get_barrier(load_h_ps),
                            )
                            pipeline_load_h.producer_commit(load_h_ps)
                            load_h_ps.advance()

                        # dk V-loop: load vnew + dh[k_iter]
                        for v_iter in cutlass.range(self.num_v_tiles):
                            tma_vnew_v = cute.domain_offset(
                                (seq_tok_offset, v_iter * BV, (Int32(0), Int32(0))),
                                tma_tensor_vnew,
                            )
                            tVNsVN, tVNgVN = self._tma_partition_A(
                                tma_atom_vnew,
                                tma_vnew_v,
                                sVnew,
                                self.vloop_gemm_tiler,
                                vloop_tiled_mma,
                                Int32(0),
                                head_idx,
                            )
                            pipeline_load_tv.producer_acquire(load_tv_ps)
                            cute.copy(
                                tma_atom_vnew,
                                tVNgVN[(None, tile_idx, 0)],
                                tVNsVN[(None, load_tv_ps.index)],
                                tma_bar_ptr=pipeline_load_tv.producer_get_barrier(load_tv_ps),
                            )
                            pipeline_load_tv.producer_commit(load_tv_ps)
                            load_tv_ps.advance()

                            tma_dh_v = cute.domain_offset(
                                (k_iter * BK, v_iter * BV, (0, 0)),
                                tma_tensor_dh,
                            )
                            tDHsDH, tDHgDH = self._tma_partition_B(
                                tma_atom_dh,
                                tma_dh_v,
                                sDh,
                                self.vloop_gemm_tiler,
                                vloop_tiled_mma,
                                head_idx,
                                i_t,
                            )
                            pipeline_load_dh.producer_acquire(load_dh_ps)
                            cute.copy(
                                tma_atom_dh,
                                tDHgDH[(None, 0, 0)],
                                tDHsDH[(None, load_dh_ps.index)],
                                tma_bar_ptr=pipeline_load_dh.producer_get_barrier(load_dh_ps),
                            )
                            pipeline_load_dh.producer_commit(load_dh_ps)
                            load_dh_ps.advance()

            # ── Warp 1: TMA S2G store (dv2, dq, dk, dg, dA) ──
            elif warp_idx == store_warp_id:
                epi_ready_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_stage)
                epi_done_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_stage)

                sEpi_for_tma = cute.group_modes(sEpi, 0, 2)
                sEpi_bf16_for_tma = cute.group_modes(sEpi_bf16, 0, 2)

                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=1,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        1,
                    ),
                )

                tidx_in_warp = cute.arch.thread_idx()[0] % Int32(32)
                universal_copy_bits = 128

                epi_copy_elems_f32 = universal_copy_bits // self.acc_dtype.width
                atom_universal_copy_f32 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.acc_dtype,
                    num_bits_per_copy=universal_copy_bits,
                )
                epi_thr_dim1_f32 = self.epi_tile[1] // epi_copy_elems_f32
                epi_thr_dim0_f32 = self.threads_per_warp // epi_thr_dim1_f32
                epi_thr_layout_f32 = cute.make_ordered_layout(
                    (epi_thr_dim0_f32, epi_thr_dim1_f32),
                    order=(1, 0),
                )
                epi_val_layout_f32 = cute.make_layout((1, epi_copy_elems_f32))
                gmem_tiled_copy_epi_f32 = cute.make_tiled_copy_tv(
                    atom_universal_copy_f32,
                    epi_thr_layout_f32,
                    epi_val_layout_f32,
                )
                epi_thr_copy_f32 = gmem_tiled_copy_epi_f32.get_slice(tidx_in_warp)
                sEpi_stage = sEpi[(None, None, 0)]
                tOsEpi_f32 = epi_thr_copy_f32.partition_S(sEpi_stage)
                tOcEpi_f32 = epi_thr_copy_f32.partition_S(cute.make_identity_tensor(self.epi_tile))
                tOrEpi_f32 = cute.make_fragment_like(tOsEpi_f32, self.acc_dtype)

                epi_copy_elems_bf16 = universal_copy_bits // self.io_dtype.width
                atom_universal_copy_bf16 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.io_dtype,
                    num_bits_per_copy=universal_copy_bits,
                )
                epi_thr_dim1_bf16 = self.epi_tile[1] // epi_copy_elems_bf16
                epi_thr_dim0_bf16 = self.threads_per_warp // epi_thr_dim1_bf16
                epi_thr_layout_bf16 = cute.make_ordered_layout(
                    (epi_thr_dim0_bf16, epi_thr_dim1_bf16),
                    order=(1, 0),
                )
                epi_val_layout_bf16 = cute.make_layout((1, epi_copy_elems_bf16))
                gmem_tiled_copy_epi_bf16 = cute.make_tiled_copy_tv(
                    atom_universal_copy_bf16,
                    epi_thr_layout_bf16,
                    epi_val_layout_bf16,
                )
                epi_thr_copy_bf16 = gmem_tiled_copy_epi_bf16.get_slice(tidx_in_warp)
                sEpi_bf16_stage = sEpi_bf16[(None, None, 0)]
                tOsEpi_bf16 = epi_thr_copy_bf16.partition_S(sEpi_bf16_stage)
                tOcEpi_bf16 = epi_thr_copy_bf16.partition_S(cute.make_identity_tensor(self.epi_tile))
                tOrEpi_bf16 = cute.make_fragment_like(tOsEpi_bf16, self.io_dtype)

                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_t = work_idx // H
                    head_idx = work_idx % H
                    batch_idx = chunk_indices[(i_t, 0)]
                    tile_idx = chunk_indices[(i_t, 1)]
                    seq_tok_offset = cu_seqlens[(batch_idx,)]
                    seq_end = cu_seqlens[(batch_idx + Int32(1),)]
                    seq_len = seq_end - seq_tok_offset
                    sub_seq_len = cutlass.min(Int32(BT), seq_len - tile_idx * Int32(BT))
                    chunk_row_base = seq_tok_offset + tile_idx * Int32(BT)
                    BK = self.BK

                    # ── Store dv2 epi-tiles (per v_iter, before k_iter loop) ──
                    for v_iter in cutlass.range(self.num_v_tiles):
                        gDv2 = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_dv2,
                            ),
                            (BT, self.BV),
                            (tile_idx, v_iter, (head_idx, Int32(0))),
                        )
                        gDv2_for_tma = cute.zipped_divide(gDv2, self.epi_tile)
                        bSG_sEpi_dv2, bSG_gDv2 = cpasync.tma_partition(
                            tma_atom_dv2,
                            0,
                            cute.make_layout(1),
                            sEpi_bf16_for_tma,
                            gDv2_for_tma,
                        )
                        epi_tile_shape_dv2 = gDv2_for_tma.shape[1]
                        epi_tile_layout_dv2 = cute.make_layout(epi_tile_shape_dv2, stride=(epi_tile_shape_dv2[1], 1))

                        for epi_idx in cutlass.range_constexpr(self.num_dv2_epi_tiles):
                            pipeline_epi_done.producer_acquire(epi_done_ps)
                            pipeline_epi_ready.consumer_wait(epi_ready_cs)
                            if sub_seq_len == Int32(BT):
                                epi_buffer = epi_idx % cute.size(bSG_sEpi_dv2, mode=[1])
                                gmem_coord = epi_tile_layout_dv2.get_hier_coord(epi_idx)
                                cute.copy(
                                    tma_atom_dv2,
                                    bSG_sEpi_dv2[(None, epi_buffer)],
                                    bSG_gDv2[(None, gmem_coord)],
                                )
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            else:
                                gmem_col_base = v_iter * Int32(self.BV) + epi_idx * Int32(32)
                                copy_partial_epi_tile_gmem_bf16(
                                    gmem_tiled_copy_epi_bf16,
                                    epi_thr_copy_bf16,
                                    tOsEpi_bf16,
                                    tOcEpi_bf16,
                                    tOrEpi_bf16,
                                    dv2_gmem.iterator,
                                    chunk_row_base,
                                    H * V,
                                    head_idx,
                                    V,
                                    gmem_col_base,
                                    sub_seq_len,
                                )
                            pipeline_epi_ready.consumer_release(epi_ready_cs)
                            epi_ready_cs.advance()
                            pipeline_epi_done.producer_commit(epi_done_ps)
                            epi_done_ps.advance()

                    # ── Unified k_iter: Store dq + dk + dg epi-tiles ──
                    for k_iter in cutlass.range(self.num_k_iters):
                        # dq epi-tiles — hybrid: full chunk uses TMA bulk
                        # store; ragged partial chunk falls back to per-thread
                        # per-thread store with `row < sub_seq_len` mask.
                        gDq = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_dq,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gDq_for_tma = cute.zipped_divide(gDq, self.epi_tile)
                        bSG_sEpi_dq, bSG_gDq = cpasync.tma_partition(
                            tma_atom_dq,
                            0,
                            cute.make_layout(1),
                            sEpi_for_tma,
                            gDq_for_tma,
                        )
                        epi_tile_shape_dq = gDq_for_tma.shape[1]
                        epi_tile_layout_dq = cute.make_layout(epi_tile_shape_dq, stride=(epi_tile_shape_dq[1], 1))

                        for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                            pipeline_epi_done.producer_acquire(epi_done_ps)
                            pipeline_epi_ready.consumer_wait(epi_ready_cs)
                            if sub_seq_len == Int32(BT):
                                epi_buffer = epi_idx % cute.size(bSG_sEpi_dq, mode=[1])
                                gmem_coord = epi_tile_layout_dq.get_hier_coord(epi_idx)
                                cute.copy(
                                    tma_atom_dq,
                                    bSG_sEpi_dq[(None, epi_buffer)],
                                    bSG_gDq[(None, gmem_coord)],
                                )
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            else:
                                gmem_col_base = k_iter * Int32(BK) + epi_idx * Int32(32)
                                copy_partial_epi_tile_gmem_f32(
                                    gmem_tiled_copy_epi_f32,
                                    epi_thr_copy_f32,
                                    tOsEpi_f32,
                                    tOcEpi_f32,
                                    tOrEpi_f32,
                                    dq_gmem.iterator,
                                    chunk_row_base,
                                    H * K,
                                    head_idx,
                                    K,
                                    gmem_col_base,
                                    sub_seq_len,
                                )
                            pipeline_epi_ready.consumer_release(epi_ready_cs)
                            epi_ready_cs.advance()
                            pipeline_epi_done.producer_commit(epi_done_ps)
                            epi_done_ps.advance()

                        # dk epi-tiles
                        gDk = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_dk,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gDk_for_tma = cute.zipped_divide(gDk, self.epi_tile)
                        bSG_sEpi_dk, bSG_gDk = cpasync.tma_partition(
                            tma_atom_dk,
                            0,
                            cute.make_layout(1),
                            sEpi_for_tma,
                            gDk_for_tma,
                        )
                        epi_tile_shape_dk = gDk_for_tma.shape[1]
                        epi_tile_layout_dk = cute.make_layout(epi_tile_shape_dk, stride=(epi_tile_shape_dk[1], 1))

                        for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                            pipeline_epi_done.producer_acquire(epi_done_ps)
                            pipeline_epi_ready.consumer_wait(epi_ready_cs)
                            if sub_seq_len == Int32(BT):
                                epi_buffer = epi_idx % cute.size(bSG_sEpi_dk, mode=[1])
                                gmem_coord = epi_tile_layout_dk.get_hier_coord(epi_idx)
                                cute.copy(
                                    tma_atom_dk,
                                    bSG_sEpi_dk[(None, epi_buffer)],
                                    bSG_gDk[(None, gmem_coord)],
                                )
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            else:
                                gmem_col_base = k_iter * Int32(BK) + epi_idx * Int32(32)
                                copy_partial_epi_tile_gmem_f32(
                                    gmem_tiled_copy_epi_f32,
                                    epi_thr_copy_f32,
                                    tOsEpi_f32,
                                    tOcEpi_f32,
                                    tOrEpi_f32,
                                    dk_gmem.iterator,
                                    chunk_row_base,
                                    H * K,
                                    head_idx,
                                    K,
                                    gmem_col_base,
                                    sub_seq_len,
                                )
                            pipeline_epi_ready.consumer_release(epi_ready_cs)
                            epi_ready_cs.advance()
                            pipeline_epi_done.producer_commit(epi_done_ps)
                            epi_done_ps.advance()

                    # ── Store dA epi-tiles (after all k_iters) ──
                    gDA = cute.local_tile(
                        cute.domain_offset(
                            (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                            tma_tensor_dA,
                        ),
                        (BT, BT),
                        (tile_idx, 0, (head_idx, Int32(0))),
                    )
                    gDA_for_tma = cute.zipped_divide(gDA, self.epi_tile)
                    bSG_sEpi_dA, bSG_gDA = cpasync.tma_partition(
                        tma_atom_dA,
                        0,
                        cute.make_layout(1),
                        sEpi_for_tma,
                        gDA_for_tma,
                    )
                    epi_tile_shape_dA = gDA_for_tma.shape[1]
                    epi_tile_layout_dA = cute.make_layout(epi_tile_shape_dA, stride=(epi_tile_shape_dA[1], 1))
                    for epi_idx in cutlass.range_constexpr(self.num_dA_epi_tiles):
                        pipeline_epi_done.producer_acquire(epi_done_ps)
                        pipeline_epi_ready.consumer_wait(epi_ready_cs)
                        if sub_seq_len == Int32(BT):
                            epi_buffer = epi_idx % cute.size(bSG_sEpi_dA, mode=[1])
                            gmem_coord = epi_tile_layout_dA.get_hier_coord(epi_idx)
                            cute.copy(
                                tma_atom_dA,
                                bSG_sEpi_dA[(None, epi_buffer)],
                                bSG_gDA[(None, gmem_coord)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        else:
                            gmem_col_base = epi_idx * Int32(32)
                            copy_partial_epi_tile_gmem_f32(
                                gmem_tiled_copy_epi_f32,
                                epi_thr_copy_f32,
                                tOsEpi_f32,
                                tOcEpi_f32,
                                tOrEpi_f32,
                                dA_gmem.iterator,
                                chunk_row_base,
                                H * Int32(BT),
                                head_idx,
                                Int32(BT),
                                gmem_col_base,
                                sub_seq_len,
                            )
                        pipeline_epi_ready.consumer_release(epi_ready_cs)
                        epi_ready_cs.advance()
                        pipeline_epi_done.producer_commit(epi_done_ps)
                        epi_done_ps.advance()

                    c_pipeline.producer_tail()

            # ── Warp 2: load beta + compute dgk_hdh from SMEM h/dh ──
            elif warp_idx == 2:
                load_beta_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                dgk_hdh_ready_ps = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_k_iters,
                )
                h_wait_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
                h_release_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
                dh_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
                lane_idx = tidx % 32
                BK = self.BK

                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_t = work_idx // H
                    head_idx = work_idx % H
                    batch_idx = chunk_indices[(i_t, 0)]
                    tile_idx = chunk_indices[(i_t, 1)]
                    seq_tok_offset = cu_seqlens[(batch_idx,)]
                    seq_end = cu_seqlens[(batch_idx + Int32(1),)]
                    sub_seq_len = cutlass.min(Int32(BT), seq_end - seq_tok_offset - tile_idx * Int32(BT))
                    chunk_tok_offset = seq_tok_offset + tile_idx * BT

                    pipeline_load_beta.producer_acquire(load_beta_ps)
                    for i in cutlass.range_constexpr(2):  # BT=64 / 32 threads = 2
                        idx = lane_idx + i * 32
                        if idx < sub_seq_len:
                            sBeta[(idx,)] = cutlass.Float32(beta_gmem[(chunk_tok_offset + idx, (head_idx, Int32(0)))])
                        else:
                            sBeta[(idx,)] = cutlass.Float32(0.0)
                    cute.arch.fence_view_async_shared()
                    pipeline_load_beta.producer_commit(load_beta_ps)
                    load_beta_ps.advance()

                    NUM_ROWS_PER_THREAD = self.BK // 32

                    for k_iter in cutlass.range(self.num_k_iters):
                        pipeline_dgk_hdh_ready.producer_acquire(dgk_hdh_ready_ps)

                        dgk_partials = cute.make_rmem_tensor((NUM_ROWS_PER_THREAD,), Float32)
                        dgk_partials.fill(Float32(0.0))

                        # Phase 1: wait for h (don't release yet)
                        for v_iter in cutlass.range(self.num_v_tiles):
                            pipeline_load_h.consumer_wait(h_wait_cs)
                            h_wait_cs.advance()

                        # Phase 2: wait for dh, read both h+dh, release both
                        for v_iter in cutlass.range(self.num_v_tiles):
                            pipeline_load_dh.consumer_wait(dh_cs)

                            sH_raw = cute.make_ptr(
                                BFloat16,
                                sH_base + h_release_cs.index * kv_bytes_per_stage,
                                cute.AddressSpace.smem,
                            )
                            sDh_raw = cute.make_ptr(
                                BFloat16,
                                sDh_base + dh_cs.index * kv_bytes_per_stage,
                                cute.AddressSpace.smem,
                            )

                            for col_chunk in cutlass.range_constexpr(self.BV // 8):
                                col_base = Int32(col_chunk * 8)
                                h_dh = cute.make_rmem_tensor((8,), Float32)
                                for r in cutlass.range_constexpr(NUM_ROWS_PER_THREAD):
                                    row = lane_idx + Int32(r * 32)
                                    h_vals = smem_load_bf16x8_sw128(sH_raw, row, col_base)
                                    dh_vals = smem_load_bf16x8_sw128(sDh_raw, row, col_base)
                                    h_dh.store(h_vals.load().to(Float32) * dh_vals.load().to(Float32))
                                    for j in cutlass.range_constexpr(8):
                                        dgk_partials[r] = dgk_partials[r] + h_dh[j]

                            pipeline_load_h.consumer_release(h_release_cs)
                            h_release_cs.advance()
                            pipeline_load_dh.consumer_release(dh_cs)
                            dh_cs.advance()

                        for r in cutlass.range_constexpr(NUM_ROWS_PER_THREAD):
                            sDgkHdh[(k_iter * Int32(BK) + lane_idx + Int32(r * 32),)] = dgk_partials[r]
                        cute.arch.fence_view_async_shared()
                        pipeline_dgk_hdh_ready.producer_commit(dgk_hdh_ready_ps)
                        dgk_hdh_ready_ps.advance()

            # ── Warp 3: TMA S2G store/reduce for dg (single-stage) ──
            elif warp_idx == 3:
                dg_ready_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_dg_stages)

                tidx_in_warp = cute.arch.thread_idx()[0] % Int32(32)
                universal_copy_bits = 128
                epi_copy_elems_f32 = universal_copy_bits // self.acc_dtype.width
                atom_universal_copy_f32 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.acc_dtype,
                    num_bits_per_copy=universal_copy_bits,
                )
                epi_thr_dim1_f32 = self.epi_tile[1] // epi_copy_elems_f32
                epi_thr_dim0_f32 = self.threads_per_warp // epi_thr_dim1_f32
                epi_thr_layout_f32 = cute.make_ordered_layout(
                    (epi_thr_dim0_f32, epi_thr_dim1_f32),
                    order=(1, 0),
                )
                epi_val_layout_f32 = cute.make_layout((1, epi_copy_elems_f32))
                gmem_tiled_copy_epi_f32 = cute.make_tiled_copy_tv(
                    atom_universal_copy_f32,
                    epi_thr_layout_f32,
                    epi_val_layout_f32,
                )
                epi_thr_copy_f32 = gmem_tiled_copy_epi_f32.get_slice(tidx_in_warp)
                cEpi_f32 = cute.make_identity_tensor(self.epi_tile)

                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_t = work_idx // H
                    head_idx = work_idx % H
                    batch_idx = chunk_indices[(i_t, 0)]
                    tile_idx = chunk_indices[(i_t, 1)]
                    seq_tok_offset = cu_seqlens[(batch_idx,)]
                    seq_end = cu_seqlens[(batch_idx + Int32(1),)]
                    sub_seq_len = cutlass.min(Int32(BT), seq_end - seq_tok_offset - tile_idx * Int32(BT))
                    chunk_row_base = seq_tok_offset + tile_idx * Int32(BT)
                    BK = self.BK

                    for k_iter in cutlass.range(self.num_k_iters):
                        gDg = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_dg,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gDg_for_tma_tile = cute.zipped_divide(gDg, self.epi_tile)
                        epi_tile_shape_dg = gDg_for_tma_tile.shape[1]
                        epi_tile_layout_dg = cute.make_layout(epi_tile_shape_dg, stride=(epi_tile_shape_dg[1], 1))

                        gDg_r = cute.local_tile(
                            cute.domain_offset(
                                (seq_tok_offset, Int32(0), (Int32(0), Int32(0))),
                                tma_tensor_dg_reduce,
                            ),
                            (BT, BK),
                            (tile_idx, k_iter, (head_idx, Int32(0))),
                        )
                        gDg_r_for_tma = cute.zipped_divide(gDg_r, self.epi_tile)

                        # ── Part 1: regular TMA store (even stage) ──
                        sDg_cur = sDg_staged[(None, None, dg_ready_cs.index)]
                        tOsDg_f32 = epi_thr_copy_f32.partition_S(sDg_cur)
                        tOcDg_f32 = epi_thr_copy_f32.partition_S(cEpi_f32)
                        tOrDg_f32 = cute.make_fragment_like(tOsDg_f32, self.acc_dtype)
                        sDg_cur_for_tma = cute.zipped_divide(sDg_cur, self.epi_tile)
                        bSG_sDg, bSG_gDg = cpasync.tma_partition(
                            tma_atom_dg,
                            0,
                            cute.make_layout(1),
                            sDg_cur_for_tma,
                            gDg_for_tma_tile,
                        )

                        pipeline_dg_ready.consumer_wait(dg_ready_cs)
                        for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                            gmem_coord = epi_tile_layout_dg.get_hier_coord(epi_idx)
                            if sub_seq_len == Int32(BT):
                                cute.copy(
                                    tma_atom_dg,
                                    bSG_sDg[(None, gmem_coord)],
                                    bSG_gDg[(None, gmem_coord)],
                                )
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                            else:
                                gmem_col_base = k_iter * Int32(BK) + epi_idx * Int32(32)
                                copy_partial_epi_tile_gmem_f32(
                                    gmem_tiled_copy_epi_f32,
                                    epi_thr_copy_f32,
                                    tOsDg_f32,
                                    tOcDg_f32,
                                    tOrDg_f32,
                                    dg_gmem.iterator,
                                    chunk_row_base,
                                    H * K,
                                    head_idx,
                                    K,
                                    gmem_col_base,
                                    sub_seq_len,
                                )
                        pipeline_dg_ready.consumer_release(dg_ready_cs)
                        dg_ready_cs.advance()

                        # ── Part 2: TMA reduce_add (odd stage) ──
                        sDg_cur2 = sDg_staged[(None, None, dg_ready_cs.index)]
                        sDg_cur2_for_tma = cute.zipped_divide(sDg_cur2, self.epi_tile)
                        bSG_sDg_r, bSG_gDg_r = cpasync.tma_partition(
                            tma_atom_dg_reduce,
                            0,
                            cute.make_layout(1),
                            sDg_cur2_for_tma,
                            gDg_r_for_tma,
                        )

                        pipeline_dg_ready.consumer_wait(dg_ready_cs)
                        if chunk_row_base + Int32(BT) <= T:
                            # For ragged but physically in-bounds chunks, OOB
                            # rows were zeroed before the SMEM write, so full
                            # tile reduce_add is a no-op for invalid rows.
                            for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                                gmem_coord = epi_tile_layout_dg.get_hier_coord(epi_idx)
                                cute.copy(
                                    tma_atom_dg_reduce,
                                    bSG_sDg_r[(None, gmem_coord)],
                                    bSG_gDg_r[(None, gmem_coord)],
                                )
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                        else:
                            # Tail chunks cannot use a full 64-row TMA reduce_add:
                            # the physical packed buffer may end before the tile does.
                            for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                                gmem_col_base = k_iter * Int32(BK) + epi_idx * Int32(32)
                                smem_col_base = epi_idx * Int32(32)
                                for row_block in cutlass.range_constexpr(16):
                                    row = Int32(row_block * 4) + tidx_in_warp // Int32(8)
                                    if row < sub_seq_len:
                                        col_low = (tidx_in_warp % Int32(8)) * Int32(4)
                                        gmem_addr = (
                                            dg_gmem.iterator
                                            + (chunk_row_base + row) * H * K
                                            + head_idx * K
                                            + gmem_col_base
                                            + col_low
                                        ).toint()
                                        old_f32 = gmem_load_f32x4(gmem_addr)
                                        add_f32 = smem_load_f32x4_sw128(
                                            sDg_raw_ptr,
                                            row,
                                            smem_col_base + col_low,
                                        )
                                        out_f32 = old_f32 + add_f32
                                        gmem_store_f32x4(gmem_addr, out_f32)
                        pipeline_dg_ready.consumer_release(dg_ready_cs)
                        dg_ready_cs.advance()

            # DMA WG done — load warp and store warp both finish here.
            # No CTA-wide sync: store warp communicates with MMA WG
            # asynchronously via bar_epi_ready / bar_epi_done mbarriers.

        # ══════════════════════════════════════════════════════════════
        # MMA WARP GROUP (warps 4-7)
        # ══════════════════════════════════════════════════════════════
        else:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            mma_warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = vloop_tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))

            # Fragments for staged SMEM operands (dq path: do, h)
            tCsDo = thr_mma.partition_A(sDo)
            tCsH = thr_mma.partition_B(sH)
            tCrDo = vloop_tiled_mma.make_fragment_A(tCsDo)
            tCrH = vloop_tiled_mma.make_fragment_B(tCsH)

            # Fragments for dk path (vnew, dh) — separate SMEM buffers
            tCsVnew = thr_mma.partition_A(sVnew)
            tCsDh = thr_mma.partition_B(sDh)
            tCrVnew = vloop_tiled_mma.make_fragment_A(tCsVnew)
            tCrDh = vloop_tiled_mma.make_fragment_B(tCsDh)

            # Fragments for dw path (dv @ h → dw, using vloop_tiled_mma)
            tCsDv = thr_mma.partition_A(sDv)
            tCrDv = vloop_tiled_mma.make_fragment_A(tCsDv)

            # Fragments for dA path (dv, v) — always m64n64 via dA_tiled_mma
            thr_dA = dA_tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
            tCsDv_dA = thr_dA.partition_A(sDv)
            tCsV_dA = thr_dA.partition_B(sV)
            tCrDv_dA = dA_tiled_mma.make_fragment_A(tCsDv_dA)
            tCrV_dA = dA_tiled_mma.make_fragment_B(tCsV_dA)
            num_k_blocks_dA = cute.size(tCrDv_dA, mode=[2])

            sEpi_no_stage = cute.slice_(sEpi, (None, None, 0))
            tCsEpi = thr_mma.partition_C(sEpi_no_stage)
            acc_shape = tCsEpi.shape
            dq_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            dw_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            dk_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            # dwkg B-operand fragment: partition sKg for dwkg_tiled_mma
            thr_dwkg = dwkg_tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
            tCsKg_B = thr_dwkg.partition_B(sKg)
            tCrKg = dwkg_tiled_mma.make_fragment_B(tCsKg_B)
            num_k_blocks_dwkg = cute.size(tCrKg, mode=[2])

            # dA_acc: always 64×64 via dwkg partition_C
            dA_acc_ref = thr_dwkg.partition_C(sA)
            dA_acc = cute.make_rmem_tensor(dA_acc_ref.shape, self.acc_dtype)

            # dkgb GEMM: A(BT,BT) @ (-dw)(BT,BK) -> (BT,BK)
            thr_dkgb = dkgb_tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
            tCsA_dkgb = thr_dkgb.partition_A(sA)
            tCsDw_dkgb = thr_dkgb.partition_B(sDw)
            tCrA_dkgb = dkgb_tiled_mma.make_fragment_A(tCsA_dkgb)
            tCrDw_dkgb = dkgb_tiled_mma.make_fragment_B(tCsDw_dkgb)
            num_k_blocks_dkgb = cute.size(tCrA_dkgb, mode=[2])
            dkgb_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            # dA post-processing GEMM 2: temp @ sA via dwkg_tiled_mma
            # B-operand = sA_row (ROW_MAJOR view of buf_A, K-major)
            tCsA_row_post = thr_dwkg.partition_B(sA_row)
            tCrA_row_post = dwkg_tiled_mma.make_fragment_B(tCsA_row_post)
            num_k_blocks_post2 = cute.size(tCrA_row_post, mode=[2])

            # dA post-processing GEMM 1: sA @ M via dA_post1_tiled_mma (always m64n64)
            thr_dA_post1 = dA_post1_tiled_mma.get_slice(
                mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
            )
            tCsA_post1 = thr_dA_post1.partition_A(sA)
            tCsDw_post1 = thr_dA_post1.partition_B(sDw_read_wide)
            tCrA_post1 = dA_post1_tiled_mma.make_fragment_A(tCsA_post1)
            tCrDw_post1 = dA_post1_tiled_mma.make_fragment_B(tCsDw_post1)
            num_k_blocks_post1 = cute.size(tCrA_post1, mode=[2])

            # dv2 GEMM: A(BT,BT) @ dv(BT,BV) -> (BT,BV)
            # B-operand uses sDv_col (BV,BT) MN-major: COL_MAJOR view of buf_dv
            thr_dv2 = dv2_tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
            tCsA_dv2 = thr_dv2.partition_A(sA)
            tCsDv_dv2 = thr_dv2.partition_B(sDv_col)
            tCrA_dv2 = dv2_tiled_mma.make_fragment_A(tCsA_dv2)
            tCrDv_dv2 = dv2_tiled_mma.make_fragment_B(tCsDv_dv2)
            num_k_blocks_dv2 = cute.size(tCrA_dv2, mode=[2])
            dv2_acc_shape = thr_dv2.partition_C(sEpi_no_stage).shape
            dv2_acc = cute.make_rmem_tensor(dv2_acc_shape, self.acc_dtype)

            num_k_blocks = cute.size(tCrDo, mode=[2])

            load_tv_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            load_h_wait_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            load_h_release_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            load_dh_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)
            load_g_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_q_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_k_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_A_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_beta_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_dv_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.vloop_stage)

            # MMA copy setup
            tiled_copy_r2s_fp32 = self._make_r2s_tiled_copy(self.acc_dtype, vloop_tiled_mma)
            mma_tidx = tidx - self.num_threads_per_warp_group * self.num_dma_warp_groups
            thr_copy_r2s = tiled_copy_r2s_fp32.get_slice(mma_tidx)
            tRS_sEpi = thr_copy_r2s.partition_D(sEpi)
            tRS_sDg = thr_copy_r2s.partition_D(sDg_write_staged)
            tRS_sKdk_write = thr_copy_r2s.partition_D(sKdk_write)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sEpi))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            # R2S tiled copy for dw with stmatrix.trans (bf16, transpose=True)
            # MMA C-layout (BT, BK) → SMEM (BK, BT) with BT contiguous
            tiled_copy_r2s_dw = self._make_stmatrix_r2s_tiled_copy(
                self.io_dtype,
                vloop_tiled_mma,
                transpose=True,
            )
            thr_copy_r2s_dw = tiled_copy_r2s_dw.get_slice(mma_tidx)
            tRS_sDw = thr_copy_r2s_dw.partition_D(sDw_write)

            # R2S for dA (based on dwkg m64n64 — always 64×64 C layout)
            tiled_copy_r2s_dA_fp32 = self._make_r2s_tiled_copy(self.acc_dtype, dwkg_tiled_mma)
            thr_copy_r2s_dA = tiled_copy_r2s_dA_fp32.get_slice(mma_tidx)
            tRS_sEpi_dA = thr_copy_r2s_dA.partition_D(sEpi)
            rD_shape_dA = cute.shape(thr_copy_r2s_dA.partition_S(sEpi))
            tRS_rD_dA = cute.make_rmem_tensor_like(cute.make_layout(rD_shape_dA[:3]), self.acc_dtype)
            size_tRS_rD_dA = cute.size(tRS_rD_dA)

            # R2S dw-transpose for dA M write (based on dwkg m64n64)
            tiled_copy_r2s_dA_dw = self._make_stmatrix_r2s_tiled_copy(
                self.io_dtype,
                dwkg_tiled_mma,
                transpose=True,
            )
            thr_copy_r2s_dA_dw = tiled_copy_r2s_dA_dw.get_slice(mma_tidx)
            tRS_sDw_wide = thr_copy_r2s_dA_dw.partition_D(sDw_write_wide)

            # R2S tiled copy for dv2: dv2_tiled_mma C layout → epi SMEM (bf16)
            tiled_copy_r2s_bf16 = self._make_r2s_tiled_copy(self.io_dtype, dv2_tiled_mma)
            thr_copy_r2s_dv2 = tiled_copy_r2s_bf16.get_slice(mma_tidx)
            tRS_sEpi_dv2 = thr_copy_r2s_dv2.partition_D(sEpi_bf16)

            rD_shape_dv2 = cute.shape(thr_copy_r2s_dv2.partition_S(sEpi_bf16))
            tRS_rD_layout_dv2 = cute.make_layout(rD_shape_dv2[:3])
            tRS_rD_dv2 = cute.make_rmem_tensor_like(tRS_rD_layout_dv2, self.io_dtype)
            size_tRS_rD_dv2 = cute.size(tRS_rD_dv2)

            # S2R tiled copy for sV: ldmatrix bulk load, aligned with dv2 MMA C partition
            tiled_copy_s2r_v = self._make_ldmatrix_c_tiled_copy(dv2_tiled_mma)
            thr_copy_s2r_v = tiled_copy_s2r_v.get_slice(mma_tidx)
            tSR_sV = thr_copy_s2r_v.partition_S(sV)
            tSR_rV_shape = cute.slice_(tSR_sV.shape, (None, None, None, 0))
            tSR_rV = cute.make_rmem_tensor(tSR_rV_shape, self.io_dtype)

            # Visitor-style kg elementwise path:
            # ldmatrix sK -> register, reuse exp_g from the dq gate, stmatrix -> sKg.
            # Uses 64×16 chunked copies to reduce register pressure.
            # 64×64 copies — kept for kg_load (dA post-processing, line ~2690)
            tiled_copy_s2r_kg = self._make_ldmatrix_c_tiled_copy(vloop_tiled_mma)
            thr_copy_s2r_kg = tiled_copy_s2r_kg.get_slice(mma_tidx)

            # 64×16 chunked copies for kg computation (k load + kg store)
            tiled_copy_r2s_kg16 = self._make_stmatrix_c_tiled_copy(
                self.io_dtype,
                q16_tiled_mma,
            )
            thr_copy_r2s_kg16 = tiled_copy_r2s_kg16.get_slice(mma_tidx)

            # 64×16 chunked q loading — reduces register pressure vs full 64×64
            tiled_copy_s2r_q16 = self._make_ldmatrix_c_tiled_copy(q16_tiled_mma)
            thr_copy_s2r_q16 = tiled_copy_s2r_q16.get_slice(mma_tidx)

            # Partitions: 64×16 chunked for k, kg, q
            tKG16_sK = thr_copy_s2r_q16.partition_S(sK)
            tKG16_sKg = thr_copy_r2s_kg16.partition_D(sKg)
            tQ16_sQ = thr_copy_s2r_q16.partition_S(sQ)
            tile16_shape_s2r = cute.slice_(tQ16_sQ.shape, (None, None, 0))
            tile16_shape_r2s = cute.slice_(tKG16_sKg.shape, (None, None, 0))
            rK16 = cute.make_rmem_tensor(tile16_shape_s2r, self.io_dtype)
            rKg16 = cute.make_rmem_tensor(tile16_shape_r2s, self.io_dtype)
            tQ16_rQ = cute.make_rmem_tensor(tile16_shape_s2r, self.io_dtype)
            num_q16_tiles = cute.size(tQ16_sQ.shape, mode=[2])
            size_per_q16 = cute.size(tile16_shape_s2r)

            tKG_sKg_load = thr_copy_s2r_kg.partition_S(sKg)
            tKG_rKg_load = cute.make_rmem_tensor(tKG_sKg_load.shape, self.io_dtype)

            # Ragged-tail sA fixup via ldmatrix -> registers -> stmatrix.
            mma_op_A_zero = cute.nvgpu.warp.MmaF16BF16Op(
                ab_dtype=self.io_dtype,
                acc_dtype=self.acc_dtype,
                shape_mnk=(16, 8, 16),
            )
            tiled_mma_A_zero = cute.make_tiled_mma(
                mma_op_A_zero,
                atom_layout_mnk=(4, 1, 1),
                permutation_mnk=(BT, BT, BT),
            )
            tiled_copy_s2r_A_zero = self._make_ldmatrix_a_tiled_copy(
                tiled_mma_A_zero,
                transpose=True,
            )
            tiled_copy_r2s_A_zero = self._make_stmatrix_a_tiled_copy(
                self.io_dtype,
                tiled_mma_A_zero,
                transpose=True,
            )
            thr_copy_s2r_A_zero = tiled_copy_s2r_A_zero.get_slice(mma_tidx)
            thr_copy_r2s_A_zero = tiled_copy_r2s_A_zero.get_slice(mma_tidx)
            thr_mma_A_zero = tiled_mma_A_zero.get_slice(mma_tidx)
            tAZ_sA = thr_copy_s2r_A_zero.partition_S(sA)
            tAZ_sA_store = thr_copy_r2s_A_zero.partition_D(sA)
            tAZ_rA_proto = thr_mma_A_zero.make_fragment_A(thr_mma_A_zero.partition_A(sA))
            tAZ_rA = cute.make_fragment_like(tAZ_rA_proto, self.io_dtype)
            cA_zero = cute.make_identity_tensor((BT, BT))
            tAZ_cA = thr_mma_A_zero.partition_A(cA_zero)

            copy_atom_s2r_g = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.acc_dtype,
                num_bits_per_copy=64,
            )
            tiled_copy_s2r_g = cute.make_tiled_copy_D(
                copy_atom_s2r_g,
                tiled_copy_r2s_fp32,
            )
            thr_copy_s2r_g = tiled_copy_s2r_g.get_slice(mma_tidx)
            tSG_sG = thr_copy_s2r_g.partition_D(sG)
            tSG_rG = cute.make_rmem_tensor(tSG_sG.shape, self.acc_dtype)

            epi_ready_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_stage)
            epi_done_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_stage)
            dg_ready_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_dg_stages)
            dgk_hdh_ready_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_k_iters,
            )

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_t = work_idx // H
                head_idx = work_idx % H
                batch_idx = chunk_indices[(i_t, 0)]
                tile_idx = chunk_indices[(i_t, 1)]
                seq_tok_offset = cu_seqlens[(batch_idx,)]
                seq_end_wu = cu_seqlens[(batch_idx + Int32(1),)]
                sub_seq_len = cutlass.min(Int32(BT), seq_end_wu - seq_tok_offset - tile_idx * Int32(BT))
                chunk_tok_offset = seq_tok_offset + tile_idx * BT
                BK = self.BK

                # Wait for A and beta early; dgk_hdh is consumed per k_iter.
                pipeline_load_A.consumer_wait(load_A_cs)
                pipeline_load_beta.consumer_wait(load_beta_cs)

                # Initialize sDb accumulator for db computation
                compute_tidx = tidx - Int32(self.num_threads_per_warp_group)
                if compute_tidx < Int32(BT):
                    sDb[(compute_tidx,)] = cutlass.Float32(0.0)
                # Partial chunk: zero sA OOB columns [sub_seq_len, BT) so the
                # dkgb = sA @ (-dw) GEMM does not consume rows pulled in by the
                # full-tile TMA load past this sequence's tail.
                if sub_seq_len < Int32(BT):
                    tAZ_rA_copy = thr_copy_s2r_A_zero.retile(tAZ_rA)
                    cute.copy(tiled_copy_s2r_A_zero, tAZ_sA, tAZ_rA_copy)
                    for i in cutlass.range_constexpr(cute.size(tAZ_rA)):
                        if tAZ_cA[i][1] >= sub_seq_len:
                            tAZ_rA[i] = self.io_dtype(0.0)
                    tAZ_rA_store = thr_copy_r2s_A_zero.retile(tAZ_rA)
                    cute.copy(tiled_copy_r2s_A_zero, tAZ_rA_store, tAZ_sA_store)
                    cute.arch.fence_view_async_shared()
                pipeline.NamedBarrier(barrier_id=BARRIER_DB_SYNC, num_threads=128).sync()

                # ═══════════════════════════════════════════════
                # dA = dv @ v^T  +  dv2 = A @ dv  (no k_iter, no g-scaling)
                # Each v_iter: dA accumulates, dv2 is complete per v_iter
                # ═══════════════════════════════════════════════
                dA_acc.fill(0.0)

                cD_dv2 = cute.make_identity_tensor((BT, self.BV))

                # Register-level db_v reduction setup (FMHA pattern)
                # Use per-thread MMA slice (runtime tidx) for identity partition
                thr_dv2_per_thread = dv2_tiled_mma.get_slice(mma_tidx)
                dv2_mn_layout = self._layout_acc_mn(dv2_tiled_mma, dv2_acc.layout)
                n_rows_dv2 = cute.size(dv2_mn_layout, mode=[0])
                tCcC_dv2 = thr_dv2_per_thread.partition_C(cD_dv2)
                coord_mn_dv2 = cute.make_tensor(tCcC_dv2.iterator, self._layout_acc_mn(dv2_tiled_mma, tCcC_dv2.layout))
                db_v_prod = cute.make_rmem_tensor(dv2_acc.layout, self.acc_dtype)
                db_v_prod_mn = cute.make_tensor(db_v_prod.iterator, dv2_mn_layout)
                partial_db_v_regs = cute.make_rmem_tensor(cute.make_layout((n_rows_dv2,)), self.acc_dtype)
                partial_db_v_regs.fill(cutlass.Float32(0.0))

                # Pre-load beta values for all rows this thread owns
                beta_regs_dv2 = cute.make_rmem_tensor(cute.make_layout((n_rows_dv2,)), self.acc_dtype)
                for i in cutlass.range_constexpr(n_rows_dv2):
                    row = coord_mn_dv2[i, 0][0]
                    beta_regs_dv2[i] = cutlass.Float32(sBeta[(row,)])

                for v_iter in cutlass.range(self.num_v_tiles):
                    pipeline_load_dv.consumer_wait(load_dv_cs)
                    pipeline_load_tv.consumer_wait(load_tv_cs)

                    # dv2 = A @ dv (complete per v_iter, K=BT) — first
                    dv2_acc.fill(0.0)
                    dv2_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks_dv2):
                        k_coord_dv2 = (None, None, k_block_idx, load_dv_cs.index)
                        cute.gemm(
                            dv2_tiled_mma,
                            dv2_acc,
                            tCrA_dv2[(None, None, k_block_idx)],
                            tCrDv_dv2[k_coord_dv2],
                            dv2_acc,
                        )
                    warpgroup.commit_group()

                    # dA += dv @ v^T (accumulates across v_iters) — second, overlaps with dv2 epi
                    dA_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks_dA):
                        k_coord_dv = (None, None, k_block_idx, load_dv_cs.index)
                        k_coord_tv = (None, None, k_block_idx, load_tv_cs.index)
                        cute.gemm(
                            dA_tiled_mma,
                            dA_acc,
                            tCrDv_dA[k_coord_dv],
                            tCrV_dA[k_coord_tv],
                            dA_acc,
                        )
                    warpgroup.commit_group()

                    # wait(1): dv2 done, dA still in flight
                    warpgroup.wait_group(1)

                    # Retile dv2_acc to S2R layout → element j aligns with tSR_rV[j]
                    tSR_rAcc_dv2 = tiled_copy_s2r_v.retile(dv2_acc)
                    # Element-wise product in S2R layout, write to db_v_prod (MMA C layout)
                    # retile is a zero-cost view, same underlying registers
                    tSR_rProd = tiled_copy_s2r_v.retile(db_v_prod)
                    # ── db_v: sum_v dv2_acc[t,v] * v[t,v] before beta scaling ──
                    # Bulk load V via ldmatrix → register, then element-wise product
                    cute.copy(
                        tiled_copy_s2r_v,
                        tSR_sV[(None, None, None, load_tv_cs.index)],
                        tSR_rV,
                    )
                    for j in cutlass.range_constexpr(cute.size(tSR_rProd)):
                        tSR_rProd[j] = tSR_rAcc_dv2[j] * cutlass.Float32(tSR_rV[j])

                    pipeline.NamedBarrier(barrier_id=BARRIER_DB_SYNC, num_threads=128).sync()

                    for i in cutlass.range_constexpr(n_rows_dv2):
                        partial_db_v_regs[i] = partial_db_v_regs[i] + db_v_prod_mn[i, None].load().reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )

                    # ── dv2 *= beta, then write to epi pipeline (overlaps with dA GEMM) ──
                    dv2_mn_view = cute.make_tensor(dv2_acc.iterator, dv2_mn_layout)
                    n_cols_dv2 = cute.size(dv2_mn_layout, mode=[1])
                    for i in cutlass.range_constexpr(n_rows_dv2):
                        for j in cutlass.range_constexpr(n_cols_dv2):
                            dv2_mn_view[i, j] = dv2_mn_view[i, j] * beta_regs_dv2[i]
                    tRS_rAcc_dv2 = tiled_copy_r2s_bf16.retile(dv2_acc)

                    for epi_idx in cutlass.range_constexpr(self.num_dv2_epi_tiles):
                        if wu_iter > 0 or v_iter > 0 or epi_idx >= self.epi_stage:
                            pipeline_epi_done.consumer_wait(epi_done_cs)
                            pipeline_epi_done.consumer_release(epi_done_cs)
                            epi_done_cs.advance()

                        pipeline_epi_ready.producer_acquire(epi_ready_ps)
                        for epi_v in cutlass.range_constexpr(size_tRS_rD_dv2):
                            tRS_rD_dv2[epi_v] = cutlass.BFloat16(tRS_rAcc_dv2[epi_idx * size_tRS_rD_dv2 + epi_v])
                        epi_buffer = epi_idx % cute.size(tRS_sEpi_dv2, mode=[3])
                        cute.copy(
                            tiled_copy_r2s_bf16,
                            tRS_rD_dv2,
                            tRS_sEpi_dv2[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_view_async_shared()
                        pipeline_epi_ready.producer_commit(epi_ready_ps)
                        epi_ready_ps.advance()

                    # wait(0): dA done
                    warpgroup.wait_group(0)

                    # Release dv/tv after dA GEMM completes (it reads dv+v SMEM)
                    pipeline_load_dv.consumer_release(load_dv_cs)
                    pipeline_load_tv.consumer_release(load_tv_cs)
                    load_dv_cs.advance()
                    load_tv_cs.advance()

                # ── db_v writeback: warp reduction + write to sDb ──
                reduction_target_dv2 = self._reduction_target_n(dv2_tiled_mma)
                red_rank_dv2 = cute.rank(reduction_target_dv2)
                for r_idx in cutlass.range_constexpr(red_rank_dv2):
                    for i in cutlass.range_constexpr(n_rows_dv2):
                        partial_db_v_regs[i] = cute.arch.warp_reduction_sum(
                            partial_db_v_regs[i],
                            threads_in_group=reduction_target_dv2.shape[r_idx],
                        )
                for i in cutlass.range_constexpr(n_rows_dv2):
                    if coord_mn_dv2[i, 0][1] == 0:
                        row = coord_mn_dv2[i, 0][0]
                        sDb[(row,)] = partial_db_v_regs[i]
                pipeline.NamedBarrier(barrier_id=BARRIER_DB_SYNC, num_threads=128).sync()

                # ═══════════════════════════════════════════════
                # Unified k_iter: dq + dw + dk + dA(-=dw@kg)
                # ═══════════════════════════════════════════════

                # db_k register reduction setup (same pattern as db_v)
                dk_mn_layout = self._layout_acc_mn(vloop_tiled_mma, dk_acc.layout)
                n_rows_dk = cute.size(dk_mn_layout, mode=[0])
                thr_dk_per_thread = vloop_tiled_mma.get_slice(mma_tidx)
                cD_dk_acc = cute.make_identity_tensor((BT, BK))
                tCcC_dk = thr_dk_per_thread.partition_C(cD_dk_acc)
                coord_mn_dk = cute.make_tensor(
                    tCcC_dk.iterator,
                    self._layout_acc_mn(vloop_tiled_mma, tCcC_dk.layout),
                )
                dbk_prod = cute.make_rmem_tensor(dk_acc.layout, self.acc_dtype)
                dbk_prod_mn = cute.make_tensor(dbk_prod.iterator, dk_mn_layout)
                partial_db_k_regs = cute.make_rmem_tensor(cute.make_layout((n_rows_dk,)), self.acc_dtype)
                partial_db_k_regs.fill(cutlass.Float32(0.0))

                for k_iter in cutlass.range(self.num_k_iters):
                    # ── dq = scale * exp2(g) * sum_v do @ h^T ──
                    dq_acc.fill(0.0)
                    vloop_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()

                    for v_iter in cutlass.range(self.num_v_tiles):
                        pipeline_load_tv.consumer_wait(load_tv_cs)
                        pipeline_load_h.consumer_wait(load_h_wait_cs)

                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_coord_do = (None, None, k_block_idx, load_tv_cs.index)
                            k_coord_h = (None, None, k_block_idx, load_h_wait_cs.index)
                            cute.gemm(
                                vloop_tiled_mma,
                                dq_acc,
                                tCrDo[k_coord_do],
                                tCrH[k_coord_h],
                                dq_acc,
                            )
                        warpgroup.commit_group()

                        if v_iter > 0:
                            warpgroup.wait_group(1)

                        pipeline_load_tv.consumer_release(load_tv_cs)
                        # h NOT released here — reused by dw loop below
                        load_tv_cs.advance()
                        load_h_wait_cs.advance()

                    # Wait for g/k TMA before WGMMA completes — ldmatrix overlaps with WGMMA.
                    # q is only needed for dg_part1 and is loaded in a separate pass
                    # after dq is gated.
                    pipeline_load_g.consumer_wait(load_g_cs)
                    pipeline_load_k.consumer_wait(load_k_cs)

                    cute.copy(tiled_copy_s2r_g, tSG_sG, tSG_rG)

                    warpgroup.wait_group(0)

                    tRS_rAcc = tiled_copy_r2s_fp32.retile(dq_acc)
                    tRS_rG = tiled_copy_r2s_fp32.retile(tSG_rG)
                    cD = cute.make_identity_tensor((BT, BK))
                    tRS_cD = thr_copy_r2s.partition_D(cD)

                    if cutlass.const_expr(DEBUG_PRINT):
                        print("=== dq_acc / R2S layout analysis ===")
                        print(f"dq_acc.shape = {dq_acc.shape}")
                        print(f"dq_acc.layout = {dq_acc.layout}")
                        print(f"tRS_rAcc size = {cute.size(tRS_rAcc)}, shape = {tRS_rAcc.shape}")
                        print(f"size_tRS_rD (per epi_tile 64x32) = {size_tRS_rD}")
                        print(f"num_epi_tiles = {self.num_epi_tiles}")
                        print("--- coordinate mapping (compile-time) ---")
                        for j in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                            print(f"  j={j}  epi={j // size_tRS_rD}  sub={j % 8 // 4}  row={tRS_cD[j][0]}  col={tRS_cD[j][1]}")
                        print("--- q16 ldmatrix (64x16 chunked) ---")
                        print(f"tQ16_sQ.shape = {tQ16_sQ.shape}, layout = {tQ16_sQ.layout}")
                        print(f"tQ16_rQ.shape = {tQ16_rQ.shape}")
                        print(f"num_q16_tiles = {num_q16_tiles}")
                        print("=== end layout analysis ===")

                    # Merged loop: kg, dq gate + cache exp2(g)/exp2(gn-g)
                    # All done in 64×16 chunks: k ldmatrix + kg stmatrix per chunk
                    k_exp_gn_g_regs = cute.make_rmem_tensor(tRS_rAcc.layout, self.acc_dtype)
                    tRS_rDbk_cache = tiled_copy_r2s_fp32.retile(dbk_prod)
                    # Ragged: chunk-last token is sub_seq_len-1, not BT-1.
                    gn_row = sub_seq_len - Int32(1)

                    for tile16 in cutlass.range_constexpr(num_q16_tiles):
                        # Load 8 k values for this 64×16 tile.
                        cute.copy(
                            tiled_copy_s2r_q16,
                            tKG16_sK[(None, None, tile16)],
                            rK16,
                        )

                        for local_j in cutlass.range_constexpr(size_per_q16):
                            j = tile16 * size_per_q16 + local_j
                            c = tRS_cD[j][1]
                            g_val = cutlass.Float32(tRS_rG[j])
                            k_val = cutlass.Float32(rK16[local_j])
                            gn_val = cutlass.Float32(sG[(gn_row, c)])

                            exp_g = cute.math.exp2(g_val)
                            exp_gn_g = cute.math.exp2(gn_val - g_val)

                            tRS_rG[j] = exp_g
                            tRS_rDbk_cache[j] = exp_gn_g

                            rKg16[local_j] = cutlass.BFloat16(k_val * exp_g)
                            k_exp_gn_g_regs[j] = k_val * exp_gn_g
                            tRS_rAcc[j] = tRS_rAcc[j] * exp_g * self.scale

                        # Store 8 kg values to sKg
                        cute.copy(
                            tiled_copy_r2s_kg16,
                            rKg16,
                            tKG16_sKg[(None, None, tile16)],
                        )

                    # k is no longer needed; q is consumed below for dg_part1.
                    pipeline_load_k.consumer_release(load_k_cs)
                    load_k_cs.advance()

                    # Epilogue: write dq[BT, BK] — epi-tiles per k_iter
                    # tRS_rAcc already retiled and gated above

                    for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                        pipeline_epi_done.consumer_wait(epi_done_cs)
                        pipeline_epi_done.consumer_release(epi_done_cs)
                        epi_done_cs.advance()

                        self._write_epi_tile(
                            epi_idx,
                            tiled_copy_r2s_fp32,
                            tRS_rAcc,
                            tRS_sEpi,
                            size_tRS_rD,
                            tRS_rD,
                            pipeline_epi_ready,
                            epi_ready_ps,
                        )
                        epi_ready_ps.advance()

                    # ── Write dg_part1 → sDg staged, signal warp 3 for TMA store ──
                    # dg_part1 = q * dq. Compute it after dq is gated and write
                    # directly to the epilogue staging fragment, avoiding a full
                    # extra fp32 register tensor for dg_part1.
                    pipeline_load_q.consumer_wait(load_q_cs)
                    pipeline_dg_ready.producer_acquire(dg_ready_ps)
                    for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                        for tile16_sub in cutlass.range_constexpr(2):
                            tile16 = epi_idx * 2 + tile16_sub
                            cute.copy(
                                tiled_copy_s2r_q16,
                                tQ16_sQ[(None, None, tile16)],
                                tQ16_rQ,
                            )
                            for local_j in cutlass.range_constexpr(size_per_q16):
                                epi_v = tile16_sub * size_per_q16 + local_j
                                j = epi_idx * size_tRS_rD + epi_v
                                q_val = cutlass.Float32(tQ16_rQ[local_j])
                                tRS_rD[epi_v] = q_val * tRS_rAcc[j]
                        cute.copy(
                            tiled_copy_r2s_fp32,
                            tRS_rD,
                            tRS_sDg[(None, None, None, epi_idx, dg_ready_ps.index)],
                        )
                    pipeline_load_q.consumer_release(load_q_cs)
                    load_q_cs.advance()
                    cute.arch.fence_view_async_shared()
                    pipeline_dg_ready.producer_commit(dg_ready_ps)
                    dg_ready_ps.advance()

                    # ── dw = dv @ h  (no g-scaling, result → sDw SMEM) ──
                    # h is still valid in SMEM from dq loop (not released)
                    dw_acc.fill(0.0)
                    vloop_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()

                    for v_iter in cutlass.range(self.num_v_tiles):
                        pipeline_load_dv.consumer_wait(load_dv_cs)

                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_coord_dv = (None, None, k_block_idx, load_dv_cs.index)
                            k_coord_h = (None, None, k_block_idx, load_h_release_cs.index)
                            cute.gemm(
                                vloop_tiled_mma,
                                dw_acc,
                                tCrDv[k_coord_dv],
                                tCrH[k_coord_h],
                                dw_acc,
                            )
                        warpgroup.commit_group()

                        if v_iter > 0:
                            warpgroup.wait_group(1)

                        pipeline_load_dv.consumer_release(load_dv_cs)
                        pipeline_load_h.consumer_release(load_h_release_cs)
                        load_dv_cs.advance()
                        load_h_release_cs.advance()

                    warpgroup.wait_group(0)

                    # ── Write -dw (fp32→bf16, transposed) to sDw via stmatrix.trans ──
                    tRS_rAcc_dw = tiled_copy_r2s_dw.retile(dw_acc)
                    rDw_shape = cute.shape(thr_copy_r2s_dw.partition_S(sDw_write))
                    tRS_rDw = cute.make_rmem_tensor_like(cute.make_layout(rDw_shape[:3]), self.io_dtype)
                    for idx in cutlass.range_constexpr(cute.size(tRS_rDw)):
                        tRS_rDw[idx] = cutlass.BFloat16(-tRS_rAcc_dw[idx])
                    cute.copy(tiled_copy_r2s_dw, tRS_rDw, tRS_sDw)
                    cute.arch.fence_view_async_shared()
                    pipeline.NamedBarrier(barrier_id=BARRIER_DW_READY, num_threads=128).sync()

                    # Convert dw_acc (C layout, fp32) → A operand (A layout, bf16, negated)
                    dw_as_a = self.make_acc_into_op(dw_acc, dwkg_tiled_mma, negate=True)

                    # ── dA -= dw @ kg: accumulate into dA_acc ──
                    dwkg_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks_dwkg):
                        cute.gemm(
                            dwkg_tiled_mma,
                            dA_acc,
                            dw_as_a[(None, None, k_block_idx)],
                            tCrKg[(None, None, k_block_idx)],
                            dA_acc,
                        )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)

                    # ── dkgb = A^T @ (-dw): A loaded transposed via COL_MAJOR SMEM ──
                    dkgb_acc.fill(0.0)
                    dkgb_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks_dkgb):
                        cute.gemm(
                            dkgb_tiled_mma,
                            dkgb_acc,
                            tCrA_dkgb[(None, None, k_block_idx)],
                            tCrDw_dkgb[(None, None, k_block_idx)],
                            dkgb_acc,
                        )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)

                    # ── dk_inter = vnew @ dh ──
                    dk_acc.fill(0.0)
                    vloop_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.fence()

                    for v_iter in cutlass.range(self.num_v_tiles):
                        pipeline_load_tv.consumer_wait(load_tv_cs)
                        pipeline_load_dh.consumer_wait(load_dh_cs)

                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_coord = (None, None, k_block_idx, load_tv_cs.index)
                            cute.gemm(
                                vloop_tiled_mma,
                                dk_acc,
                                tCrVnew[k_coord],
                                tCrDh[k_coord],
                                dk_acc,
                            )
                        warpgroup.commit_group()

                        if v_iter > 0:
                            warpgroup.wait_group(1)

                        pipeline_load_tv.consumer_release(load_tv_cs)
                        pipeline_load_dh.consumer_release(load_dh_cs)
                        load_tv_cs.advance()
                        load_dh_cs.advance()

                    cute.copy(tiled_copy_s2r_kg, tKG_sKg_load, tKG_rKg_load)

                    warpgroup.wait_group(0)

                    # dk = exp2(gn - g) * dk_inter + dkgb * exp2(g) * beta
                    tRS_rDk = tiled_copy_r2s_fp32.retile(dk_acc)
                    tRS_rDkgb = tiled_copy_r2s_fp32.retile(dkgb_acc)
                    cD_dk = cute.make_identity_tensor((BT, BK))
                    tRS_cD_dk = thr_copy_r2s.partition_D(cD_dk)

                    # Register arrays for dgk and db_k reductions
                    kdk_regs = cute.make_rmem_tensor(tRS_rDk.layout, self.acc_dtype)
                    dg_part2_regs = cute.make_rmem_tensor(tRS_rDk.layout, self.acc_dtype)
                    tRS_rDbk = tiled_copy_r2s_fp32.retile(dbk_prod)
                    tRS_rKg_ld = tiled_copy_r2s_fp32.retile(tKG_rKg_load)

                    for j in cutlass.range_constexpr(cute.size(tRS_rDk)):
                        r = tRS_cD_dk[j][0]
                        c = tRS_cD_dk[j][1]
                        beta_val = cutlass.Float32(sBeta[(r,)])
                        kg_val = cutlass.Float32(tRS_rKg_ld[j])
                        dk_inter_j = tRS_rDk[j]
                        dkgb_j = tRS_rDkgb[j]

                        exp_gn_g_j = tRS_rDbk[j]
                        exp_g_j = tRS_rG[j]

                        kdk_regs[j] = k_exp_gn_g_regs[j] * dk_inter_j
                        tRS_rDbk[j] = dkgb_j * kg_val

                        dg_part2_regs[j] = kg_val * dkgb_j * beta_val - k_exp_gn_g_regs[j] * dk_inter_j

                        tRS_rDk[j] = exp_gn_g_j * dk_inter_j + dkgb_j * exp_g_j * beta_val

                        # Ragged: zero OOB rows before any SMEM epilogue write.
                        # kdk needs this for the column reduction over BT rows;
                        # dg_part2 needs it so in-bounds tail chunks can still
                        # use full 64-row TMA reduce_add without polluting the
                        # next sequence.
                        if r >= sub_seq_len:
                            kdk_regs[j] = cutlass.Float32(0.0)
                            dg_part2_regs[j] = cutlass.Float32(0.0)
                    # Epilogue: write dk[BT, BK]
                    tRS_rAcc = tRS_rDk

                    for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                        pipeline_epi_done.consumer_wait(epi_done_cs)
                        pipeline_epi_done.consumer_release(epi_done_cs)
                        epi_done_cs.advance()

                        self._write_epi_tile(
                            epi_idx,
                            tiled_copy_r2s_fp32,
                            tRS_rAcc,
                            tRS_sEpi,
                            size_tRS_rD,
                            tRS_rD,
                            pipeline_epi_ready,
                            epi_ready_ps,
                        )
                        epi_ready_ps.advance()

                    # ── db_k: register row reduction (accumulate across k_iters) ──
                    for i in cutlass.range_constexpr(n_rows_dk):
                        partial_db_k_regs[i] = partial_db_k_regs[i] + dbk_prod_mn[i, None].load().reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )

                    # Save gn before sG is overwritten (needed by dgk)
                    # ld128: each of 16 threads reads 4 consecutive gn values
                    my_gn = cute.make_rmem_tensor((4,), self.acc_dtype)
                    my_gn.fill(cutlass.Float32(0.0))
                    if compute_tidx < Int32(BK // 4):
                        gn_col_base = compute_tidx * Int32(4)
                        my_gn.store(smem_load_f32x4_sw128(sG_raw_ptr, sub_seq_len - Int32(1), gn_col_base))

                    # ── dgk: m_last * (exp2(gn)*sum_v(h*dh) + sum_t(kdk)) ──

                    # Write kdk_regs → sKdk via stmatrix (separate buffer)
                    for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = kdk_regs[epi_idx * size_tRS_rD + epi_v]
                        cute.copy(
                            tiled_copy_r2s_fp32,
                            tRS_rD,
                            tRS_sKdk_write[(None, None, None, epi_idx)],
                        )

                    cute.arch.fence_view_async_shared()
                    pipeline.NamedBarrier(barrier_id=BARRIER_DG_COMPUTE, num_threads=128).sync()
                    pipeline_dgk_hdh_ready.consumer_wait(dgk_hdh_ready_cs)

                    # Column reduction via ld128: 16 threads × 4 cols each
                    # Compute dgk and write to sDgkHdh for broadcast
                    if compute_tidx < Int32(BK // 4):
                        col_base = compute_tidx * Int32(4)
                        dgk = cute.make_rmem_tensor((4,), self.acc_dtype)
                        dgk.fill(cutlass.Float32(0.0))
                        for row in cutlass.range(BT, unroll_full=True):
                            vals = smem_load_f32x4_sw128(sKdk_raw_ptr, Int32(row), col_base)
                            for ci in cutlass.range_constexpr(4):
                                dgk[ci] = dgk[ci] + vals[ci]
                        hdh_off = k_iter * Int32(BK) + col_base
                        for ci in cutlass.range_constexpr(4):
                            hdh_val = cutlass.Float32(sDgkHdh[(hdh_off + Int32(ci),)])
                            dgk[ci] = cute.math.exp2(my_gn[ci]) * hdh_val + dgk[ci]
                            sDgkHdh[(hdh_off + Int32(ci),)] = dgk[ci]

                    cute.arch.fence_view_async_shared()
                    pipeline.NamedBarrier(barrier_id=BARRIER_DG_COMPUTE, num_threads=128).sync()

                    # Acquire sDg stage BEFORE writing — ensures warp 3
                    # has finished TMA-reading this stage from a prior round.
                    pipeline_dg_ready.producer_acquire(dg_ready_ps)

                    # Write dg_part2 → sDg staged via stmatrix (without dgk)
                    tRS_rDgP2 = tiled_copy_r2s_fp32.retile(dg_part2_regs)
                    for epi_idx in cutlass.range_constexpr(self.num_epi_tiles):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = tRS_rDgP2[epi_idx * size_tRS_rD + epi_v]
                        cute.copy(
                            tiled_copy_r2s_fp32,
                            tRS_rD,
                            tRS_sDg[(None, None, None, epi_idx, dg_ready_ps.index)],
                        )

                    # Fence + barrier: ensure stmatrix of dg_part2 is visible
                    cute.arch.fence_view_async_shared()
                    pipeline.NamedBarrier(barrier_id=BARRIER_DG_COMPUTE, num_threads=128).sync()

                    # 16 threads add dgk directly to sDg SMEM last row
                    # Ragged: last row = sub_seq_len - 1 (== BT-1 for full chunk).
                    dgk_row = sub_seq_len - Int32(1)
                    if compute_tidx < Int32(BK // 4):
                        col_base = compute_tidx * Int32(4)
                        for ci in cutlass.range_constexpr(4):
                            col = col_base + Int32(ci)
                            dgk_off = k_iter * Int32(BK) + col
                            old_val = cutlass.Float32(sDg_staged[(dgk_row, col, dg_ready_ps.index)])
                            sDg_staged[(dgk_row, col, dg_ready_ps.index)] = old_val + cutlass.Float32(sDgkHdh[(dgk_off,)])

                    pipeline_dgk_hdh_ready.consumer_release(dgk_hdh_ready_cs)
                    dgk_hdh_ready_cs.advance()

                    # ── dg_part2+dgk epilogue: signal warp 3 for TMA reduce_add ──
                    cute.arch.fence_view_async_shared()
                    pipeline_dg_ready.producer_commit(dg_ready_ps)
                    dg_ready_ps.advance()

                    # Release g after dk/dg are done with it
                    pipeline_load_g.consumer_release(load_g_cs)
                    load_g_cs.advance()

                # ═══════════════════════════════════════════════
                # dA post-processing via WGMMA:
                # dA_final = -lower_tri(sA @ (lower_tri(dA_raw * beta) @ sA))
                # GEMM 1: M @ sA  (dwkg pattern: RMEM A, sA_row as B)
                # GEMM 2: sA @ temp  (dkgb pattern: sA as A, temp in sDw as B)
                # ═══════════════════════════════════════════════

                # Step 1: mask + beta in dA_acc
                tRS_rAcc_dA = tiled_copy_r2s_dA_fp32.retile(dA_acc)
                cD_dA = cute.make_identity_tensor((BT, BT))
                tRS_cD_dA = thr_copy_r2s_dA.partition_D(cD_dA)

                for j in cutlass.range_constexpr(cute.size(tRS_rAcc_dA)):
                    r = tRS_cD_dA[j][0]
                    c = tRS_cD_dA[j][1]
                    if r > c:
                        tRS_rAcc_dA[j] = tRS_rAcc_dA[j] * cutlass.Float32(sBeta[(c,)])
                    else:
                        tRS_rAcc_dA[j] = cutlass.Float32(0.0)

                # Step 2: GEMM 1 — M @ sA_row via dwkg_tiled_mma
                # Convert M (dA_acc, C layout) → A operand for dwkg.
                m_as_a = self.make_acc_into_op(dA_acc, dwkg_tiled_mma)
                dA_acc.fill(0.0)
                dwkg_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks_post2):
                    cute.gemm(
                        dwkg_tiled_mma,
                        dA_acc,
                        m_as_a[(None, None, k_block_idx)],
                        tCrA_row_post[(None, None, k_block_idx)],
                        dA_acc,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                # db_k writeback + db GMEM write — overlap with WGMMA GEMM 1
                reduction_target_dk = self._reduction_target_n(vloop_tiled_mma)
                red_rank_dk = cute.rank(reduction_target_dk)
                for r_idx in cutlass.range_constexpr(red_rank_dk):
                    for i in cutlass.range_constexpr(n_rows_dk):
                        partial_db_k_regs[i] = cute.arch.warp_reduction_sum(
                            partial_db_k_regs[i],
                            threads_in_group=reduction_target_dk.shape[r_idx],
                        )
                for i in cutlass.range_constexpr(n_rows_dk):
                    if coord_mn_dk[i, 0][1] == 0:
                        row = coord_mn_dk[i, 0][0]
                        sDb[(row,)] = cutlass.Float32(sDb[(row,)]) + partial_db_k_regs[i]

                # Ensure all warps' sDb writes are visible before reading
                pipeline.NamedBarrier(barrier_id=BARRIER_DB_SYNC, num_threads=128).sync()

                # Ragged: only write db rows that belong to this sequence's chunk.
                if compute_tidx < sub_seq_len:
                    db_gmem[(chunk_tok_offset + compute_tidx, (head_idx, Int32(0)))] = sDb[(compute_tidx,)]

                # Step 3: write temp (in dA_acc) → bf16 → sDw_wide via stmatrix.trans.
                # temp becomes the K-major B operand for GEMM 2.
                tRS_rAcc_dA_dw = tiled_copy_r2s_dA_dw.retile(dA_acc)
                rM_shape = cute.shape(thr_copy_r2s_dA_dw.partition_S(sDw_write_wide))
                tRS_rM = cute.make_rmem_tensor_like(cute.make_layout(rM_shape[:3]), self.io_dtype)
                for idx in cutlass.range_constexpr(cute.size(tRS_rM)):
                    tRS_rM[idx] = cutlass.BFloat16(tRS_rAcc_dA_dw[idx])
                cute.copy(tiled_copy_r2s_dA_dw, tRS_rM, tRS_sDw_wide)
                cute.arch.fence_view_async_shared()
                pipeline.NamedBarrier(barrier_id=BARRIER_DB_SYNC, num_threads=128).sync()

                # Step 4: GEMM 2 — sA @ temp → dA_acc (always m64n64)
                dA_acc.fill(0.0)
                dA_post1_tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks_post1):
                    cute.gemm(
                        dA_post1_tiled_mma,
                        dA_acc,
                        tCrA_post1[(None, None, k_block_idx)],
                        tCrDw_post1[(None, None, k_block_idx)],
                        dA_acc,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                # Step 5: mask + negate
                tRS_rAcc_dA = tiled_copy_r2s_dA_fp32.retile(dA_acc)
                cD_dA = cute.make_identity_tensor((BT, BT))
                tRS_cD_dA = thr_copy_r2s_dA.partition_D(cD_dA)

                for j in cutlass.range_constexpr(cute.size(tRS_rAcc_dA)):
                    r = tRS_cD_dA[j][0]
                    c = tRS_cD_dA[j][1]
                    if r > c:
                        tRS_rAcc_dA[j] = -tRS_rAcc_dA[j]
                    else:
                        tRS_rAcc_dA[j] = cutlass.Float32(0.0)

                # Release A and beta after post-processing
                pipeline_load_A.consumer_release(load_A_cs)
                load_A_cs.advance()
                pipeline_load_beta.consumer_release(load_beta_cs)
                load_beta_cs.advance()

                # ═══════════════════════════════════════════════
                # dA epilogue: write dA (2 epi-tiles) — after all k_iters
                # ═══════════════════════════════════════════════
                tRS_rAcc = tiled_copy_r2s_dA_fp32.retile(dA_acc)

                for epi_idx in cutlass.range_constexpr(self.num_dA_epi_tiles):
                    pipeline_epi_done.consumer_wait(epi_done_cs)
                    pipeline_epi_done.consumer_release(epi_done_cs)
                    epi_done_cs.advance()

                    self._write_epi_tile(
                        epi_idx,
                        tiled_copy_r2s_dA_fp32,
                        tRS_rAcc,
                        tRS_sEpi_dA,
                        size_tRS_rD_dA,
                        tRS_rD_dA,
                        pipeline_epi_ready,
                        epi_ready_ps,
                    )
                    epi_ready_ps.advance()

        return


# =====================================================================
# Compilation cache
# =====================================================================

_bwd_wy_kernel_cache: dict = {}


def _compile_bwd_wy_variant(
    H: int,
    K: int,
    V: int,
    scale: float,
    chunk_size: int,
    beta_dtype: type[cutlass.Numeric],
    use_fast_math: bool,
    bk: int = 32,
    bv: int = 64,
    min_occupancy: int = 2,
):
    kernel_obj = ChunkKdaBwdWyDqkgFusedSM90(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        scale=scale,
        use_fast_math=use_fast_math,
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )

    sym_b = cute.sym_int()
    sym_nt = cute.sym_int()
    sym_cu = cute.sym_int()
    sym_ci = cute.sym_int()

    do_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    h_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_nt, H, K, V),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=128,
    )
    vnew_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dh_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_nt, H, K, V),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=128,
    )
    dq_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dk_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    ci_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ci, 2), stride_order=(1, 0), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    g_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dv_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    v_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    A_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (1, sym_b, H, chunk_size), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    dA_fake = make_fake_compact_tensor(
        cutlass.Float32, (1, sym_b, H, chunk_size), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    dv2_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    db_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, H), stride_order=(2, 1, 0), assumed_align=128)
    beta_fake = make_fake_compact_tensor(beta_dtype, (1, sym_b, H), stride_order=(2, 1, 0), assumed_align=128)

    dg_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    compiled_fn = cute.compile(
        kernel_obj,
        do_fake,
        h_fake,
        vnew_fake,
        dh_fake,
        g_fake,
        q_fake,
        k_fake,
        dq_fake,
        dk_fake,
        dg_fake,
        dv_fake,
        v_fake,
        A_fake,
        dA_fake,
        dv2_fake,
        db_fake,
        beta_fake,
        cu_fake,
        ci_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Int32(1),
        stream_fake,
        options=COMPILE_OPTIONS,
    )
    return compiled_fn


def _get_compiled_bwd_wy(
    H: int,
    K: int,
    V: int,
    scale: float,
    chunk_size: int,
    beta_dtype: torch.dtype,
    bk: int = 32,
    bv: int = 64,
    min_occupancy: int = 2,
):
    key = (H, K, V, scale, chunk_size, beta_dtype, USE_FAST_MATH, bk, bv, min_occupancy)
    if key not in _bwd_wy_kernel_cache:
        _bwd_wy_kernel_cache[key] = _compile_bwd_wy_variant(
            H,
            K,
            V,
            scale,
            chunk_size,
            _torch_to_cutlass_dtype[beta_dtype],
            USE_FAST_MATH,
            bk=bk,
            bv=bv,
            min_occupancy=min_occupancy,
        )
    return _bwd_wy_kernel_cache[key]


# =====================================================================
# Public Python wrapper — FLA-compatible
# =====================================================================


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
    *,
    bk: int = 32,
    bv: int = 64,
    min_occupancy: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM90 wrapper for the WY dq+kg fused backward kernel.

    Returns:
        (dq, dk, dv, db, dg, dA), matching FLA's output order.
    """
    from fla.ops.utils.index import prepare_chunk_indices

    from cula.utils import prepare_uniform_cu_seqlens

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    device = q.device

    if scale is None:
        scale = K**-0.5

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(B, T, device, torch.int32)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    assert cu_seqlens.dtype == torch.int32
    assert do.dtype == torch.bfloat16
    assert h.dtype == torch.bfloat16
    assert g.dtype == torch.float32
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert A.dtype == torch.bfloat16
    assert beta.dtype in _torch_to_cutlass_dtype, f"SM90 kernel only supports fp32/bf16 beta, got {beta.dtype}"

    T_total = B * T
    num_seqs = cu_seqlens.shape[0] - 1
    total_nt_val = chunk_indices.shape[0]
    ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(K), Int32(V))

    dq = torch.empty(1, T_total, H, K, dtype=torch.float32, device=device)
    dk = torch.empty(1, T_total, H, K, dtype=torch.float32, device=device)
    dv_out = torch.empty(1, T_total, H, V, dtype=torch.bfloat16, device=device)
    db = torch.empty(1, T_total, H, dtype=torch.float32, device=device)
    dg = torch.empty(1, T_total, H, K, dtype=torch.float32, device=device)
    dA = torch.empty(1, T_total, H, BT, dtype=torch.float32, device=device)

    if B != 1:
        do = do.reshape(1, T_total, H, V)
        h = h.reshape(1, total_nt_val, H, K, V)
        g = g.reshape(1, T_total, H, K)
        q = q.reshape(1, T_total, H, K)
        k = k.reshape(1, T_total, H, K)
        v_new = v_new.reshape(1, T_total, H, V)
        dh = dh.reshape(1, total_nt_val, H, K, V)
        dv = dv.reshape(1, T_total, H, V)
        v = v.reshape(1, T_total, H, V)
        A = A.reshape(1, T_total, H, BT)
        beta = beta.reshape(1, T_total, H)

    compiled_fn = _get_compiled_bwd_wy(H, K, V, scale, chunk_size, beta.dtype, bk=bk, bv=bv, min_occupancy=min_occupancy)

    compiled_fn(
        do,
        h,
        v_new,
        dh,
        g,
        q,
        k,
        dq,
        dk,
        dg,
        dv,
        v,
        A,
        dA,
        dv_out,
        db,
        beta,
        cu_seqlens,
        chunk_indices,
        ps,
        Int32(total_nt_val),
    )

    if B != 1:
        dq = dq.reshape(B, T, H, K)
        dk = dk.reshape(B, T, H, K)
        dv_out = dv_out.reshape(B, T, H, V)
        db = db.reshape(B, T, H)
        dg = dg.reshape(B, T, H, K)
        dA = dA.reshape(B, T, H, BT)

    return dq, dk, dv_out, db, dg, dA
