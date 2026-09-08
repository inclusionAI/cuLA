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
Chunk Gated Delta Rule Backward DHU Kernel (SM90 WGMMA)

Hopper tensor-core path aligned with cula/ops/chunk_delta_h.py:
- fixed chunk size BT=64
- K=V=128, BV=64
- non-varlen tensors [B, T, H, D] and packed varlen tensors
- state layout [B, NT, H, K, V] or [B, NT, H, V, K]

The recurrence follows FLA's bwd_dhu:
    dv2 = dv + K @ dh
    dh  = decay(dh) + scale * Q^T @ do - W^T @ dv2

Each CTA owns one BV tile and one (batch, head). WGMMA computes the three
64x64 GEMMs per chunk while CUDA threads carry dh in registers.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

from cula.utils import USE_FAST_MATH, assert_hopper

BT = 64
BV = 64
BK = 128
NUM_THREADS = 224


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkDeltaRuleBwdDHUSm90:
    def __init__(
        self,
        num_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        is_varlen: bool,
        use_g: bool,
        use_gk: bool,
        use_dht: bool,
        use_dh0: bool,
        use_exp2: bool,
        transpose_state_layout: bool,
        scale: float,
        use_fast_math: bool = True,
    ):
        assert head_dim_k == 128 and head_dim_v == 128, (
            f"SM90 bwd_dhu currently aligns with ChunkDeltaRuleFwdH and requires K=V=128, got K={head_dim_k}, V={head_dim_v}"
        )
        self.H = num_heads
        self.K = head_dim_k
        self.V = head_dim_v
        self.is_varlen = is_varlen
        self.use_g = use_g
        self.use_gk = use_gk
        self.use_dht = use_dht
        self.use_dh0 = use_dh0
        self.use_exp2 = use_exp2
        self.transpose_state_layout = transpose_state_layout
        self.scale = scale
        self.use_fast_math = use_fast_math

        self.BT = BT
        self.BV = BV
        self.BK = BK
        self.threads_per_warp = 32
        self.num_compute_warps = 4
        self.num_compute_threads = self.threads_per_warp * self.num_compute_warps
        self.load_warp_id = 4
        self.load_current_warp_id = 5
        self.store_warp_id = 6
        self.num_threads = NUM_THREADS
        self.num_regs_compute = 256
        self.num_regs_other = 40
        self.k_stage = 3
        self.dv_stage = 2
        self.do_stage = 2
        self.q_stage = 3
        self.w_stage = 3
        self.gk_stage = 3
        self.dh_store_stage = 2
        self.dv2_store_stage = 2
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 128

        # K=BK=128, so the carried dh state is a single BV x BK register tile.
        self.kdh_mma_tiler = (self.BV, self.BT, self.BK)
        self.update_mma_tiler = (self.BV, self.BK, self.BT)
        self.atom_layout_mnk = (1, 1, 1)
        self.num_v_tile_mcast_ctas = 2
        self.cluster_shape_mnk = (self.num_v_tile_mcast_ctas, 1, 1)
        self.gk_precompute_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_compute_threads,
        )

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        g_in: cute.Tensor,
        gk_in: cute.Tensor,
        dht_in: cute.Tensor,
        dh0_in: cute.Tensor,
        do_in: cute.Tensor,
        dh_in: cute.Tensor,
        dv_in: cute.Tensor,
        dv2_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,
        chunk_offsets_in: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32],
        stream: cuda.CUstream,
    ):
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        w_ptr = w_in.iterator
        g_ptr = g_in.iterator
        gk_ptr = gk_in.iterator
        dht_ptr = dht_in.iterator
        dh0_ptr = dh0_in.iterator
        do_ptr = do_in.iterator
        dh_ptr = dh_in.iterator
        dv_ptr = dv_in.iterator
        dv2_ptr = dv2_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator
        chunk_offsets_ptr = chunk_offsets_in.iterator

        B, T, N, NT_total = problem_size

        # ===================== GMEM layouts =====================
        g_layout = cute.make_layout(
            (B, T, self.H),
            stride=(T * self.H, self.H, 1),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((N + 1,)))
        chunk_offsets = cute.make_tensor(chunk_offsets_ptr, cute.make_layout((N + 1,)))

        # dh TMA store view: (V, K) tile with layout selected by the requested state layout.
        if cutlass.const_expr(self.transpose_state_layout):
            dh_tma_layout = cute.make_layout(
                (self.V, self.K, (NT_total, self.H, B)),
                stride=(self.K, 1, (self.H * self.K * self.V, self.K * self.V, NT_total * self.H * self.K * self.V)),
            )
        else:
            dh_tma_layout = cute.make_layout(
                (self.V, self.K, (NT_total, self.H, B)),
                stride=(1, self.V, (self.H * self.K * self.V, self.K * self.V, NT_total * self.H * self.K * self.V)),
            )
        dh_tma_tile = (self.BV, self.BK)
        dh_smem_layout_enum = (
            utils.LayoutEnum.ROW_MAJOR if cutlass.const_expr(self.transpose_state_layout) else utils.LayoutEnum.COL_MAJOR
        )
        dh_tma = cute.make_tensor(dh_ptr, dh_tma_layout)

        if cutlass.const_expr(self.transpose_state_layout):
            final_layout = cute.make_layout(
                (N, self.H, self.V, self.K),
                stride=(self.H * self.K * self.V, self.K * self.V, self.K, 1),
            )
        else:
            final_layout = cute.make_layout(
                (N, self.H, self.K, self.V),
                stride=(self.H * self.K * self.V, self.K * self.V, self.V, 1),
            )
        dht = cute.make_tensor(dht_ptr, final_layout)
        dh0 = cute.make_tensor(dh0_ptr, final_layout)
        if cutlass.const_expr(self.transpose_state_layout):
            dht_tma_layout = cute.make_layout(
                (self.V, self.K, (self.H, N)),
                stride=(self.K, 1, (self.K * self.V, self.H * self.K * self.V)),
            )
        else:
            dht_tma_layout = cute.make_layout(
                (self.V, self.K, (self.H, N)),
                stride=(1, self.V, (self.K * self.V, self.H * self.K * self.V)),
            )
        dht_tma = cute.make_tensor(dht_ptr, dht_tma_layout)
        dh0_tma = cute.make_tensor(dh0_ptr, dht_tma_layout)

        # TMA operand views. Varlen shifts the T dimension with domain_offset below.
        tk_layout = cute.make_layout((T, self.K, (self.H, B)), stride=(self.H * self.K, 1, (self.K, T * self.H * self.K)))
        k_tk = cute.make_tensor(k_ptr, tk_layout)

        kt_layout = cute.make_layout((self.K, T, (self.H, B)), stride=(1, self.H * self.K, (self.K, T * self.H * self.K)))
        q_kt = cute.make_tensor(q_ptr, kt_layout)
        w_kt = cute.make_tensor(w_ptr, kt_layout)
        gk_kt = cute.make_tensor(gk_ptr, kt_layout)

        vt_layout = cute.make_layout((self.V, T, (self.H, B)), stride=(1, self.H * self.V, (self.V, T * self.H * self.V)))
        do_vt = cute.make_tensor(do_ptr, vt_layout)
        dv_vt = cute.make_tensor(dv_ptr, vt_layout)
        dv2_vt_layout = cute.make_layout((self.V, T, (self.H, B)), stride=(1, self.V, (T * self.V, T * self.H * self.V)))
        dv2_vt = cute.make_tensor(dv2_ptr, dv2_vt_layout)
        dv2_layout = cute.make_layout(
            (B, self.H, T, self.V),
            stride=(T * self.H * self.V, T * self.V, self.V, 1),
        )
        dv2 = cute.make_tensor(dv2_ptr, dv2_layout)

        # ===================== MMA setup =====================
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.kdh_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        update_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.update_mma_tiler[:2],
        )
        qdo_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.update_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        # ===================== SMEM layouts =====================
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            self.kdh_mma_tiler,
            self.io_dtype,
            self.k_stage,
        )
        dv_smem_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            self.dv_stage,
        )
        do_smem_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            self.do_stage,
        )
        q_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            self.q_stage,
        )
        w_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            self.w_stage,
        )
        dv2_smem_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            self.dv2_store_stage,
        )
        gk_smem_layout_staged = cute.make_layout(
            (self.BK, 1, self.gk_stage),
            stride=(1, self.BK, self.BK),
        )
        dht_smem_layout = (
            cute.make_layout((self.BV, self.BK), stride=(self.BK, 1))
            if cutlass.const_expr(self.transpose_state_layout)
            else cute.make_layout((self.BV, self.BK), stride=(1, self.BV))
        )
        dh_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.io_dtype,
            dh_smem_layout_enum,
            dh_tma_tile,
            self.dh_store_stage,
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_load_mcast_op = cpasync.CopyBulkTensorTileG2SMulticastOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # ===================== TMA descriptors =====================
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_mcast_op,
            k_tk,
            cute.slice_(k_smem_layout_staged, (None, None, 0)),
            (self.BT, self.BK),
            num_multicast=self.num_v_tile_mcast_ctas,
        )
        tma_atom_dv, tma_tensor_dv = cpasync.make_tiled_tma_atom(
            tma_load_op,
            dv_vt,
            cute.slice_(dv_smem_layout_staged, (None, None, 0)),
            (self.BV, self.BT),
        )
        tma_atom_do, tma_tensor_do = cpasync.make_tiled_tma_atom(
            tma_load_op,
            do_vt,
            cute.slice_(do_smem_layout_staged, (None, None, 0)),
            (self.BV, self.BT),
        )
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_mcast_op,
            q_kt,
            cute.slice_(q_smem_layout_staged, (None, None, 0)),
            (self.BK, self.BT),
            num_multicast=self.num_v_tile_mcast_ctas,
        )
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_mcast_op,
            w_kt,
            cute.slice_(w_smem_layout_staged, (None, None, 0)),
            (self.BK, self.BT),
            num_multicast=self.num_v_tile_mcast_ctas,
        )
        tma_atom_gk, tma_tensor_gk = cpasync.make_tiled_tma_atom(
            tma_load_op,
            gk_kt,
            cute.slice_(gk_smem_layout_staged, (None, None, 0)),
            (self.BK, 1),
        )
        tma_atom_dht, tma_tensor_dht = cpasync.make_tiled_tma_atom(
            tma_load_op,
            dht_tma,
            dht_smem_layout,
            (self.BV, self.BK),
        )
        tma_atom_dh0, tma_tensor_dh0 = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dh0_tma,
            dht_smem_layout,
            (self.BV, self.BK),
        )
        tma_atom_dh, tma_tensor_dh = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dh_tma,
            cute.slice_(dh_smem_layout_staged, (None, None, 0)),
            dh_tma_tile,
        )
        tma_atom_dv2, tma_tensor_dv2 = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dv2_vt,
            cute.slice_(dv2_smem_layout_staged, (None, None, 0)),
            (self.BV, self.BT),
        )
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0)))
        self.tma_dv_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(dv_smem_layout_staged, (None, None, 0)))
        self.tma_do_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(do_smem_layout_staged, (None, None, 0)))
        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(q_smem_layout_staged, (None, None, 0)))
        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(w_smem_layout_staged, (None, None, 0)))
        self.tma_gk_bytes = cute.size_in_bytes(cutlass.Float32, cute.slice_(gk_smem_layout_staged, (None, None, 0)))
        self.tma_dht_bytes = cute.size_in_bytes(cutlass.Float32, dht_smem_layout)

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            load_k_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_dv_mbar: cute.struct.MemRange[Int64, self.dv_stage * 2]
            load_do_mbar: cute.struct.MemRange[Int64, self.do_stage * 2]
            load_q_mbar: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_gk_mbar: cute.struct.MemRange[Int64, self.gk_stage * 2]
            load_dht_mbar: cute.struct.MemRange[Int64, 2]
            store_dh_mbar: cute.struct.MemRange[Int64, self.dh_store_stage * 2]
            store_dh0_mbar: cute.struct.MemRange[Int64, 2]
            store_dv2_mbar: cute.struct.MemRange[Int64, self.dv2_store_stage * 2]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDo: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(do_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.gk_stage],
                128,
            ]
            sG: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT * 2],
                128,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDv2: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dv2_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dh_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            g,
            dht,
            dh0,
            dv2,
            cu_seqlens,
            chunk_offsets,
            problem_size,
            tiled_mma,
            update_tiled_mma,
            qdo_tiled_mma,
            k_smem_layout_staged,
            dv_smem_layout_staged,
            do_smem_layout_staged,
            dv2_smem_layout_staged,
            q_smem_layout_staged,
            w_smem_layout_staged,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_dv,
            tma_tensor_dv,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_gk,
            tma_tensor_gk,
            tma_atom_dht,
            tma_tensor_dht,
            tma_atom_dh0,
            tma_tensor_dh0,
            dht_smem_layout,
            tma_atom_dh,
            tma_tensor_dh,
            dh_smem_layout_staged,
            tma_atom_dv2,
            tma_tensor_dv2,
        ).launch(
            grid=[cute.ceil_div(self.V, self.BV), N * self.H, 1],
            block=[self.num_threads, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g: cute.Tensor,
        dht: cute.Tensor,
        dh0: cute.Tensor,
        dv2: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32],
        tiled_mma: cute.TiledMma,
        update_tiled_mma: cute.TiledMma,
        qdo_tiled_mma: cute.TiledMma,
        k_smem_layout_staged: cute.ComposedLayout,
        dv_smem_layout_staged: cute.ComposedLayout,
        do_smem_layout_staged: cute.ComposedLayout,
        dv2_smem_layout_staged: cute.ComposedLayout,
        q_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_dv: cute.CopyAtom,
        tma_tensor_dv: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        tma_atom_dht: cute.CopyAtom,
        tma_tensor_dht: cute.Tensor,
        tma_atom_dh0: cute.CopyAtom,
        tma_tensor_dh0: cute.Tensor,
        dht_smem_layout: cute.Layout,
        tma_atom_dh: cute.CopyAtom,
        tma_tensor_dh: cute.Tensor,
        dh_smem_layout_staged: cute.ComposedLayout,
        tma_atom_dv2: cute.CopyAtom,
        tma_tensor_dv2: cute.Tensor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        B, T, N, NT_total = problem_size

        # ===================== Block indices =====================
        v_tile_idx, bh_idx, _ = cute.arch.block_idx()
        bidx = bh_idx // self.H
        hidx = bh_idx - bidx * self.H
        data_bidx = bidx
        tok_offset = Int32(0)
        seq_len = T
        NT = (T + self.BT - 1) // self.BT
        chunk_off = Int32(0)
        if cutlass.const_expr(self.is_varlen):
            data_bidx = Int32(0)
            tok_offset = cu_seqlens[bidx]
            seq_len = cu_seqlens[bidx + 1] - tok_offset
            NT = (seq_len + self.BT - 1) // self.BT
            chunk_off = chunk_offsets[bidx]
        v_tile_base = v_tile_idx * self.BV

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ===================== SMEM views =====================
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sDv = storage.sDv.get_tensor(dv_smem_layout_staged.outer, swizzle=dv_smem_layout_staged.inner)
        sDo = storage.sDo.get_tensor(do_smem_layout_staged.outer, swizzle=do_smem_layout_staged.inner)
        sGK = storage.sGK.get_tensor(cute.make_layout((self.BK, 1, self.gk_stage), stride=(1, self.BK, self.BK)))
        sG = storage.sG.get_tensor(cute.make_layout((self.BT, 2), stride=(1, self.BT)))
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sW = storage.sW.get_tensor(w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner)
        sDv2 = storage.sDv2.get_tensor(dv2_smem_layout_staged.outer, swizzle=dv2_smem_layout_staged.inner)
        sDh = storage.sDh.get_tensor(dh_smem_layout_staged.outer, swizzle=dh_smem_layout_staged.inner)
        sDht = cute.make_tensor(cute.recast_ptr(storage.sDh.data_ptr(), dtype=cutlass.Float32), dht_smem_layout)
        sDh0 = cute.make_tensor(cute.recast_ptr(storage.sK.data_ptr(), dtype=cutlass.Float32), dht_smem_layout)

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        v_tile_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk,
            cluster_coord_mnk,
            mode=0,
        )

        # ===================== Pipelines =====================
        load_k_P, load_k_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.num_v_tile_mcast_ctas),
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            mcast_mode_mn=(0, 1),
        ).make_participants()
        load_dv_P, load_dv_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.dv_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps),
            tx_count=self.tma_dv_bytes,
            barrier_storage=storage.load_dv_mbar.data_ptr(),
        ).make_participants()
        load_do_P, load_do_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.do_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps),
            tx_count=self.tma_do_bytes,
            barrier_storage=storage.load_do_mbar.data_ptr(),
        ).make_participants()
        load_q_P, load_q_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.num_v_tile_mcast_ctas),
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            mcast_mode_mn=(0, 1),
        ).make_participants()
        load_w_P, load_w_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.num_v_tile_mcast_ctas),
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            mcast_mode_mn=(0, 1),
        ).make_participants()
        if cutlass.const_expr(self.use_gk):
            load_gk_P, load_gk_C = pipeline.PipelineTmaAsync.create(
                num_stages=self.gk_stage,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(self.num_compute_warps),
                tx_count=self.tma_gk_bytes,
                barrier_storage=storage.load_gk_mbar.data_ptr(),
            ).make_participants()
        if cutlass.const_expr(self.use_dht):
            load_dht_P, load_dht_C = pipeline.PipelineTmaAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(self.num_compute_warps),
                tx_count=self.tma_dht_bytes,
                barrier_storage=storage.load_dht_mbar.data_ptr(),
            ).make_participants()
        if cutlass.const_expr(self.use_dh0):
            store_dh0_P, store_dh0_C = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(self.num_compute_threads),
                consumer_group=make_thread_cooperative_group(self.threads_per_warp),
                barrier_storage=storage.store_dh0_mbar.data_ptr(),
            ).make_participants()
        store_dh_P, store_dh_C = pipeline.PipelineAsync.create(
            num_stages=self.dh_store_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_threads),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.store_dh_mbar.data_ptr(),
        ).make_participants()
        store_dv2_P, store_dv2_C = pipeline.PipelineAsync.create(
            num_stages=self.dv2_store_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_threads),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.store_dv2_mbar.data_ptr(),
        ).make_participants()

        # ===================== TMA partitions =====================
        # Varlen shifts token-indexed tensors by tok_offset; dh uses chunk_off
        # because state storage is compact across sequences.
        if cutlass.const_expr(self.is_varlen):
            tma_tensor_k_use = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
            tma_tensor_dv_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_dv)
            tma_tensor_do_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_do)
            tma_tensor_q_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_q)
            tma_tensor_w_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_w)
            tma_tensor_dh_use = cute.domain_offset((0, 0, (chunk_off, 0, 0)), tma_tensor_dh)
            tma_tensor_dv2_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_dv2)
            if cutlass.const_expr(self.use_gk):
                tma_tensor_gk_use = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_gk)
            tma_tensor_dht_use = tma_tensor_dht
        else:
            tma_tensor_k_use = tma_tensor_k
            tma_tensor_dv_use = tma_tensor_dv
            tma_tensor_do_use = tma_tensor_do
            tma_tensor_q_use = tma_tensor_q
            tma_tensor_w_use = tma_tensor_w
            tma_tensor_dh_use = tma_tensor_dh
            tma_tensor_dv2_use = tma_tensor_dv2
            if cutlass.const_expr(self.use_gk):
                tma_tensor_gk_use = tma_tensor_gk
            tma_tensor_dht_use = tma_tensor_dht

        v_tile_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        v_tile_cta_crd = cluster_coord_mnk[0]
        k_gmem_tile = cute.flat_divide(
            tma_tensor_k_use[None, None, (hidx, data_bidx)],
            (self.BT, self.BK),
        )
        bSG_sK, bSG_gK = cpasync.tma_partition(
            tma_atom_k,
            v_tile_cta_crd,
            v_tile_cta_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(k_gmem_tile, 0, 2),
        )
        _, bSG_sDv, bSG_gDv = self._epilog_partition(
            tma_atom_dv, tma_tensor_dv_use[None, None, (hidx, data_bidx)], (self.BV, self.BT), sDv
        )
        _, bSG_sDo, bSG_gDo = self._epilog_partition(
            tma_atom_do, tma_tensor_do_use[None, None, (hidx, data_bidx)], (self.BV, self.BT), sDo
        )
        q_gmem_tile = cute.flat_divide(
            tma_tensor_q_use[None, None, (hidx, data_bidx)],
            (self.BK, self.BT),
        )
        bSG_sQ, bSG_gQ = cpasync.tma_partition(
            tma_atom_q,
            v_tile_cta_crd,
            v_tile_cta_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(q_gmem_tile, 0, 2),
        )
        w_gmem_tile = cute.flat_divide(
            tma_tensor_w_use[None, None, (hidx, data_bidx)],
            (self.BK, self.BT),
        )
        bSG_sW, bSG_gW = cpasync.tma_partition(
            tma_atom_w,
            v_tile_cta_crd,
            v_tile_cta_layout,
            cute.group_modes(sW, 0, 2),
            cute.group_modes(w_gmem_tile, 0, 2),
        )
        if cutlass.const_expr(self.use_gk):
            _, bSG_sGK, bSG_gGK = self._epilog_partition(
                tma_atom_gk, tma_tensor_gk_use[None, None, (hidx, data_bidx)], (self.BK, 1), sGK
            )
        if cutlass.const_expr(self.use_dht):
            _, bSG_sDht, bSG_gDht = self._epilog_partition(
                tma_atom_dht, tma_tensor_dht_use[None, None, (hidx, bidx)], (self.BV, self.BK), sDht
            )
        if cutlass.const_expr(self.use_dh0):
            _, bSG_sDh0, bSG_gDh0 = self._epilog_partition(
                tma_atom_dh0, tma_tensor_dh0[None, None, (hidx, bidx)], (self.BV, self.BK), sDh0
            )
        _, bSG_sDh, bSG_gDh = self._epilog_partition(
            tma_atom_dh, tma_tensor_dh_use[None, None, (None, hidx, data_bidx)], (self.BV, self.BK), sDh
        )
        _, bSG_sDv2, bSG_gDv2 = self._epilog_partition(
            tma_atom_dv2, tma_tensor_dv2_use[None, None, (hidx, data_bidx)], (self.BV, self.BT), sDv2
        )

        is_compute_warp = warp_idx < self.num_compute_warps
        local_tidx = tidx % self.num_compute_threads
        if is_compute_warp:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ===================== MMA fragments =====================
        thr_mma = tiled_mma.get_slice(local_tidx)
        update_thr_mma = update_tiled_mma.get_slice(local_tidx)

        tKsB = thr_mma.partition_B(sK)
        tKrB = thr_mma.make_fragment_B(tKsB)
        tUsB = update_thr_mma.partition_B(sQ)
        tWsB = update_thr_mma.partition_B(sW)
        tDv2sA = update_thr_mma.partition_A(sDv2)
        tDv2rA = update_thr_mma.make_fragment_A(tDv2sA)
        tUrB = update_thr_mma.make_fragment_B(tUsB)
        tWrB = update_thr_mma.make_fragment_B(tWsB)
        if cutlass.const_expr(self.use_g):
            qdo_thr_mma = qdo_tiled_mma.get_slice(local_tidx)
            qdo_tUsB = qdo_thr_mma.partition_B(sQ)
            qdo_tUrB = qdo_thr_mma.make_fragment_B(qdo_tUsB)
            rDo = thr_mma.make_fragment_C(thr_mma.partition_shape_C((self.BV, self.BT)))
        else:
            tUsDo = update_thr_mma.partition_A(sDo)
            tUrDo = update_thr_mma.make_fragment_A(tUsDo)

        cDV = cute.make_identity_tensor((self.BV, self.BT))
        tCcDV = thr_mma.partition_C(cDV)
        acc_dv = thr_mma.make_fragment_C(thr_mma.partition_shape_C((self.BV, self.BT)))

        cState = cute.make_identity_tensor((self.BV, self.BK))
        tUcState = update_thr_mma.partition_C(cState)
        state_shape = update_thr_mma.partition_shape_C((self.BV, self.BK))
        rState = update_thr_mma.make_fragment_C(state_shape)
        acc_qdo = update_thr_mma.make_fragment_C(state_shape)
        dh_smem_layout_enum = (
            utils.LayoutEnum.ROW_MAJOR if cutlass.const_expr(self.transpose_state_layout) else utils.LayoutEnum.COL_MAJOR
        )
        dh_copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            dh_smem_layout_enum,
            elem_ty_d=self.io_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        dh_copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                dh_smem_layout_enum.is_m_major_c(),
                4,
            ),
            self.io_dtype,
        )
        tiled_copy_dh_atom = cute.make_tiled_copy_C_atom(dh_copy_atom, update_tiled_mma)
        tiled_copy_dh_r2s = cute.make_tiled_copy_S(dh_copy_atom_r2s, tiled_copy_dh_atom)
        thr_copy_dh_r2s = tiled_copy_dh_r2s.get_slice(local_tidx)
        tRS_sDh = thr_copy_dh_r2s.partition_D(sDh)
        rDh_shape = cute.shape(thr_copy_dh_r2s.partition_S(sDh))
        tRS_rDh_layout = cute.make_layout(rDh_shape[:3])

        # =========================================================================
        # WARP SPECIALIZATION
        #   load_warp_id         : preloads K, dv, and optional gk for the next reverse chunk
        #   load_current_warp_id : loads do, q, and w for the current reverse chunk
        #   compute warps        : carry dh in registers and run WGMMA
        #   store_warp_id        : stores dh and dv2 after compute warps publish SMEM tiles
        # =========================================================================
        # ===== Reverse chunk loop =====
        # Pipeline: preload(prev chunk) -> Phase 1 (publish dh + K@dh)
        # -> Phase 2 (dv2) -> Phase 3 (QDO + decay) -> Phase 4 (WDV + dh update).
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_dv)
            if cutlass.const_expr(self.use_gk):
                cpasync.prefetch_descriptor(tma_atom_gk)
            if cutlass.const_expr(self.use_dht):
                cpasync.prefetch_descriptor(tma_atom_dht)
                dht_h = load_dht_P.acquire_and_advance()
                cute.copy(
                    tma_atom_dht,
                    bSG_gDht[(None, v_tile_idx, 0)],
                    bSG_sDht[None],
                    tma_bar_ptr=dht_h.barrier,
                )

            if NT > 0:
                first_chunk = NT - 1
                k_h = load_k_P.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    bSG_gK[(None, first_chunk, 0)],
                    bSG_sK[None, k_h.index],
                    tma_bar_ptr=k_h.barrier,
                    mcast_mask=v_tile_mcast_mask,
                )
                dv_h = load_dv_P.acquire_and_advance()
                cute.copy(
                    tma_atom_dv,
                    bSG_gDv[(None, v_tile_idx, first_chunk)],
                    bSG_sDv[None, dv_h.index],
                    tma_bar_ptr=dv_h.barrier,
                )
                if cutlass.const_expr(self.use_gk):
                    gk_h = load_gk_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_gk,
                        bSG_gGK[(None, 0, seq_len - 1)],
                        bSG_sGK[None, gk_h.index],
                        tma_bar_ptr=gk_h.barrier,
                    )

            for chunk_rev in cutlass.range(0, NT, unroll=0):
                chunk_idx = NT - 1 - chunk_rev
                next_chunk_idx = chunk_idx - 1
                if next_chunk_idx >= 0:
                    k_h = load_k_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        bSG_gK[(None, next_chunk_idx, 0)],
                        bSG_sK[None, k_h.index],
                        tma_bar_ptr=k_h.barrier,
                        mcast_mask=v_tile_mcast_mask,
                    )
                    dv_h = load_dv_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_dv,
                        bSG_gDv[(None, v_tile_idx, next_chunk_idx)],
                        bSG_sDv[None, dv_h.index],
                        tma_bar_ptr=dv_h.barrier,
                    )
                    if cutlass.const_expr(self.use_gk):
                        next_gk_idx = cutlass.min(next_chunk_idx * self.BT + self.BT, seq_len) - 1
                        gk_h = load_gk_P.acquire_and_advance()
                        cute.copy(
                            tma_atom_gk,
                            bSG_gGK[(None, 0, next_gk_idx)],
                            bSG_sGK[None, gk_h.index],
                            tma_bar_ptr=gk_h.barrier,
                        )

        elif warp_idx == self.load_current_warp_id:
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_w)

            for chunk_rev in cutlass.range(0, NT, unroll=0):
                chunk_idx = NT - 1 - chunk_rev
                do_h = load_do_P.acquire_and_advance()
                cute.copy(
                    tma_atom_do, bSG_gDo[(None, v_tile_idx, chunk_idx)], bSG_sDo[None, do_h.index], tma_bar_ptr=do_h.barrier
                )
                q_h = load_q_P.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    bSG_gQ[(None, 0, chunk_idx)],
                    bSG_sQ[None, q_h.index],
                    tma_bar_ptr=q_h.barrier,
                    mcast_mask=v_tile_mcast_mask,
                )
                w_h = load_w_P.acquire_and_advance()
                cute.copy(
                    tma_atom_w,
                    bSG_gW[(None, 0, chunk_idx)],
                    bSG_sW[None, w_h.index],
                    tma_bar_ptr=w_h.barrier,
                    mcast_mask=v_tile_mcast_mask,
                )
        elif is_compute_warp:
            # Initialize carried dh state in registers.  dht is loaded by the
            # load warp into the sDh backing buffer before sDh is used for
            # per-chunk output stores.
            if cutlass.const_expr(self.use_dht):
                dht_h = load_dht_C.wait_and_advance()
            for ei in cutlass.range(cute.size(rState), unroll_full=True):
                v_rel, k_rel = tUcState[ei]
                init = Float32(0.0)
                if cutlass.const_expr(self.use_dht):
                    init = sDht[v_rel, k_rel].to(self.acc_dtype)
                rState[ei] = init
            if cutlass.const_expr(self.use_dht):
                dht_h.release()

            for chunk_rev in cutlass.range(0, NT, unroll=0):
                chunk_idx = NT - 1 - chunk_rev
                chunk_start = chunk_idx * self.BT
                chunk_end = cutlass.min(chunk_start + self.BT, seq_len)
                remaining = chunk_end - chunk_start

                # ========================================
                # Phase 1: Publish dh + start K @ dh
                # ========================================
                # Publish carried dh to the store pipeline before the GEMM chain,
                # matching chunk_delta_h.py's h_out overlap pattern.
                rState_bf16 = cute.make_rmem_tensor(rState.shape, self.io_dtype)
                rState_bf16.store(rState.load().to(self.io_dtype))
                dh_h = store_dh_P.acquire_and_advance()
                tRS_rState = tiled_copy_dh_r2s.retile(rState_bf16)
                tRS_rDh_out = cute.make_rmem_tensor_like(tRS_rDh_layout, self.io_dtype)
                tRS_rDh_out.store(tRS_rState.load())
                cute.copy(
                    tiled_copy_dh_r2s,
                    tRS_rDh_out,
                    tRS_sDh[(None, None, None, dh_h.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                dh_h.commit()

                # dv2 = dv + K @ dh. Compute the equivalent dh @ K^T tile so
                # the register-carried state can feed WGMMA as an RMEM A operand.
                acc_dv.fill(0.0)
                k_wait = load_k_C.wait_and_advance()
                rState_op = self.make_acc_into_op(rState_bf16, tiled_mma.tv_layout_A, self.io_dtype)
                cute.nvgpu.warpgroup.fence()
                for kp in cutlass.range(cute.size(tKrB, mode=[2]), unroll_full=True):
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        tiled_mma,
                        acc_dv,
                        rState_op[None, None, kp],
                        tKrB[None, None, kp, k_wait.index],
                        acc_dv,
                    )
                cute.nvgpu.warpgroup.commit_group()
                if cutlass.const_expr(self.use_g):
                    if local_tidx < self.BT:
                        t_idx = chunk_start + local_tidx
                        g_last = g[data_bidx, tok_offset + chunk_end - 1, hidx].to(self.acc_dtype)
                        g_decay = Float32(0.0)
                        g_exp = Float32(0.0)
                        if t_idx < seq_len:
                            g_cur = g[data_bidx, tok_offset + t_idx, hidx].to(self.acc_dtype)
                            if cutlass.const_expr(self.use_exp2):
                                g_decay = cute.exp2(g_last - g_cur, fastmath=self.use_fast_math)
                                g_exp = cute.exp2(g_cur, fastmath=self.use_fast_math)
                            else:
                                g_decay = cute.exp(g_last - g_cur, fastmath=self.use_fast_math)
                                g_exp = cute.exp(g_cur, fastmath=self.use_fast_math)
                        sG[local_tidx, 0] = g_decay
                        sG[local_tidx, 1] = g_exp
                    cute.arch.barrier(barrier_id=2, number_of_threads=self.num_compute_threads)
                    do_wait_early = load_do_C.wait_and_advance()
                    q_wait_early = load_q_C.wait_and_advance()
                    for ei in cutlass.range(cute.size(rDo), unroll_full=True):
                        v_rel, t_rel = tCcDV[ei]
                        t_idx = chunk_start + t_rel
                        do_scaled = Float32(0.0)
                        if t_idx < seq_len:
                            do_scaled = sDo[v_rel, t_rel, do_wait_early.index].to(self.acc_dtype) * sG[t_rel, 1]
                        rDo[ei] = do_scaled
                    rDo_op = self.make_acc_into_op(rDo, qdo_tiled_mma.tv_layout_A, self.io_dtype)
                    do_wait_early.release()

                    acc_qdo.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(qdo_tUrB, mode=[2]), unroll_full=True):
                        qdo_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            qdo_tiled_mma,
                            acc_qdo,
                            rDo_op[None, None, kp],
                            qdo_tUrB[None, None, kp, q_wait_early.index],
                            acc_qdo,
                        )
                    cute.nvgpu.warpgroup.commit_group()

                    g_state_scale = sG[remaining - 1, 1].to(self.acc_dtype)
                    for ei in cutlass.range(cute.size(rState), unroll_full=True):
                        rState[ei] = rState[ei] * g_state_scale
                    if cutlass.const_expr(self.use_gk):
                        gk_wait_early = load_gk_C.wait_and_advance()
                        gk_last = sGK[local_tidx, 0, gk_wait_early.index].to(self.acc_dtype)
                        if cutlass.const_expr(self.use_exp2):
                            k_decay = cute.exp2(gk_last, fastmath=self.use_fast_math)
                        else:
                            k_decay = cute.exp(gk_last, fastmath=self.use_fast_math)
                        sGK[local_tidx, 0, gk_wait_early.index] = k_decay
                        self.gk_precompute_bar.arrive_and_wait()
                        for ei in cutlass.range(cute.size(rState), unroll_full=True):
                            v_rel, k_rel = tUcState[ei]
                            rState[ei] = rState[ei] * sGK[k_rel, 0, gk_wait_early.index]
                        gk_wait_early.release()
                    cute.nvgpu.warpgroup.wait_group(1)
                else:
                    # Phase 3 is independent of K@dh, so overlap QDO and optional gk decay
                    # with the first GEMM in the no-scalar-g fast path.  For
                    # varlen tails, zero padded do positions before QDO so TMA
                    # overfetch into the next sequence cannot contribute.
                    do_wait_early = load_do_C.wait_and_advance()
                    q_wait_early = load_q_C.wait_and_advance()
                    if cutlass.const_expr(self.use_gk):
                        gk_wait_early = load_gk_C.wait_and_advance()
                    if cutlass.const_expr(self.is_varlen):
                        if remaining < self.BT:
                            linear_do = local_tidx
                            while linear_do < self.BV * self.BT:
                                v_rel = linear_do // self.BT
                                t_rel = linear_do - v_rel * self.BT
                                t_idx = chunk_start + t_rel
                                do_scaled = Float32(0.0)
                                if t_idx < seq_len:
                                    do_scaled = sDo[v_rel, t_rel, do_wait_early.index].to(self.acc_dtype)
                                sDo[v_rel, t_rel, do_wait_early.index] = do_scaled.to(self.io_dtype)
                                linear_do += self.num_compute_threads
                            cute.arch.barrier(barrier_id=2, number_of_threads=self.num_compute_threads)
                    acc_qdo.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tUrDo, mode=[2]), unroll_full=True):
                        update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            update_tiled_mma,
                            acc_qdo,
                            tUrDo[None, None, kp, do_wait_early.index],
                            tUrB[None, None, kp, q_wait_early.index],
                            acc_qdo,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    if cutlass.const_expr(self.use_gk):
                        gk_last = sGK[local_tidx, 0, gk_wait_early.index].to(self.acc_dtype)
                        if cutlass.const_expr(self.use_exp2):
                            k_decay = cute.exp2(gk_last, fastmath=self.use_fast_math)
                        else:
                            k_decay = cute.exp(gk_last, fastmath=self.use_fast_math)
                        sGK[local_tidx, 0, gk_wait_early.index] = k_decay
                        self.gk_precompute_bar.arrive_and_wait()
                    cute.nvgpu.warpgroup.wait_group(1)
                k_wait.release()

                # ========================================
                # Phase 2: dv2 = dv + K @ dh
                # ========================================
                dv_wait = load_dv_C.wait_and_advance()
                dv_stage = dv_wait.index
                dv2_store_h = store_dv2_P.acquire_and_advance()
                dv2_stage = dv2_store_h.index
                for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                    v_rel, t_rel = tCcDV[ei]
                    t_idx = chunk_start + t_rel
                    out = Float32(0.0)
                    if t_idx < seq_len:
                        out = acc_dv[ei]
                        if cutlass.const_expr(self.use_g):
                            out = out * sG[t_rel, 0]
                        out = out + sDv[v_rel, t_rel, dv_stage].to(self.acc_dtype)
                    out_bf16 = out.to(self.io_dtype)
                    sDv2[v_rel, t_rel, dv2_stage] = out_bf16
                    if remaining < self.BT and t_idx < seq_len:
                        dv2[data_bidx, hidx, tok_offset + chunk_start + t_rel, v_tile_base + v_rel] = out_bf16
                cute.arch.fence_proxy("async.shared", space="cta")
                dv2_store_h.commit()
                dv_wait.release()

                # ========================================
                # Phase 3/4: dh += scale * do^T @ q - dv2^T @ w
                # ========================================
                if cutlass.const_expr(not self.use_g):
                    w_wait = load_w_C.wait_and_advance()
                    acc_wdv = update_thr_mma.make_fragment_C(state_shape)
                    acc_wdv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2rA, mode=[2]), unroll_full=True):
                        update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            update_tiled_mma,
                            acc_wdv,
                            tDv2rA[None, None, kp, dv2_stage],
                            tWrB[None, None, kp, w_wait.index],
                            acc_wdv,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    q_wait_early.release()

                    for ei in cutlass.range(cute.size(rState), unroll_full=True):
                        update = acc_qdo[ei] * Float32(self.scale) - acc_wdv[ei]
                        if cutlass.const_expr(self.use_gk):
                            v_rel, k_rel = tUcState[ei]
                            rState[ei] = rState[ei] * sGK[k_rel, 0, gk_wait_early.index] + update
                        else:
                            rState[ei] = rState[ei] + update
                    w_wait.release()
                    do_wait_early.release()
                    if cutlass.const_expr(self.use_gk):
                        gk_wait_early.release()
                else:
                    # Phase 4: WDV and final dh update.
                    w_wait = load_w_C.wait_and_advance()
                    acc_wdv = update_thr_mma.make_fragment_C(state_shape)
                    acc_wdv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2rA, mode=[2]), unroll_full=True):
                        update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            update_tiled_mma,
                            acc_wdv,
                            tDv2rA[None, None, kp, dv2_stage],
                            tWrB[None, None, kp, w_wait.index],
                            acc_wdv,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    q_wait_early.release()

                    for ei in cutlass.range(cute.size(rState), unroll_full=True):
                        update = acc_qdo[ei] * Float32(self.scale) - acc_wdv[ei]
                        rState[ei] = rState[ei] + update
                    w_wait.release()

            if cutlass.const_expr(self.use_dh0):
                dh0_h = store_dh0_P.acquire_and_advance()
                for ei in cutlass.range(cute.size(rState), unroll_full=True):
                    v_rel, k_rel = tUcState[ei]
                    sDh0[v_rel, k_rel] = rState[ei]
                cute.arch.fence_proxy("async.shared", space="cta")
                dh0_h.commit()

        elif warp_idx == self.store_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_dv2)
            if cutlass.const_expr(self.use_dh0):
                cpasync.prefetch_descriptor(tma_atom_dh0)

            for chunk_rev in cutlass.range(0, NT, unroll=0):
                chunk_idx = NT - 1 - chunk_rev
                chunk_start = chunk_idx * self.BT
                chunk_end = cutlass.min(chunk_start + self.BT, seq_len)
                remaining = chunk_end - chunk_start

                dh_h = store_dh_C.wait_and_advance()
                cute.copy(tma_atom_dh, bSG_sDh[None, dh_h.index], bSG_gDh[(None, v_tile_idx, 0, chunk_idx)])
                cute.arch.cp_async_bulk_commit_group()

                dv2_store_h = store_dv2_C.wait_and_advance()
                # Tail chunks skip TMA because the tile would cross sequence
                # bounds. The store pipeline itself keeps sDv2 stages from
                # being overwritten before this warp releases them.
                if remaining >= self.BT:
                    cute.copy(
                        tma_atom_dv2,
                        bSG_sDv2[None, dv2_store_h.index],
                        bSG_gDv2[(None, v_tile_idx, chunk_idx)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                dh_h.release()
                dv2_store_h.release()

            if cutlass.const_expr(self.use_dh0):
                dh0_h = store_dh0_C.wait_and_advance()
                cute.copy(tma_atom_dh0, bSG_sDh0[None], bSG_gDh0[(None, v_tile_idx, 0)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                dh0_h.release()

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
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

    @staticmethod
    def _convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, element_type):
        operand = cute.make_rmem_tensor_like(
            self._convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            element_type,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(element_type))
        return operand


@functools.lru_cache(maxsize=64)
def _compile_bwd_dhu_sm90(
    H: int,
    K: int,
    V: int,
    is_varlen: bool,
    use_g: bool,
    use_gk: bool,
    use_dht: bool,
    use_dh0: bool,
    use_exp2: bool,
    transpose_state_layout: bool,
    scale: float,
):
    """Compile one bwd_dhu kernel variant.

    B, T, N, and NT are symbolic during compilation and are passed as runtime
    problem_size values, matching the forward kernel's dynamic-shape pattern.
    """
    kernel = ChunkDeltaRuleBwdDHUSm90(
        num_heads=H,
        head_dim_k=K,
        head_dim_v=V,
        is_varlen=is_varlen,
        use_g=use_g,
        use_gk=use_gk,
        use_dht=use_dht,
        use_dh0=use_dh0,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
        scale=scale,
        use_fast_math=USE_FAST_MATH,
    )

    sym_b = cute.sym_int()
    sym_t = cute.sym_int()
    sym_n = cute.sym_int()
    sym_nt = cute.sym_int()
    sym_meta = cute.sym_int()

    q_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_t, H, K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    k_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_t, H, K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    w_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_t, H, K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    do_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_t, H, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    dv_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_t, H, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    dv2_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, H, sym_t, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    g_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_b, sym_t, H),
        stride_order=(2, 1, 0),
        assumed_align=128,
    )
    gk_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_b, sym_t, H, K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    if transpose_state_layout:
        dht_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_n, H, V, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh0_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_n, H, V, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_b, sym_nt, H, V, K),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )
    else:
        dht_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_n, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh0_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_n, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_b, sym_nt, H, K, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_meta,), assumed_align=128)
    offsets_fake = make_fake_compact_tensor(cutlass.Int32, (sym_meta,), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel,
        q_fake,
        k_fake,
        w_fake,
        g_fake,
        gk_fake,
        dht_fake,
        dh0_fake,
        do_fake,
        dh_fake,
        dv_fake,
        dv2_fake,
        cu_fake,
        offsets_fake,
        (Int32(1), Int32(1), Int32(1), Int32(1)),
        stream_fake,
        options="--enable-tvm-ffi",
    )


def chunk_gated_delta_rule_bwd_dhu_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = BT,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """FLA-compatible wrapper for the SM90 WGMMA bwd_dhu path."""
    assert_hopper(q.device)
    if chunk_size != BT:
        raise NotImplementedError(f"SM90 bwd_dhu only supports chunk_size={BT}.")

    B, T, H, K = q.shape
    V = do.shape[-1]
    is_varlen = cu_seqlens is not None
    if is_varlen and B != 1:
        raise ValueError("varlen mode expects packed inputs with shape [1, total_T, H, D].")
    if K != 128 or V != 128:
        raise NotImplementedError(f"SM90 bwd_dhu currently aligns with fwd and only supports K=V=128, got K={K}, V={V}.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("q, k, and w must be bfloat16 for the SM90 bwd_dhu path.")
    if do.dtype != torch.bfloat16 or dv.dtype != torch.bfloat16:
        raise TypeError("do and dv must be bfloat16 for the SM90 bwd_dhu path.")
    if not q.is_contiguous() or not k.is_contiguous() or not w.is_contiguous():
        raise ValueError("q, k, and w must be contiguous.")
    if not do.is_contiguous() or not dv.is_contiguous():
        raise ValueError("do and dv must be contiguous.")
    if h0 is not None and (h0.dtype != torch.float32 or not h0.is_contiguous()):
        raise ValueError("h0 must be contiguous float32.")
    if cu_seqlens is not None and (cu_seqlens.device != q.device or not cu_seqlens.is_contiguous()):
        raise ValueError("cu_seqlens must be contiguous and on the same CUDA device as q.")
    if chunk_indices is not None and (chunk_indices.device != q.device or not chunk_indices.is_contiguous()):
        raise ValueError("chunk_indices must be contiguous and on the same CUDA device as q.")
    if chunk_offsets is not None and (chunk_offsets.device != q.device or not chunk_offsets.is_contiguous()):
        raise ValueError("chunk_offsets must be contiguous and on the same CUDA device as q.")

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT).int()
        elif chunk_offsets.dtype != torch.int32:
            chunk_offsets = chunk_offsets.int()
        cu_seqlens_arg = cu_seqlens.int() if cu_seqlens.dtype != torch.int32 else cu_seqlens
    else:
        N = B
        NT = (T + BT - 1) // BT
        cu_seqlens_arg = torch.empty(B + 1, device=q.device, dtype=torch.int32)
        chunk_offsets = torch.empty(B + 1, device=q.device, dtype=torch.int32)
    scale_value = 1.0 if scale is None else float(scale)

    state_shape = (N, H, V, K) if transpose_state_layout else (N, H, K, V)
    dh = q.new_empty(B, NT, H, V, K) if transpose_state_layout else q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2_storage = q.new_empty(B, H, T, V)
    dv2 = dv2_storage.permute(0, 2, 1, 3)

    g_arg = g if g is not None else torch.empty(B, T, H, device=q.device, dtype=torch.float32)
    gk_arg = gk if gk is not None else torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    dht_arg = dht if dht is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    dh0_arg = dh0 if dh0 is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    if g is not None and (g.dtype != torch.float32 or not g.is_contiguous()):
        raise ValueError("g must be contiguous float32.")
    if g is not None and tuple(g.shape) != (B, T, H):
        raise ValueError(f"g must have shape {(B, T, H)}, got {tuple(g.shape)}.")
    if gk is not None and (gk.dtype != torch.float32 or not gk.is_contiguous()):
        raise ValueError("gk must be contiguous float32.")
    if gk is not None and tuple(gk.shape) != (B, T, H, K):
        raise ValueError(f"gk must have shape {(B, T, H, K)}, got {tuple(gk.shape)}.")
    if dht is not None and (dht.dtype != torch.float32 or not dht.is_contiguous()):
        raise ValueError("dht must be contiguous float32.")
    if dht is not None and tuple(dht.shape) != state_shape:
        raise ValueError(f"dht must have shape {state_shape} for this state layout, got {tuple(dht.shape)}.")
    if h0 is not None and tuple(h0.shape) != state_shape:
        raise ValueError(f"h0 must have shape {state_shape} for this state layout, got {tuple(h0.shape)}.")

    compiled = _compile_bwd_dhu_sm90(
        H,
        K,
        V,
        is_varlen,
        g is not None,
        gk is not None,
        dht is not None,
        h0 is not None,
        use_exp2,
        transpose_state_layout,
        scale_value,
    )
    problem_size = (Int32(B), Int32(T), Int32(N), Int32(NT))
    compiled(q, k, w, g_arg, gk_arg, dht_arg, dh0_arg, do, dh, dv, dv2_storage, cu_seqlens_arg, chunk_offsets, problem_size)
    return dh, dh0, dv2


chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu_sm90
