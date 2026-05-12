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
SM90 CuTe DSL implementation for chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.

This is the first Hopper tensor-core path:
  - fixed chunk size BT=64
  - BV=64, matching cula/ops/chunk_delta_h.py
  - non-varlen tensors [B, T, H, D]
  - state layout [B, NT, H, K, V] or [B, NT, H, V, K]
  - non-persistent scheduling

The recurrence is the Triton bwd_dhu recurrence:
    dv2 = dv + K @ dh
    dh  = decay(dh) + scale * Q^T @ do - W^T @ dv2

Each CTA owns one BV tile and one (batch, head).  WGMMA computes the three
64x64 GEMMs per chunk; scalar CUDA code only stages operands and applies the
elementwise recurrence.
"""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

from cula.utils import USE_FAST_MATH, assert_hopper

BT = 64
BV = 64
NUM_THREADS = 128


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkDeltaRuleBwdDHUSm90:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_sequences: int,
        total_chunks: int,
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
        assert head_dim_k in (64, 128, 256), f"SM90 bwd_dhu supports K in {{64, 128, 256}}, got {head_dim_k}"
        assert head_dim_v % BV == 0, f"SM90 bwd_dhu tensor-core path requires V to be a multiple of {BV}, got {head_dim_v}"
        self.B = batch_size
        self.T = seq_len
        self.N = num_sequences
        self.NT = total_chunks
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
        self.num_v_tiles = (head_dim_v + BV - 1) // BV
        self.num_threads = NUM_THREADS
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 1024

        self.mma_tiler = (BT, BV, head_dim_k)
        self.update_mma_tiler = (BV, head_dim_k, BT)
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

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

        NT_total = self.NT

        q_layout = cute.make_layout(
            (self.B, self.T, self.H, self.K),
            stride=(self.T * self.H * self.K, self.H * self.K, self.K, 1),
        )
        q = cute.make_tensor(q_ptr, q_layout)
        k = cute.make_tensor(k_ptr, q_layout)
        w = cute.make_tensor(w_ptr, q_layout)

        v_layout = cute.make_layout(
            (self.B, self.T, self.H, self.V),
            stride=(self.T * self.H * self.V, self.H * self.V, self.V, 1),
        )
        do = cute.make_tensor(do_ptr, v_layout)
        dv = cute.make_tensor(dv_ptr, v_layout)
        dv2 = cute.make_tensor(dv2_ptr, v_layout)

        g_layout = cute.make_layout(
            (self.B, self.T, self.H),
            stride=(self.T * self.H, self.H, 1),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        gk_layout = cute.make_layout(
            (self.B, self.T, self.H, self.K),
            stride=(self.T * self.H * self.K, self.H * self.K, self.K, 1),
        )
        gk = cute.make_tensor(gk_ptr, gk_layout)
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((self.N + 1,)))
        chunk_offsets = cute.make_tensor(chunk_offsets_ptr, cute.make_layout((self.N + 1,)))

        if cutlass.const_expr(self.transpose_state_layout):
            state_layout = cute.make_layout(
                (self.B, NT_total, self.H, self.V, self.K),
                stride=(
                    NT_total * self.H * self.K * self.V,
                    self.H * self.K * self.V,
                    self.K * self.V,
                    self.K,
                    1,
                ),
            )
        else:
            state_layout = cute.make_layout(
                (self.B, NT_total, self.H, self.K, self.V),
                stride=(
                    NT_total * self.H * self.K * self.V,
                    self.H * self.K * self.V,
                    self.K * self.V,
                    self.V,
                    1,
                ),
            )
        dh = cute.make_tensor(dh_ptr, state_layout)

        if cutlass.const_expr(self.transpose_state_layout):
            final_layout = cute.make_layout(
                (self.N, self.H, self.V, self.K),
                stride=(self.H * self.K * self.V, self.K * self.V, self.K, 1),
            )
        else:
            final_layout = cute.make_layout(
                (self.N, self.H, self.K, self.V),
                stride=(self.H * self.K * self.V, self.K * self.V, self.V, 1),
            )
        dht = cute.make_tensor(dht_ptr, final_layout)
        dh0 = cute.make_tensor(dh0_ptr, final_layout)

        tk_layout = cute.make_layout(
            (self.T, self.K, (self.H, self.B)), stride=(self.H * self.K, 1, (self.K, self.T * self.H * self.K))
        )
        k_tk = cute.make_tensor(k_ptr, tk_layout)

        kt_layout = cute.make_layout(
            (self.K, self.T, (self.H, self.B)), stride=(1, self.H * self.K, (self.K, self.T * self.H * self.K))
        )
        q_kt = cute.make_tensor(q_ptr, kt_layout)
        w_kt = cute.make_tensor(w_ptr, kt_layout)

        vt_layout = cute.make_layout(
            (self.V, self.T, (self.H, self.B)), stride=(1, self.H * self.V, (self.V, self.T * self.H * self.V))
        )
        do_vt = cute.make_tensor(do_ptr, vt_layout)
        dv_vt = cute.make_tensor(dv_ptr, vt_layout)
        dv2_vt = cute.make_tensor(dv2_ptr, vt_layout)

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.mma_tiler[:2],
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

        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR,
            self.mma_tiler,
            self.io_dtype,
            1,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            self.mma_tiler,
            self.io_dtype,
            1,
        )
        update_a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        update_b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        dv_smem_layout_staged = cute.make_layout((self.BV, self.BT, 1), stride=(1, self.BV, self.BV * self.BT))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            k_tk,
            cute.slice_(a_smem_layout_staged, (None, None, 0)),
            (self.BT, self.K),
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
            cute.slice_(update_a_smem_layout_staged, (None, None, 0)),
            (self.BV, self.BT),
        )
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            q_kt,
            cute.slice_(update_b_smem_layout_staged, (None, None, 0)),
            (self.K, self.BT),
        )
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            w_kt,
            cute.slice_(update_b_smem_layout_staged, (None, None, 0)),
            (self.K, self.BT),
        )
        tma_atom_dv2, tma_tensor_dv2 = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dv2_vt,
            cute.slice_(dv_smem_layout_staged, (None, None, 0)),
            (self.BV, self.BT),
        )
        self.tma_kdv_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(a_smem_layout_staged, (None, None, 0))
        ) + cute.size_in_bytes(self.io_dtype, cute.slice_(dv_smem_layout_staged, (None, None, 0)))
        self.tma_qdo_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(update_a_smem_layout_staged, (None, None, 0))
        ) + cute.size_in_bytes(self.io_dtype, cute.slice_(update_b_smem_layout_staged, (None, None, 0)))
        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, cute.slice_(update_b_smem_layout_staged, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            load_kdv_mbar: cute.struct.MemRange[Int64, 2]
            load_qdo_mbar: cute.struct.MemRange[Int64, 2]
            load_w_mbar: cute.struct.MemRange[Int64, 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sUA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(update_a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sUB: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(update_b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sDv2T: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            q,
            k,
            w,
            g,
            gk,
            dht,
            dh0,
            do,
            dh,
            dv,
            dv2,
            cu_seqlens,
            chunk_offsets,
            tiled_mma,
            update_tiled_mma,
            a_smem_layout_staged,
            b_smem_layout_staged,
            update_a_smem_layout_staged,
            update_b_smem_layout_staged,
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
            tma_atom_dv2,
            tma_tensor_dv2,
        ).launch(
            grid=[cute.ceil_div(self.V, self.BV), self.N * self.H, 1],
            block=[self.num_threads, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        w: cute.Tensor,
        g: cute.Tensor,
        gk: cute.Tensor,
        dht: cute.Tensor,
        dh0: cute.Tensor,
        do: cute.Tensor,
        dh: cute.Tensor,
        dv: cute.Tensor,
        dv2: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        tiled_mma: cute.TiledMma,
        update_tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        update_a_smem_layout_staged: cute.ComposedLayout,
        update_b_smem_layout_staged: cute.ComposedLayout,
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
        tma_atom_dv2: cute.CopyAtom,
        tma_tensor_dv2: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        i_v_tile, i_bh, _ = cute.arch.block_idx()
        i_n = i_bh // self.H
        i_h = i_bh - i_n * self.H
        data_b = i_n
        state_b = i_n
        seq_start = Int32(0)
        seq_len = self.T
        NT = (self.T + self.BT - 1) // self.BT
        chunk_base = Int32(0)
        if cutlass.const_expr(self.is_varlen):
            data_b = Int32(0)
            state_b = Int32(0)
            seq_start = cu_seqlens[i_n]
            seq_len = cu_seqlens[i_n + 1] - seq_start
            NT = (seq_len + self.BT - 1) // self.BT
            chunk_base = chunk_offsets[i_n]
        v_base = i_v_tile * self.BV

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sUA = storage.sUA.get_tensor(update_a_smem_layout_staged.outer, swizzle=update_a_smem_layout_staged.inner)
        sUB = storage.sUB.get_tensor(update_b_smem_layout_staged.outer, swizzle=update_b_smem_layout_staged.inner)
        sDv2T = storage.sDv2T.get_tensor(cute.make_layout((BV, BT, 1), stride=(1, BV, BV * BT)))

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_dv)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_dv2)

        load_kdv_P, load_kdv_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_threads // 32),
            tx_count=self.tma_kdv_bytes,
            barrier_storage=storage.load_kdv_mbar.data_ptr(),
        ).make_participants()
        load_qdo_P, load_qdo_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_threads // 32),
            tx_count=self.tma_qdo_bytes,
            barrier_storage=storage.load_qdo_mbar.data_ptr(),
        ).make_participants()
        load_w_P, load_w_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.num_threads // 32),
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar.data_ptr(),
        ).make_participants()

        thr_mma = tiled_mma.get_slice(tidx)
        update_thr_mma = update_tiled_mma.get_slice(tidx)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA)
        tCrB = thr_mma.make_fragment_B(tCsB)
        tUsA = update_thr_mma.partition_A(sUA)
        tUsB = update_thr_mma.partition_B(sUB)
        tUrA = update_thr_mma.make_fragment_A(tUsA)
        tUrB = update_thr_mma.make_fragment_B(tUsB)

        cDV = cute.make_identity_tensor((BT, BV))
        tCcDV = thr_mma.partition_C(cDV)
        acc_dv = thr_mma.make_fragment_C(thr_mma.partition_shape_C((BT, BV)))

        cState = cute.make_identity_tensor((BV, self.K))
        tUcState = update_thr_mma.partition_C(cState)
        rState = update_thr_mma.make_fragment_C(update_thr_mma.partition_shape_C((BV, self.K)))
        acc_qdo = update_thr_mma.make_fragment_C(update_thr_mma.partition_shape_C((BV, self.K)))
        acc_wdv = update_thr_mma.make_fragment_C(update_thr_mma.partition_shape_C((BV, self.K)))

        if cutlass.const_expr(self.is_varlen):
            tma_tensor_k_use = cute.domain_offset((seq_start, 0, (0, 0)), tma_tensor_k)
            tma_tensor_dv_use = cute.domain_offset((0, seq_start, (0, 0)), tma_tensor_dv)
            tma_tensor_do_use = cute.domain_offset((0, seq_start, (0, 0)), tma_tensor_do)
            tma_tensor_q_use = cute.domain_offset((0, seq_start, (0, 0)), tma_tensor_q)
            tma_tensor_w_use = cute.domain_offset((0, seq_start, (0, 0)), tma_tensor_w)
            tma_tensor_dv2_use = cute.domain_offset((0, seq_start, (0, 0)), tma_tensor_dv2)
        else:
            tma_tensor_k_use = tma_tensor_k
            tma_tensor_dv_use = tma_tensor_dv
            tma_tensor_do_use = tma_tensor_do
            tma_tensor_q_use = tma_tensor_q
            tma_tensor_w_use = tma_tensor_w
            tma_tensor_dv2_use = tma_tensor_dv2

        _, bSG_sK, bSG_gK = self._epilog_partition(
            tma_atom_k, tma_tensor_k_use[None, None, (i_h, data_b)], (self.BT, self.K), sA
        )
        _, bSG_sDv, bSG_gDv = self._epilog_partition(
            tma_atom_dv, tma_tensor_dv_use[None, None, (i_h, data_b)], (self.BV, self.BT), sDv2T
        )
        _, bSG_sDo, bSG_gDo = self._epilog_partition(
            tma_atom_do, tma_tensor_do_use[None, None, (i_h, data_b)], (self.BV, self.BT), sUA
        )
        _, bSG_sQ, bSG_gQ = self._epilog_partition(
            tma_atom_q, tma_tensor_q_use[None, None, (i_h, data_b)], (self.K, self.BT), sUB
        )
        _, bSG_sW, bSG_gW = self._epilog_partition(
            tma_atom_w, tma_tensor_w_use[None, None, (i_h, data_b)], (self.K, self.BT), sUB
        )
        _, bSG_sDv2, bSG_gDv2 = self._epilog_partition(
            tma_atom_dv2, tma_tensor_dv2_use[None, None, (i_h, data_b)], (self.BV, self.BT), sDv2T
        )

        # Initialize carried dh state.
        for ei in cutlass.range(cute.size(rState), unroll_full=True):
            v_rel, k_idx = tUcState[ei]
            v_idx = v_base + v_rel
            init = Float32(0.0)
            if cutlass.const_expr(self.use_dht):
                if cutlass.const_expr(self.transpose_state_layout):
                    init = dht[i_n, i_h, v_idx, k_idx].to(self.acc_dtype)
                else:
                    init = dht[i_n, i_h, k_idx, v_idx].to(self.acc_dtype)
            rState[ei] = init

        for chunk_rev in cutlass.range(0, NT, unroll=0):
            i_t = NT - 1 - chunk_rev
            chunk_start = i_t * self.BT
            chunk_end = cutlass.min(chunk_start + self.BT, seq_len)
            last_idx = chunk_end - 1
            g_last = Float32(0.0)
            g_last_exp = Float32(1.0)
            if cutlass.const_expr(self.use_g):
                g_last = g[data_b, seq_start + last_idx, i_h].to(self.acc_dtype)
                if cutlass.const_expr(self.use_exp2):
                    g_last_exp = cute.exp2(g_last, fastmath=self.use_fast_math)
                else:
                    g_last_exp = cute.exp(g_last, fastmath=self.use_fast_math)

            # Store dh before applying this chunk's reverse update.
            for ei in cutlass.range(cute.size(rState), unroll_full=True):
                v_rel, k_idx = tUcState[ei]
                v_idx = v_base + v_rel
                if cutlass.const_expr(self.transpose_state_layout):
                    dh[state_b, chunk_base + i_t, i_h, v_idx, k_idx] = rState[ei].to(dh.element_type)
                else:
                    dh[state_b, chunk_base + i_t, i_h, k_idx, v_idx] = rState[ei].to(dh.element_type)
            cute.arch.sync_threads()

            # dv2 = dv + K @ dh.
            acc_dv.fill(0.0)
            if warp_idx == 0:
                kdv_h = load_kdv_P.acquire_and_advance()
                cute.copy(tma_atom_k, bSG_gK[(None, i_t, 0)], bSG_sK[None, kdv_h.index], tma_bar_ptr=kdv_h.barrier)
                cute.copy(
                    tma_atom_dv,
                    bSG_gDv[(None, i_v_tile, i_t)],
                    bSG_sDv[None, kdv_h.index],
                    tma_bar_ptr=kdv_h.barrier,
                )

            for ei in cutlass.range(cute.size(rState), unroll_full=True):
                v_rel, k_idx = tUcState[ei]
                sB[v_rel, k_idx, 0] = rState[ei].to(self.io_dtype)

            kdv_wait = load_kdv_C.wait_and_advance()
            cute.arch.sync_threads()

            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE,
                    cutlass.Boolean(kp != 0),
                )
                cute.gemm(
                    tiled_mma,
                    acc_dv,
                    tCrA[None, None, kp, 0],
                    tCrB[None, None, kp, 0],
                    acc_dv,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            kdv_wait.release()
            cute.arch.sync_threads()

            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                t_rel, v_rel = tCcDV[ei]
                t_idx = chunk_start + t_rel
                v_idx = v_base + v_rel
                out = Float32(0.0)
                if t_idx < seq_len:
                    out = acc_dv[ei]
                    if cutlass.const_expr(self.use_g):
                        g_cur = g[data_b, seq_start + t_idx, i_h].to(self.acc_dtype)
                        if cutlass.const_expr(self.use_exp2):
                            g_decay = cute.exp2(g_last - g_cur, fastmath=self.use_fast_math)
                        else:
                            g_decay = cute.exp(g_last - g_cur, fastmath=self.use_fast_math)
                        out = out * g_decay
                    out = out + sDv2T[v_rel, t_rel, 0].to(self.acc_dtype)
                sDv2T[v_rel, t_rel, 0] = out.to(self.io_dtype)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp_idx == 0:
                cute.copy(tma_atom_dv2, bSG_sDv2[None, 0], bSG_gDv2[(None, i_v_tile, i_t)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()

            # Apply state decay after dv2, before accumulating QO - WV into dh.
            if cutlass.const_expr(self.use_g):
                for ei in cutlass.range(cute.size(rState), unroll_full=True):
                    rState[ei] = rState[ei] * g_last_exp
            if cutlass.const_expr(self.use_gk):
                for ei in cutlass.range(cute.size(rState), unroll_full=True):
                    v_rel, k_idx = tUcState[ei]
                    gk_last = gk[data_b, seq_start + last_idx, i_h, k_idx].to(self.acc_dtype)
                    if cutlass.const_expr(self.use_exp2):
                        k_decay = cute.exp2(gk_last, fastmath=self.use_fast_math)
                    else:
                        k_decay = cute.exp(gk_last, fastmath=self.use_fast_math)
                    rState[ei] = rState[ei] * k_decay

            # dh += scale * do^T @ q - dv2^T @ w.
            if warp_idx == 0:
                qdo_h = load_qdo_P.acquire_and_advance()
                cute.copy(tma_atom_do, bSG_gDo[(None, i_v_tile, i_t)], bSG_sDo[None, qdo_h.index], tma_bar_ptr=qdo_h.barrier)
                cute.copy(tma_atom_q, bSG_gQ[(None, 0, i_t)], bSG_sQ[None, qdo_h.index], tma_bar_ptr=qdo_h.barrier)
            qdo_wait = load_qdo_C.wait_and_advance()
            cute.arch.sync_threads()

            if cutlass.const_expr(self.use_g):
                linear_q = tidx
                while linear_q < self.K * self.BT:
                    k_rel = linear_q // self.BT
                    t_rel = linear_q - k_rel * self.BT
                    t_idx = chunk_start + t_rel
                    q_scaled = Float32(0.0)
                    if t_idx < seq_len:
                        g_cur = g[data_b, seq_start + t_idx, i_h].to(self.acc_dtype)
                        if cutlass.const_expr(self.use_exp2):
                            g_exp = cute.exp2(g_cur, fastmath=self.use_fast_math)
                        else:
                            g_exp = cute.exp(g_cur, fastmath=self.use_fast_math)
                        q_scaled = sUB[k_rel, t_rel, 0].to(self.acc_dtype) * g_exp
                    sUB[k_rel, t_rel, 0] = q_scaled.to(self.io_dtype)
                    linear_q += self.num_threads
                cute.arch.sync_threads()

            acc_qdo.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tUrA, mode=[2]), unroll_full=True):
                update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    update_tiled_mma,
                    acc_qdo,
                    tUrA[None, None, kp, 0],
                    tUrB[None, None, kp, 0],
                    acc_qdo,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            qdo_wait.release()
            cute.arch.sync_threads()

            if warp_idx == 0:
                w_h = load_w_P.acquire_and_advance()
                cute.copy(tma_atom_w, bSG_gW[(None, 0, i_t)], bSG_sW[None, w_h.index], tma_bar_ptr=w_h.barrier)

            linear = tidx
            while linear < self.BV * self.BT:
                v_rel = linear // self.BT
                t_rel = linear - v_rel * self.BT
                sUA[v_rel, t_rel, 0] = sDv2T[v_rel, t_rel, 0]
                linear += self.num_threads
            w_wait = load_w_C.wait_and_advance()
            cute.arch.sync_threads()

            acc_wdv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tUrA, mode=[2]), unroll_full=True):
                update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    update_tiled_mma,
                    acc_wdv,
                    tUrA[None, None, kp, 0],
                    tUrB[None, None, kp, 0],
                    acc_wdv,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            w_wait.release()

            for ei in cutlass.range(cute.size(rState), unroll_full=True):
                update = acc_qdo[ei] * Float32(self.scale) - acc_wdv[ei]
                rState[ei] = rState[ei] + update
            cute.arch.sync_threads()

        if cutlass.const_expr(self.use_dh0):
            for ei in cutlass.range(cute.size(rState), unroll_full=True):
                v_rel, k_idx = tUcState[ei]
                v_idx = v_base + v_rel
                if cutlass.const_expr(self.transpose_state_layout):
                    dh0[i_n, i_h, v_idx, k_idx] = rState[ei]
                else:
                    dh0[i_n, i_h, k_idx, v_idx] = rState[ei]

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


def _as_cute(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16)


@functools.lru_cache(maxsize=64)
def _compile_bwd_dhu_sm90(
    B: int,
    T: int,
    N: int,
    NT: int,
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
    kernel = ChunkDeltaRuleBwdDHUSm90(
        batch_size=B,
        seq_len=T,
        num_sequences=N,
        total_chunks=NT,
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

    q_fake = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k_fake = torch.empty_like(q_fake)
    w_fake = torch.empty_like(q_fake)
    do_fake = torch.empty(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    dv_fake = torch.empty_like(do_fake)
    dv2_fake = torch.empty_like(do_fake)
    g_fake = torch.empty(B, T, H, device="cuda", dtype=torch.float32)
    gk_fake = torch.empty(B, T, H, K, device="cuda", dtype=torch.float32)
    if transpose_state_layout:
        dht_fake = torch.empty(N, H, V, K, device="cuda", dtype=torch.float32)
        dh0_fake = torch.empty_like(dht_fake)
        dh_fake = torch.empty(B, NT, H, V, K, device="cuda", dtype=torch.bfloat16)
    else:
        dht_fake = torch.empty(N, H, K, V, device="cuda", dtype=torch.float32)
        dh0_fake = torch.empty_like(dht_fake)
        dh_fake = torch.empty(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    cu_fake = torch.empty(N + 1, device="cuda", dtype=torch.int32)
    offsets_fake = torch.empty(N + 1, device="cuda", dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    return cute.compile(
        kernel,
        _as_cute(q_fake),
        _as_cute(k_fake),
        _as_cute(w_fake),
        _as_cute(g_fake),
        _as_cute(gk_fake),
        _as_cute(dht_fake),
        _as_cute(dh0_fake),
        _as_cute(do_fake),
        _as_cute(dh_fake),
        _as_cute(dv_fake),
        _as_cute(dv2_fake),
        _as_cute(cu_fake),
        _as_cute(offsets_fake),
        stream=stream,
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
    if K not in (64, 128, 256):
        raise NotImplementedError(f"SM90 bwd_dhu only supports K in {{64, 128, 256}}, got K={K}.")
    if V % BV != 0:
        raise NotImplementedError(f"SM90 bwd_dhu WGMMA path requires V to be a multiple of {BV}, got V={V}.")
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

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT).int()
        cu_seqlens_arg = cu_seqlens.int() if cu_seqlens.dtype != torch.int32 else cu_seqlens
    else:
        N = B
        NT = math.ceil(T / BT)
        cu_seqlens_arg = torch.arange(B + 1, device=q.device, dtype=torch.int32) * T
        chunk_offsets = torch.arange(B + 1, device=q.device, dtype=torch.int32) * NT
    scale_value = 1.0 if scale is None else float(scale)

    state_shape = (N, H, V, K) if transpose_state_layout else (N, H, K, V)
    dh = q.new_empty(B, NT, H, V, K) if transpose_state_layout else q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

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
        B,
        T,
        N,
        NT,
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
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(q, k, w, g_arg, gk_arg, dht_arg, dh0_arg, do, dh, dv, dv2, cu_seqlens_arg, chunk_offsets, stream)
    return dh, dh0, dv2


chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu_sm90
