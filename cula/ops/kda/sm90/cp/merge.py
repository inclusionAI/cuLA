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


"""CuTeDSL merge kernel for intracard-CP (SM90).

Carries kept in SMEM across segment iterations — eliminates gmem round-trips.
Uses SM80 TF32 MMA (mma.sync.m16n8k8), available on SM80+.

Layout: SM90 uses separate b_seg/m_seg [S,H,D,D] fp32 in bhvk order.
Matmul: carries[i+1] = carries[i] @ M_seg[i] + B_seg[i]  (carry on left).

Compared to SM100 merge:
 - A=carry (sH), B=transition (sM) — SM100 has A=transition, B=state
 - Row-tiles carry with BR=64 (SM100 col-tiles state with BV=64)
 - 4 warps × 16 rows (SM100: 4 warps × 32 rows)
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cula.ops.kda.sm90._common import _stream_key
from cula.ops.kda.sm90.k2 import D, _get_current_custream
from cula.ops.ptx import cvt_f32_to_tf32, mma_m16n8k8_tf32

_BR = 64
_BN = 64
_M_THR = 8
_N_THR = 16
_NUM_THREADS = _M_THR * _N_THR  # 128
_VEC = 4
_PAD = 8


class Merge:
    """SMEM-resident carry merge.

    Grid: (D // BR, n_seqs, H).  Each CTA handles BR=64 rows of carry.
    4 warps × 16 rows/warp = 64 rows.  MMA: 1 M-tile × 16 N-tiles × 16 K-iters.
    """

    def __init__(self, H: int, has_init: int = 0):
        self.H = H
        self.has_init = int(has_init)
        self.num_row_tiles = D // _BR  # 2
        self.num_col_tiles = D // _BN  # 2

    @cute.jit
    def __call__(
        self,
        b_seg: cute.Tensor,
        m_seg: cute.Tensor,
        carries: cute.Tensor,
        init: cute.Tensor,
        first_ptr: cute.Tensor,
        nseg_ptr: cute.Tensor,
        num_seqs: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        sH_layout = cute.make_layout((_BR, D), stride=(D + _PAD, 1))
        sM_layout = cute.make_layout((D, D), stride=(D + _PAD, 1))
        sB_layout = cute.make_layout((_BR, D), stride=(D + _PAD, 1))

        @cute.struct
        class SharedStorage:
            sH: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sH_layout)],
                128,
            ]
            sM: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sM_layout)],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sB_layout)],
                128,
            ]

        self.shared_storage_ty = SharedStorage

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=_VEC * 32,
        )
        thr_layout = cute.make_layout((_M_THR, _N_THR), stride=(_N_THR, 1))
        val_layout = cute.make_layout((1, _VEC))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=_VEC * 32,
        )
        tiled_store = cute.make_tiled_copy_tv(store_atom, thr_layout, val_layout)

        self.kernel(
            b_seg,
            m_seg,
            carries,
            init,
            first_ptr,
            nseg_ptr,
            sH_layout,
            sM_layout,
            sB_layout,
            tiled_copy,
            tiled_store,
        ).launch(
            grid=(self.num_row_tiles, num_seqs, self.H),
            block=(_NUM_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        b_seg: cute.Tensor,
        m_seg: cute.Tensor,
        carries: cute.Tensor,
        init: cute.Tensor,
        first_ptr: cute.Tensor,
        nseg_ptr: cute.Tensor,
        sH_layout: cute.Layout,
        sM_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy: cute.TiledCopy,
        tiled_store: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i_r, i_seq, i_h = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_ty)
        sH = cute.make_tensor(storage.sH.data_ptr(), sH_layout)
        sM = cute.make_tensor(storage.sM.data_ptr(), sM_layout)
        sB = cute.make_tensor(storage.sB.data_ptr(), sB_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        thr_store = tiled_store.get_slice(tidx)

        first = first_ptr[i_seq]
        n_seg = nseg_ptr[i_seq]

        t_m = tidx // _N_THR
        t_n = tidx % _N_THR

        if cutlass.const_expr(self.has_init):
            g_init = init[i_seq, i_h, None, None]
            for j in cutlass.range_constexpr(self.num_col_tiles):
                gI = cute.local_tile(g_init, tiler=(_BR, _BN), coord=(i_r, j))
                sHj = cute.local_tile(sH, tiler=(_BR, _BN), coord=(0, j))
                cute.copy(
                    tiled_copy,
                    thr_copy.partition_S(gI),
                    thr_copy.partition_D(sHj),
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
        else:
            _rows_per_thr: cutlass.Constexpr[int] = _BR // _M_THR
            _col_stride: cutlass.Constexpr[int] = _N_THR * _VEC
            _n_col_groups: cutlass.Constexpr[int] = D // _col_stride
            for ri in cutlass.range_constexpr(_rows_per_thr):
                r = t_m + _M_THR * ri
                for g in cutlass.range_constexpr(_n_col_groups):
                    for c in cutlass.range_constexpr(_VEC):
                        sH[r, g * _col_stride + t_n * _VEC + c] = cutlass.Float32(0.0)
            cute.arch.barrier()

        g_c0 = carries[first, i_h, None, None]
        for j in cutlass.range_constexpr(self.num_col_tiles):
            gC = cute.local_tile(g_c0, tiler=(_BR, _BN), coord=(i_r, j))
            sHj = cute.local_tile(sH, tiler=(_BR, _BN), coord=(0, j))
            cute.copy(
                tiled_store,
                thr_store.partition_S(sHj),
                thr_store.partition_D(gC),
            )

        seg_idx = cutlass.Int32(0)
        idx = cutlass.Int32(0)

        for idx in cutlass.range(0, n_seg - 1, unroll=0):
            seg_idx = first + idx

            g_b = b_seg[seg_idx, i_h, None, None]
            for j in cutlass.range_constexpr(self.num_col_tiles):
                gB = cute.local_tile(g_b, tiler=(_BR, _BN), coord=(i_r, j))
                sBj = cute.local_tile(sB, tiler=(_BR, _BN), coord=(0, j))
                cute.copy(
                    tiled_copy,
                    thr_copy.partition_S(gB),
                    thr_copy.partition_D(sBj),
                )

            g_m = m_seg[seg_idx, i_h, None, None]
            for j in cutlass.range_constexpr(self.num_col_tiles):
                gM = cute.local_tile(g_m, tiler=(D, _BN), coord=(0, j))
                sMj = cute.local_tile(sM, tiler=(D, _BN), coord=(0, j))
                cute.copy(
                    tiled_copy,
                    thr_copy.partition_S(gM),
                    thr_copy.partition_D(sMj),
                )

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # A = sH (carry rows), B = sM (transition cols)
            warp_id = tidx // 32
            lane = tidx % 32
            q = lane // 4
            rp = lane % 4

            M_TILES: cutlass.Constexpr[int] = 1
            N_TILES: cutlass.Constexpr[int] = D // 8  # 16
            K_TILES: cutlass.Constexpr[int] = D // 8  # 16

            acc = cute.make_rmem_tensor(
                cute.make_layout((M_TILES, N_TILES, 4)),
                cutlass.Float32,
            )

            for mi in cutlass.range_constexpr(M_TILES):
                row_a = warp_id * 16 + mi * 16 + q
                row_b = row_a + 8
                for nj in cutlass.range_constexpr(N_TILES):
                    col_a = nj * 8 + rp * 2
                    acc[mi, nj, 0] = sB[row_a, col_a]
                    acc[mi, nj, 1] = sB[row_a, col_a + 1]
                    acc[mi, nj, 2] = sB[row_b, col_a]
                    acc[mi, nj, 3] = sB[row_b, col_a + 1]

            a_frag = cute.make_rmem_tensor(
                cute.make_layout((M_TILES, 4)),
                cutlass.Int32,
            )
            b_frag = cute.make_rmem_tensor(
                cute.make_layout((N_TILES, 2)),
                cutlass.Int32,
            )

            for ki in cutlass.range_constexpr(K_TILES):
                k_base = ki * 8
                for mi in cutlass.range_constexpr(M_TILES):
                    row_a = warp_id * 16 + mi * 16 + q
                    row_b = row_a + 8
                    a_frag[mi, 0] = cvt_f32_to_tf32(sH[row_a, k_base + rp])
                    a_frag[mi, 1] = cvt_f32_to_tf32(sH[row_b, k_base + rp])
                    a_frag[mi, 2] = cvt_f32_to_tf32(sH[row_a, k_base + rp + 4])
                    a_frag[mi, 3] = cvt_f32_to_tf32(sH[row_b, k_base + rp + 4])
                for nj in cutlass.range_constexpr(N_TILES):
                    col_b = nj * 8 + q
                    b_frag[nj, 0] = cvt_f32_to_tf32(sM[k_base + rp, col_b])
                    b_frag[nj, 1] = cvt_f32_to_tf32(sM[k_base + rp + 4, col_b])
                for mi in cutlass.range_constexpr(M_TILES):
                    for nj in cutlass.range_constexpr(N_TILES):
                        d0, d1, d2, d3 = mma_m16n8k8_tf32(
                            a_frag[mi, 0],
                            a_frag[mi, 1],
                            a_frag[mi, 2],
                            a_frag[mi, 3],
                            b_frag[nj, 0],
                            b_frag[nj, 1],
                            acc[mi, nj, 0],
                            acc[mi, nj, 1],
                            acc[mi, nj, 2],
                            acc[mi, nj, 3],
                        )
                        acc[mi, nj, 0] = d0
                        acc[mi, nj, 1] = d1
                        acc[mi, nj, 2] = d2
                        acc[mi, nj, 3] = d3

            cute.arch.barrier()
            for mi in cutlass.range_constexpr(M_TILES):
                row_a = warp_id * 16 + mi * 16 + q
                row_b = row_a + 8
                for nj in cutlass.range_constexpr(N_TILES):
                    col_a = nj * 8 + rp * 2
                    sH[row_a, col_a] = acc[mi, nj, 0]
                    sH[row_a, col_a + 1] = acc[mi, nj, 1]
                    sH[row_b, col_a] = acc[mi, nj, 2]
                    sH[row_b, col_a + 1] = acc[mi, nj, 3]

            cute.arch.barrier()
            g_c_out = carries[seg_idx + 1, i_h, None, None]
            for j in cutlass.range_constexpr(self.num_col_tiles):
                gC = cute.local_tile(g_c_out, tiler=(_BR, _BN), coord=(i_r, j))
                sHj = cute.local_tile(sH, tiler=(_BR, _BN), coord=(0, j))
                cute.copy(
                    tiled_store,
                    thr_store.partition_S(sHj),
                    thr_store.partition_D(gC),
                )

            cute.arch.barrier()


def _compile_merge(H: int, has_init: int):
    kernel_obj = Merge(H=H, has_init=has_init)

    sym_s = cute.sym_int()
    sym_n = cute.sym_int()
    sym_nseq = cute.sym_int()

    b_seg_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_s, H, D, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    m_seg_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_s, H, D, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    carries_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_s, H, D, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    init_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_n, H, D, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    first_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nseq,), assumed_align=16)
    nseg_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nseq,), assumed_align=16)
    stream_fake = make_fake_stream()

    return cute.compile(
        kernel_obj,
        b_seg_fake,
        m_seg_fake,
        carries_fake,
        init_fake,
        first_fake,
        nseg_fake,
        cutlass.Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=32)
def _get_compiled_merge(H: int, has_init: int):
    return _compile_merge(H, has_init)


# Cached per_seq-derived tensors and dummy init: building them per call
# costs two synchronous pageable H2D copies plus a memset.
_PER_SEQ_TENSOR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_PER_SEQ_TENSOR_CACHE_MAXSIZE = 64
_DUMMY_INIT_CACHE: dict[tuple, torch.Tensor] = {}
_DUMMY_INIT_CACHE_MAXSIZE = 64


def _get_per_seq_tensors(per_seq: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (per_seq, _stream_key(device))
    cached = _PER_SEQ_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_PER_SEQ_TENSOR_CACHE) >= _PER_SEQ_TENSOR_CACHE_MAXSIZE:
        _PER_SEQ_TENSOR_CACHE.pop(next(iter(_PER_SEQ_TENSOR_CACHE)))
    firsts = torch.tensor([f for f, _ in per_seq], dtype=torch.int32, device=device)
    nsegs = torch.tensor([n for _, n in per_seq], dtype=torch.int32, device=device)
    _PER_SEQ_TENSOR_CACHE[key] = (firsts, nsegs)
    return firsts, nsegs


def _get_dummy_init(H: int, device: torch.device) -> torch.Tensor:
    key = (H, _stream_key(device))
    cached = _DUMMY_INIT_CACHE.get(key)
    if cached is None:
        if len(_DUMMY_INIT_CACHE) >= _DUMMY_INIT_CACHE_MAXSIZE:
            _DUMMY_INIT_CACHE.pop(next(iter(_DUMMY_INIT_CACHE)))
        cached = torch.zeros(1, H, D, D, dtype=torch.float32, device=device)
        _DUMMY_INIT_CACHE[key] = cached
    return cached


def launch_merge(
    carries: torch.Tensor,
    m_seg: torch.Tensor,
    b_seg: torch.Tensor,
    per_seq: list[tuple[int, int]],
    init_bhvk: torch.Tensor | None,
) -> torch.Tensor:
    """CuTeDSL merge: SMEM-resident carry, TF32 MMA."""
    n_seqs = len(per_seq)
    H = carries.shape[1]
    device = carries.device

    firsts, nsegs = _get_per_seq_tensors(tuple(per_seq), device)
    has_init = 1 if init_bhvk is not None else 0
    init_arg = init_bhvk if init_bhvk is not None else _get_dummy_init(H, device)

    compiled_fn = _get_compiled_merge(H, has_init)
    stream = _get_current_custream()

    # tvm-ffi launch: torch tensors pass straight through, positional args unvalidated.
    compiled_fn(
        b_seg,
        m_seg,
        carries,
        init_arg,
        firsts,
        nsegs,
        n_seqs,
        stream,
    )
    return carries
