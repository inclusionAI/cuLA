# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge step for Intra-Card Context Parallel chunk_delta_h.

Implements the prefix-scan merge:
    For each original sequence split into sub-sequences [s0, s1, ..., s_{n-1}]:
        h0_s0 = initial_state (or zero)
        h0_s1 = m_s0 @ h0_s0 + he_s0
        ...

Input:  hm [S_split, H, K, V+K] fp32 — packed (he, m) from pre_scan
Output: h  [num_non_first, H, K, V] fp32
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T as _T


# ---------------------------------------------------------------------------
# Inline PTX helpers: SM80 warp-level TF32 MMA (mma.sync.m16n8k8.tf32.tf32.f32)
# ---------------------------------------------------------------------------
def _to_ir(v, loc=None, ip=None):
    """Convert DSL Numeric to an MLIR Value; pass through if already a Value."""
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    return v


@cutlass.dsl_user_op
def _cvt_f32_to_tf32(f, *, loc=None, ip=None):
    """Round-to-nearest convert fp32 -> tf32 (stored as i32 bit pattern)."""
    f_ir = _to_ir(f, loc=loc, ip=ip)
    result = _llvm.inline_asm(
        _T.i32(),
        [f_ir],
        "cvt.rna.tf32.f32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cutlass.dsl_user_op
def _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """One mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 instruction.

    Inputs:
      a0..a3: tf32 bits (Int32) — A fragment of 16x8 tile
      b0..b1: tf32 bits (Int32) — B fragment of 8x8 tile
      c0..c3: Float32 — accumulator in
    Returns:
      (d0, d1, d2, d3) Float32 — accumulator out
    """
    ins = [
        _to_ir(a0, loc=loc, ip=ip),
        _to_ir(a1, loc=loc, ip=ip),
        _to_ir(a2, loc=loc, ip=ip),
        _to_ir(a3, loc=loc, ip=ip),
        _to_ir(b0, loc=loc, ip=ip),
        _to_ir(b1, loc=loc, ip=ip),
        _to_ir(c0, loc=loc, ip=ip),
        _to_ir(c1, loc=loc, ip=ip),
        _to_ir(c2, loc=loc, ip=ip),
        _to_ir(c3, loc=loc, ip=ip),
    ]
    struct_ty = ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>")
    ret = _llvm.inline_asm(
        struct_ty,
        ins,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = _llvm.extractvalue(_T.f32(), ret, [0], loc=loc, ip=ip)
    d1 = _llvm.extractvalue(_T.f32(), ret, [1], loc=loc, ip=ip)
    d2 = _llvm.extractvalue(_T.f32(), ret, [2], loc=loc, ip=ip)
    d3 = _llvm.extractvalue(_T.f32(), ret, [3], loc=loc, ip=ip)
    return (
        cutlass.Float32(d0),
        cutlass.Float32(d1),
        cutlass.Float32(d2),
        cutlass.Float32(d3),
    )


# ---------------------------------------------------------------------------
# Compile-time constants (thread/vector layout)
# ---------------------------------------------------------------------------
_BV_DEFAULT = 64
_M_THR = 8  # threads along rows of the (K, BV) tile
_N_THR = 16  # threads along cols of the (K, BV) tile
_NUM_THREADS = _M_THR * _N_THR  # 128
_VEC = 4  # 128-bit vectorized fp32 cp.async


class ChunkDeltaRuleMerge:
    """Prefix-scan merge kernel.

    H/K/V/BV kept as Python ints on ``self`` so layout construction is static.
    """

    def __init__(self, H: int, K: int, V: int, BV: int = _BV_DEFAULT, has_h0: int = 0):
        assert V % BV == 0, f"V={V} not divisible by BV={BV}"
        assert K % _M_THR == 0, f"K={K} not divisible by M_THR={_M_THR}"
        assert BV % _N_THR == 0, f"BV={BV} not divisible by N_THR={_N_THR}"
        assert (BV // _N_THR) == _VEC, f"BV/N_THR must equal VEC={_VEC}"
        assert K % _N_THR == 0, f"K={K} not divisible by N_THR={_N_THR}"
        assert (K // _N_THR) % _VEC == 0, "K/N_THR must be a multiple of VEC"
        self.H = H
        self.K = K
        self.V = V
        self.BV = BV
        self.has_h0 = int(has_h0)
        self.rows_per_thr = K // _M_THR
        self.cols_per_thr = BV // _N_THR  # == _VEC
        self.num_v_tiles = V // BV

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        hm: cute.Tensor,
        h_out: cute.Tensor,
        h0: cute.Tensor,
        seq_starts: cute.Tensor,
        seq_counts: cute.Tensor,
        init_offsets: cute.Tensor,
        split_seq_ids: cute.Tensor,
        num_split_seqs: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # +8 fp32 pad on the leading dim to eliminate SMEM bank conflicts:
        # without padding, row strides 128 / 64 are both multiples of 32 banks,
        # causing 4-8-way conflicts in the mma fragment loads/stores.
        _PAD: cutlass.Constexpr[int] = 8
        sM_layout = cute.make_layout((self.K, self.K), stride=(self.K + _PAD, 1))
        sHe_layout = cute.make_layout((self.K, self.BV), stride=(self.BV + _PAD, 1))
        sH_layout = cute.make_layout((self.K, self.BV), stride=(self.BV + _PAD, 1))

        @cute.struct
        class SharedStorage:
            sM: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sM_layout)],
                128,
            ]
            sHe: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sHe_layout)],
                128,
            ]
            sH: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sH_layout)],
                128,
            ]

        self.shared_storage_ty = SharedStorage

        # cp.async 128-bit vectorized copy atom (G->S loads).
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=_VEC * 32,
        )
        thr_layout = cute.make_layout((_M_THR, _N_THR), stride=(_N_THR, 1))
        val_layout = cute.make_layout((1, _VEC))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        # Universal 128-bit copy atom (S->G stores) sharing the same T/V layout
        # so gmem writes to h_out are coalesced (128B/warp).
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=_VEC * 32,
        )
        tiled_store = cute.make_tiled_copy_tv(store_atom, thr_layout, val_layout)

        self.kernel(
            hm,
            h_out,
            h0,
            seq_starts,
            seq_counts,
            init_offsets,
            split_seq_ids,
            sM_layout,
            sHe_layout,
            sH_layout,
            tiled_copy,
            tiled_store,
        ).launch(
            grid=(self.num_v_tiles, num_split_seqs, self.H),
            block=(_NUM_THREADS, 1, 1),
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        hm: cute.Tensor,
        h_out: cute.Tensor,
        h0: cute.Tensor,
        seq_starts: cute.Tensor,
        seq_counts: cute.Tensor,
        init_offsets: cute.Tensor,
        split_seq_ids: cute.Tensor,
        sM_layout: cute.Layout,
        sHe_layout: cute.Layout,
        sH_layout: cute.Layout,
        tiled_copy: cute.TiledCopy,
        tiled_store: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i_v, i_seq, i_h = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_ty)
        sM = cute.make_tensor(storage.sM.data_ptr(), sM_layout)
        sHe = cute.make_tensor(storage.sHe.data_ptr(), sHe_layout)
        sH = cute.make_tensor(storage.sH.data_ptr(), sH_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        thr_store = tiled_store.get_slice(tidx)

        ss_start = seq_starts[i_seq]
        n_ss = seq_counts[i_seq]
        init_base = init_offsets[i_seq]

        t_m = tidx // _N_THR
        t_n = tidx % _N_THR

        # --- Initialize sH from h0 or zero ---
        if cutlass.const_expr(self.has_h0):
            orig_id = split_seq_ids[i_seq]
            g_full = h0[orig_id, i_h, None, None]  # (K, V)
            gH0_tile = cute.local_tile(
                g_full,
                tiler=(self.K, self.BV),
                coord=(0, i_v),
            )
            tAgH = thr_copy.partition_S(gH0_tile)
            tAsH = thr_copy.partition_D(sH)
            cute.copy(tiled_copy, tAgH, tAsH)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
        else:
            for i in cutlass.range_constexpr(self.rows_per_thr):
                r = t_m + _M_THR * i
                for c in cutlass.range_constexpr(self.cols_per_thr):
                    sH[r, t_n * _VEC + c] = cutlass.Float32(0.0)
            cute.arch.barrier()

        # --- Main prefix-scan loop ---
        # Pre-declare loop-scratch scalars so their dsl types are stable across
        # the has_h0 / !has_h0 control-flow merge and into the dynamic loop.
        r = t_m
        out_idx = cutlass.Int32(0)
        i_ss = cutlass.Int32(0)
        # Number of BV-wide column tiles in b_m (K cols).
        m_col_tiles: cutlass.Constexpr[int] = self.K // self.BV
        for idx in cutlass.range(0, n_ss, unroll=0):
            i_ss = ss_start + idx

            g_hm = hm[i_ss, i_h, None, None]  # (K, V+K)

            # Load b_he [K, BV] from cols [i_v*BV, (i_v+1)*BV) of g_hm.
            gHe_tile = cute.local_tile(
                g_hm,
                tiler=(self.K, self.BV),
                coord=(0, i_v),
            )
            tAgHe = thr_copy.partition_S(gHe_tile)
            tAsHe = thr_copy.partition_D(sHe)
            cute.copy(tiled_copy, tAgHe, tAsHe)

            # Load b_m [K, K] as m_col_tiles BV-wide tiles (cols V..V+K).
            base_tile = self.num_v_tiles  # col-tile index where m starts
            for j in cutlass.range_constexpr(m_col_tiles):
                gM_j = cute.local_tile(
                    g_hm,
                    tiler=(self.K, self.BV),
                    coord=(0, base_tile + j),
                )
                sM_j = cute.local_tile(
                    sM,
                    tiler=(self.K, self.BV),
                    coord=(0, j),
                )
                tAgM = thr_copy.partition_S(gM_j)
                tAsM = thr_copy.partition_D(sM_j)
                cute.copy(tiled_copy, tAgM, tAsM)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # --- Compute new_b_h = b_m @ b_h + b_he via SM80 TF32 MMA ---
            # Warp-level mma.sync.m16n8k8 tiling of the (K=128, BV=64) output.
            # 4 warps per CTA: each warp owns rows [warp*32, warp*32 + 32) and
            # all BV cols. Within a warp, 2 M-tiles × 8 N-tiles × 16 K-iters.
            warp_id = tidx // 32
            lane = tidx % 32
            q = lane // 4
            rp = lane % 4

            M_TILES: cutlass.Constexpr[int] = 2  # (warp rows = 32) / 16
            N_TILES: cutlass.Constexpr[int] = self.BV // 8
            K_TILES: cutlass.Constexpr[int] = self.K // 8

            # Accumulator: [M_TILES, N_TILES, 4] fp32 per lane.
            acc = cute.make_rmem_tensor(
                cute.make_layout((M_TILES, N_TILES, 4)),
                cutlass.Float32,
            )
            # Initialize acc from sHe using the MMA D-fragment ownership.
            for mi in cutlass.range_constexpr(M_TILES):
                row_a = warp_id * 32 + mi * 16 + q
                row_b = row_a + 8
                for nj in cutlass.range_constexpr(N_TILES):
                    col_a = nj * 8 + rp * 2
                    acc[mi, nj, 0] = sHe[row_a, col_a]
                    acc[mi, nj, 1] = sHe[row_a, col_a + 1]
                    acc[mi, nj, 2] = sHe[row_b, col_a]
                    acc[mi, nj, 3] = sHe[row_b, col_a + 1]

            # K-reduction. For each k-tile: pre-cvt A (per M-tile) and B
            # (per N-tile) once, then call 2*8 MMAs reusing them.
            a_frag = cute.make_rmem_tensor(
                cute.make_layout((M_TILES, 4)),
                cutlass.Int32,  # tf32 bits
            )
            b_frag = cute.make_rmem_tensor(
                cute.make_layout((N_TILES, 2)),
                cutlass.Int32,  # tf32 bits
            )
            for ki in cutlass.range_constexpr(K_TILES):
                k_base = ki * 8
                # Pre-load + cvt A. For m16n8k8 TF32, A[16x8] per-lane:
                #   a0: (q,    rp),     a1: (q+8, rp)
                #   a2: (q,    rp+4),   a3: (q+8, rp+4)
                for mi in cutlass.range_constexpr(M_TILES):
                    row_a = warp_id * 32 + mi * 16 + q
                    row_b = row_a + 8
                    a_frag[mi, 0] = _cvt_f32_to_tf32(sM[row_a, k_base + rp])
                    a_frag[mi, 1] = _cvt_f32_to_tf32(sM[row_b, k_base + rp])
                    a_frag[mi, 2] = _cvt_f32_to_tf32(sM[row_a, k_base + rp + 4])
                    a_frag[mi, 3] = _cvt_f32_to_tf32(sM[row_b, k_base + rp + 4])
                # Pre-load + cvt B. For m16n8k8 TF32, B[8x8] per-lane (col-major):
                #   b0: (rp,   q)
                #   b1: (rp+4, q)
                for nj in cutlass.range_constexpr(N_TILES):
                    col_b = nj * 8 + q
                    b_frag[nj, 0] = _cvt_f32_to_tf32(sH[k_base + rp, col_b])
                    b_frag[nj, 1] = _cvt_f32_to_tf32(sH[k_base + rp + 4, col_b])
                # MMAs
                for mi in cutlass.range_constexpr(M_TILES):
                    for nj in cutlass.range_constexpr(N_TILES):
                        d0, d1, d2, d3 = _mma_m16n8k8_tf32(
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

            # --- Write acc → sH (for next iter) and h_out (when not last) ---
            cute.arch.barrier()
            for mi in cutlass.range_constexpr(M_TILES):
                row_a = warp_id * 32 + mi * 16 + q
                row_b = row_a + 8
                for nj in cutlass.range_constexpr(N_TILES):
                    col_a = nj * 8 + rp * 2
                    sH[row_a, col_a] = acc[mi, nj, 0]
                    sH[row_a, col_a + 1] = acc[mi, nj, 1]
                    sH[row_b, col_a] = acc[mi, nj, 2]
                    sH[row_b, col_a + 1] = acc[mi, nj, 3]

            if idx < n_ss - 1:
                # Coalesced 128-bit stores from sH -> h_out via shared thread
                # layout (matches loader). acc was already scattered to sH
                # above, so read from sH (same barrier covers the hand-off).
                cute.arch.barrier()
                out_idx = init_base + idx
                g_out = h_out[out_idx, i_h, None, None]  # (K, V)
                gOut_tile = cute.local_tile(
                    g_out,
                    tiler=(self.K, self.BV),
                    coord=(0, i_v),
                )
                tSsH = thr_store.partition_S(sH)
                tSgO = thr_store.partition_D(gOut_tile)
                cute.copy(tiled_store, tSsH, tSgO)

            cute.arch.barrier()


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------
def _compile_merge_variant(H: int, K: int, V: int, has_h0: int):
    kernel_obj = ChunkDeltaRuleMerge(H=H, K=K, V=V, BV=_BV_DEFAULT, has_h0=has_h0)

    sym_s = cute.sym_int()
    sym_nnf = cute.sym_int()
    sym_nss = cute.sym_int()
    sym_nss1 = cute.sym_int()
    sym_n = cute.sym_int()

    hm_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_s, H, K, V + K),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    h_out_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_nnf, H, K, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    h0_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_n, H, K, V),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    starts_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nss,), assumed_align=16)
    counts_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nss,), assumed_align=16)
    init_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nss1,), assumed_align=16)
    sid_fake = make_fake_compact_tensor(cutlass.Int32, (sym_nss,), assumed_align=16)

    stream_fake = make_fake_stream()

    return cute.compile(
        kernel_obj,
        hm_fake,
        h_out_fake,
        h0_fake,
        starts_fake,
        counts_fake,
        init_fake,
        sid_fake,
        cutlass.Int32(1),
        stream_fake,
    )


@functools.lru_cache(maxsize=32)
def _get_compiled_merge(H: int, K: int, V: int, has_h0: int):
    return _compile_merge_variant(H, K, V, has_h0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def merge_fwd(
    hm: torch.Tensor,
    seq_starts: list[int],
    seq_counts: list[int],
    init_offsets: list[int],
    split_seq_ids: list[int],
    h0: torch.Tensor | None,
    num_non_first: int,
) -> torch.Tensor:
    """Prefix-scan merge using a single CuTeDSL kernel launch."""
    assert hm.dtype == torch.float32, f"hm must be fp32, got {hm.dtype}"
    _, H, K, VK = hm.shape
    V = VK - K
    device = hm.device
    num_split_seqs = len(split_seq_ids)

    h_out = hm.new_empty(num_non_first, H, K, V)

    starts_gpu = torch.tensor(seq_starts, dtype=torch.int32, device=device)
    counts_gpu = torch.tensor(seq_counts, dtype=torch.int32, device=device)
    init_off_gpu = torch.tensor(init_offsets, dtype=torch.int32, device=device)
    sid_gpu = torch.tensor(split_seq_ids, dtype=torch.int32, device=device)

    if h0 is not None:
        h0_arg = h0
        has_h0 = 1
    else:
        h0_arg = hm.new_zeros(1, H, K, V)
        has_h0 = 0

    compiled_fn = _get_compiled_merge(H, K, V, has_h0)
    stream_ptr = torch.cuda.current_stream(device).cuda_stream

    compiled_fn(
        from_dlpack(hm, assumed_align=128),
        from_dlpack(h_out, assumed_align=128),
        from_dlpack(h0_arg, assumed_align=128),
        from_dlpack(starts_gpu, assumed_align=16),
        from_dlpack(counts_gpu, assumed_align=16),
        from_dlpack(init_off_gpu, assumed_align=16),
        from_dlpack(sid_gpu, assumed_align=16),
        cutlass.Int32(num_split_seqs),
        cuda.CUstream(stream_ptr),
    )

    return h_out
