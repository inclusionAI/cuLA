# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KDA Fused Forward Prefill — SM80 CuTe DSL (Step 2: working mma.sync QK)."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op
from cutlass.cute.typing import BFloat16, Float32

from cula.utils import USE_FAST_MATH, assert_ampere

BT = 64
BK = 128
NUM_THREADS = 128
NUM_STAGES = 1

MMA_INST_SHAPE = (16, 8, 16)
ATOM_LAYOUT_MNK = (4, 8, 8)
PERMUTATION_MNK = (
    ATOM_LAYOUT_MNK[0] * MMA_INST_SHAPE[0],
    ATOM_LAYOUT_MNK[1] * MMA_INST_SHAPE[1],
    ATOM_LAYOUT_MNK[2] * MMA_INST_SHAPE[2],
)


@cute.kernel
def _kda_qk_sm80(
    tiled_mma: cute.TiledMma,
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    o: cute.Tensor,
    scale: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int],
    D: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx, head_idx, batch_idx = cute.arch.block_idx()
    lane_id = tidx % 32

    _smem_layout = cute.make_layout((C, D, NUM_STAGES), stride=(D, 1, C * D))
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(BFloat16, _smem_layout, 128)
    sK = smem.allocate_tensor(BFloat16, _smem_layout, 128)
    sC = smem.allocate_tensor(Float32, cute.make_layout((BT, BT), stride=(BT, 1)), 128)

    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            k_idx = i * 32 + lane_id
            sQ[(row, k_idx, 0)] = q[(chunk_idx * BT + row, head_idx, k_idx)]
            sK[(row, k_idx, 0)] = k[(chunk_idx * BT + row, head_idx, k_idx)]
    cute.arch.barrier()

    r_q = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
    r_e = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)

    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            k_idx = i * 32 + lane_id
            r_q[i] = Float32(sQ[(row, k_idx, 0)])
            r_k[i] = Float32(sK[(row, k_idx, 0)])
            r_e[i] = g[(chunk_idx * BT + row, head_idx, k_idx)]
        for i in cutlass.range_constexpr(4):
            r_q[i] = r_q[i] * cute.exp2(r_e[i], fastmath=USE_FAST_MATH)
            r_k[i] = r_k[i] * cute.exp2(r_e[i], fastmath=USE_FAST_MATH)
            sQ[(row, i * 32 + lane_id, 0)] = BFloat16(r_q[i])
            sK[(row, i * 32 + lane_id, 0)] = BFloat16(r_k[i])
    cute.arch.barrier()

    # ── Ampere MMA pipeline ──
    thr_mma = tiled_mma.get_slice(tidx)

    # Partition SMEM tensors for MMA
    tCsA = thr_mma.partition_A(sQ)
    tCsB = thr_mma.partition_B(sK)
    tCsC = thr_mma.partition_C(sC)

    # Create register fragments from stage-0 SMEM
    rA = tiled_mma.make_fragment_A(tCsA[(None, None, None, 0)])
    rB = tiled_mma.make_fragment_B(tCsB[(None, None, None, 0)])
    rC = tiled_mma.make_fragment_C(tCsC)    
    rC.fill(0.0)

    # S2R: ldmatrix SMEM→registers
    atom_s2r = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        BFloat16,
    )
    tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
    tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
    thr_s2r_A = tiled_s2r_A.get_slice(tidx)
    thr_s2r_B = tiled_s2r_B.get_slice(tidx)

    sQ_pipe = thr_s2r_A.partition_S(sQ)[(None, None, None, 0)]
    sK_pipe = thr_s2r_B.partition_S(sK)[(None, None, None, 0)]
    rA_view = thr_s2r_A.retile(rA)
    rB_view = thr_s2r_B.retile(rB)

    num_kb = cute.size(rA, mode=[2])
    for kb in cutlass.range_constexpr(num_kb):
        cute.copy(tiled_s2r_A, sQ_pipe[(None, None, kb)], rA_view[(None, None, kb)])
        cute.copy(tiled_s2r_B, sK_pipe[(None, None, kb)], rB_view[(None, None, kb)])
        cute.gemm(tiled_mma, rC, rA[(None, None, kb)], rB[(None, None, kb)], rC)

    # ── Epilogue: R2S accumulator → SMEM, then S2G → output ──
    cute.autovec_copy(rC, tCsC)
    cute.arch.barrier()

    # Map MMA C-partition to output: use identity layout for thread→(row,col)
    c_ident = cute.make_identity_tensor((BT, BT))
    tC_ident = thr_mma.partition_C(c_ident)
    for i in cutlass.range_constexpr(cute.size(tC_ident)):
        coord = tC_ident[i]
        row = coord[0]
        col = coord[1]
        val = tCsC[i]
        o[(chunk_idx * BT + row, head_idx, col)] = BFloat16(Float32(val))


    cute.arch.barrier()


class KDAFusedFwdSM80:
    def __init__(self, chunk_size=64, head_dim=128, scale=None, safe_gate=False):
        assert_ampere()
        self.chunk_size = chunk_size
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.safe_gate = safe_gate

    @cute.jit
    def __call__(
        self, q:cute.Tensor, k:cute.Tensor, g:cute.Tensor, o:cute.Tensor,
        beta:cute.Tensor, s0:cute.Tensor, s1:cute.Tensor,
        cu:cute.Tensor, ws:cute.Tensor,
        problem_size: tuple[cute.Int32,cute.Int32,cute.Int32,cute.Int32], stream,
    ):
        B, S, H, _D = problem_size
        C = self.chunk_size
        D = self.head_dim  # Python int → Constexpr, NOT from problem_size

        op = MmaF16BF16Op(ab_dtype=BFloat16, acc_dtype=Float32, shape_mnk=MMA_INST_SHAPE)
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout(ATOM_LAYOUT_MNK), permutation_mnk=PERMUTATION_MNK)
        num_chunks = cute.ceil_div(S, C)

        _kda_qk_sm80(tiled_mma, q, k, g, o,
            scale=self.scale, C=C, D=D, S=S, H=H, K=D, V=D,
        ).launch(grid=(num_chunks,H,B), block=[NUM_THREADS,1,1],
                 smem=(2*C*D*NUM_STAGES*2)+(4*C*C)+2048, stream=stream)