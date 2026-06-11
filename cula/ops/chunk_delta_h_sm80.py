"""Chunk Delta-H Forward — SM80 CuTe DSL (state update via mma.sync)."""

import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op
from cutlass.cute.typing import BFloat16, Float32
from cula.utils import USE_FAST_MATH, assert_ampere

BT=64; BK=128; BV=64; NT=128; NS=1
MS=(16,8,16); AL=(4,8,8); PM=(AL[0]*MS[0],AL[1]*MS[1],AL[2]*MS[2])
AK=(4,8,4); PK=(AK[0]*MS[0],AK[1]*MS[1],AK[2]*MS[2])

@cute.kernel
def _chunk_delta_h_sm80(
    tiled_mma_wh: cute.TiledMma, tiled_mma_kv: cute.TiledMma,
    k: cute.Tensor, w: cute.Tensor, g: cute.Tensor,
    h_state: cute.Tensor, o_state: cute.Tensor,
    decay: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int], D: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], K: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx, head_idx, v_tile = cute.arch.block_idx()
    lane_id = tidx % 32
    v_offset = v_tile * BV
    _smem_layout = cute.make_layout((C, D, NS), stride=(D, 1, C * D))
    smem = cutlass.utils.SmemAllocator()

    # 3D SMEM for operands (matching QK kernel pattern)
    sK = smem.allocate_tensor(BFloat16, _smem_layout, 128)
    sW = smem.allocate_tensor(BFloat16, _smem_layout, 128)
    sH = smem.allocate_tensor(BFloat16, cute.make_layout((BV,BK,1), stride=(BK,1,BV*BK)), 128)
    # 2D SMEM for accumulators (matching QK kernel sC pattern)
    sWH = smem.allocate_tensor(Float32, cute.make_layout((BV,BT), stride=(BT,1)), 128)

    # Load K, W, state (v_offset shifts which V-tile of state we read)
    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            ki = i*32+lane_id
            sK[(row,ki,0)] = k[(chunk_idx*BT+row,head_idx,ki)]
            sW[(row,ki,0)] = w[(chunk_idx*BT+row,head_idx,ki)]
    for row in cutlass.range_constexpr(BV):
        for i in cutlass.range_constexpr(4):
            ki = i*32+lane_id
            sH[(row,ki,0)] = BFloat16(h_state[(0,0,v_offset+row,ki)])
    cute.arch.barrier()

    # Gate K only: K_g = exp2(g) * K.  W arrives pre-gated (M^{-1}@K_gated)
    rk = cute.make_rmem_tensor(cute.make_layout((4,)),Float32)
    re = cute.make_rmem_tensor(cute.make_layout((4,)),Float32)
    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            ki=i*32+lane_id
            rk[i]=Float32(sK[(row,ki,0)])
            re[i]=g[(chunk_idx*BT+row,head_idx,ki)]
        for i in cutlass.range_constexpr(4):
            rk[i]=rk[i]*cute.exp2(re[i],fastmath=USE_FAST_MATH)
            sK[(row,i*32+lane_id,0)]=BFloat16(rk[i])
    cute.arch.barrier()

    # === WH MMA: state[B,128] @ W^T[64,128] → WH_acc[B,64] ===
    thr = tiled_mma_wh.get_slice(tidx)
    tA=thr.partition_A(sH); tB=thr.partition_B(sW); tC=thr.partition_C(sWH)
    rA=tiled_mma_wh.make_fragment_A(tA[(None,None,None,0)])
    rB=tiled_mma_wh.make_fragment_B(tB[(None,None,None,0)])
    rC=tiled_mma_wh.make_fragment_C(tC)
    rC.fill(0.0)

    atom= cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False,4),BFloat16)
    tsA=cute.make_tiled_copy_A(atom,tiled_mma_wh); tsB=cute.make_tiled_copy_B(atom,tiled_mma_wh)
    stA=tsA.get_slice(tidx); stB=tsB.get_slice(tidx)
    pA=stA.partition_S(sH)[(None,None,None,0)]; pB=stB.partition_S(sW)[(None,None,None,0)]
    vA=stA.retile(rA); vB=stB.retile(rB)

    nk=cute.size(rA,mode=[2])
    for kb in cutlass.range_constexpr(nk):
        cute.copy(tsA,pA[(None,None,kb)],vA[(None,None,kb)])
        cute.copy(tsB,pB[(None,None,kb)],vB[(None,None,kb)])
        cute.gemm(tiled_mma_wh,rC,rA[(None,None,kb)],rB[(None,None,kb)],rC)
    cute.autovec_copy(rC,tC)
    cute.arch.barrier()

    # ── Epilogue: write WH_acc [BV,BT] → o_state[0,0,0:BV,0:BT] ──
    c_ident = cute.make_identity_tensor((BV, BT))
    tC_id = thr.partition_C(c_ident)
    for i in cutlass.range_constexpr(cute.size(tC_id)):
        coord = tC_id[i]
        row, col = coord[0], coord[1]
        o_state[(0, 0, v_offset+row, col)] = Float32(tC[i])
    cute.arch.barrier()


class ChunkDeltaHFwdSM80:
    def __init__(self,chunk_size=64,head_dim_k=128,head_dim_v=128):
        assert_ampere(); self.C=chunk_size; self.D=head_dim_k; self.V=head_dim_v

    @cute.jit
    def __call__(self,k:cute.Tensor,w:cute.Tensor,g:cute.Tensor,
                 h:cute.Tensor,o:cute.Tensor,b:cute.Tensor,
                 problem_size:tuple[cute.Int32,cute.Int32,cute.Int32,cute.Int32,cute.Int32],
                 decay:cutlass.Constexpr[float],stream):
        B,S,H,_,_=problem_size
        NV = (self.V + BV - 1) // BV
        op=MmaF16BF16Op(BFloat16,Float32,MS)
        tw=cute.make_tiled_mma(op,cute.make_layout(AL),permutation_mnk=PM)
        tk=cute.make_tiled_mma(op,cute.make_layout(AK),permutation_mnk=PK)
        nc=cute.ceil_div(S,self.C)
        _chunk_delta_h_sm80(tw,tk,k,w,g,h,o,decay=decay,C=self.C,D=self.D,S=S,H=H,V=self.V,K=self.D
        ).launch(grid=(nc,H,NV),block=[NT,1,1],
                 smem=(2*self.C*self.D*NS*2)+(2*BV*BK)+(4*BV*BT)+2048,stream=stream)