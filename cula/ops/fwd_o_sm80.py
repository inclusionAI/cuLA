"""Fused Output — SM80 CuTe DSL (Q@state + tril(A)@V → output)."""

import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op
from cutlass.cute.typing import BFloat16, Float32
from cula.utils import USE_FAST_MATH, assert_ampere

BT=64; BK=128; BV=64; NT=128; NS=1
MS=(16,8,16); AL=(4,8,8); PM=(AL[0]*MS[0],AL[1]*MS[1],AL[2]*MS[2])

@cute.kernel
def _fwd_o_sm80(
    tiled_mma: cute.TiledMma,
    q: cute.Tensor, h: cute.Tensor, g: cute.Tensor, o: cute.Tensor,
    scale: cutlass.Constexpr[float],
    C: cutlass.Constexpr[int], D: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], K: cutlass.Constexpr[int],
):
    tidx,_,_=cute.arch.thread_idx()
    chunk_idx,head_idx,v_tile=cute.arch.block_idx()
    lane_id=tidx%32
    v_offset = v_tile * BV
    _smem_layout = cute.make_layout((C, D, NS), stride=(D, 1, C * D))
    smem=cutlass.utils.SmemAllocator()
    sQ=smem.allocate_tensor(BFloat16,_smem_layout,128)
    sH=smem.allocate_tensor(BFloat16,cute.make_layout((BV,BK,1),stride=(BK,1,BV*BK)),128)
    sO=smem.allocate_tensor(Float32,cute.make_layout((BT,BV),stride=(BV,1)),128)

    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            ki=i*32+lane_id
            sQ[(row,ki,0)]=q[(chunk_idx*BT+row,head_idx,ki)]
    for row in cutlass.range_constexpr(BV):
        for i in cutlass.range_constexpr(4):
            ki=i*32+lane_id
            sH[(row,ki,0)]=BFloat16(h[(0,0,v_offset+row,ki)])
    cute.arch.barrier()

    # Gate Q: Q_g = exp(g) * Q (before MMA)
    rq = cute.make_rmem_tensor(cute.make_layout((4,)),Float32)
    re = cute.make_rmem_tensor(cute.make_layout((4,)),Float32)
    for row in cutlass.range_constexpr(BT):
        for i in cutlass.range_constexpr(4):
            ki=i*32+lane_id
            rq[i]=Float32(sQ[(row,ki,0)])
            re[i]=g[(chunk_idx*BT+row,head_idx,ki)]
        for i in cutlass.range_constexpr(4):
            rq[i]=rq[i]*cute.exp2(re[i],fastmath=USE_FAST_MATH)
            sQ[(row,i*32+lane_id,0)]=BFloat16(rq[i])
    cute.arch.barrier()

    thr=tiled_mma.get_slice(tidx)
    tA=thr.partition_A(sQ); tB=thr.partition_B(sH); tC=thr.partition_C(sO)
    rA=tiled_mma.make_fragment_A(tA[(None,None,None,0)])
    rB=tiled_mma.make_fragment_B(tB[(None,None,None,0)])
    rC=tiled_mma.make_fragment_C(tC)
    rC.fill(0.0)

    atom=cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False,4),BFloat16)
    tsA=cute.make_tiled_copy_A(atom,tiled_mma); tsB=cute.make_tiled_copy_B(atom,tiled_mma)
    stA=tsA.get_slice(tidx); stB=tsB.get_slice(tidx)
    pA=stA.partition_S(sQ)[(None,None,None,0)]; pB=stB.partition_S(sH)[(None,None,None,0)]
    vA=stA.retile(rA); vB=stB.retile(rB)

    nk=cute.size(rA,mode=[2])
    for kb in cutlass.range_constexpr(nk):
        cute.copy(tsA,pA[(None,None,kb)],vA[(None,None,kb)])
        cute.copy(tsB,pB[(None,None,kb)],vB[(None,None,kb)])
        cute.gemm(tiled_mma,rC,rA[(None,None,kb)],rB[(None,None,kb)],rC)
    cute.autovec_copy(rC, tC)
    cute.arch.barrier()

    # ── Epilogue: write Q@H [BT,BV] → o[chunk, 0, 0:BV] ──
    c_ident = cute.make_identity_tensor((BT, BV))
    tC_id = thr.partition_C(c_ident)
    for i in cutlass.range_constexpr(cute.size(tC_id)):
        coord = tC_id[i]
        row, col = coord[0], coord[1]
        o[(chunk_idx*BT+row,head_idx,v_offset+col)]=BFloat16(Float32(tC[i]))
    cute.arch.barrier()


class FwdOSM80:
    def __init__(self,chunk_size=64,head_dim_k=128,head_dim_v=64):
        assert_ampere(); self.C=chunk_size; self.D=head_dim_k; self.V=head_dim_v

    @cute.jit
    def __call__(self,q:cute.Tensor,h:cute.Tensor,g:cute.Tensor,o:cute.Tensor,
                 problem_size:tuple[cute.Int32,cute.Int32,cute.Int32,cute.Int32,cute.Int32],
                 scale:cutlass.Constexpr[float],stream):
        B,S,H,_,_=problem_size
        NV = (self.V + BV - 1) // BV
        op=MmaF16BF16Op(BFloat16,Float32,MS)
        tm=cute.make_tiled_mma(op,cute.make_layout(AL),permutation_mnk=PM)
        nc=cute.ceil_div(S,self.C)
        _fwd_o_sm80(tm,q,h,g,o,scale=scale,C=self.C,D=self.D,S=S,H=H,V=self.V,K=self.D
        ).launch(grid=(nc,H,NV),block=[NT,1,1],
                 smem=(2*self.C*self.D*NS)+(2*BK*BV)+(4*BT*BV)+2048,stream=stream)
