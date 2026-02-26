import time
import random
import os
import sys

import torch
from flashla.kda.interface import kda_bwd_intra

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, List, Union, Optional, overload

from dataclasses import dataclass

# Ensure repo root is on sys.path so `cutile_test` is importable when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from cutile_test.kda_bwd_intra_cutile import cutile_kda_bwd_chunk_intra
except ImportError:
    cutile_kda_bwd_chunk_intra = None

torch.backends.cuda.matmul.allow_tf32 = True

@dataclass
class KDAParams:
    B: int 
    T: int
    H: int
    K: int
    q: torch.Tensor
    k: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    dAqk: torch.Tensor = None
    dAkk: torch.Tensor = None
    dq: torch.Tensor = None
    dk: torch.Tensor = None
    db: torch.Tensor = None
    dg: torch.Tensor = None
    cu_seqlens: torch.Tensor = None
    chunk_indices: torch.Tensor = None
    chunk_size: int = None
    
def fla_chunk_kda_bwd_intra(params: KDAParams, safe_gate: bool = False):
    q = params.q
    k = params.k
    g = params.g
    beta = params.beta
    dAqk = params.dAqk
    dAkk = params.dAkk
    dq = params.dq
    dk = params.dk
    db = params.db
    dg = params.dg
    cu_seqlens = params.cu_seqlens
    chunk_indices = params.chunk_indices
    chunk_size = params.chunk_size
    return chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, chunk_size, safe_gate)

def cutile_chunk_kda_bwd_intra(params: KDAParams, safe_gate: bool = False):
    q = params.q
    k = params.k
    g = params.g
    beta = params.beta
    dAqk = params.dAqk
    dAkk = params.dAkk
    dq = params.dq
    dk = params.dk
    db = params.db
    dg = params.dg
    cu_seqlens = params.cu_seqlens
    chunk_indices = params.chunk_indices
    chunk_size = params.chunk_size
    return cutile_kda_bwd_chunk_intra(
        q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )

def kda_bwd_intra_kernel_ref(dq_out, dk_out, db_out, dg_out, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, chunk_size, i_kc, i_t, i_bh):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = 32
    NC = (BT + BC - 1) // BC
    NK = (K + BK - 1) // BK
    NT = (T + BT - 1) // BT

    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    i_ti = i_t * BT + i_i * BC
    i_ki = i_k * BK

    b_g = g[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    b_b = beta[i_b, i_ti:i_ti+BC, i_h]
    b_dq2 = torch.zeros(BC, BK, device='cuda:0', dtype=torch.float32)
    b_dk2 = torch.zeros(BC, BK, device='cuda:0', dtype=torch.float32)
    if i_i > 0:
        b_gn = g[i_b, i_ti, i_h, i_ki:i_ki+BK]
        for j in range(0, i_i):
            i_tj = i_t * BT + j * BC

            b_k = k[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK]
            b_gk = g[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK]
            b_kg = b_k * torch.exp2(b_gn[None, :] - b_gk).squeeze(0)
            b_dAqk = dAqk[i_b, i_ti:i_ti+BC, i_h, j*BC:(j+1)*BC]
            b_dAkk = dAkk[i_b, i_ti:i_ti+BC, i_h, j*BC:(j+1)*BC]
            b_dq2 += b_dAqk.matmul(b_kg)
            b_dk2 += b_dAkk.matmul(b_kg)

        b_gqn = torch.exp2(b_g - b_gn[None, :])
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn
       
    o_i = torch.arange(0, BC, device='cpu')
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H*BT + o_i 

    b_q = q[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    b_k = k[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]

    b_gn = g[i_b, i_ti + min(BC // 2, T - i_ti - 1), i_h, i_ki:i_ki+BK].to(torch.float32)

    b_dAqk = dAqk[i_b, i_ti:i_ti+BC, i_h, i_i * BC:(i_i + 1) * BC]
    b_dAkk = dAkk[i_b, i_ti:i_ti+BC, i_h, i_i * BC:(i_i + 1) * BC]

    m_i_diag_qk = ((o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)).to('cuda:0')
    m_j_diag_qk = ((i_ti + o_i[:, None]) < T).to('cuda:0')
    b_dAqk_diag_qk = torch.where(m_i_diag_qk, b_dAqk, 0.0)
    b_dAkk_diag_qk = torch.where(m_i_diag_qk, b_dAkk, 0.0)
    b_g_diag_qk = torch.where(m_j_diag_qk, b_g - b_gn[None, :], 0.0)
    b_k_exp_diag_qk = torch.where(m_j_diag_qk, torch.exp2(b_g_diag_qk), 0.0)
    exp_neg_b_g_diag_qk = torch.where(m_j_diag_qk, torch.exp2(-b_g_diag_qk), 0.0)
    tmp_dq2 = b_dAqk_diag_qk.matmul(b_k * exp_neg_b_g_diag_qk)
    tmp_dk2 = b_dAkk_diag_qk.matmul(b_k * exp_neg_b_g_diag_qk)
    b_dq2 += tmp_dq2 * b_k_exp_diag_qk
    b_dk2 += tmp_dk2 * b_k_exp_diag_qk
    
    # for j in range(0, min(BC, T - i_ti)):
    #     b_dAqk = dAqk[i_b, i_ti:i_ti+BC, i_h, i_i * BC + j]
    #     b_dAkk = dAkk[i_b, i_ti:i_ti+BC, i_h, i_i * BC + j]
    #     b_kj = k[i_b, i_ti + j, i_h, i_ki:i_ki+BK].to(torch.float32)
    #     b_gkj = g[i_b, i_ti + j, i_h, i_ki:i_ki+BK].to(torch.float32)

    #     m_i = o_i[:, None] >= j
    #     b_gqk = torch.exp2(b_g - b_gkj[None, :])
        
    #     for idx_c in range(BC):
    #         for idx_k in range(BK):
    #             if m_i[idx_c]:
    #                 b_dq2[idx_c, idx_k] += b_dAqk[idx_c] * b_kj[idx_k] * b_gqk[idx_c, idx_k]
    #                 b_dk2[idx_c, idx_k] += b_dAkk[idx_c] * b_kj[idx_k] * b_gqk[idx_c, idx_k]

    
    b_db = torch.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]
    

    b_dg = b_q * b_dq2
    b_dq2 = b_dq2 + dq[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    dq_out[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK] = b_dq2.to(torch.float16)
    db_out[i_k, i_b, i_ti:i_ti+BC, i_h] = b_db.to(torch.float32)

    b_dkt = torch.zeros(BC, BK, device='cuda:0', dtype=torch.float32)
    if i_i < NC - 1:
        b_gn = g[i_b, i_ti + BC, i_h, i_ki:i_ki+BK]
        for j in range(i_i + 1, NC):
            i_tj = i_t * BT + j * BC
            b_b = beta[i_b, i_tj:i_tj+BC, i_h]

            b_q = q[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK]
            b_kb = k[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK] * b_b[:, None]
            b_gk = g[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK]

            b_dAqk = dAqk[i_b, i_tj:i_tj+BC, i_h, i_i*BC:(i_i+1)*BC].transpose(0, 1)
            b_dAkk = dAkk[i_b, i_tj:i_tj+BC, i_h, i_i*BC:(i_i+1)*BC].transpose(0, 1)
            
            o_j = i_t * BT + j * BC + o_i
            m_j = o_j < T
            b_gkn = torch.exp2(b_gk - b_gn[None, :])
            b_qg = torch.zeros(BC, BK, device='cuda:0', dtype=torch.float32)
            b_kbg = torch.zeros(BC, BK, device='cuda:0', dtype=torch.float32)
            for idx_c in range(BC):
                for idx_k in range(BK):
                    if m_j[idx_c]:
                        b_qg[idx_c, idx_k] = b_q[idx_c, idx_k] * b_gkn[idx_c, idx_k]
                        b_kbg[idx_c, idx_k] = b_kb[idx_c, idx_k] * b_gkn[idx_c, idx_k]
            # if (i_k == 0 and i_i == 0 and j == 1):
            #     print(f"k: {k[i_b, i_tj:i_tj+BC, i_h, i_ki:i_ki+BK]}")
            #     print(f"b_kb: {b_kb}")
            #     print(f"b_b: {b_b}")
            #     print(f"b_gkn: {b_gkn}")
            #     print(f"b_qg: {b_qg}")
            #     print(f"b_kbg: {b_kbg}")
            b_dkt += b_dAqk.matmul(b_qg)
            b_dkt += b_dAkk.matmul(b_kbg)
        # if (i_k == 0):
        #     print(f"i_i: {i_i}, j: {j}")
        #     print(f"b_dkt before: {b_dkt}")
        b_dkt *= torch.exp2(b_gn[None, :] - b_g).to(torch.float32)
    tmp_b_dkt = torch.zeros_like(b_dkt, device='cuda:0', dtype=torch.float32)
    
    o_dA = i_ti * H*BT + i_i * BC + o_i
    # for j in range(0, min(BC, T - i_ti)):
    #     b_dAqk = dAqk[i_b, i_ti + j, i_h, i_i * BC:(i_i+1)*BC]
    #     b_dAkk = dAkk[i_b, i_ti + j, i_h, i_i * BC:(i_i+1)*BC]

    #     b_qj = q[i_b, i_ti + j, i_h, i_ki:i_ki+BK].to(torch.float32)
    #     b_kbj = k[i_b, i_ti + j, i_h, i_ki:i_ki+BK].to(torch.float32) * beta[i_b, i_ti + j, i_h]
    #     b_gkj = g[i_b, i_ti + j, i_h, i_ki:i_ki+BK].to(torch.float32)

    #     m_i = o_i[:, None] <= j
    #     b_gkq = torch.exp2(b_gkj[None, :] - b_g)
    #     for idx_c in range(BC):
    #         for idx_k in range(BK):
    #             if m_i[idx_c]:
    #                 tmp_b_dkt[idx_c, idx_k] += b_dAqk[idx_c] * b_qj[idx_k] * b_gkq[idx_c, idx_k]
    #                 tmp_b_dkt[idx_c, idx_k] += b_dAkk[idx_c] * b_kbj[idx_k] * b_gkq[idx_c, idx_k]
    b_gn = g[i_b, i_ti + min(BC // 2, T - i_ti - 1), i_h, i_ki:i_ki+BK].to(torch.float32)
    b_b = beta[i_b, i_ti:i_ti+BC, i_h]
    b_q = q[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    b_dAqk = dAqk[i_b, i_ti:i_ti+BC, i_h, i_i * BC:(i_i + 1) * BC].transpose(0, 1)
    b_dAkk = dAkk[i_b, i_ti:i_ti+BC, i_h, i_i * BC:(i_i + 1) * BC].transpose(0, 1)

    m_i_diag_qk = ((o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)).to('cuda:0')
    b_dAqk_diag_qk = torch.where(m_i_diag_qk, b_dAqk, 0.0)
    b_dAkk_diag_qk = torch.where(m_i_diag_qk, b_dAkk, 0.0)
    b_g_diag_qk = torch.where(m_j_diag_qk, b_g - b_gn[None, :], 0.0)
    b_k_exp_diag_qk = torch.where(m_j_diag_qk, torch.exp2(b_g_diag_qk), 0.0)
    exp_neg_b_g_diag_qk = torch.where(m_j_diag_qk, torch.exp2(-b_g_diag_qk), 0.0)
    b_q_exp = b_q * b_k_exp_diag_qk
    b_kb_exp = b_k * b_b[:, None] * b_k_exp_diag_qk
    # if (i_k == 0):
    #     print("i_i: ", i_i)
    #     print("b_q: ", b_q)
    #     print("b_k: ", b_k)
    #     print("b_b: ", b_b)
    #     print("b_g_diag_qk: ", b_g_diag_qk)
    #     print("b_k_exp_diag_qk: ", b_k_exp_diag_qk)
    #     print("b_q_exp: ", b_q_exp)
    #     print("b_kb_exp: ", b_kb_exp)
    #     print("exp_neg_b_g_diag_qk: ", exp_neg_b_g_diag_qk)
    tmp_b_dkt = b_dAqk_diag_qk.matmul(b_q_exp)
    tmp_b_dkt += b_dAkk_diag_qk.matmul(b_kb_exp)
    # if (i_k == 0): 
    #     print(f"i_i: {i_i}")
    #     print("tmp_b_dkt before scaling: ", tmp_b_dkt)
    tmp_b_dkt *= exp_neg_b_g_diag_qk
    # if (i_i == 0 and i_k == 0):
    #     print(f"b_dg(old): {b_dg[0, :32]}")
    #     print(f"b_dk2: {b_dk2[0, :32] - tmp_b_dkt[0, :32]}")
    #     print(f"b_dkt: {b_dkt[0, :32]}")
    #     print(f"b_k: {b_k[0, :32]}")
    #     print(f"dg(new): {dg[0, 0, 0, :32]}")
    b_dkt += tmp_b_dkt
    b_dg += (b_dk2 - b_dkt) * b_k # + dg[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    b_dk2 += dk[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK]
    b_dk2 += b_dkt
    dk_out[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK] = b_dk2.to(torch.float16)
    dg_out[i_b, i_ti:i_ti+BC, i_h, i_ki:i_ki+BK] = b_dg.to(torch.float32)

def kda_bwd_intra_cuda(params: KDAParams):
    q = params.q.to('cuda:0')
    k = params.k.to('cuda:0')
    g = params.g.to('cuda:0')
    beta = params.beta.to('cuda:0')
    dAqk = params.dAqk.to('cuda:0')
    dAkk = params.dAkk.to('cuda:0')
    dq = params.dq.to('cuda:0')
    dk = params.dk.to('cuda:0')
    db = params.db.to('cuda:0')
    dg = params.dg.to('cuda:0')
    if params.cu_seqlens is not None:
        cu_seqlens = params.cu_seqlens.to('cuda:0')
    else:
        cu_seqlens = None
    if params.chunk_indices is not None:
        chunk_indices = params.chunk_indices.to('cuda:0')
    else:
        chunk_indices = None
    chunk_size = params.chunk_size
    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    return kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, dq_out, dk_out, db_out, dg_out, chunk_size, tile_counter)


def kda_bwd_intra_ref(params: KDAParams):
    q = params.q.to('cuda:0')
    k = params.k.to('cuda:0')
    g = params.g.to('cuda:0')
    beta = params.beta.to('cuda:0')
    dAqk = params.dAqk.to('cuda:0')
    dAkk = params.dAkk.to('cuda:0')
    dq = params.dq.to('cuda:0')
    dk = params.dk.to('cuda:0')
    db = params.db.to('cuda:0')
    dg = params.dg.to('cuda:0')
    cu_seqlens = params.cu_seqlens
    chunk_indices = params.chunk_indices

    chunk_size = params.chunk_size

    # cu_seqlens_cpu = cu_seqlens.to('cpu')
    # chunk_indices_cpu = chunk_indices.to('cpu')

    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = 32
    NC = (BT + BC - 1) // BC
    NK = (K + BK - 1) // BK
    NT = (T + BT - 1) // BT
    grid = (NK * NC, NT, B * H)
    print(f"NK: {NK}, NC: {NC}, NT: {NT}, B: {B}, H: {H}")

    dq_out = torch.empty_like(q, dtype=torch.float16)
    dk_out = torch.empty_like(k, dtype=torch.float16)
    db_out = beta.new_empty(NK, *beta.shape, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    
    for i in range(B * H):
        for j in range(NT):
            for i_kc in range(NC * NK):
                kda_bwd_intra_kernel_ref(dq_out, dk_out, db_out, dg_out, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, chunk_size, i_kc, j, i)
    # kda_bwd_intra_kernel_ref(dq_out_cpu, dk_out_cpu, db_out_cpu, dg_out_cpu, q_cpu, k_cpu, g_cpu, beta_cpu, dAqk_cpu, dAkk_cpu, dq_cpu, dk_cpu, db_cpu, dg_cpu, cu_seqlens_cpu, chunk_indices_cpu, chunk_size, 0, 0, 0)

    db_out = db_out.sum(0).add_(db)
    dg_out = chunk_local_cumsum(
        dg_out.add_(dg),
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        #chunk_indices=chunk_indices,
    )
    return dq_out, dk_out, db_out, dg_out
    # for i in range(B * H):

def generate_data(seed=42, B=4, T=128, H=4, K=128, varlen = False):
    torch.manual_seed(seed)
    random.seed(seed)
    BT = 64
    cu_seqlens = torch.zeros(B + 1, device='cuda:0', dtype=torch.int32)
    cu_tile_indices = torch.zeros(B + 1, device='cuda:0', dtype=torch.int32)
    TILE_NUM = int(0)
    if varlen:
        cu_seqlens[0] = 0
        cu_tile_indices[0] = 0
        for i in range(1, B + 1):
            seq_len = min(random.normalvariate(T, T / 2), T*2)
            cu_seqlens[i] = cu_seqlens[i - 1] + seq_len
            TILE_NUM += int((seq_len + 64 - 1) // 64)
            cu_tile_indices[i] = TILE_NUM
    else:
        for i in range(B + 1):
            cu_seqlens[i] = i * T
        TILE_NUM = B * ((T + 64 - 1) // 64)
    chunk_indices = torch.zeros(TILE_NUM*2, device='cuda:0', dtype=torch.int32)
    
    acc_num = 0
    for i in range(B):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        current_tile_num = (seq_len + 64 - 1) // 64
        for j in range(current_tile_num):
            chunk_indices[acc_num * 2 + j * 2] = i
            chunk_indices[acc_num * 2 + j * 2 + 1] = j
        acc_num += current_tile_num
    chunk_indices = chunk_indices.reshape(TILE_NUM, 2)

    total_len = cu_seqlens[-1]
    
    # T = T.to('cuda:0')
    q = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.bfloat16) 
    k = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.bfloat16)
    g = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.float32) / 10
    beta = torch.randn(1, total_len, H, device='cuda', dtype=torch.bfloat16)
    dAqk = torch.randn(1, total_len, H, BT, device='cuda', dtype=torch.float32)
    dAkk = torch.randn(1, total_len, H, BT, device='cuda', dtype=torch.float32)
    dq = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.float32)
    dk = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.float32)
    db = torch.randn(1, total_len, H, device='cuda', dtype=torch.float32)
    dg = torch.randn(1, total_len, H, K, device='cuda', dtype=torch.float32)
    
    return KDAParams(B=B, T=T, H=H, K=K, q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)

def load_data(data_path, B = None, T = None, H = None, dtype=torch.bfloat16):
    q = torch.load(os.path.join(data_path, "q.pt")).to(dtype)
    k = torch.load(os.path.join(data_path, "k.pt")).to(dtype)
    g = torch.load(os.path.join(data_path, "g.pt"))
    beta = torch.load(os.path.join(data_path, "beta.pt")).to(dtype)
    dAqk = torch.load(os.path.join(data_path, "dAqk.pt"))
    dAkk = torch.load(os.path.join(data_path, "dAkk.pt"))
    dq = torch.load(os.path.join(data_path, "dq.pt"))
    dk = torch.load(os.path.join(data_path, "dk.pt"))
    db = torch.load(os.path.join(data_path, "db.pt"))
    dg = torch.load(os.path.join(data_path, "dg.pt"))
    if B is None:
        B, T, H, K = q.shape
    else:
        K = q.shape[3]
        q = q[:B, :T, :H, :].to(dtype).contiguous()
        k = k[:B, :T, :H, :].to(dtype).contiguous()
        g = g[:B, :T, :H, :].contiguous()
        beta = beta[:B, :T, :H].to(dtype).contiguous()
        dAqk = dAqk[:B, :T, :H, :].contiguous()
        dAkk = dAkk[:B, :T, :H, :].contiguous()
        dq = dq[:B, :T, :H, :].contiguous()
        dk = dk[:B, :T, :H, :].to(dtype).contiguous()
        db = db[:B, :T, :H].contiguous()
        dg = dg[:B, :T, :H, :].contiguous()
    
    q = q.reshape(1, T*B, H, K)
    k = k.reshape(1, T*B, H, K)
    g = g.reshape(1, T*B, H, K)
    beta = beta.reshape(1, T*B, H)
    dAqk = dAqk.reshape(1, T*B, H, 64)
    dAkk = dAkk.reshape(1, T*B, H, 64)
    dq = dq.reshape(1, T*B, H, K)
    dk = dk.reshape(1, T*B, H, K)
    db = db.reshape(1, T*B, H)
    dg = dg.reshape(1, T*B, H, K)

    cu_seqlens = torch.zeros(B + 1, device='cuda:0', dtype=torch.int32)
    TILE_NUM = (T+64-1)//64
    chunk_indices = torch.zeros(B*TILE_NUM*2, device='cuda:0', dtype=torch.int32)
    for i in range(B):
        for j in range(TILE_NUM):
            chunk_indices[i * TILE_NUM * 2 + j * 2] = i
            chunk_indices[i * TILE_NUM * 2 + j * 2 + 1] = j
    cu_seqlens[0] = 0
    for i in range(1, B + 1):
        cu_seqlens[i] = cu_seqlens[i - 1] + T

    params = KDAParams(B=B, T=T, H=H, K=K, q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=64)
    return params

def test_kda_bwd_intra():
    load_path = "/home/scratch.zeyuw_gpu/gitlab_repo/nv-moonshot-kda-internal/dataset"
    # params = load_data(load_path, B=1, T=64, H=1)
    # params = load_data(load_path)
    # params = generate_data(seed=42, B=148, T=128, H=1, K=128, varlen=False)
    # params = generate_data(seed=42, B=1, T=640, H=1, K=128, varlen=False)
    params = generate_data(seed=42, B=10, T=800, H=96, K=128, varlen=True)
    # params = generate_data(seed=42, B=16, T=8192, H=96, K=128, varlen=False)
    dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)

    for i in range(100):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)

    dq_out_baseline, dk_out_baseline, db_out_baseline, dg_out_baseline = kda_bwd_intra_cuda(params)
    for i in range(20):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)
        assert torch.equal(dq_out, dq_out_baseline)
        assert torch.equal(dk_out, dk_out_baseline)
        assert torch.equal(db_out, db_out_baseline)
        assert torch.equal(dg_out, dg_out_baseline)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        dq_out, dk_out, db_out, dg_out = kda_bwd_intra_cuda(params)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"CUDA Time taken: {(end_time - start_time) * 1000/100:.6f} ms")
    

    # fla
    for i in range(10):
        dq2, dk2, db2, dg2 = fla_chunk_kda_bwd_intra(params, True)
    # print(f"dq2: {dq2}")
    # print(f"dk2: {dk2}")
    # print(f"db2: {db2}")
    # print(f"dg2: {dg2}")
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        dq2, dk2, db2, dg2 = fla_chunk_kda_bwd_intra(params, True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"FLA Time taken: {(end_time - start_time) * 1000/100:.6f} ms")

    assert_close("dq", dq2, dq_out, 0.008)
    assert_close("dk", dk2, dk_out, 0.008)
    assert_close("db", db2, db_out, 0.02)
    assert_close("dg", dg2, dg_out, 0.02)


def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    dtype: torch.dtype,
    tma: bool,
):
    torch.manual_seed(42)
    if not tma:
        os.environ["FLA_USE_TMA"] = "0"
    else:
        os.environ["FLA_USE_TMA"] = "1"

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float)
    if use_gate_in_kernel:
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
    else:
        g = F.logsigmoid(g) / gate_logit_normalizer
        g = g * (torch.rand_like(g) > mask_p)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # ref, ref_ht = naive_recurrent_kda(
    #     q=F.normalize(q.clone(), p=2, dim=-1),
    #     k=F.normalize(k.clone(), p=2, dim=-1),
    #     v=v.clone(),
    #     g=(naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
    #     beta=beta.clone(),
    #     scale=scale,
    #     initial_state=h0.clone(),
    #     output_final_state=True,
    # )
    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # if use_gate_in_kernel:
    #     ref_dA, A_log.grad = A_log.grad, None
    #     ref_dbias, dt_bias.grad = dt_bias.grad, None
    # ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    # q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None
    print("init done")

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
    )
    print("forward done")
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    print("backward done")
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    # assert_close("o", ref, tri, 0.005)
    # assert_close("ht", ref_ht, tri_ht, 0.005)
    # assert_close("dq", ref_dq, tri_dq, 0.008)
    # assert_close("dk", ref_dk, tri_dk, 0.008)
    # assert_close("dv", ref_dv, tri_dv, 0.008)
    # assert_close("dg", ref_dg, tri_dg, 0.02)
    # assert_close("db", ref_db, tri_db, 0.02)
    # if use_gate_in_kernel:
    #     assert_close("dA", ref_dA, tri_dA, 0.001)
    #     assert_close("dbias", ref_dbias, tri_dbias, 0.001)
    # assert_close("dh0", ref_dh0, tri_dh0, 0.008)

    
if __name__ == "__main__":
    test_kda_bwd_intra()
   