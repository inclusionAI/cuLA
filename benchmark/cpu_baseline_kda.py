"""
CPU Baseline for KDA Chunkwise - Two Chunks Case

This script computes step-by-step intermediate values for debugging.
Uses the same seed and initialization order as bench_kda.py.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from benchmark.utils import set_seed

# Constants - must match bench_kda.py
B, H, D = 1, 1, 128
S = T = 128
CHUNK_SIZE = 64
SEED = 42

def cpu_chunk_kda_detailed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # RAW g (not cumsum), same as naive_recurrent_kda expects
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    verbose: bool = True,
    dump_chunk1_file: str = None,  # File path to dump chunk 1 intermediate values
):
    """
    Detailed chunk-wise KDA implementation with intermediate value logging.
    
    NOTE: g should be RAW (not cumsum), we do cumsum internally per chunk.
    This matches what naive_recurrent_kda expects.
    
    Formula:
        A[i,j] = sum_d( k[i,d] * exp(g[i,d] - g[j,d]) * k[j,d] )  for i > j
        A = -A * beta  (lower triangular, masked)
        A = (I - A)^{-1} * diag(beta)  (Neumann series approximation)
        
        w = A @ (exp(g) * k)
        u = A @ v
        
        For each chunk i:
            pseudoV = u - w @ S
            O_intra = tril(Q * exp(g_Q) @ K^T) @ pseudoV
            O_inter = Q * exp(g_Q) @ S
            O = O_inter + O_intra
            
            S = S * exp(g_last) + K^T * exp(g_last - g_K) @ pseudoV
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT  # Number of chunks
    
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CPU Baseline KDA - Detailed Computation")
        print(f"B={B}, T={T}, H={H}, D={K}, chunk_size={BT}, num_chunks={NT}")
        print(f"{'='*80}")
    
    # Reshape to chunks: [B, H, NT, BT, D]
    q = rearrange(q, 'b (n c) h d -> b h n c d', c=BT).to(torch.float)
    k = rearrange(k, 'b (n c) h d -> b h n c d', c=BT).to(torch.float)
    v = rearrange(v, 'b (n c) h d -> b h n c d', c=BT).to(torch.float)
    g = rearrange(g, 'b (n c) h d -> b h n c d', c=BT).to(torch.float)
    beta = rearrange(beta, 'b (n c) h -> b h n c', c=BT).to(torch.float)
    
    q = q * scale
    
    # Cumsum within each chunk (same as naive_chunk_kda does)
    g = g.cumsum(-2)  # [B, H, NT, BT, D]
    
    if verbose:
        print(f"\nAfter reshaping and g_cumsum:")
        print(f"  q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        print(f"  g shape: {g.shape}, beta shape: {beta.shape}")
    
    # =========================================================================
    # Step 1: Compute A matrix (KK attention with decay)
    # A[i,j] = sum_d( k[i,d] * exp(g[i,d] - g[j,d]) * k[j,d] ) for i > j
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 1: Computing A matrix (KK^T with gating)")
        print(f"{'='*60}")
    
    mask_lower = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    
    A = torch.zeros(B, H, NT, BT, BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]  # [B, H, NT, D]
        g_i = g[..., i:i+1, :]  # [B, H, NT, 1, D]
        # A[..., i] = sum over d of: k[..., :, d] * exp(g[..., :, d] - g_i[d]) * k_i[d]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    
    A = A * beta[..., None]  # Scale by beta
    A = -A.masked_fill(mask_lower, 0)  # Keep only lower triangular, negate
    
    if verbose:
        print(f"  A (before inverse) shape: {A.shape}")
        print(f"  A[0,0,0] (chunk 0):\n{A[0,0,0,:8,:8]}")
        print(f"  A[0,0,1] (chunk 1):\n{A[0,0,1,:8,:8]}")
    
    # =========================================================================
    # Step 2: Compute (I - A)^{-1} using Neumann series
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 2: Computing M = (I - A)^{-1} * diag(beta)")
        print(f"{'='*60}")
    
    # Neumann series: (I - A)^{-1} = I + A + A^2 + A^3 + ...
    # For lower triangular A, this terminates
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    
    M = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]
    
    if verbose:
        print(f"  M (after inverse) shape: {M.shape}")
        print(f"  M[0,0,0] (chunk 0) first 8x8:\n{M[0,0,0,:8,:8]}")
        print(f"  M[0,0,1] (chunk 1) first 8x8:\n{M[0,0,1,:8,:8]}")
    
    # =========================================================================
    # Step 3: Compute w = M @ (exp(g) * k) and u = M @ v
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 3: Computing M matrix (already done above)")
        print(f"{'='*60}")
    
    # Note: w and u are no longer precomputed globally
    # Instead, pseudoV = M @ (v - k*exp(g) @ S) is computed per chunk
    
    if verbose:
        print(f"  M shape: {M.shape}")
    
    # =========================================================================
    # Step 4: Main loop over chunks
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("Step 4: Main loop over chunks")
        print(f"{'='*60}")
    
    # Initialize state S: [B, H, D_k, D_v]
    S = torch.zeros(B, H, K, V, dtype=torch.float, device=q.device)
    
    # Track state after each chunk for debugging
    states_after_chunk = []
    
    # Output tensor
    o = torch.zeros_like(v)
    
    # Mask for QK attention (upper triangular masked out)
    mask_qk = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    
    for chunk_idx in range(NT):
        # Save state at start of this chunk (for debugging)
        S_at_chunk_start = S.clone()
        
        if verbose:
            print(f"\n--- Chunk {chunk_idx} ---")
        
        # Extract chunk data
        q_i = q[:, :, chunk_idx]  # [B, H, BT, D]
        k_i = k[:, :, chunk_idx]  # [B, H, BT, D]
        v_i = v[:, :, chunk_idx]  # [B, H, BT, D]
        g_i = g[:, :, chunk_idx]  # [B, H, BT, D]
        M_i = M[:, :, chunk_idx]  # [B, H, BT, BT]
        
        if verbose:
            print(f"  q_i[:,:,:2,:4]:\n{q_i[0,0,:2,:4]}")
            print(f"  k_i[:,:,:2,:4]:\n{k_i[0,0,:2,:4]}")
            print(f"  g_i[:,:,:2,:4]:\n{g_i[0,0,:2,:4]}")
        
        # Compute QK attention matrix with gating
        # A_qk[j, i] = sum_d( q[j,d] * exp(g[j,d] - g[i,d]) * k[i,d] )
        A_qk = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, chunk_idx, j]  # [B, H, D]
            g_j = g[:, :, chunk_idx, j:j+1, :]  # [B, H, 1, D]
            # A_qk[..., j] shape: [B, H, BT]
            A_qk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        
        # Mask upper triangular (causal mask)
        A_qk = A_qk.masked_fill(mask_qk, 0)
        
        if verbose:
            print(f"  A_qk (QK attention, masked) first 8x8:\n{A_qk[0,0,:8,:8]}")
        
        # PseudoV = M @ (v - k*exp(g) @ S)
        # Step 1: k*exp(g) @ S: [B, H, BT, D] @ [B, H, D, D] -> [B, H, BT, D]
        kg_i = k_i * g_i.exp()  # k * exp(g)
        kS = torch.einsum('b h c k, b h k v -> b h c v', kg_i, S)
        
        # Step 2: v - kS
        v_minus_kS = v_i - kS
        
        # Step 3: pseudoV = M @ (v - kS)
        pseudoV = M_i @ v_minus_kS
        
        if verbose:
            print(f"  S (state before update) [:,:,:4,:4]:\n{S[0,0,:4,:4]}")
            print(f"  V [128x64] (transposed):\n{v_i[0,0].T}")
            print(f"  k*exp(g) [:,:,:2,:4]:\n{kg_i[0,0,:2,:4]}")
            print(f"  kS = k*exp(g) @ S [:,:,:2,:4]:\n{kS[0,0,:2,:4]}")
            print(f"  v - kS [128x64] (transposed):\n{v_minus_kS[0,0].T}")
            print(f"  pseudoV [128x64] (transposed):\n{pseudoV[0,0].T}")
        
        # O_intra = A_qk @ pseudoV
        O_intra = A_qk @ pseudoV
        
        if verbose:
            print(f"  O_intra = A_qk @ pseudoV [:,:,:2,:4]:\n{O_intra[0,0,:2,:4]}")
        
        # O_inter = (q * exp(g)) @ S
        # q * exp(g): [B, H, BT, D]
        # S: [B, H, D, D]
        qg = q_i * g_i.exp()
        O_inter = torch.einsum('b h c k, b h k v -> b h c v', qg, S)
        
        if verbose:
            print(f"  q * exp(g) [:,:,:2,:4]:\n{qg[0,0,:2,:4]}")
            print(f"  O_inter = qg @ S [:,:,:2,:4]:\n{O_inter[0,0,:2,:4]}")
        
        # O = O_inter + O_intra
        o[:, :, chunk_idx] = O_inter + O_intra
        
        if verbose:
            print(f"  O = O_inter + O_intra [:,:,:2,:4]:\n{o[0,0,chunk_idx,:2,:4]}")
        
        # =====================================================================
        # Update state for next chunk
        # S_new = S * exp(g_last) + K^T * exp(g_last - g_K) @ pseudoV
        # NOTE: g_last - g_i is non-positive (since g is cumsum), so exp is <= 1
        # =====================================================================
        g_last = g_i[:, :, -1]  # [B, H, D] - last row's g values
        
        if verbose:
            print(f"  g_last (g[-1] of chunk) [:,:,:8]:\n{g_last[0,0,:8]}")
        
        # S * exp(g_last): broadcast [B, H, D, D] * [B, H, D, 1]
        S_decay = S * rearrange(g_last.exp(), 'b h k -> b h k 1')
        
        if verbose:
            print(f"  exp(g_last) [:,:,:8]:\n{g_last.exp()[0,0,:8]}")
            print(f"  S * exp(g_last) [:,:,:4,:4]:\n{S_decay[0,0,:4,:4]}")
        
        # K^T * exp(g_last - g_K) @ pseudoV
        # NOTE: g_last - g_i (not g_i - g_last)!
        g_diff = g_last[:, :, None, :] - g_i  # [B, H, BT, D], g_last >= g_i, so diff <= 0
        k_weighted = k_i * g_diff.exp()  # K * exp(g_last - g)
        
        if verbose:
            print(f"  g_last - g_i (for K weighting) [:,:,:4,:4]:\n{g_diff[0,0,:4,:4]}")
            print(f"  exp(g_last - g_i) [:,:,:4,:4]:\n{g_diff.exp()[0,0,:4,:4]}")
        
        # K^T @ pseudoV: [B, H, D, BT] @ [B, H, BT, D] -> [B, H, D, D]
        # But we have k_weighted: [B, H, BT, D], pseudoV: [B, H, BT, D]
        # We want: sum over BT of k_weighted[t] outer pseudoV[t]
        KV_update = torch.einsum('b h c k, b h c v -> b h k v', k_weighted, pseudoV)
        
        if verbose:
            print(f"  KV_update = K_weighted^T @ pseudoV [:,:,:4,:4]:\n{KV_update[0,0,:4,:4]}")
        
        S = S_decay + KV_update
        
        if verbose:
            print(f"  S_new = S_decay + KV_update [:,:,:4,:4]:\n{S[0,0,:4,:4]}")
        
        if verbose:
            print(f"  S_new = S_decay + KV_update [:,:,:4,:4]:\n{S[0,0,:4,:4]}")
        
        # Dump all chunk intermediate values to file
        if dump_chunk1_file is not None:
            # Use 'w' mode for first chunk, 'a' mode for subsequent chunks
            mode = 'w' if chunk_idx == 0 else 'a'
            with open(dump_chunk1_file, mode) as f:
                if chunk_idx == 0:
                    f.write("=" * 80 + "\n")
                    f.write("CPU Baseline KDA - All Chunks Intermediate Values\n")
                    f.write("=" * 80 + "\n")
                
                f.write("\n")
                f.write("#" * 80 + "\n")
                f.write(f"# CHUNK {chunk_idx}\n")
                f.write("#" * 80 + "\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"1. State at START of chunk {chunk_idx} (V^T @ K format)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {S_at_chunk_start.shape} -> transposed to [D_v, D_k] = [128, 128]\n")
                f.write(f"State (V^T @ K):\n{S_at_chunk_start[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"2. V (original value)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {v_i.shape} -> transposed to [128, 64]\n")
                f.write(f"V [128x64] (transposed):\n{v_i[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"3. k*exp(g) (kg, used in kS = kg @ S)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {kg_i.shape} = [BT, D] = [64, 128]\n")
                f.write(f"k*exp(g):\n{kg_i[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"4. k*exp(g) @ S (kS, used in PseudoV = M @ (v - kS))\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {kS.shape} -> transposed to [128, 64]\n")
                f.write(f"kS (transposed, D x BT = 128 x 64):\n{kS[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"5. V - kS (v_minus_kS)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {v_minus_kS.shape} -> transposed to [128, 64]\n")
                f.write(f"V - kS [128x64] (transposed):\n{v_minus_kS[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"6. Q * exp(g) (Qg, used in O_inter = Qg @ S)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {qg.shape}\n")
                f.write(f"Qg:\n{qg[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"7. A_qk (QK attention matrix with gating, for O_intra)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {A_qk.shape}\n")
                f.write(f"A_qk:\n{A_qk[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"8. PseudoV = M @ (v - kS)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {pseudoV.shape} -> transposed to [128, 64]\n")
                f.write(f"PseudoV [128x64] (transposed):\n{pseudoV[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"9. O_intra = A_qk @ PseudoV\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {O_intra.shape}\n")
                f.write(f"O_intra:\n{O_intra[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"10. O_inter = Qg @ S (contribution from previous state)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {O_inter.shape}\n")
                f.write(f"O_inter:\n{O_inter[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"11. O = O_inter + O_intra (final output for chunk {chunk_idx})\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {o[:,:,chunk_idx].shape}\n")
                f.write(f"O:\n{o[0,0,chunk_idx].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"12. g_last (last row of g_cumsum in chunk {chunk_idx})\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {g_last.shape}\n")
                f.write(f"g_last:\n{g_last[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"13. exp(g_last) (decay factor for state)\n")
                f.write("-" * 60 + "\n")
                f.write(f"exp(g_last):\n{g_last.exp()[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"14. S_decay = S * exp(g_last) (state after decay, V^T @ K format)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {S_decay.shape} -> transposed to [D_v, D_k] = [128, 128]\n")
                f.write(f"S_decay (V^T @ K):\n{S_decay[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"15. g_diff = g_last - g_i (for K weighting)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {g_diff.shape}\n")
                f.write(f"g_diff:\n{g_diff[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"16. exp(g_diff) = exp(g_last - g_i)\n")
                f.write("-" * 60 + "\n")
                f.write(f"exp(g_diff):\n{g_diff.exp()[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"17. K_weighted = K * exp(g_last - g_i)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {k_weighted.shape}\n")
                f.write(f"K_weighted:\n{k_weighted[0,0].cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"18. KV_update = K_weighted^T @ PseudoV (V^T @ K format)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {KV_update.shape} -> transposed to [D_v, D_k] = [128, 128]\n")
                f.write(f"KV_update (V^T @ K):\n{KV_update[0,0].T.cpu().numpy()}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write(f"19. S_new = S_decay + KV_update (state after chunk {chunk_idx}, V^T @ K format)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Shape: {S.shape} -> transposed to [D_v, D_k] = [128, 128]\n")
                f.write(f"S_new (V^T @ K):\n{S[0,0].T.cpu().numpy()}\n\n")
                
            if chunk_idx == NT - 1:
                # Add footer after last chunk
                with open(dump_chunk1_file, 'a') as f:
                    f.write("=" * 80 + "\n")
                print(f"\n*** All chunks intermediate values dumped to: {dump_chunk1_file} ***\n")
    
    # Reshape output back to original shape
    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Final output shape: {o.shape}")
        print(f"Output [0, :8, 0, :4]:\n{o[0, :8, 0, :4]}")
        print(f"Output [0, 64:72, 0, :4] (chunk 1 start):\n{o[0, 64:72, 0, :4]}")
        print(f"{'='*80}")
    
    return o, S


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    
    # Use the same seed as bench_kda.py
    set_seed(SEED)
    
    scale = D ** (-0.5)
    
    # Initialize in the same order as bench_kda.py
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid()
    
    # Apply logsigmoid to g (same as bench_kda.py when use_gate_in_kernel=False)
    g = F.logsigmoid(g)
    
    # Apply L2 norm to q and k (same as cutedsl_kda_prefill with use_qk_l2norm_in_kernel=True)
    q_normed, _ = l2norm_fwd(q)
    k_normed, _ = l2norm_fwd(k)
    
    print(f"Input shapes:")
    print(f"  q: {q_normed.shape}, k: {k_normed.shape}, v: {v.shape}")
    print(f"  g: {g.shape}, beta: {beta.shape}")
    print(f"  scale: {scale}")
    
    print(f"\nFirst few values:")
    print(f"  q_normed[0,0,0,:8]: {q_normed[0,0,0,:8]}")
    print(f"  k_normed[0,0,0,:8]: {k_normed[0,0,0,:8]}")
    print(f"  v[0,0,0,:8]: {v[0,0,0,:8]}")
    print(f"  g[0,0,0,:8] (raw, before cumsum): {g[0,0,0,:8]}")
    print(f"  beta[0,0,0]: {beta[0,0,0]}")
    
    # Run detailed CPU baseline (pass raw g, cumsum is done inside)
    # Dump chunk 1 intermediate values to file for debugging
    dump_file = "/ossfs/workspace/flashla/benchmark/cpu_chunk1_debug.txt"
    o_cpu, final_state = cpu_chunk_kda_detailed(
        q=q_normed.clone(),
        k=k_normed.clone(),
        v=v.clone(),
        g=g.clone(),  # Raw g, cumsum done internally
        beta=beta.clone(),
        scale=scale,
        chunk_size=CHUNK_SIZE,
        verbose=True,
        dump_chunk1_file=dump_file,
    )
    
    # Compare with FLA's naive_recurrent_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    
    o_ref, _ = naive_recurrent_kda(
        q=q_normed.clone(),
        k=k_normed.clone(),
        v=v.clone(),
        g=g.clone(),  # Same raw g
        beta=beta.clone(),
        scale=scale,
        initial_state=None,
        output_final_state=False,
    )
    
    print(f"\n{'='*80}")
    print("Comparison with naive_recurrent_kda:")
    abs_err = (o_cpu - o_ref).abs().max().item()
    rel_err = ((o_cpu - o_ref).square().mean().sqrt() / (o_ref.square().mean().sqrt() + 1e-8)).item()
    print(f"  Absolute error: {abs_err}")
    print(f"  Relative error: {rel_err}")
    print(f"\n  o_cpu[0, :4, 0, :4]:\n{o_cpu[0, :4, 0, :4]}")
    print(f"  o_ref[0, :4, 0, :4]:\n{o_ref[0, :4, 0, :4]}")
    print(f"\n  o_cpu[0, 64:68, 0, :4] (chunk 1 start):\n{o_cpu[0, 64:68, 0, :4]}")
    print(f"  o_ref[0, 64:68, 0, :4] (chunk 1 start):\n{o_ref[0, 64:68, 0, :4]}")
    print(f"{'='*80}")
    
    if rel_err < 0.01:
        print("\n✓ CPU baseline matches naive_recurrent_kda!")
    else:
        print(f"\n✗ CPU baseline does NOT match naive_recurrent_kda (rel_err={rel_err:.4f})")
    
    return o_cpu, final_state


if __name__ == "__main__":
    main()
