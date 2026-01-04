# Copyright (c) 2024-2025 FlashLA Authors
# Licensed under the MIT License

"""
PyTorch Reference Implementation of Linear Attention with Exponential Decay

This module provides baseline implementations for accuracy comparison:
1. naive_linear_attn_decay: O(n²) loop-based reference implementation
2. chunkwise_linear_attn_decay: O(n) chunkwise parallel implementation

Mathematical formulation:
    O_t = sum_{i=1}^{t} λ^{t-i} * Q_t * K_i^T * V_i
    
Where λ = exp(-s) is the per-head exponential decay factor.

The chunkwise algorithm decomposes computation into:
    - Intra-chunk: Attention within a block with decay mask
    - Inter-chunk: Attention from accumulated state with query decay
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def naive_linear_attn_decay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """
    Naive O(n²) implementation of linear attention with exponential decay.
    
    This is the reference implementation for accuracy verification.
    NOT recommended for production use due to quadratic complexity.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, d_k]
        k: Key tensor of shape [batch, heads, seq_len, d_k]
        v: Value tensor of shape [batch, heads, seq_len, d_v]
        s: Log-lambda decay tensor of shape [heads] or [batch, heads]
           Decay factor λ = exp(-s), where s > 0
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, d_v]
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    
    # Normalize s shape to [b, h, 1, 1] for broadcasting
    if s.dim() == 1:
        s = s.view(1, h, 1, 1)
    elif s.dim() == 2:
        s = s.view(b, h, 1, 1)
    else:
        s = s.view(b, h, 1, 1)
    
    # Create position indices
    positions = torch.arange(n, device=q.device, dtype=q.dtype)
    
    # Compute position difference matrix: [n, n]
    # pos_diff[i, j] = i - j
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    
    # Create causal decay mask: [1, 1, n, n]
    # decay[i, j] = exp(-s * (i - j)) if i >= j else 0
    # Using -inf for masked positions before exp
    decay_mask = torch.where(
        pos_diff >= 0,
        -s * pos_diff.float(),  # -s * (i - j)
        torch.tensor(float('-inf'), device=q.device, dtype=q.dtype)
    )
    decay_mask = torch.exp(decay_mask)  # [b, h, n, n]
    
    # Compute attention: QK^T with decay mask
    # [b, h, n, d] @ [b, h, d, n] -> [b, h, n, n]
    qk = torch.matmul(q, k.transpose(-2, -1))
    
    # Apply causal decay mask
    qk = qk * decay_mask
    
    # Compute output: [b, h, n, n] @ [b, h, n, e] -> [b, h, n, e]
    o = torch.matmul(qk, v)
    
    return o


def naive_linear_attn_decay_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recurrent O(n) implementation of linear attention with exponential decay.
    
    This demonstrates the RNN-like state update mechanism.
    Sequential implementation - slow but memory efficient.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, d_k]
        k: Key tensor of shape [batch, heads, seq_len, d_k]
        v: Value tensor of shape [batch, heads, seq_len, d_v]
        s: Log-lambda decay tensor of shape [heads] or [batch, heads]
    
    Returns:
        Tuple of:
            - Output tensor of shape [batch, heads, seq_len, d_v]
            - Final state tensor of shape [batch, heads, d_k, d_v]
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    
    # Normalize s shape
    if s.dim() == 1:
        s = s.view(1, h, 1, 1)
    elif s.dim() == 2:
        s = s.view(b, h, 1, 1)
    
    # Decay factor: λ = exp(-s)
    decay = torch.exp(-s)  # [b, h, 1, 1]
    
    # Initialize state: S_0 = 0
    state = torch.zeros(b, h, d, e, device=q.device, dtype=torch.float32)
    
    outputs = []
    
    for t in range(n):
        # Get current tokens
        q_t = q[:, :, t:t+1, :]  # [b, h, 1, d]
        k_t = k[:, :, t:t+1, :]  # [b, h, 1, d]
        v_t = v[:, :, t:t+1, :]  # [b, h, 1, e]
        
        # State update: S_t = λ * S_{t-1} + K_t^T @ V_t
        # K_t^T: [b, h, d, 1], V_t: [b, h, 1, e] -> [b, h, d, e]
        kv_t = torch.matmul(k_t.transpose(-2, -1), v_t.float())
        state = decay * state + kv_t
        
        # Output: O_t = Q_t @ S_t
        # Q_t: [b, h, 1, d], S_t: [b, h, d, e] -> [b, h, 1, e]
        o_t = torch.matmul(q_t.float(), state)
        outputs.append(o_t)
    
    # Stack outputs: [b, h, n, e]
    o = torch.cat(outputs, dim=2)
    
    return o.to(q.dtype), state


def chunkwise_linear_attn_decay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunkwise O(n) implementation of linear attention with exponential decay.
    
    This is the parallel-friendly implementation that matches the Triton kernel.
    Decomposes computation into intra-chunk and inter-chunk components.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, d_k]
        k: Key tensor of shape [batch, heads, seq_len, d_k]
        v: Value tensor of shape [batch, heads, seq_len, d_v]
        s: Log-lambda decay tensor of shape [heads] or [batch, heads]
        chunk_size: Size of each chunk (BLOCK in Triton implementation)
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, d_v]
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    
    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - n % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
    
    n_padded = q.shape[2]
    num_chunks = n_padded // chunk_size
    
    # Normalize s shape to [b, h]
    if s.dim() == 1:
        s = s.unsqueeze(0).expand(b, -1)  # [b, h]
    
    # Reshape to chunks: [b, h, num_chunks, chunk_size, dim]
    q_chunks = q.view(b, h, num_chunks, chunk_size, d)
    k_chunks = k.view(b, h, num_chunks, chunk_size, d)
    v_chunks = v.view(b, h, num_chunks, chunk_size, e)
    
    # Compute decay factors
    # Position indices within chunk: [chunk_size]
    chunk_pos = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
    
    # q_decay: exp(-s * position) for each position in chunk
    # Shape: [b, h, chunk_size, 1]
    q_decay = torch.exp(-s.view(b, h, 1, 1) * chunk_pos.view(1, 1, -1, 1))
    
    # k_trans_decay: exp(-s * (chunk_size - position)) for state accumulation
    # Shape: [b, h, 1, chunk_size]
    k_trans_decay = torch.exp(-s.view(b, h, 1, 1) * (chunk_size - chunk_pos).view(1, 1, 1, -1))
    
    # block_decay: exp(-s * chunk_size) for state transition between chunks
    # Shape: [b, h, 1, 1]
    block_decay = torch.exp(-s.view(b, h, 1, 1) * chunk_size)
    
    # Intra-chunk causal decay mask
    # pos_diff[i, j] = i - j
    pos_diff = chunk_pos.unsqueeze(1) - chunk_pos.unsqueeze(0)  # [chunk_size, chunk_size]
    
    # diag_decay[i, j] = exp(-s * (i - j)) if i >= j else 0
    diag_decay = torch.where(
        pos_diff >= 0,
        torch.exp(-s.view(b, h, 1, 1) * pos_diff.view(1, 1, chunk_size, chunk_size)),
        torch.zeros(1, device=q.device, dtype=torch.float32)
    )  # [b, h, chunk_size, chunk_size]
    
    # Initialize state: [b, h, d, e]
    state = torch.zeros(b, h, d, e, device=q.device, dtype=torch.float32)
    
    outputs = []
    
    for c in range(num_chunks):
        # Get current chunk
        q_c = q_chunks[:, :, c]  # [b, h, chunk_size, d]
        k_c = k_chunks[:, :, c]  # [b, h, chunk_size, d]
        v_c = v_chunks[:, :, c]  # [b, h, chunk_size, e]
        
        # ===== Intra-chunk computation =====
        # QK^T within chunk: [b, h, chunk_size, chunk_size]
        qk_intra = torch.matmul(q_c.float(), k_c.float().transpose(-2, -1))
        
        # Apply causal decay mask
        qk_intra = qk_intra * diag_decay  # [b, h, chunk_size, chunk_size]
        
        # Output from intra-chunk: [b, h, chunk_size, e]
        o_intra = torch.matmul(qk_intra, v_c.float())
        
        # ===== Inter-chunk computation =====
        # Output from previous state: Q @ S * q_decay
        # [b, h, chunk_size, d] @ [b, h, d, e] -> [b, h, chunk_size, e]
        o_inter = torch.matmul(q_c.float(), state) * q_decay
        
        # ===== Combine outputs =====
        o_c = o_intra + o_inter
        outputs.append(o_c)
        
        # ===== Update state for next chunk =====
        # K^T with decay: [b, h, d, chunk_size] * [b, h, 1, chunk_size]
        k_trans_decayed = k_c.float().transpose(-2, -1) * k_trans_decay
        
        # New contribution: K^T_decayed @ V
        # [b, h, d, chunk_size] @ [b, h, chunk_size, e] -> [b, h, d, e]
        kv_new = torch.matmul(k_trans_decayed, v_c.float())
        
        # State update: S = block_decay * S + KV_new
        state = block_decay * state + kv_new
    
    # Concatenate outputs: [b, h, n_padded, e]
    o = torch.cat(outputs, dim=2)
    
    # Remove padding
    if pad_len > 0:
        o = o[:, :, :n, :]
    
    return o.to(q.dtype)


def chunkwise_linear_attn_decay_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Fully parallel chunkwise implementation using cumulative operations.
    
    This version pre-computes all chunk states in parallel, avoiding the
    sequential loop in chunkwise_linear_attn_decay.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, d_k]
        k: Key tensor of shape [batch, heads, seq_len, d_k]
        v: Value tensor of shape [batch, heads, seq_len, d_v]
        s: Log-lambda decay tensor of shape [heads] or [batch, heads]
        chunk_size: Size of each chunk
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, d_v]
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    
    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - n % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
    
    n_padded = q.shape[2]
    num_chunks = n_padded // chunk_size
    
    # Normalize s shape
    if s.dim() == 1:
        s = s.unsqueeze(0).expand(b, -1)
    
    # Reshape to chunks
    q_chunks = q.view(b, h, num_chunks, chunk_size, d).float()
    k_chunks = k.view(b, h, num_chunks, chunk_size, d).float()
    v_chunks = v.view(b, h, num_chunks, chunk_size, e).float()
    
    # Compute decay factors
    chunk_pos = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
    
    # Query decay for inter-chunk: [b, h, 1, chunk_size, 1]
    q_decay = torch.exp(-s.view(b, h, 1, 1, 1) * chunk_pos.view(1, 1, 1, -1, 1))
    
    # Key decay for state: [b, h, 1, 1, chunk_size]
    k_trans_decay = torch.exp(-s.view(b, h, 1, 1, 1) * (chunk_size - chunk_pos).view(1, 1, 1, 1, -1))
    
    # Block decay: [b, h, 1, 1]
    block_decay = torch.exp(-s.view(b, h, 1, 1) * chunk_size)
    
    # Intra-chunk decay mask: [b, h, 1, chunk_size, chunk_size]
    pos_diff = chunk_pos.unsqueeze(1) - chunk_pos.unsqueeze(0)
    diag_decay = torch.where(
        pos_diff >= 0,
        torch.exp(-s.view(b, h, 1, 1, 1) * pos_diff.view(1, 1, 1, chunk_size, chunk_size)),
        torch.zeros(1, device=q.device, dtype=torch.float32)
    )
    
    # ===== Compute intra-chunk attention for all chunks in parallel =====
    # [b, h, num_chunks, chunk_size, d] @ [b, h, num_chunks, d, chunk_size]
    qk_intra = torch.matmul(q_chunks, k_chunks.transpose(-2, -1))  # [b, h, num_chunks, chunk_size, chunk_size]
    qk_intra = qk_intra * diag_decay  # Apply decay mask
    o_intra = torch.matmul(qk_intra, v_chunks)  # [b, h, num_chunks, chunk_size, e]
    
    # ===== Compute KV for each chunk =====
    # K^T with decay: [b, h, num_chunks, d, chunk_size]
    k_trans_decayed = k_chunks.transpose(-2, -1) * k_trans_decay
    # KV per chunk: [b, h, num_chunks, d, e]
    kv_chunks = torch.matmul(k_trans_decayed, v_chunks)
    
    # ===== Compute cumulative state with decay =====
    # We need: state[c] = sum_{i=0}^{c-1} block_decay^{c-1-i} * kv[i]
    # This is a weighted cumsum with exponential decay
    
    # Create decay weights for cumsum: [num_chunks, num_chunks]
    chunk_indices = torch.arange(num_chunks, device=q.device, dtype=torch.float32)
    decay_matrix = chunk_indices.unsqueeze(1) - chunk_indices.unsqueeze(0)  # [num_chunks, num_chunks]
    
    # decay_weights[i, j] = block_decay^{i-j-1} if i > j else 0
    # For state[i], we sum kv[0..i-1] with weights block_decay^{i-1-j}
    # block_decay: [b, h, 1, 1], decay_matrix: [num_chunks, num_chunks]
    # Result should be [b, h, num_chunks, num_chunks]
    decay_powers = decay_matrix.unsqueeze(0).unsqueeze(0) - 1  # [1, 1, num_chunks, num_chunks]
    decay_weights = torch.where(
        decay_matrix > 0,
        block_decay ** decay_powers,
        torch.zeros(1, device=q.device, dtype=torch.float32)
    )  # [b, h, num_chunks, num_chunks]
    
    # Compute cumulative states: [b, h, num_chunks, d, e]
    # states[c] = sum_{i<c} decay_weights[c,i] * kv_chunks[i]
    # Reshape for batch matmul
    kv_flat = kv_chunks.view(b, h, num_chunks, d * e)  # [b, h, num_chunks, d*e]
    states_flat = torch.matmul(decay_weights, kv_flat)  # [b, h, num_chunks, d*e]
    states = states_flat.view(b, h, num_chunks, d, e)  # [b, h, num_chunks, d, e]
    
    # ===== Compute inter-chunk attention =====
    # o_inter[c] = Q[c] @ state[c] * q_decay
    # [b, h, num_chunks, chunk_size, d] @ [b, h, num_chunks, d, e] -> [b, h, num_chunks, chunk_size, e]
    o_inter = torch.matmul(q_chunks, states) * q_decay
    
    # ===== Combine =====
    o = o_intra + o_inter  # [b, h, num_chunks, chunk_size, e]
    
    # Reshape back
    o = o.view(b, h, n_padded, e)
    
    # Remove padding
    if pad_len > 0:
        o = o[:, :, :n, :]
    
    return o.to(q.dtype)


# Alias for the recommended implementation
linear_attn_decay = chunkwise_linear_attn_decay


def test_implementations():
    """Test that all implementations produce the same results."""
    torch.manual_seed(42)
    
    b, h, n, d, e = 2, 4, 128, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    q = torch.randn(b, h, n, d, device=device, dtype=dtype) / 10
    k = torch.randn(b, h, n, d, device=device, dtype=dtype) / 10
    v = torch.randn(b, h, n, e, device=device, dtype=dtype) / 10
    s = torch.rand(h, device=device, dtype=torch.float32) * 0.1 + 0.01  # Small positive decay
    
    print(f"Testing on {device} with shape [b={b}, h={h}, n={n}, d={d}, e={e}]")
    print(f"Decay factors (lambda = exp(-s)): {torch.exp(-s).tolist()}")
    
    # Reference: naive implementation
    o_naive = naive_linear_attn_decay(q, k, v, s)
    print(f"Naive output shape: {o_naive.shape}")
    
    # Recurrent implementation
    o_recurrent, final_state = naive_linear_attn_decay_recurrent(q, k, v, s)
    diff_recurrent = (o_naive - o_recurrent).abs().max().item()
    print(f"Recurrent vs Naive max diff: {diff_recurrent:.2e}")
    
    # Chunkwise implementation
    for chunk_size in [32, 64]:
        o_chunk = chunkwise_linear_attn_decay(q, k, v, s, chunk_size=chunk_size)
        diff_chunk = (o_naive - o_chunk).abs().max().item()
        print(f"Chunkwise (chunk={chunk_size}) vs Naive max diff: {diff_chunk:.2e}")
    
    # Parallel chunkwise implementation
    o_parallel = chunkwise_linear_attn_decay_parallel(q, k, v, s, chunk_size=64)
    diff_parallel = (o_naive - o_parallel).abs().max().item()
    print(f"Parallel Chunkwise vs Naive max diff: {diff_parallel:.2e}")
    
    print("\nAll tests passed!" if max(diff_recurrent, diff_chunk, diff_parallel) < 1e-4 else "\nSome tests failed!")


if __name__ == "__main__":
    test_implementations()
