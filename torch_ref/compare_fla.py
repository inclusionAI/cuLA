# Copyright (c) 2024-2025 FlashLA Authors
# Licensed under the MIT License

"""
Comparison script between FLA (flash-linear-attention) and our PyTorch reference implementation.

FLA uses `g_gamma` as the per-head scalar decay parameter where:
    - State update: h = exp(g_gamma) * h + k^T @ v
    - Output: o = q @ h * scale

Our implementation uses `s` (log-lambda) where:
    - State update: h = exp(-s) * h + k^T @ v  
    - Output: o = q @ h

The key difference is the sign convention:
    - FLA: g_gamma can be negative (decay) or positive (growth)
    - Ours: s > 0 always means decay (lambda = exp(-s) < 1)

To align: g_gamma = -s (our s equals negative of FLA's g_gamma)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def naive_linear_attn_fla_style(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FLA-style linear attention with g_gamma decay parameter.
    
    This matches the FLA fused_recurrent kernel semantics:
        h_t = exp(g_gamma) * h_{t-1} + k_t^T @ v_t
        o_t = q_t @ h_t * scale
    
    Args:
        q: Query tensor of shape [batch, seq_len, heads, d_k]
        k: Key tensor of shape [batch, seq_len, heads, d_k]
        v: Value tensor of shape [batch, seq_len, heads, d_v]
        g_gamma: Per-head decay tensor of shape [heads]
                 Typically negative for decay (e.g., g_gamma = -0.1 means lambda = exp(-0.1) ≈ 0.9)
        scale: Scale factor, defaults to 1/sqrt(d_k)
    
    Returns:
        Output tensor of shape [batch, seq_len, heads, d_v]
    """
    b, t, h, d = q.shape
    e = v.shape[-1]
    
    if scale is None:
        scale = d ** -0.5
    
    # Decay factor per head: [h]
    decay = torch.exp(g_gamma)  # exp(g_gamma), typically < 1 for decay
    
    # Initialize state: [b, h, d, e]
    state = torch.zeros(b, h, d, e, device=q.device, dtype=torch.float32)
    
    outputs = []
    
    for i in range(t):
        # Get current tokens: [b, h, d] and [b, h, e]
        q_t = q[:, i]  # [b, h, d]
        k_t = k[:, i]  # [b, h, d]
        v_t = v[:, i]  # [b, h, e]
        
        # State decay: h = decay * h
        state = decay.view(1, h, 1, 1) * state
        
        # State update: h += k^T @ v
        # [b, h, d, 1] @ [b, h, 1, e] -> [b, h, d, e]
        kv = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # outer product
        state = state + kv.float()
        
        # Output: o = q @ h * scale
        # [b, h, 1, d] @ [b, h, d, e] -> [b, h, e]
        o_t = torch.einsum('bhd,bhde->bhe', q_t.float(), state) * scale
        outputs.append(o_t)
    
    # Stack outputs: [b, t, h, e]
    o = torch.stack(outputs, dim=1)
    
    return o.to(q.dtype)


def naive_linear_attn_decay_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Our style linear attention with s (log-lambda) decay parameter.
    Using [B, T, H, D] layout to match FLA.
    
    State update: h_t = exp(-s) * h_{t-1} + k_t^T @ v_t
    Output: o_t = q_t @ h_t * scale
    
    Args:
        q: Query tensor of shape [batch, seq_len, heads, d_k]
        k: Key tensor of shape [batch, seq_len, heads, d_k]
        v: Value tensor of shape [batch, seq_len, heads, d_v]
        s: Log-lambda decay tensor of shape [heads], s > 0 means decay
        scale: Scale factor, defaults to 1/sqrt(d_k)
    
    Returns:
        Output tensor of shape [batch, seq_len, heads, d_v]
    """
    b, t, h, d = q.shape
    e = v.shape[-1]
    
    if scale is None:
        scale = d ** -0.5
    
    # Decay factor per head: lambda = exp(-s)
    decay = torch.exp(-s)  # exp(-s), < 1 for s > 0
    
    # Initialize state: [b, h, d, e]
    state = torch.zeros(b, h, d, e, device=q.device, dtype=torch.float32)
    
    outputs = []
    
    for i in range(t):
        # Get current tokens
        q_t = q[:, i]  # [b, h, d]
        k_t = k[:, i]  # [b, h, d]
        v_t = v[:, i]  # [b, h, e]
        
        # State decay: h = decay * h
        state = decay.view(1, h, 1, 1) * state
        
        # State update: h += k^T @ v
        kv = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        state = state + kv.float()
        
        # Output: o = q @ h * scale
        o_t = torch.einsum('bhd,bhde->bhe', q_t.float(), state) * scale
        outputs.append(o_t)
    
    o = torch.stack(outputs, dim=1)
    
    return o.to(q.dtype)


def compare_with_fla():
    """Compare our implementation with FLA's fused_recurrent_simple_gla."""
    try:
        from fla.ops.simple_gla import fused_recurrent_simple_gla
        has_fla = True
    except ImportError:
        print("FLA not installed. Install with: pip install flash-linear-attention")
        has_fla = False
        raise
    
    torch.manual_seed(42)
    
    # Test parameters
    b, t, h, d, e = 2, 128, 4, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    # Create inputs in FLA's expected layout: [B, T, H, D]
    q = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    k = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    v = torch.randn(b, t, h, e, device=device, dtype=dtype) / 10
    
    # Decay parameters
    # FLA uses g_gamma where decay = exp(g_gamma)
    # For decay behavior, g_gamma should be negative
    g_gamma = -torch.rand(h, device=device, dtype=torch.float32) * 0.1 - 0.01  # [-0.11, -0.01]
    
    # Our convention: s > 0, decay = exp(-s)
    # To align: s = -g_gamma
    s = -g_gamma
    
    scale = d ** -0.5
    
    print(f"Testing on {device} with shape [B={b}, T={t}, H={h}, D={d}, E={e}]")
    print(f"g_gamma (FLA style): {g_gamma.tolist()}")
    print(f"s (our style): {s.tolist()}")
    print(f"Decay factors (exp(g_gamma) = exp(-s)): {torch.exp(g_gamma).tolist()}")
    print()
    
    # Our FLA-style implementation
    o_fla_style = naive_linear_attn_fla_style(q, k, v, g_gamma, scale)
    print(f"Our FLA-style output shape: {o_fla_style.shape}")
    
    # Our s-style implementation  
    o_our_style = naive_linear_attn_decay_bthd(q, k, v, s, scale)
    print(f"Our s-style output shape: {o_our_style.shape}")
    
    # Compare the two implementations
    diff_styles = (o_fla_style - o_our_style).abs().max().item()
    print(f"Difference between FLA-style and s-style: {diff_styles:.2e}")
    
    if has_fla:
        # Compare with actual FLA implementation
        o_fla, _ = fused_recurrent_simple_gla(q, k, v, g=None, g_gamma=g_gamma, scale=scale)
        print(f"FLA output shape: {o_fla.shape}")
        
        diff_fla = (o_fla_style - o_fla).abs().max().item()
        print(f"Difference between our FLA-style and actual FLA: {diff_fla:.2e}")
        
        diff_fla_our = (o_our_style - o_fla).abs().max().item()
        print(f"Difference between our s-style and actual FLA: {diff_fla_our:.2e}")
        
        if diff_fla < 1e-3 and diff_fla_our < 1e-3:
            print("\n✓ All implementations match!")
        else:
            print("\n✗ Implementations differ significantly")
    else:
        if diff_styles < 1e-6:
            print("\n✓ Internal consistency check passed!")
        else:
            print("\n✗ Internal implementations differ")


def compare_with_torch_ref():
    """Compare with our torch_ref implementation (BHND layout)."""
    import sys
    sys.path.insert(0, '/ossfs/workspace/flashla')
    from torch_ref.linear_attn_decay import naive_linear_attn_decay_recurrent
    
    torch.manual_seed(42)
    
    b, t, h, d, e = 2, 64, 4, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Create inputs
    # torch_ref expects: [B, H, N, D]
    # This script uses: [B, T, H, D]
    q_bthd = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    k_bthd = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    v_bthd = torch.randn(b, t, h, e, device=device, dtype=dtype) / 10
    
    # Convert to [B, H, N, D] for torch_ref
    q_bhnd = q_bthd.transpose(1, 2)  # [B, H, T, D]
    k_bhnd = k_bthd.transpose(1, 2)
    v_bhnd = v_bthd.transpose(1, 2)
    
    # Decay parameter
    s = torch.rand(h, device=device, dtype=torch.float32) * 0.1 + 0.01  # [0.01, 0.11]
    
    scale = d ** -0.5
    
    print(f"Comparing torch_ref (BHND) with BTHD implementation")
    print(f"s (log-lambda): {s.tolist()}")
    print(f"Decay factors: {torch.exp(-s).tolist()}")
    print()
    
    # Our BTHD implementation
    o_bthd = naive_linear_attn_decay_bthd(q_bthd, k_bthd, v_bthd, s, scale)
    
    # torch_ref implementation (no scale in its recurrent version, we'll add it)
    o_bhnd, _ = naive_linear_attn_decay_recurrent(q_bhnd, k_bhnd, v_bhnd, s)
    o_bhnd = o_bhnd * scale  # Apply scale
    
    # Convert back to BTHD for comparison
    o_bhnd_as_bthd = o_bhnd.transpose(1, 2)  # [B, T, H, E]
    
    diff = (o_bthd - o_bhnd_as_bthd).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    
    if diff < 1e-5:
        print("✓ Implementations match!")
    else:
        print("✗ Implementations differ")


def run_full_comparison():
    """Run full comparison between all implementations."""
    print("=" * 60)
    print("FLA vs Our Implementation Comparison")
    print("=" * 60)
    print()
    
    try:
        from fla.ops.simple_gla import fused_recurrent_simple_gla
        has_fla = True
    except ImportError:
        has_fla = False
        print("Note: FLA not installed, will only run internal comparisons")
        print()
        raise
    
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        (1, 64, 4, 64, 64),
        (2, 128, 8, 128, 128),
        (4, 256, 4, 64, 64),
    ]
    
    for b, t, h, d, e in configs:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16
        
        print(f"Config: B={b}, T={t}, H={h}, D={d}, E={e}")
        
        q = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
        k = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
        v = torch.randn(b, t, h, e, device=device, dtype=dtype) / 10
        
        # g_gamma for FLA: negative for decay
        g_gamma = -torch.rand(h, device=device, dtype=torch.float32) * 0.1 - 0.01
        s = -g_gamma  # Our convention
        scale = d ** -0.5
        
        # Our implementation
        o_ours = naive_linear_attn_fla_style(q, k, v, g_gamma, scale)
        
        if has_fla:
            o_fla, _ = fused_recurrent_simple_gla(q, k, v, g=None, g_gamma=g_gamma, scale=scale)
            diff = (o_ours - o_fla).abs().max().item()
            status = "✓" if diff < 1e-3 else "✗"
            print(f"  {status} Max diff vs FLA: {diff:.2e}")
        else:
            # Self-consistency check
            o_check = naive_linear_attn_decay_bthd(q, k, v, s, scale)
            diff = (o_ours - o_check).abs().max().item()
            status = "✓" if diff < 1e-6 else "✗"
            print(f"  {status} Self-consistency: {diff:.2e}")
        
        print()
    
    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)


def compare_with_fla_lightning_attn():
    """
    Compare with FLA's fused_recurrent_lightning_attn directly.
    
    FLA's lightning attention computes g_gamma automatically based on:
        g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * range(H)
    
    This creates a per-head decay pattern that varies with layer depth.
    """
    try:
        from fla.ops.simple_gla import fused_recurrent_simple_gla
    except ImportError:
        print("FLA not installed. Install with: pip install flash-linear-attention")
        raise
    
    torch.manual_seed(42)
    
    # Test parameters
    b, t, h, d, e = 2, 128, 8, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    # Layer configuration
    num_layers = 12
    layer_idx = 6  # Middle layer
    
    print("=" * 60)
    print("FLA Lightning Attention Direct Comparison")
    print("=" * 60)
    print()
    print(f"Config: B={b}, T={t}, H={h}, D={d}, E={e}")
    print(f"Layer: {layer_idx}/{num_layers}")
    print()
    
    # Create inputs in FLA's expected layout: [B, T, H, D]
    q = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    k = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    v = torch.randn(b, t, h, e, device=device, dtype=dtype) / 10
    
    # Compute g_gamma the same way FLA's lightning_attn does
    # g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * range(H)
    g_gamma = -(8 / h * (1 - layer_idx / num_layers)) * torch.arange(h, device=device, dtype=torch.float32)
    
    print(f"g_gamma (computed like FLA): {g_gamma.tolist()}")
    print(f"Decay factors per head: {torch.exp(g_gamma).tolist()}")
    print()
    
    scale = d ** -0.5
    
    # 1. FLA's simple_gla with computed g_gamma (matches lightning_attn internally)
    o_fla, _ = fused_recurrent_simple_gla(
        q, k, v, 
        g=None, 
        g_gamma=g_gamma, 
        scale=scale
    )
    print(f"FLA simple_gla output shape: {o_fla.shape}")
    
    # 2. Our naive implementation
    o_ours = naive_linear_attn_fla_style(q, k, v, g_gamma, scale)
    print(f"Our implementation output shape: {o_ours.shape}")
    
    print()
    
    # Compare results
    diff = (o_fla - o_ours).abs().max().item()
    
    print("Comparison Results:")
    print(f"  FLA simple_gla vs Our impl: {diff:.2e}")
    
    if diff < 1e-3:
        print("\n✓ Implementations match!")
    else:
        print("\n✗ Implementations differ")
    
    print()
    
    # Test different layer indices
    print("-" * 60)
    print("Testing across different layers (simulating lightning_attn):")
    print("-" * 60)
    
    for layer_idx in [0, 3, 6, 9, 11]:
        g_gamma_layer = -(8 / h * (1 - layer_idx / num_layers)) * torch.arange(h, device=device, dtype=torch.float32)
        
        o_fla, _ = fused_recurrent_simple_gla(q, k, v, g=None, g_gamma=g_gamma_layer, scale=scale)
        o_ours = naive_linear_attn_fla_style(q, k, v, g_gamma_layer, scale)
        
        diff = (o_fla - o_ours).abs().max().item()
        status = "✓" if diff < 1e-3 else "✗"
        
        decay_range = f"[{torch.exp(g_gamma_layer).min().item():.4f}, {torch.exp(g_gamma_layer).max().item():.4f}]"
        print(f"  Layer {layer_idx:2d}: {status} diff={diff:.2e}, decay_range={decay_range}")


def compare_fla_chunk_vs_recurrent():
    """
    Compare FLA's chunk-based and recurrent implementations.
    
    FLA provides multiple implementations:
    - fused_recurrent: Sequential RNN-style, O(n) memory
    - fused_chunk: Chunkwise parallel, better GPU utilization
    """
    try:
        from fla.ops.simple_gla import fused_recurrent_simple_gla
        from fla.ops.simple_gla import fused_chunk_simple_gla
    except ImportError:
        print("FLA not installed or chunk implementation not available")
        raise
    
    torch.manual_seed(42)
    
    b, t, h, d, e = 2, 256, 4, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("FLA Chunk vs Recurrent Comparison")
    print("=" * 60)
    print()
    
    q = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    k = torch.randn(b, t, h, d, device=device, dtype=dtype) / 10
    v = torch.randn(b, t, h, e, device=device, dtype=dtype) / 10
    
    g_gamma = -torch.rand(h, device=device, dtype=torch.float32) * 0.1 - 0.01
    scale = d ** -0.5
    
    print(f"Config: B={b}, T={t}, H={h}, D={d}, E={e}")
    print(f"g_gamma: {g_gamma.tolist()}")
    print()
    
    # Recurrent implementation
    o_recurrent, _ = fused_recurrent_simple_gla(q, k, v, g=None, g_gamma=g_gamma, scale=scale)
    
    # Chunk implementation
    o_chunk, _ = fused_chunk_simple_gla(q, k, v, g=None, g_gamma=g_gamma, scale=scale)
    
    # Our implementation
    o_ours = naive_linear_attn_fla_style(q, k, v, g_gamma, scale)
    
    diff_rec_chunk = (o_recurrent - o_chunk).abs().max().item()
    diff_rec_ours = (o_recurrent - o_ours).abs().max().item()
    diff_chunk_ours = (o_chunk - o_ours).abs().max().item()
    
    print("Results:")
    print(f"  FLA recurrent vs FLA chunk: {diff_rec_chunk:.2e}")
    print(f"  FLA recurrent vs Our impl:  {diff_rec_ours:.2e}")
    print(f"  FLA chunk vs Our impl:      {diff_chunk_ours:.2e}")
    
    all_match = all(d < 1e-3 for d in [diff_rec_chunk, diff_rec_ours, diff_chunk_ours])
    print(f"\n{'✓ All match!' if all_match else '✗ Some differ'}")


if __name__ == "__main__":
    print("Running FLA comparison...")
    print()
    compare_with_fla()
    print()
    print("-" * 60)
    print()
    print("Running torch_ref comparison...")
    print()
    compare_with_torch_ref()
    print()
    print("-" * 60)
    print()
    run_full_comparison()
    print()
    print("-" * 60)
    print()
    print("Running FLA Lightning Attention direct comparison...")
    print()
    compare_with_fla_lightning_attn()
    print()
    print("-" * 60)
    print()
    print("Running FLA Chunk vs Recurrent comparison...")
    print()
    compare_fla_chunk_vs_recurrent()
