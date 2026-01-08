"""
Torch Reference: Print Q, K, G elementwise gating results for first chunk

This computes and prints:
- g_cumsum: cumulative sum of gate values within chunk
- Q' = Q * exp(g_cumsum)
- K_inter = K * exp(g_cumsum)  
- K_intra = K * exp(-g_cumsum)

Output format matches SMEM layout (C, D) = (64, 64) for comparison with CUDA kernel.

Supports optional KDA CUDA kernel invocation for comparison.
"""

import argparse
import sys
import time

import torch
import numpy as np


def torch_ref_gating(Q, K, G_raw, C, verbose=False):
    """
    Compute torch reference gating values.
    
    Args:
        Q: Query tensor [B, S, H, D]
        K: Key tensor [B, S, H, D]
        G_raw: Gate tensor [B, S, H, D]
        C: Chunk size
        verbose: Whether to print detailed output
    
    Returns:
        dict with keys: g_cumsum, q_gated, k_inter, k_intra
    """
    B, S, H, D = Q.shape
    
    # Compute g_cumsum (chunkwise cumsum along sequence dimension)
    num_chunks = S // C
    G_chunked = G_raw.float().view(B, num_chunks, C, H, D)
    G_cumsum_chunked = torch.cumsum(G_chunked, dim=2)
    G_cumsum = G_cumsum_chunked.view(B, S, H, D)
    
    # Compute gated values for all positions
    exp_g = torch.exp(G_cumsum)
    exp_neg_g = torch.exp(-G_cumsum)
    
    q_gated = Q.float() * exp_g      # Q' = Q * exp(g)
    k_inter = K.float() * exp_g      # K_inter = K * exp(g)
    k_intra = K.float() * exp_neg_g  # K_intra = K * exp(-g)
    
    return {
        'g_cumsum': G_cumsum,
        'q_gated': q_gated,
        'k_inter': k_inter,
        'k_intra': k_intra,
    }


def print_gating_results(Q, K, G_raw, C, D, output_file='torch_ref.txt'):
    """Print detailed gating results for first chunk and save to file."""
    B, S, H, _ = Q.shape
    
    results = torch_ref_gating(Q, K, G_raw, C)
    
    # Extract first chunk, head 0, batch 0: shape (C, D)
    q_chunk0 = Q[0, :C, 0, :].float()
    k_chunk0 = K[0, :C, 0, :].float()
    g_chunk0 = results['g_cumsum'][0, :C, 0, :]
    g_raw = G_raw[0, :C, 0, :]
    q_gated = results['q_gated'][0, :C, 0, :]
    k_inter = results['k_inter'][0, :C, 0, :]
    k_intra = results['k_intra'][0, :C, 0, :]
    
    # Build output content
    lines = []
    lines.append("=" * 80)
    lines.append("Torch Reference: KDA Gating Results for Chunk 0, Head 0, Batch 0")
    lines.append("=" * 80)
    lines.append(f"Shape: ({C}, {D}) = (rows, cols)")
    lines.append("")

    lines.append("g_raw:")
    lines.append(str(g_raw))
    lines.append("")
    
    lines.append("g_cumsum:")
    lines.append(str(g_chunk0))
    lines.append("")
    
    lines.append("Q' = Q * exp(g_cumsum):")
    lines.append(str(q_gated))
    lines.append("")
    
    lines.append("K_inter = K * exp(g_cumsum):")
    lines.append(str(k_inter))
    lines.append("")
    
    lines.append("K_intra = K * exp(-g_cumsum):")
    lines.append(str(k_intra))
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("Summary Statistics:")
    lines.append("=" * 80)
    lines.append(f"g_cumsum: min={g_chunk0.min().item():.6f}, max={g_chunk0.max().item():.6f}")
    lines.append(f"Q':       min={q_gated.min().item():.6f}, max={q_gated.max().item():.6f}")
    lines.append(f"K_inter:  min={k_inter.min().item():.6f}, max={k_inter.max().item():.6f}")
    lines.append(f"K_intra:  min={k_intra.min().item():.6f}, max={k_intra.max().item():.6f}")
    
    # Write to file
    content = "\n".join(lines)
    with open(output_file, 'w') as f:
        f.write(content)
    print(f"Gating results written to {output_file}")
    
    return results


def run_kda_cuda(Q, K, V, G, decay_factor=0.95, chunk_size=64):
    """
    Run KDA CUDA kernel and return output.
    
    Args:
        Q: Query tensor [B, S, H, D]
        K: Key tensor [B, S, H, D]
        V: Value tensor [B, S, H, D]
        G: Gate tensor [B, S, H, D]
        decay_factor: Decay factor for attention
        chunk_size: Chunk size for attention computation
    
    Returns:
        Output tensor [B, S, H, D]
    """
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack
    
    # Import KDA kernel
    sys.path.insert(0, '/ossfs/workspace/flashla')
    from flashla.kda import KDAChunkwise
    
    B, S, H, D = Q.shape
    
    # Per-head decay coefficients [H]
    decay = torch.full((H,), decay_factor, device="cuda", dtype=torch.float32)
    
    # Create output tensor
    O = torch.zeros_like(Q)
    
    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    g_cute = from_dlpack(G)
    o_cute = from_dlpack(O)
    decay_cute = from_dlpack(decay)
    
    # Create kernel instance
    attn_kernel = KDAChunkwise(
        chunk_size=chunk_size,
        qk_acc_dtype=cutlass.Float32,
        kv_acc_dtype=cutlass.Float32,
        io_dtype=cutlass.BFloat16,
    )
    
    # Get default stream
    stream = cutlass_torch.default_stream()
    
    # Compile kernel
    print("Compiling KDA kernel...")
    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
        (B, S, H, D),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")
    
    # Execute kernel
    print("Executing KDA kernel...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    compiled(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
        (B, S, H, D),
        stream,
    )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Execution time: {elapsed*1000:.4f} ms")
    
    return O


def compare_with_kda(Q, K, V, G, C, decay_factor=0.95):
    """
    Compare torch reference with KDA CUDA kernel output.
    
    Args:
        Q, K, V, G: Input tensors [B, S, H, D]
        C: Chunk size
        decay_factor: Decay factor
    
    Returns:
        dict with comparison results
    """
    B, S, H, D = Q.shape
    
    print("\n" + "=" * 80)
    print("Running KDA CUDA Kernel for Comparison")
    print("=" * 80)
    
    # Run KDA kernel
    O_kda = run_kda_cuda(Q, K, V, G, decay_factor=decay_factor, chunk_size=C)
    
    print("\n" + "=" * 80)
    print("KDA Output Statistics:")
    print("=" * 80)
    print(f"O_kda shape: {O_kda.shape}")
    print(f"O_kda: min={O_kda.min().item():.6f}, max={O_kda.max().item():.6f}, mean={O_kda.mean().item():.6f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(O_kda).any().item()
    has_inf = torch.isinf(O_kda).any().item()
    print(f"O_kda has NaN: {has_nan}, has Inf: {has_inf}")
    
    return {
        'O_kda': O_kda,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Torch Reference: KDA Gating Results with optional CUDA kernel comparison"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", "-s", type=int, default=64, help="Sequence length")
    parser.add_argument("--num_heads", "-n", type=int, default=1, help="Number of heads")
    parser.add_argument("--head_dim", "-d", type=int, default=128, help="Head dimension")
    parser.add_argument("--chunk_size", "-c", type=int, default=64, help="Chunk size")
    parser.add_argument("--decay", type=float, default=0.95, help="Decay factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_kda", action="store_true", help="Run KDA CUDA kernel for comparison")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed gating values")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Match kernel configuration
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim
    C = args.chunk_size
    
    print(f"Configuration: B={B}, S={S}, H={H}, D={D}, C={C}")
    print(f"Decay factor: {args.decay}")
    print()
    
    # Create inputs matching kernel format
    Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    G_raw = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda') * 0.1

    # Print detailed gating results (always)
    results = print_gating_results(Q, K, G_raw, C, D)
    
    # Run KDA CUDA kernel if requested
    if args.run_kda:

        num_chunks = S // C
        # G_chunked = G_raw.float().view(B, num_chunks, C, H, D)
        G_chunked = G_raw.float().view(B, num_chunks, C, H, D)
        G_cumsum_chunked = torch.cumsum(G_chunked, dim=2)
        G_cumsum = G_cumsum_chunked.view(B, S, H, D).bfloat16()

        comparison = compare_with_kda(Q, K, V, G_cumsum, C, decay_factor=args.decay)
        print("\n" + "=" * 80)
        print("Comparison Complete")
        print("=" * 80)


if __name__ == "__main__":
    main()
