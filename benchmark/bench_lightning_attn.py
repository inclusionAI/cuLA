import torch
import time

from fla.ops.lightning_attn import chunk_lightning_attn
from fla.utils import assert_close, device

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

from flashla.lightning_attn import LinearAttentionChunkwiseDecay

import torch
from einops import rearrange


PRINT_DEBUG = False

def print_chunkwise(t, name):
    if not PRINT_DEBUG:
        return
    print(f"--------{name}:")
    c = t.shape[1] // 64
    for i in range(c):
        beg = i*64
        end = beg + 64
        print(t[:, beg:end])


def test_triton_lightning_attn(
  args,
  Q,
  K,
  V,  
  decay,
  problem_size,
  layer_idx,
) -> torch.Tensor:
    B, S, H, D = problem_size
    (
        chunk_size,
        acc_dtype,
        io_dtype,
        iterations,
        num_layers,
    ) = args

    # warmup
    for _ in range(2):
        tri, _ = chunk_lightning_attn(Q, K, V, scale=1, num_layers=num_layers, layer_idx=layer_idx, output_final_state=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    tri = None
    for _ in range(iterations):
        tri, _ = chunk_lightning_attn(Q, K, V, scale=1, num_layers=num_layers, layer_idx=layer_idx, output_final_state=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Triton Execution time: {elapsed*1000/iterations:.2f} ms (average over {iterations} iterations)")
    return tri, elapsed

def test_cutedsl_lightning_attn(
  args,
  Q,
  K,
  V,  
  decay,
  problem_size,
) -> torch.Tensor:
    B, S, H, D = problem_size
    (
        chunk_size,
        acc_dtype,
        io_dtype,
        iterations,
        num_layers,
    ) = args
    attn_kernel = LinearAttentionChunkwiseDecay(
        chunk_size=chunk_size,
        qk_acc_dtype=acc_dtype,
        kv_acc_dtype=acc_dtype,
        io_dtype=io_dtype,
    )

    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    decay_cute = from_dlpack(decay)
    
    O = torch.zeros_like(Q)
    o_cute = from_dlpack(O)

    # Get default stream
    stream = cutlass.torch.default_stream()

    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
        (B, S, H, D),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print(f"B, S, H, D: {(B, S, H, D)}")

    # warm up
    for _ in range(2):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (B, S, H, D),
            stream,
        )
    torch.cuda.synchronize()

    # Run
    start = time.perf_counter()
    for _ in range(iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (B, S, H, D),
            stream,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\nCuteDSL Execution time: {elapsed*1000/iterations:.2f} ms (average over {iterations} iterations)")

    return O, elapsed

def benchmark_lightning_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float | None,
    dtype: torch.dtype,
    layer_idx: int = 0,
    num_layers: int = 24,
    iterations: int = 10,
):
    """
    Benchmark Lightning Attention with decay.
    
    Args:
        B: Batch size
        T: Sequence length
        H: Number of heads
        D: Head dimension
        scale: Attention scale (default 1.0)
        dtype: Data type for tensors
        layer_idx: Layer index (for FLA decay calculation)
        num_layers: Number of layers (for FLA decay calculation)
        iterations: Number of benchmark iterations
    """
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    
    # Per-head decay parameters
    # Match FLA's decay calculation: g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * range(H)
    # FLA's gamma is negative, our decay_s is positive, so: decay_s[h] = -gamma[h]
    decay = 8 / H * (1 - layer_idx / num_layers) * torch.arange(H, dtype=torch.float32, device=device)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Lightning Attention with Decay")
    print(f"{'='*60}")
    print(f"Problem size: B={B}, T={T}, H={H}, D={D}")
    print(f"Layer: {layer_idx}/{num_layers}")
    print(f"Decay range: [{decay[0]:.4f}, {decay[-1]:.4f}]")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        args = (
            64,  # chunk_size
            cutlass.Float32,  # acc_dtype
            cutlass.BFloat16,  # io_dtype
            iterations,
            num_layers,
        )

        # Test Triton (FLA) implementation
        print("Testing Triton (FLA) implementation...")
        tri, triton_elapsed = test_triton_lightning_attn(args, q, k, v, decay, problem_size=(B, T, H, D), layer_idx=layer_idx)
        print_chunkwise(tri, 'TRITON_O')

        # Test CuteDSL implementation
        print("\nTesting CuteDSL implementation...")
        cutedsl_o, cutedsl_elapsed = test_cutedsl_lightning_attn(args, q, k, v, decay, problem_size=(B, T, H, D))
        print_chunkwise(cutedsl_o, 'CUTEDSL_O')

        # Compare outputs
        print("\n" + "="*60)
        print("Numerical Comparison")
        print("="*60)
        max_diff = torch.max(torch.abs(tri - cutedsl_o)).item()
        mean_diff = torch.mean(torch.abs(tri - cutedsl_o)).item()
        rel_error = mean_diff / (torch.mean(torch.abs(tri)).item() + 1e-8)
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        
        # Note: FLA uses per-head decay based on head index
        try:
            assert_close('o', tri, cutedsl_o, 0.03)
            print("✓ Outputs match within tolerance (0.03)")
        except AssertionError as e:
            print(f"✗ Outputs differ: {e}")

        # Performance comparison
        print("\n" + "="*60)
        print("Performance Results")
        print("="*60)
        speedup = triton_elapsed / cutedsl_elapsed
        print(f"Triton time:  {triton_elapsed*1000/iterations:.3f} ms/iter")
        print(f"CuteDSL time: {cutedsl_elapsed*1000/iterations:.3f} ms/iter")
        print(f"Speedup (Triton/CuteDSL): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"✓ CuteDSL is {speedup:.2f}x faster than Triton")
        else:
            print(f"✗ Triton is {1/speedup:.2f}x faster than CuteDSL")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Default configuration
    B, T, H, D = 64, 4096, 64, 128
    
    # Alternative configurations (uncomment to test):
    # B, T, H, D = 1, 256, 4, 128        # Small test
    # B, T, H, D = 2, 2048, 16, 128      # Medium test
    # B, T, H, D = 32, 8192, 32, 128     # Large test
    
    # Benchmark with layer_idx=12 (middle layer, moderate decay)
    benchmark_lightning_attn(
        B, T, H, D, 
        scale=1., 
        dtype=torch.bfloat16, 
        layer_idx=12,
        num_layers=24,
        iterations=10
    )
    
    # Test with different layers to see decay effects
    # for layer_idx in [0, 6, 12, 18, 23]:
    #     benchmark_lightning_attn(
    #         B, T, H, D, 
    #         scale=1., 
    #         dtype=torch.bfloat16, 
    #         layer_idx=layer_idx,
    #         num_layers=24,
    #         iterations=10
    #     )
    
    # Optional: Test different decay values
    # for decay_val in [0.05, 0.1, 0.2, 0.5]:
    #     benchmark_lightning_attn(
    #         B, T, H, D, 
    #         scale=1., 
    #         dtype=torch.bfloat16, 
    #         decay_value=decay_val,
    #         num_layers=24,
    #         iterations=10
    #     )
