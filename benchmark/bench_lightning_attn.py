import torch
import time
import ctypes

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


def reset_cuda_error():
    """Reset CUDA error state after an error occurs."""
    try:
        # Get CUDA error and reset
        torch.cuda.synchronize()
        # Call cudaGetLastError to clear the error
        libcudart = ctypes.CDLL('libcudart.so')
        libcudart.cudaGetLastError()
        torch.cuda.empty_cache()
    except:
        pass

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
        tri = None
        triton_elapsed = None
        triton_error = None
        print("Testing Triton (FLA) implementation...")
        try:
            tri, triton_elapsed = test_triton_lightning_attn(args, q, k, v, decay, problem_size=(B, T, H, D), layer_idx=layer_idx)
            print_chunkwise(tri, 'TRITON_O')
        except Exception as e:
            triton_error = str(e)
            print(f"✗ Triton (FLA) failed: {e}")
            # Clear CUDA error state
            reset_cuda_error()

        # Test CuteDSL implementation
        cutedsl_o = None
        cutedsl_elapsed = None
        cutedsl_error = None
        print("\nTesting CuteDSL implementation...")
        try:
            cutedsl_o, cutedsl_elapsed = test_cutedsl_lightning_attn(args, q, k, v, decay, problem_size=(B, T, H, D))
            print_chunkwise(cutedsl_o, 'CUTEDSL_O')
        except Exception as e:
            cutedsl_error = str(e)
            print(f"✗ CuteDSL failed: {e}")
            # Clear CUDA error state
            reset_cuda_error()

        # Check if both failed
        if tri is None and cutedsl_o is None:
            print("\n" + "="*60)
            print("Both Implementations Failed")
            print("="*60)
            print(f"✗ Triton (FLA) error: {triton_error}")
            print(f"✗ CuteDSL error: {cutedsl_error}")
            print("="*60 + "\n")
            
            return {
                'max_diff': float('nan'),
                'mean_diff': float('nan'),
                'rel_error': float('nan'),
                'triton_time': float('nan'),
                'cutedsl_time': float('nan'),
                'speedup': float('nan'),
                'triton_error': triton_error,
                'cutedsl_error': cutedsl_error,
            }
        
        # If only one succeeded, return partial results
        if tri is None or cutedsl_o is None:
            print("\n" + "="*60)
            print("Partial Results (one implementation failed)")
            print("="*60)
            if tri is None:
                print(f"✗ Triton (FLA) failed: {triton_error}")
                print(f"✓ CuteDSL succeeded: {cutedsl_elapsed*1000/iterations:.3f} ms/iter")
            else:
                print(f"✓ Triton (FLA) succeeded: {triton_elapsed*1000/iterations:.3f} ms/iter")
                print(f"✗ CuteDSL failed: {cutedsl_error}")
            print("="*60 + "\n")
            
            return {
                'max_diff': float('nan'),
                'mean_diff': float('nan'),
                'rel_error': float('nan'),
                'triton_time': triton_elapsed * 1000 / iterations if triton_elapsed else float('nan'),
                'cutedsl_time': cutedsl_elapsed * 1000 / iterations if cutedsl_elapsed else float('nan'),
                'speedup': float('nan'),
                'triton_error': triton_error,
                'cutedsl_error': cutedsl_error,
            }

        # Compare outputs (both succeeded)
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
        
        # Return results for plotting
        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rel_error': rel_error,
            'triton_time': triton_elapsed * 1000 / iterations,
            'cutedsl_time': cutedsl_elapsed * 1000 / iterations,
            'speedup': speedup,
            'triton_error': None,
            'cutedsl_error': None,
        }


def run_benchmark_suite():
    """
    Run comprehensive benchmarks with multiple configurations.
    Tests different batch sizes and head counts, then plots results.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test configurations
    batch_sizes = [2, 8, 16]  # Removed 64 to avoid CUDA context corruption
    num_heads_list = [64, 128]  # Test H=64 and H=128
    seq_lens = [4096, 32768, 65536]  # Test 4K, 32K, and 64K sequence lengths
    head_dim = 128
    layer_idx = 12
    num_layers = 24
    iterations = 10
    
    results = []
    configs = []
    
    print("\n" + "="*80)
    print("Running Comprehensive Benchmark Suite")
    print("="*80)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num heads: {num_heads_list}")
    print(f"Seq lens: {seq_lens}, Head dim: {head_dim}")
    print(f"Layer: {layer_idx}/{num_layers}, Iterations: {iterations}")
    print("="*80 + "\n")
    
    for B in batch_sizes:
        for T in seq_lens:
            for H in num_heads_list:
                # Skip configurations that cause int32 overflow
                # B*T*H*D must be < 2^31 (2,147,483,648) for signed int32
                # B=16, T=32768: 16*32768*64*128 = 4,294,967,296 > 2^32 (overflow!)
                total_elements = B * T * H * head_dim
                if total_elements > 2_147_483_648:  # 2^31
                    print(f"\n{'='*60}")
                    print(f"Skipping B={B}, T={T}, H={H} (int32 overflow: {total_elements:,} > 2^31)")
                    print(f"{'='*60}\n")
                    continue
                
                config_name = f"B={B}, T={T}, H={H}"
                configs.append(config_name)
                
                print(f"\n{'='*60}")
                print(f"Testing configuration: {config_name}")
                print(f"{'='*60}")
                
                result = benchmark_lightning_attn(
                    B=B,
                    T=T,
                    H=H,
                    D=head_dim,
                    scale=1.0,
                    dtype=torch.bfloat16,
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                    iterations=iterations,
                )
                results.append(result)
    
    # Plot results
    plot_benchmark_results(configs, results, batch_sizes, num_heads_list, seq_lens)
    
    return results


def plot_benchmark_results(configs, results, batch_sizes, num_heads_list, seq_lens):
    """
    Create visualization of benchmark results.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data
    max_diffs = [r['max_diff'] for r in results]
    mean_diffs = [r['mean_diff'] for r in results]
    rel_errors = [r['rel_error'] for r in results]
    triton_times = [r['triton_time'] for r in results]
    cutedsl_times = [r['cutedsl_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Lightning Attention Benchmark: CuteDSL vs Triton (FLA)', fontsize=16, fontweight='bold')
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Plot 1: Max Difference
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, max_diffs, color='coral')
    ax1.set_ylabel('Max Absolute Difference', fontsize=10)
    ax1.set_title('Maximum Difference', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, max_diffs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Mean Difference
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, mean_diffs, color='lightblue')
    ax2.set_ylabel('Mean Absolute Difference', fontsize=10)
    ax2.set_title('Mean Difference', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, mean_diffs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Relative Error
    ax3 = axes[0, 2]
    bars3 = ax3.bar(x, [r*100 for r in rel_errors], color='lightgreen')
    ax3.set_ylabel('Relative Error (%)', fontsize=10)
    ax3.set_title('Relative Error', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=3.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Tolerance (3%)')
    ax3.legend(fontsize=8)
    for i, (bar, val) in enumerate(zip(bars3, rel_errors)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val*100:.2f}%',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Execution Time Comparison
    ax4 = axes[1, 0]
    bars4a = ax4.bar(x - width/2, triton_times, width, label='Triton (FLA)', color='steelblue')
    bars4b = ax4.bar(x + width/2, cutedsl_times, width, label='CuteDSL', color='orange')
    ax4.set_ylabel('Time (ms/iter)', fontsize=10)
    ax4.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Speedup
    ax5 = axes[1, 1]
    bars5 = ax5.bar(x, speedups, color='gold')
    ax5.set_ylabel('Speedup (Triton/CuteDSL)', fontsize=10)
    ax5.set_title('Performance Speedup', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax5.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars5, speedups)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}x',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Grouped comparison by batch size and sequence length
    ax6 = axes[1, 2]
    
    # Build speedup matrix with actual data (handle skipped configs)
    n_batch = len(batch_sizes)
    n_seqlen = len(seq_lens)
    speedup_matrix = np.full((n_batch, n_seqlen), np.nan)
    
    # Fill matrix with available results
    for config, result in zip(configs, results):
        # Parse config string like "B=16, T=32768, H=64"
        parts = config.split(', ')
        B_val = int(parts[0].split('=')[1])
        T_val = int(parts[1].split('=')[1])
        
        b_idx = batch_sizes.index(B_val)
        t_idx = seq_lens.index(T_val)
        speedup_matrix[b_idx, t_idx] = result['speedup']
    
    x_pos = np.arange(n_batch)
    width = 0.35
    
    for i, T in enumerate(seq_lens):
        offset = (i - n_seqlen/2 + 0.5) * width
        # Only plot non-NaN values
        data = speedup_matrix[:, i]
        ax6.bar(x_pos + offset, np.nan_to_num(data), width, label=f'T={T}')
    
    ax6.set_ylabel('Speedup (Triton/CuteDSL)', fontsize=10)
    ax6.set_title('Speedup by Batch Size and Seq Length', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Batch Size', fontsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'B={b}' for b in batch_sizes], fontsize=9)
    ax6.legend(fontsize=9)
    ax6.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'benchmark_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Benchmark results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("\n" + "="*110)
    print("BENCHMARK SUMMARY TABLE")
    print("="*110)
    print(f"{'Config':<25} {'Max Diff':>12} {'Mean Diff':>12} {'Rel Err %':>10} "
          f"{'Triton(ms)':>12} {'CuteDSL(ms)':>12} {'Speedup':>10}")
    print("-"*110)
    for cfg, res in zip(configs, results):
        if np.isnan(res['speedup']):
            # Check which implementation failed
            error_msg = ""
            if res.get('triton_error') and res.get('cutedsl_error'):
                error_msg = f"BOTH FAILED (T: {res['triton_error'][:30]}..., C: {res['cutedsl_error'][:30]}...)"
            elif res.get('triton_error'):
                error_msg = f"Triton FAILED: {res['triton_error'][:60]}..."
            elif res.get('cutedsl_error'):
                error_msg = f"CuteDSL FAILED: {res['cutedsl_error'][:60]}..."
            else:
                error_msg = "FAILED"
            print(f"{cfg:<25} {error_msg}")
        else:
            print(f"{cfg:<25} {res['max_diff']:>12.2f} {res['mean_diff']:>12.4f} {res['rel_error']*100:>9.2f}% "
                  f"{res['triton_time']:>12.3f} {res['cutedsl_time']:>12.3f} {res['speedup']:>9.2f}x")
    print("="*110 + "\n")
    
    # Calculate and print averages (excluding failed tests)
    valid_speedups = [s for s in speedups if not np.isnan(s)]
    valid_rel_errors = [r for r in rel_errors if not np.isnan(r)]
    if valid_speedups:
        avg_speedup = np.mean(valid_speedups)
        avg_rel_error = np.mean(valid_rel_errors) * 100
        print(f"Average Speedup: {avg_speedup:.2f}x (from {len(valid_speedups)} successful tests)")
        print(f"Average Relative Error: {avg_rel_error:.2f}%")
    else:
        print("No successful tests to calculate averages.")
    print()


if __name__ == '__main__':
    # Run comprehensive benchmark suite
    run_benchmark_suite()
    
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
