import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from benchmarks.utils import set_seed
import time
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import sys
import pathlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).parent))
from flashla.kda_fully_fused import KDAChunkwise

# Configuration
CHUNK_SIZE = 64
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 10
SEED = 42
D = 128

# Test configurations
SEQLENS = [128, 512, 1024, 4096, 8192]
NUM_HEADS = [16, 64]
BATCH_SIZES = [1, 2, 4]

# Maximum addressable size (INT32_MAX / sizeof(bf16))
MAX_ADDRESSABLE_ELEMENTS = (2**31 - 1) // 2  # ~1B elements

def check_size_limit(B, S, H, D):
    """Check if tensor size exceeds INT32 addressing limit."""
    total_elements = B * S * H * D
    return total_elements <= MAX_ADDRESSABLE_ELEMENTS

compiled_kernel_cache = {}

def flashkda_impl(q, k, v, g, beta, scale, B, S, H, D):
    """FlashKDA implementation with caching."""
    # Apply cumsum
    g = chunk_local_cumsum(
        g=g,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=None,
        chunk_indices=None
    )

    # L2 norm
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # Convert to dlpack
    q_cute = from_dlpack(q)
    k_cute = from_dlpack(k)
    v_cute = from_dlpack(v)
    g_cute = from_dlpack(g)
    beta_cute = from_dlpack(beta)
    
    o = torch.empty_like(q)
    o_cute = from_dlpack(o)

    stream = cutlass_torch.default_stream()

    # Cache key
    cache_key = (B, S, H, D)
    
    if cache_key not in compiled_kernel_cache:
        # Create and compile kernel
        attn_kernel = KDAChunkwise(
            chunk_size=CHUNK_SIZE,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=scale,
        )

        compiled = cute.compile(
            attn_kernel,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )
        compiled_kernel_cache[cache_key] = compiled
    
    compiled = compiled_kernel_cache[cache_key]
    
    # Run kernel
    compiled(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        beta_cute.iterator,
        (B, S, H, D),
        stream,
    )
    
    return o

def fla_impl(q, k, v, g, beta, scale, B, S, H, D):
    """FLA implementation."""
    o, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=None,
        dt_bias=None,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )
    return o

def benchmark_config(B, S, H, D, device):
    """Benchmark a single configuration."""
    print(f"\nBenchmarking B={B}, S={S}, H={H}, D={D}")
    
    # Check size limit
    if not check_size_limit(B, S, H, D):
        print(f"  Skipped: Exceeds INT32 addressing limit")
        return None
    
    dtype = torch.bfloat16
    scale = D ** (-0.5)
    
    set_seed(SEED)
    
    # Generate inputs
    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, S, H, D, dtype=dtype, device=device))
    beta = torch.randn(B, S, H, dtype=torch.float, device=device).sigmoid()
    
    results = {}
    
    # Benchmark FlashKDA
    try:
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            o_flash = flashkda_impl(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), scale, B, S, H, D)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(BENCHMARK_ITERATIONS):
            o_flash = flashkda_impl(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), scale, B, S, H, D)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        flashkda_time = elapsed * 1000 / BENCHMARK_ITERATIONS
        results['flashkda_ms'] = flashkda_time
        print(f"  FlashKDA: {flashkda_time:.3f} ms")
    except Exception as e:
        print(f"  FlashKDA failed: {e}")
        results['flashkda_ms'] = None
    
    # Benchmark FLA
    try:
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            o_fla = fla_impl(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), scale, B, S, H, D)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(BENCHMARK_ITERATIONS):
            o_fla = fla_impl(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), scale, B, S, H, D)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        fla_time = elapsed * 1000 / BENCHMARK_ITERATIONS
        results['fla_ms'] = fla_time
        print(f"  FLA: {fla_time:.3f} ms")
    except Exception as e:
        print(f"  FLA failed: {e}")
        results['fla_ms'] = None
    
    # Calculate speedup
    if results.get('flashkda_ms') and results.get('fla_ms'):
        speedup = results['fla_ms'] / results['flashkda_ms']
        results['speedup'] = speedup
        print(f"  Speedup: {speedup:.2f}x")
    else:
        results['speedup'] = None
    
    return results

def run_benchmarks():
    """Run all benchmark configurations."""
    device = torch.device("cuda")
    all_results = []
    
    print("="*80)
    print("KDA Performance Benchmark: FlashKDA vs FLA")
    print("="*80)
    print(f"Configuration:")
    print(f"  D (head_dim): {D}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"  Benchmark iterations: {BENCHMARK_ITERATIONS}")
    print(f"  Sequence lengths: {SEQLENS}")
    print(f"  Number of heads: {NUM_HEADS}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print("="*80)
    
    for B in BATCH_SIZES:
        for H in NUM_HEADS:
            for S in SEQLENS:
                results = benchmark_config(B, S, H, D, device)
                if results:
                    all_results.append({
                        'B': B,
                        'S': S,
                        'H': H,
                        'D': D,
                        **results
                    })
    
    return pd.DataFrame(all_results)

def plot_results(df, output_dir):
    """Generate plots for benchmark results."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Filter valid results
    df_valid = df[df['flashkda_ms'].notna() & df['fla_ms'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No valid results to plot")
        return
    
    # Generate all-in-one comprehensive plot first
    plot_all_in_one(df_valid, output_dir)
    
    # Generate individual detailed plots
    plot_individual_results(df_valid, output_dir)

def plot_individual_results(df_valid, output_dir):
    """Generate individual plots for detailed analysis."""
    
    # Plot 1: Performance comparison (grouped bar chart)
    for B in df_valid['B'].unique():
        for H in df_valid['H'].unique():
            df_subset = df_valid[(df_valid['B'] == B) & (df_valid['H'] == H)]
            
            if len(df_subset) == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(df_subset))
            width = 0.35
            
            ax.bar(x - width/2, df_subset['flashkda_ms'], width, label='FlashKDA', alpha=0.8)
            ax.bar(x + width/2, df_subset['fla_ms'], width, label='FLA', alpha=0.8)
            
            ax.set_xlabel('Sequence Length', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.set_title(f'KDA Performance Comparison (B={B}, H={H}, D={D})', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(df_subset['S'].astype(str))
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'performance_B{B}_H{H}.png', dpi=150)
            plt.close()
    
    # Plot 2: Speedup heatmap
    for B in df_valid['B'].unique():
        df_subset = df_valid[df_valid['B'] == B]
        
        if len(df_subset) == 0:
            continue
        
        # Create pivot table
        pivot = df_subset.pivot_table(values='speedup', index='H', columns='S', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0, 
                    vmin=0.5, vmax=1.5, ax=ax, cbar_kws={'label': 'Speedup (FLA/FlashKDA)'})
        ax.set_title(f'FlashKDA Speedup over FLA (B={B}, D={D})', fontsize=14)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Number of Heads', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'speedup_heatmap_B{B}.png', dpi=150)
        plt.close()
    
    # Plot 3: Scaling with sequence length
    for B in df_valid['B'].unique():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, H in enumerate(sorted(df_valid['H'].unique())):
            df_subset = df_valid[(df_valid['B'] == B) & (df_valid['H'] == H)].sort_values('S')
            
            if len(df_subset) == 0:
                continue
            
            # Plot absolute time
            ax = axes[0]
            ax.plot(df_subset['S'], df_subset['flashkda_ms'], marker='o', label=f'FlashKDA (H={H})', linewidth=2)
            ax.plot(df_subset['S'], df_subset['fla_ms'], marker='s', label=f'FLA (H={H})', linewidth=2, linestyle='--')
        
        axes[0].set_xlabel('Sequence Length', fontsize=12)
        axes[0].set_ylabel('Time (ms)', fontsize=12)
        axes[0].set_title(f'Performance Scaling (B={B}, D={D})', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # Plot speedup
        for H in sorted(df_valid['H'].unique()):
            df_subset = df_valid[(df_valid['B'] == B) & (df_valid['H'] == H)].sort_values('S')
            
            if len(df_subset) == 0:
                continue
            
            axes[1].plot(df_subset['S'], df_subset['speedup'], marker='o', label=f'H={H}', linewidth=2)
        
        axes[1].set_xlabel('Sequence Length', fontsize=12)
        axes[1].set_ylabel('Speedup (FLA/FlashKDA)', fontsize=12)
        axes[1].set_title(f'Speedup vs Sequence Length (B={B}, D={D})', fontsize=14)
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'scaling_B{B}.png', dpi=150)
        plt.close()
    
    print(f"Individual plots saved to {output_dir}/")

def plot_all_in_one(df_valid, output_dir):
    """Generate comprehensive all-in-one visualization."""
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = sns.color_palette("husl", len(df_valid['B'].unique()) * len(df_valid['H'].unique()))
    
    # Subplot 1: Overall speedup distribution
    ax1 = fig.add_subplot(gs[0, 0])
    speedups = df_valid['speedup']
    ax1.hist(speedups, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax1.axvline(x=speedups.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({speedups.mean():.2f}x)')
    ax1.set_xlabel('Speedup (FLA/FlashKDA)', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Speedup Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Performance comparison - all configs
    ax2 = fig.add_subplot(gs[0, 1:])
    df_sorted = df_valid.sort_values(['B', 'H', 'S'])
    x_labels = [f"B{row['B']}_H{row['H']}_S{row['S']}" for _, row in df_sorted.iterrows()]
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    ax2.bar(x_pos - width/2, df_sorted['flashkda_ms'], width, label='FlashKDA', alpha=0.8, color='#2E86AB')
    ax2.bar(x_pos + width/2, df_sorted['fla_ms'], width, label='FLA', alpha=0.8, color='#A23B72')
    ax2.set_xlabel('Configuration', fontsize=10)
    ax2.set_ylabel('Time (ms)', fontsize=10)
    ax2.set_title('Performance Comparison - All Configurations', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos[::max(1, len(x_pos)//15)])  # Show subset of labels
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_pos)//15))], 
                         rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Speedup heatmap (combined for all batch sizes)
    ax3 = fig.add_subplot(gs[1, 0])
    # Create a combined key for batch and heads
    df_heatmap = df_valid.copy()
    df_heatmap['B_H'] = df_heatmap['B'].astype(str) + '_' + df_heatmap['H'].astype(str)
    pivot = df_heatmap.pivot_table(values='speedup', index='B_H', columns='S', aggfunc='mean')
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                vmin=0.5, vmax=2.0, ax=ax3, cbar_kws={'label': 'Speedup'}, 
                annot_kws={'fontsize': 8})
    ax3.set_title('Speedup Heatmap (B_H vs SeqLen)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sequence Length', fontsize=10)
    ax3.set_ylabel('Batch_Heads', fontsize=10)
    
    # Subplot 4 & 5: Scaling curves by sequence length (separate for each metric)
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    
    color_idx = 0
    for B in sorted(df_valid['B'].unique()):
        for H in sorted(df_valid['H'].unique()):
            df_subset = df_valid[(df_valid['B'] == B) & (df_valid['H'] == H)].sort_values('S')
            if len(df_subset) == 0:
                continue
            
            color = colors[color_idx % len(colors)]
            label = f'B={B}, H={H}'
            
            # Absolute time
            ax4.plot(df_subset['S'], df_subset['flashkda_ms'], marker='o', 
                    label=f'{label} (Flash)', linewidth=2, color=color, alpha=0.8)
            ax4.plot(df_subset['S'], df_subset['fla_ms'], marker='s', 
                    label=f'{label} (FLA)', linewidth=2, color=color, alpha=0.5, linestyle='--')
            
            # Speedup
            ax5.plot(df_subset['S'], df_subset['speedup'], marker='o', 
                    label=label, linewidth=2, color=color)
            
            color_idx += 1
    
    ax4.set_xlabel('Sequence Length', fontsize=10)
    ax4.set_ylabel('Time (ms)', fontsize=10)
    ax4.set_title('Performance Scaling', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    ax5.set_xlabel('Sequence Length', fontsize=10)
    ax5.set_ylabel('Speedup (FLA/FlashKDA)', fontsize=10)
    ax5.set_title('Speedup vs Sequence Length', fontsize=12, fontweight='bold')
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax5.set_xscale('log')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Best/Worst cases
    ax6 = fig.add_subplot(gs[2, 0])
    best_idx = df_valid['speedup'].idxmax()
    worst_idx = df_valid['speedup'].idxmin()
    best = df_valid.loc[best_idx]
    worst = df_valid.loc[worst_idx]
    
    cases = ['Best\nSpeedup', 'Worst\nSpeedup']
    flash_times = [best['flashkda_ms'], worst['flashkda_ms']]
    fla_times = [best['fla_ms'], worst['fla_ms']]
    
    x_pos = np.arange(len(cases))
    width = 0.35
    ax6.bar(x_pos - width/2, flash_times, width, label='FlashKDA', alpha=0.8, color='#2E86AB')
    ax6.bar(x_pos + width/2, fla_times, width, label='FLA', alpha=0.8, color='#A23B72')
    
    ax6.set_ylabel('Time (ms)', fontsize=10)
    ax6.set_title('Best & Worst Cases', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(cases, fontsize=9)
    ax6.legend(fontsize=9)
    
    # Add text annotations
    for i, (flash_t, fla_t, case) in enumerate(zip(flash_times, fla_times, [best, worst])):
        speedup = case['speedup']
        ax6.text(i, max(flash_t, fla_t) * 1.1, 
                f"{speedup:.2f}x\nB={int(case['B'])}, H={int(case['H'])}, S={int(case['S'])}", 
                ha='center', fontsize=8, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Subplot 7: Summary statistics table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Configurations', f"{len(df_valid)}"],
        ['Average Speedup', f"{df_valid['speedup'].mean():.2f}x"],
        ['Median Speedup', f"{df_valid['speedup'].median():.2f}x"],
        ['Best Speedup', f"{df_valid['speedup'].max():.2f}x"],
        ['Worst Speedup', f"{df_valid['speedup'].min():.2f}x"],
        ['Avg FlashKDA Time', f"{df_valid['flashkda_ms'].mean():.2f} ms"],
        ['Avg FLA Time', f"{df_valid['fla_ms'].mean():.2f} ms"],
        ['FlashKDA Faster', f"{(df_valid['speedup'] > 1.0).sum()} / {len(df_valid)} configs"],
        ['FLA Faster', f"{(df_valid['speedup'] < 1.0).sum()} / {len(df_valid)} configs"],
    ]
    
    table = ax7.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
    
    ax7.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(f'FlashKDA vs FLA - Comprehensive Performance Analysis (D={D})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'all_in_one_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"All-in-one summary plot saved to {output_dir / 'all_in_one_summary.png'}")

def plot_individual_results(df_valid, output_dir):
    """Generate individual plots for detailed analysis."""

def generate_markdown_report(df, output_path):
    """Generate markdown report with benchmark results."""
    with open(output_path, 'w') as f:
        f.write("# KDA Performance Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Configuration:**\n")
        f.write(f"- Head dimension (D): {D}\n")
        f.write(f"- Chunk size: {CHUNK_SIZE}\n")
        f.write(f"- Warmup iterations: {WARMUP_ITERATIONS}\n")
        f.write(f"- Benchmark iterations: {BENCHMARK_ITERATIONS}\n")
        f.write(f"- Device: {torch.cuda.get_device_name()}\n\n")
        
        f.write("## Summary\n\n")
        
        # Add all-in-one plot reference at the top
        f.write("### Comprehensive Overview\n\n")
        f.write("![All-in-One Summary](all_in_one_summary.png)\n\n")
        
        f.write("### Statistics\n\n")
        
        df_valid = df[df['flashkda_ms'].notna() & df['fla_ms'].notna()].copy()
        
        if len(df_valid) > 0:
            avg_speedup = df_valid['speedup'].mean()
            max_speedup = df_valid['speedup'].max()
            min_speedup = df_valid['speedup'].min()
            
            f.write(f"- **Average Speedup:** {avg_speedup:.2f}x\n")
            f.write(f"- **Best Speedup:** {max_speedup:.2f}x\n")
            f.write(f"- **Worst Speedup:** {min_speedup:.2f}x\n")
            f.write(f"- **Total Configurations Tested:** {len(df)}\n")
            f.write(f"- **Successful Configurations:** {len(df_valid)}\n\n")
        
        # Results by batch size
        for B in sorted(df['B'].unique()):
            f.write(f"## Batch Size = {B}\n\n")
            
            for H in sorted(df[df['B'] == B]['H'].unique()):
                df_subset = df[(df['B'] == B) & (df['H'] == H)].copy()
                
                f.write(f"### Heads = {H}\n\n")
                f.write("| Seq Len | FlashKDA (ms) | FLA (ms) | Speedup |\n")
                f.write("|---------|---------------|----------|----------|\n")
                
                for _, row in df_subset.iterrows():
                    flash_str = f"{row['flashkda_ms']:.3f}" if pd.notna(row['flashkda_ms']) else "N/A"
                    fla_str = f"{row['fla_ms']:.3f}" if pd.notna(row['fla_ms']) else "N/A"
                    speedup_str = f"{row['speedup']:.2f}x" if pd.notna(row['speedup']) else "N/A"
                    
                    f.write(f"| {row['S']} | {flash_str} | {fla_str} | {speedup_str} |\n")
                
                f.write("\n")
                
                # Add plot reference
                plot_file = f"performance_B{B}_H{H}.png"
                f.write(f"![Performance Comparison]({plot_file})\n\n")
            
            # Add heatmap reference
            heatmap_file = f"speedup_heatmap_B{B}.png"
            f.write(f"### Speedup Heatmap\n\n")
            f.write(f"![Speedup Heatmap]({heatmap_file})\n\n")
            
            # Add scaling plot reference
            scaling_file = f"scaling_B{B}.png"
            f.write(f"### Performance Scaling\n\n")
            f.write(f"![Performance Scaling]({scaling_file})\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        f.write("| B | S | H | D | FlashKDA (ms) | FLA (ms) | Speedup |\n")
        f.write("|---|---|---|---|---------------|----------|----------|\n")
        
        for _, row in df.iterrows():
            flash_str = f"{row['flashkda_ms']:.3f}" if pd.notna(row['flashkda_ms']) else "N/A"
            fla_str = f"{row['fla_ms']:.3f}" if pd.notna(row['fla_ms']) else "N/A"
            speedup_str = f"{row['speedup']:.2f}x" if pd.notna(row['speedup']) else "N/A"
            
            f.write(f"| {row['B']} | {row['S']} | {row['H']} | {row['D']} | {flash_str} | {fla_str} | {speedup_str} |\n")
        
        f.write("\n")
        
        # Notes
        f.write("## Notes\n\n")
        f.write("- Speedup is calculated as FLA time / FlashKDA time\n")
        f.write("- Speedup > 1.0 means FlashKDA is faster\n")
        f.write("- Speedup < 1.0 means FLA is faster\n")
        f.write("- Configurations exceeding INT32 addressing limit are skipped\n")
    
    print(f"\nMarkdown report saved to {output_path}")

def main():
    """Main benchmark execution."""
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(f"benchmark_results/kda_performance_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    df = run_benchmarks()
    
    # Save raw results to CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate plots
    plot_results(df, output_dir)
    
    # Generate markdown report
    markdown_path = output_dir / "benchmark_report.md"
    generate_markdown_report(df, markdown_path)
    
    print("\n" + "="*80)
    print("Benchmark completed successfully!")
    print(f"Results directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
