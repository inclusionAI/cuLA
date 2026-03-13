import pandas as pd
import pathlib

def calculate_speedups():
    # Paths
    uniform_path = "/ossfs/workspace/kevinzeng/flashla/bench_safe_gate/Performance_B2_H64.csv"
    varlen_path = "/ossfs/workspace/kevinzeng/flashla/bench_varlen_safe_gate/Performance_varlen_NSEQ8_H64_VAR1.0.csv"
    
    # Process uniform
    df_uniform = pd.read_csv(uniform_path)
    df_uniform['T'] = df_uniform['T'].astype(int)
    df_uniform['speedup'] = df_uniform['fla'] / df_uniform['flashla']
    
    print("### Uniform Sequence Length (B=2, H=64, D=128)")
    print("| T | flash-linear-attention (ms) | flashla (ms) | Speedup |")
    print("|---|---------------------------|--------------|---------|")
    for _, row in df_uniform.iterrows():
        print(f"| {int(row['T'])} | {row['fla']:.3f} | {row['flashla']:.3f} | **{row['speedup']:.3f}x** |")
        
    print("\n### Varlen Sequence Length (NUM_SEQS=8, H=64, D=128)")
    print("| Total Length | flash-linear-attention (ms) | flashla (ms) | Speedup |")
    print("|--------------|---------------------------|--------------|---------|")
    
    # Process varlen
    df_varlen = pd.read_csv(varlen_path)
    df_varlen['total_len'] = df_varlen['total_len'].astype(int)
    df_varlen['speedup'] = df_varlen['fla'] / df_varlen['flashla']
    
    for _, row in df_varlen.iterrows():
        print(f"| {int(row['total_len'])} | {row['fla']:.3f} | {row['flashla']:.3f} | **{row['speedup']:.3f}x** |")

if __name__ == "__main__":
    calculate_speedups()
