"""Sweep register allocation for warp groups."""
import sys, pathlib, torch, torch.nn.functional as F, triton
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from benchmark.utils import set_seed, exclusive_cumsum

torch.backends.cuda.matmul.allow_tf32 = True
D = 128; DTYPE = torch.bfloat16; DEV = torch.device("cuda")

def bench(fn):
    return triton.testing.do_bench(fn, warmup=15, rep=80, quantiles=[0.5, 0.2, 0.8])

# Test config: H=64, T=32768, N=8 (512 CTAs, where flashla wins)
H = 64; total_T = 32768; N = 8
set_seed(42)
base = total_T // N; rem = total_T % N
seq_lens = [base]*(N-1) + [base+rem]
cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=DEV)
q = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEV), p=2, dim=-1)
k = F.normalize(torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEV), p=2, dim=-1)
v = torch.randn(1, total_T, H, D, dtype=DTYPE, device=DEV)
g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float, device=DEV)).clamp(-5, 0)
beta = torch.randn(1, total_T, H, dtype=torch.float32, device=DEV).sigmoid()
h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEV)
scale = D ** -0.5

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: H={H}, T={total_T}, N={N}, CTAs={N*H}")
print()

# Sweep configurations: (num_regs_cuda, num_regs_subchunk, num_regs_others)
configs = [
    # Baseline
    (248, 192, 64,  "baseline"),
    # Reduce cuda core regs
    (224, 192, 64,  "cuda=224"),
    (200, 192, 64,  "cuda=200"),
    (192, 192, 64,  "cuda=192"),
    (160, 192, 64,  "cuda=160"),
    (128, 192, 64,  "cuda=128"),
    # Reduce subchunk regs
    (248, 160, 64,  "subchunk=160"),
    (248, 128, 64,  "subchunk=128"),
    (248, 96,  64,  "subchunk=96"),
    # Reduce both
    (224, 160, 64,  "cuda=224,sub=160"),
    (200, 160, 64,  "cuda=200,sub=160"),
    (192, 160, 64,  "cuda=192,sub=160"),
    (192, 128, 64,  "cuda=192,sub=128"),
    # Increase others (give more to load/mma/epi)
    (248, 192, 96,  "others=96"),
    (248, 192, 128, "others=128"),
    # Reduce cuda, increase others
    (224, 192, 96,  "cuda=224,others=96"),
    (200, 192, 96,  "cuda=200,others=96"),
]

print(f"{'Label':<25} {'cuda':>5} {'sub':>5} {'others':>7} {'ms':>8} {'vs base':>8}")
print("-" * 62)

baseline_ms = None
from flashla.kda_fully_fused import KDAChunkwise
from flashla.kda_wrapper import flash_kda_prefill, compiled_kernel_cache
import cutlass
import cutlass.cute as cute

for num_regs_cuda, num_regs_subchunk, num_regs_others, label in configs:
    # Clear compile cache to force recompile with new reg settings
    compiled_kernel_cache.clear()
    
    # Monkey-patch the default values
    original_init = KDAChunkwise.__init__
    def patched_init(self, chunk_size=64, qk_acc_dtype=cutlass.Float32,
                     kv_acc_dtype=cutlass.Float32, acc_dtype=cutlass.Float32,
                     io_dtype=cutlass.BFloat16, scale=1.0, safe_gate=False,
                     has_initial_state=False, output_final_state=False,
                     is_varlen=False,
                     _nrc=num_regs_cuda, _nrs=num_regs_subchunk, _nro=num_regs_others,
                     num_regs_cuda=None, num_regs_subchunk=None, num_regs_others=None):
        original_init(self, chunk_size=chunk_size, qk_acc_dtype=qk_acc_dtype,
                      kv_acc_dtype=kv_acc_dtype, acc_dtype=acc_dtype,
                      io_dtype=io_dtype, scale=scale, safe_gate=safe_gate,
                      has_initial_state=has_initial_state, output_final_state=output_final_state,
                      is_varlen=is_varlen,
                      num_regs_cuda=_nrc, num_regs_subchunk=_nrs, num_regs_others=_nro)
    
    KDAChunkwise.__init__ = patched_init
    
    try:
        ms, _, _ = bench(lambda: flash_kda_prefill(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=h0, output_final_state=True,
            safe_gate=True, cu_seqlens=cu))
        
        if baseline_ms is None:
            baseline_ms = ms
        ratio = baseline_ms / ms
        print(f"{label:<25} {num_regs_cuda:>5} {num_regs_subchunk:>5} {num_regs_others:>7} {ms:>7.3f} {ratio:>7.2f}x")
    except Exception as e:
        print(f"{label:<25} {num_regs_cuda:>5} {num_regs_subchunk:>5} {num_regs_others:>7}  ERROR: {e}")
    finally:
        KDAChunkwise.__init__ = original_init

print("\nDone.")
