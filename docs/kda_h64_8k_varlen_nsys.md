# KDA backward Nsight Systems breakdown on GB200

## Scope

- GPU: NVIDIA GB200
- PyTorch 2.11.0+cu130; Nsight Systems 2025.4.1
- Shape: `H=HV=64`, `K=V=128`, total 8192 tokens
- Packed sequence lengths: `[257, 997, 2048, 4890]`
- `safe_gate=True`, `lower_bound=-5`, chunk size 64
- Both output and final-state gradients are included
- cuLA and FLA use `disable_recompute=True`
- cuLA uses SM100 CuTeDSL WY/dKQG plus tcgen05 CuTeDSL intra
- FLA uses its Triton backend (`FLA_TILELANG=0`)

Each report captures 30 backward calls after JIT compilation and three warmup
calls. Capture uses `cudaProfilerApi`, so initialization and compilation are not
included. Percentages below are from `nsys stats --report cuda_gpu_kern_sum`.

## Total captured GPU kernel time

| Backend | GPU kernel time / backward | Kernel launches / backward |
|---|---:|---:|
| CAKE | 2.038 ms | 2 |
| cuLA | 2.313 ms | 17 |
| FLA Triton | 3.419 ms | 16 |

The Nsight aggregate gives cuLA a 1.478x speedup over FLA and CAKE a 1.135x
speedup over cuLA. The independently measured CUDA-event medians were 2.077,
2.367, and 3.458 ms respectively, so the profiler and benchmark agree within a
few percent.

## CAKE breakdown

| Stage / kernel | Time / backward | GPU time |
|---|---:|---:|
| Persistent fused backward (`kernel_flashkda_backward_persistent_c16`) | 1.913 ms | 93.9% |
| FP32 parameter reduction (`kernel_flashkda_backward_param_reduce_c16_partial`) | 0.124 ms | 6.1% |

CAKE fuses the token/head work into one persistent kernel and leaves only the
global parameter-gradient reduction as a second launch.

## cuLA breakdown

| Stage / kernel group | Time / backward | GPU time |
|---|---:|---:|
| CuTeDSL intra backward | 0.664 ms | 28.7% |
| CuTeDSL WY/dKQG fused | 0.500 ms | 21.6% |
| Recurrent state gradient (`dhu`) | 0.340 ms | 14.7% |
| Direct-copy/layout kernels | 0.173 ms | 7.5% |
| Gate local reverse cumsum | 0.127 ms | 5.5% |
| `dAv` | 0.122 ms | 5.3% |
| BF16 copy/cast kernels | 0.117 ms | 5.0% |
| Safe-gate backward | 0.109 ms | 4.7% |
| Q/K L2-normalization backward | 0.108 ms | 4.7% |
| BF16/FP32 reductions | 0.050 ms | 2.2% |
| Sigmoid backward and fill | 0.005 ms | 0.2% |

The two optimized CuTeDSL kernels account for 50.3% of cuLA GPU time. Copies,
casts, and standalone reductions account for about 14.9%, and are the largest
remaining fusion opportunity outside the recurrent-state kernel.

## FLA Triton breakdown

| Stage / kernel group | Time / backward | GPU time |
|---|---:|---:|
| Triton intra backward | 1.280 ms | 37.4% |
| Triton WY/dKQG fused | 1.174 ms | 34.4% |
| Recurrent state gradient (`dhu`) | 0.340 ms | 9.9% |
| Q/K L2-normalization backward | 0.145 ms | 4.2% |
| Gate local reverse cumsum | 0.127 ms | 3.7% |
| `dAv` | 0.121 ms | 3.5% |
| Safe-gate backward | 0.110 ms | 3.2% |
| Copies, casts, reductions, sigmoid, and add | 0.122 ms | 3.6% |

FLA spends 71.8% of its GPU time in intra and WY/dKQG. The corresponding cuLA
kernels are 1.93x and 2.35x faster. The recurrent-state, `dAv`, cumsum, and gate
kernels are effectively the same speed between cuLA and FLA, which explains why
the end-to-end cuLA speedup is lower than the speedup of either CuTeDSL kernel.

## Main conclusions

1. CAKE wins by launch and dataflow fusion: two kernels per backward versus 17
   for cuLA and 16 for FLA.
2. cuLA's advantage over FLA comes almost entirely from the SM100 CuTeDSL intra
   and WY/dKQG kernels.
3. After those two kernels, cuLA's most useful optimization target is eliminating
   the copy/cast/reduction tail, followed by fusing more work around `dhu`.
4. For this exact shape, CuTeDSL intra with saved forward intermediates is the
   fastest tested cuLA combination: 1.271x faster in backward than Triton intra,
   and 1.105x faster than recomputing the forward intermediates.

The raw reports are stored on the GB200 host under
`/ossfs/workspace/cuLA-fast-cutedsl-bwd/benchmarks/` as
`nsys_kda_h64_8k_varlen_{cake,cula,fla}.nsys-rep`.
