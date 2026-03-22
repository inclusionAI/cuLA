<div align="center">

# cuLA — CUDA Linear Attention

**High-performance CUDA kernels for linear attention variants, written in [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) and CUTLASS C++.**

</div>

## Introduction

Linear attention mechanisms reformulate the standard attention computation by replacing the softmax with a feature map $\phi$, enabling the output to be expressed via a recurrent state:

$$S_i = S_{i-1} + \phi(k_i) v_i^T, \quad o_i = \phi(q_i)^T S_i$$

This recurrence reduces the complexity from $O(N^2)$ (standard attention) to $O(N)$, making linear attention particularly attractive for long-sequence modeling in LLMs. Recent variants — such as [GLA](https://arxiv.org/abs/2312.06635), [KDA](http://arxiv.org/abs/2510.26692), [GDN](https://arxiv.org/abs/2412.06464), and [Lightning Attention](https://arxiv.org/abs/2405.17381) — further enhance expressiveness with gating, delta updates, and chunkwise decomposition.

**cuLA** provides hand-tuned CUDA implementations of these linear attention variants, targeting NVIDIA Blackwell (SM100) and Hopper (SM90) GPUs. It is designed as a submodule of [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention), sharing the same interface — adopting cuLA requires only a one-line import change. For ease of maintenance, cuLA is currently developed as a standalone library; the end goal is for users to seamlessly access these kernels through FLA. Since FLA already has a kernel dispatch mechanism in place, integration will be ready soon.

## Installation

```bash
git clone <repo-url>
git submodule update --init --recursive

# Install PyTorch (CUDA 13.0)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install flash-linear-attention for benchmark repro
cd third_party/flash-linear-attention
pip install -e .
cd ../..

# Install cuLA
pip install -e . --no-build-isolation
```

> **Requirements:** Python 3.12+, CUDA Toolkit 13.0, NVCC 12.9+ (SM100a support), PyTorch 2.9.1+

## Quick Start

### KDA (Kimi Delta Attention)

cuLA is a drop-in replacement for [FLA](https://github.com/fla-org/flash-linear-attention) — just change the import:

```python
import torch
from flashla.kda.chunk import chunk_kda  # <-- one-line change from fla.ops.kda

B, T, H, K, V = 2, 2048, 4, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1   # gate (log space)
beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

# Forward
o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=1.0,
    A_log=A_log,
    dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

# Backward
do = torch.randn_like(o)
o.backward(do)

print(f'Output shape: {o.shape}')             # [2, 2048, 4, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 4, 128, 128]
```

**Notes:**
- `safe_gate=True` is required (leverages M=16 TensorCore acceleration).
- `beta` and `initial_state` must be `float32`.
- `cu_seqlens` (for variable-length sequences) must be `int32`.
- Matrix inversion uses FP16 precision.

## Benchmarks

All benchmarks run on a single **NVIDIA GB200** GPU with **CUDA Toolkit 13.0**, **PyTorch 2.9.1**, **Triton 3.5.1**.

FLA baseline: [flash-linear-attention@5da31d19](https://github.com/fla-org/flash-linear-attention/tree/5da31d199456ee4004f70186f3391d309e26ca98).

### KDA — Fixed-Length (H=64, D=128, bf16)

| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |
|---|---|-----------------|-----------|---------|
| 1 | 512 | 0.572 | 0.530 | **1.08x** |
| 1 | 1024 | 0.553 | 0.523 | **1.06x** |
| 1 | 4096 | 0.885 | 0.586 | **1.51x** |
| 1 | 8192 | 1.686 | 1.099 | **1.53x** |
| 1 | 16384 | 3.298 | 2.131 | **1.55x** |
| 2 | 512 | 0.567 | 0.534 | **1.06x** |
| 2 | 1024 | 0.557 | 0.520 | **1.07x** |
| 2 | 4096 | 1.563 | 1.102 | **1.42x** |
| 2 | 8192 | 3.048 | 2.145 | **1.42x** |
| 2 | 16384 | 6.043 | 4.231 | **1.43x** |

### KDA — Variable-Length (H=64, D=128, bf16)

| Config | FLA Triton (ms) | cuLA (ms) | Speedup |
|--------|-----------------|-----------|---------|
| 1seq, T=4096 | 0.893 | 0.586 | **1.52x** |
| 1seq, T=8192 | 1.689 | 1.097 | **1.54x** |
| 1seq, T=16384 | 3.296 | 2.133 | **1.55x** |
| 20seqs, T=4096, uniform | 0.902 | 0.671 | **1.34x** |
| 20seqs, T=8192, uniform | 1.610 | 1.168 | **1.38x** |
| 25seqs, T=8192, uniform | 1.629 | 1.182 | **1.38x** |
| 20seqs, T=16384, uniform | 3.059 | 2.181 | **1.40x** |
| 25seqs, T=16384, uniform | 3.070 | 2.187 | **1.40x** |
| 20seqs, T=4096, skewed | 0.872 | 0.608 | **1.43x** |
| 20seqs, T=8192, skewed | 1.698 | 1.111 | **1.53x** |
| 20seqs, T=16384, skewed | 3.340 | 2.134 | **1.57x** |
| 20seqs, T=4096, tail-heavy | 0.910 | 0.617 | **1.48x** |
| 20seqs, T=8192, tail-heavy | 1.757 | 1.125 | **1.56x** |
| 20seqs, T=16384, tail-heavy | 3.489 | 2.155 | **1.62x** |

To reproduce:

```bash
python benchmarks/bench_kda.py --mode both
```

### Lightning Attention — Prefill (H=64, D=128, bf16)

| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |
|---|---|-----------------|-----------|---------|
| 1 | 1024 | 0.094 | 0.070 | **1.34x** |
| 1 | 4096 | 0.205 | 0.154 | **1.33x** |
| 1 | 8192 | 0.393 | 0.291 | **1.35x** |
| 1 | 16384 | 0.765 | 0.561 | **1.36x** |
| 2 | 1024 | 0.109 | 0.064 | **1.71x** |
| 2 | 4096 | 0.386 | 0.175 | **2.21x** |
| 2 | 8192 | 0.753 | 0.326 | **2.31x** |
| 2 | 16384 | 1.486 | 0.631 | **2.36x** |

### Lightning Attention — Variable-Length (H=64, D=128, bf16)

Persistent CuTe DSL kernel vs FLA Triton varlen, 126 configs (N=5..25 seqs, T=1K..32K, uniform/skewed/random).

| N (seqs) | T | Dist | cuLA (ms) | FLA Triton (ms) | Speedup |
|----------|---|------|-----------|-----------------|---------|
| 5 | 1024 | uniform | 0.079 | 0.184 | **2.33x** |
| 5 | 8192 | skewed | 0.216 | 0.404 | **1.87x** |
| 10 | 16384 | skewed | 0.401 | 0.731 | **1.82x** |
| 16 | 8192 | uniform | 0.250 | 0.403 | **1.61x** |
| 16 | 32768 | uniform | 0.724 | 1.298 | **1.79x** |
| 20 | 16384 | skewed | 0.445 | 0.732 | **1.65x** |
| 20 | 32768 | skewed | 0.776 | 1.452 | **1.87x** |
| 25 | 32768 | random | 0.793 | 1.449 | **1.83x** |

Summary (126 configs): **avg=1.58x**, min=0.94x, max=2.33x. Persistent vs Non-persistent output is **bit-exact**.

To reproduce:

```bash
python benchmarks/bench_lightning_attn.py --modes no_state h0_ht varlen
```

## Tests

```bash
# End-to-end test (forward + backward) against FLA Triton implementation
python -m pytest tests/test_kda_e2e_compare_fla.py -v

# End-to-end test against naive KDA reference
python -m pytest tests/test_kda.py -v
```

<details>
<summary>Sample test output</summary>

```
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B1-T63-H1-D128-...]    PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T500-H3-D128-...]   PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T1000-H3-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B3-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T2048-H8-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk_varlen[...]             PASSED
...
======================= 17 passed in 40.95s =======================
```

</details>

CUDA kernel tuning is significantly more labor-intensive than Triton — contributions from the open-source community are warmly welcomed!

## Roadmap

### **Current Status**

- [x] **Modular KDA Forward (Blackwell)** — compatible with Context Parallelism (CP)
  - [x] `chunk_intra_subchunk`
  - [x] `chunk_gated_delta_h`
  - [x] `chunk_fwd_o`
- [x] **Fused KDA Forward (Hopper)**
- [x] **Fused Lightning Prefill (Blackwell)**
- [x] **Lightning Decode (Hopper & Blackwell)**

### **Roadmap**

* [ ] Integrate into [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) via FLA's kernel dispatch mechanism
* [ ] More fusions.

**Train**

* [x] Modular KDA Forward (sm100, compatible with Kimi CP)
  * [x] kda chunk intra
  * [x] chunk gated delta h
  * [ ] recompute wu
  * [x] chunk fwd o

* [ ] Modular GDN Forward / Backward Kernels (compatible with Kimi CP)

* [ ] More backward supports

* [ ] Kernel-level compute-communication overlapping CP linear attention kernels (via **nvshmem**)

**Inference**

* [x] Lightning prefill kernel (sm100)

* [x] Lightning decode kernel (sm90 & sm100)

* [x] Fused KDA prefill kernel (sm90)

* [ ] Fused KDA prefill kernel (sm100)

* [ ] Small B/H optimizations

* [ ] MTP support

* [ ] More aggressive fusion of small neighbor kernels like cumsum for inference scenarios.

## Acknowledgements

This project is inspired by [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL). We thank [FLA-org](https://github.com/fla-org) and NVIDIA for their great work.

## Contact

If you're interested in an internship or job opportunity, feel free to reach out: **shuyan.ycf@antgroup.com**  / **chaofanyu@gmail.com**

No cuda experiences are required as long as you're a quick leaner.
