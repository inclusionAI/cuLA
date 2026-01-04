# Linear Attention with Headwise Decay — High-Level Design Document

## 1. Overview

`flashla/linear_attn.py` implements **Chunkwise Linear Attention with Per-Head Decay** targeting NVIDIA Blackwell (SM100) GPUs using NVIDIA's **CuTe DSL** (part of CUTLASS 3.x). The kernel decomposes the linear attention computation into *intra-chunk* (local) and *inter-chunk* (global) components to maximize GPU utilization while maintaining O(N) time and memory complexity.

### Key Features
| Feature | Description |
|---------|-------------|
| **Chunkwise Computation** | Sequence is processed in fixed-size chunks (default 64) to enable efficient tiling and double-buffering |
| **Per-Head Decay (λ)** | Each attention head can have an independent exponential decay factor for flexible dependency modeling |
| **Warp-Specialized Kernel** | Different warps handle loading, compute (MMA), CUDA-core post-processing, and epilogue/store independently |
| **Tensor Memory (TMEM)** | Accumulator tensors reside in SM100's dedicated tensor memory for high-bandwidth MMA operations |
| **TMA (Tensor Memory Accelerator)** | Asynchronous bulk data movement between global memory, shared memory, and tensor memory |

---

## 2. Mathematical Formulation

Linear attention with headwise decay follows the Lightning Attention algorithm:

### Intra-Chunk (Local) Computation
Within each chunk of size $C$:

$$P = \text{tril}(Q K^T)$$
$$O_{\text{intra}} = P \cdot V$$

where $\text{tril}(\cdot)$ applies a causal (lower-triangular) mask.

### Inter-Chunk (Global) State Accumulation
The hidden state $S$ (shape $[D, D]$) accumulates across chunks:

$$S_i = \lambda^C \cdot S_{i-1} + K_i^T V_i$$

The inter-chunk output contribution:

$$O_{\text{inter}} = S \cdot Q^T$$

### Final Output
$$O = O_{\text{intra}} + O_{\text{inter}}$$

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LinearAttentionChunkwise                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  __init__()         Configure tile shapes, warp assignments, TMEM layout    │
│  __call__()         JIT entry point: setup tensors, TMA descriptors, launch │
│  kernel()           @cute.kernel: warp-specialized execution                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Class Initialization

```python
LinearAttentionChunkwise(
    chunk_size=64,          # Tokens per chunk
    qk_acc_dtype=Float32,   # QK accumulator precision
    kv_acc_dtype=Float32,   # KV state accumulator precision
    io_dtype=BFloat16,      # Input/output precision
)
```

**Tile Shapes (MMA Tilers)**:
| Operation | Tile Shape (M, N, K) | Description |
|-----------|---------------------|-------------|
| `qk_mma_tiler` | (64, 64, 128) | $Q \times K^T$ → attention scores |
| `kv_mma_tiler` | (128, 128, 64) | $V^T \times K$ → state update |
| `vp_mma_tiler` | (128, 64, 64) | $V \times P$ → intra-chunk output |
| `sq_mma_tiler` | (128, 64, 128) | $S \times Q^T$ → inter-chunk output |

### 3.2 Warp Specialization

The kernel uses 7 warps (224 threads) with distinct roles:

| Warp ID | Role | Description |
|---------|------|-------------|
| 0–3 | **CUDA Core Warps** | Post-process MMA outputs: causal masking, dtype conversion, TMEM↔RMEM transfers |
| 4 | **MMA Warp** | Execute tensor-core MMAs (QK, KV, VP, SQ) |
| 5 | **Load Warp** | Issue TMA loads for Q, K, V from global to shared memory |
| 6 | **Epilogue Warp** | TMA stores from shared memory to global memory |

### 3.3 Memory Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│                      Global Memory (HBM)                       │
│   Q, K, V: [B, S, H, D]    O: [B, S, H, D]                     │
└────────────────────────────────────────────────────────────────┘
                              │ TMA
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   Shared Memory (SMEM)                         │
│   sQ, sK, sV: staged buffers    sP: masked scores              │
│   sO: output staging            sKV: state staging             │
└────────────────────────────────────────────────────────────────┘
                              │ TMEM Copy
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   Tensor Memory (TMEM)                         │
│   tCtAccQK: QK accumulator      tCtAccPV: intra-chunk output   │
│   tCtAccKV: state (FP32)        tCtAccSQ: inter-chunk output   │
│   State16: state (BF16 for MMA operand A)                      │
└────────────────────────────────────────────────────────────────┘
```

**TMEM Layout Planning** (`_plan_tmem_offsets`):
- Computes column offsets for each accumulator tensor
- Ensures total usage ≤ 512 columns (SM100 TMEM capacity)
- Allocates separate space for FP32 state and BF16 state copy

---

## 4. Kernel Execution Flow

### 4.1 Per-Chunk Pipeline (Simplified)

```
For each chunk i in [0, S/C):
    
    ┌─────────────────────────────────────────────────────────────┐
    │ LOAD WARP (warp 5)                                          │
    │   TMA.load(Q_i → sQ)                                        │
    │   TMA.load(K_i → sK)                                        │
    │   TMA.load(V_i → sV)                                        │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ MMA WARP (warp 4)                                           │
    │   if i > 0:                                                 │
    │       SQ = State × Q^T  (inter-chunk)                       │
    │   QK = Q × K^T                                              │
    │   VP = V × P            (after CUDA warps apply mask)       │
    │   State += V^T × K      (accumulate state)                  │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ CUDA CORE WARPS (warps 0–3)                                 │
    │   Load QK from TMEM → registers                             │
    │   Apply causal mask: P[i,j] = 0 if j > i                    │
    │   Convert FP32 → BF16, store P to SMEM                      │
    │   Convert State FP32 → BF16, store to TMEM (for SQ MMA)     │
    │   Load O_intra, O_inter from TMEM                           │
    │   O = O_intra + O_inter                                     │
    │   Store O to SMEM                                           │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ EPILOGUE WARP (warp 6)                                      │
    │   TMA.store(sO → O_i)                                       │
    └─────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline Synchronization

The kernel uses multiple **named barriers** and **pipeline objects** for producer-consumer synchronization:

| Pipeline | Producer | Consumer | Purpose |
|----------|----------|----------|---------|
| `load_q/k/v_*` | Load Warp | MMA Warp | Q/K/V ready in SMEM |
| `mma_s0_*` | MMA Warp | CUDA Warps | QK scores ready in TMEM |
| `p_*` | CUDA Warps | MMA Warp | Masked P ready in SMEM |
| `kv_*` | MMA Warp | CUDA Warps | State (FP32) ready in TMEM |
| `kv16_*` | CUDA Warps | MMA Warp | State (BF16) ready in TMEM |
| `o_intra_*` | MMA Warp | CUDA Warps | Intra-chunk output ready |
| `o_inter_*` | MMA Warp | CUDA Warps | Inter-chunk output ready |
| `smem_o_*` | CUDA Warps | Epilogue Warp | Output ready in SMEM |

---

## 5. Key Implementation Details

### 5.1 Tensor Layouts

Input tensors follow PyTorch's row-major convention `[B, S, H, D]` with strides `(S*H*D, H*D, D, 1)`.

The kernel internally permutes to optimize for MMA operand requirements:
- **Q, K**: `(S, D, (H, B))` — K-major for MMA operand A/B
- **V, O**: `(D, S, (H, B))` — Column-major for MMA operand A and epilogue store

### 5.2 TMA Descriptor Setup

TMA (Tensor Memory Accelerator) descriptors are created for bulk async copies:

```python
tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
    tma_load_op,      # CopyBulkTensorTileG2SOp
    q,                # Global tensor
    q_smem_layout,    # Target SMEM layout (with swizzle)
    qk_mma_tiler,     # Tile shape
    qk_tiled_mma,     # Associated MMA
    cluster_shape,    # (1,1,1) — no multicast
)
```

### 5.3 Causal Masking

Applied in CUDA core warps after loading QK scores from TMEM:

```python
def apply_mask(self, acc_qk, index_qk, p, ...):
    for i in range(size(acc_qk)):
        index_q, index_k = index_qk[i]
        if index_q < index_k:
            acc_qk[i] = 0.0
            p[i] = BFloat16(0.0)
        else:
            p[i] = acc_qk[i].to(BFloat16)
```

### 5.4 State Accumulation

The FP32 state tensor accumulates across chunks:

```python
kv_tiled_mma = exec_mma(
    ...,
    tCtAcc=tCtAccKV,        # State accumulator in TMEM
    tCrA=tCrV,              # V^T (from SMEM)
    tCrB=tCrK_kv,           # K (from SMEM)
    always_acc=True if idx != 0 else False,  # Accumulate after first chunk
)
```

For use as MMA operand A (requires BF16), CUDA warps convert and copy to a separate TMEM region.

---

## 6. Shared Memory Layout

The `SharedStorage` struct organizes all SMEM allocations:

```python
@cute.struct
class SharedStorage:
    # Pipeline barriers (mbarriers)
    load_q_mbar_ptr: MemRange[Int64, q_stage * 2]
    load_k_mbar_ptr: MemRange[Int64, k_stage * 2]
    load_v_mbar_ptr: MemRange[Int64, v_stage * 2]
    s_mbar_ptr: MemRange[Int64, acc_stage * 2]
    ...
    
    # TMEM address holder
    tmem_holding_buf: Int32
    
    # Staged SMEM tensors
    sO: Align[MemRange[io_dtype, cosize(o_smem_layout_staged)], 1024]
    sQ: Align[MemRange[q_dtype, cosize(q_smem_layout_staged)], 1024]
    sK: Align[MemRange[k_dtype, cosize(k_smem_layout_staged)], 1024]
    sV: Align[MemRange[v_dtype, cosize(v_smem_layout_staged)], 1024]
    sP: Align[MemRange[v_dtype, cosize(p_smem_layout_staged)], 1024]
```

All tensors are 1024-byte aligned for optimal TMA performance.

---

## 7. Grid and Block Configuration

```python
grid = (
    1,              # Chunks processed sequentially (loop inside kernel)
    num_heads,      # Parallelize across heads
    batch_size,     # Parallelize across batches
)
block = (224, 1, 1)  # 7 warps × 32 threads
cluster = (1, 1, 1)  # Single-CTA cluster (no multicast)
```

---

## 8. Performance Considerations

### Strengths
1. **Warp specialization** hides latency by overlapping load, compute, and store
2. **Double-buffering** (2 stages) enables pipelined execution
3. **TMEM accumulation** keeps MMA outputs on-chip until needed
4. **TMA bulk copies** maximize memory bandwidth utilization

### Current Limitations / TODOs
1. **Fixed chunk size (64)** and head dimension (128) — parameterization needed
2. **Single-CTA cluster** — potential for 2-CTA optimization
3. **No variable-length (varlen) support** — sequence length must be divisible by chunk size
4. **No initial state input** — always starts from zero state
5. **Per-head decay not fully utilized** — decay pointer passed but not applied in current implementation

---

## 9. Usage Example

```python
from flashla import LinearAttentionChunkwise
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

# Create kernel
kernel = LinearAttentionChunkwise(
    chunk_size=64,
    qk_acc_dtype=cutlass.Float32,
    io_dtype=cutlass.BFloat16,
)

# Prepare inputs (PyTorch tensors on CUDA)
Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
O = torch.zeros_like(Q)
decay = torch.full((H,), 0.95, device="cuda", dtype=torch.float32)

# Convert to CuTe pointers
q_cute, k_cute, v_cute, o_cute = map(from_dlpack, [Q, K, V, O])
decay_cute = from_dlpack(decay)

# Compile and run
stream = cutlass_torch.default_stream()
compiled = cute.compile(kernel, q_cute.iterator, k_cute.iterator, ...)
compiled(q_cute.iterator, k_cute.iterator, v_cute.iterator, 
         o_cute.iterator, decay_cute.iterator, (B, S, H, D), stream)
```

---

## 10. File Structure

```
flashla/
├── __init__.py              # Exports LinearAttentionChunkwise
├── linear_attn.py           # Main implementation (this file)
├── flashla_interface.py     # C++ extension interface (Lightning Attention)
└── _version.py              # Version info (setuptools_scm)

benchmark/
└── bench_linear_attn.py     # Benchmark comparing CuTe DSL vs Triton
```

---

## 11. References

1. **Lightning Attention**: Qin et al., "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
2. **CUTLASS 3.x / CuTe DSL**: NVIDIA CUTLASS library documentation
3. **SM100 (Blackwell) Architecture**: NVIDIA Blackwell GPU Architecture Whitepaper
