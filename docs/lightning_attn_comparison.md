# Lightning Attention 2: Comparative Analysis Report

## Overview

This document provides a detailed comparative analysis of two Triton implementations from OpenNLPLab's [lightning-attention](https://github.com/OpenNLPLab/lightning-attention) repository:

1. **`lightning_attn2_no_decay.py`**: Vanilla linear attention without exponential decay
2. **`lightning_attn2.py`**: Linear attention with headwise exponential decay (lambda)

Both implementations follow the chunkwise linear attention algorithm, but differ fundamentally in how they handle temporal information decay across sequence positions.

---

## 1. Mathematical Formulation

### 1.1 Vanilla Linear Attention (No Decay)

The standard causal linear attention is defined as:

$$
O_t = \sum_{i=1}^{t} Q_t K_i^\top V_i = Q_t \sum_{i=1}^{t} K_i^\top V_i = Q_t \cdot S_t
$$

Where the hidden state $S_t$ accumulates without decay:

$$
S_t = S_{t-1} + K_t^\top V_t
$$

**Intra-chunk computation** (within a block):
$$
O_{\text{intra}} = \text{tril}(QK^\top) \cdot V
$$

**Inter-chunk computation** (from previous blocks):
$$
O_{\text{inter}} = Q \cdot S
$$

### 1.2 Linear Attention with Decay

The decay-enhanced variant introduces a headwise exponential decay factor $\lambda_h = e^{-s_h}$, where $s_h > 0$ is the log-lambda parameter per attention head:

$$
O_t = \sum_{i=1}^{t} \lambda^{t-i} Q_t K_i^\top V_i
$$

The hidden state now decays exponentially:

$$
S_t = \lambda \cdot S_{t-1} + K_t^\top V_t
$$

**Intra-chunk computation** (with position-aware decay):
$$
O_{\text{intra}} = (QK^\top \odot D) \cdot V
$$

Where $D_{ij} = \exp(-s \cdot (i-j))$ for $i \geq j$, and $-\infty$ otherwise.

**Inter-chunk computation** (with query decay):
$$
O_{\text{inter}} = (Q \cdot S) \odot \exp(-s \cdot \text{offset})
$$

---

## 2. API Differences

| Aspect | No Decay | With Decay |
|--------|----------|------------|
| **Forward signature** | `forward(ctx, q, k, v)` | `forward(ctx, q, k, v, s)` |
| **Backward return** | `return dq, dk, dv` | `return dq, dk, dv, None, None` |
| **Input parameters** | 3 tensors | 4 tensors (+ `s` for log-lambda) |
| **S parameter shape** | N/A | `(h,)` - one scalar per head |

### API Usage

```python
# No decay version
from lightning_attn.ops.triton import lightning_attn2_no_decay
o = lightning_attn2_no_decay(q, k, v)

# With decay version
from lightning_attn.ops.triton import lightning_attn2
s = torch.randn(h, device=q.device, dtype=torch.float32)  # log-lambda per head
o = lightning_attn2(q, k, v, s)
```

---

## 3. Forward Pass Implementation Differences

### 3.1 Kernel Parameters

**No Decay (`_fwd_kernel`):**
```python
@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    b: tl.constexpr, h: tl.constexpr, n: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr,
    BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr,
):
```

**With Decay (`_fwd_kernel`):**
```python
@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    S,  # log lambda - ADDITIONAL PARAMETER
    b: tl.constexpr, h: tl.constexpr, n: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr,
    BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr,
):
```

### 3.2 Decay Factor Computation (Only in Decay Version)

```python
# Load headwise decay parameter
s = tl.load(S_block_ptr)

# Query position decay: exp(-s * position_in_block)
q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])

# Key transpose decay: exp(-s * (BLOCK - position))
k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))

# Block-level decay for state accumulation
block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

# Diagonal decay matrix for intra-chunk attention
index = off_block[:, None] - off_block[None, :]
s_index = s * index
s_index = tl.where(index >= 0, -s_index, float("-inf"))
diag_decay = tl.exp(s_index)
```

### 3.3 Causal Masking

**No Decay** (Simple binary mask):
```python
index = off_block[:, None] - off_block[None, :]
qk = tl.dot(q, k_trans)
qk = tl.where(index >= 0, qk, 0)  # Binary causal mask
```

**With Decay** (Exponential decay mask):
```python
index = off_block[:, None] - off_block[None, :]
s_index = s * index
s_index = tl.where(index >= 0, -s_index, float("-inf"))
diag_decay = tl.exp(s_index)  # Smooth exponential decay

qk = tl.dot(q, k_trans) * diag_decay  # Apply decay to attention
```

### 3.4 Output Computation

**No Decay:**
```python
o_intra = tl.dot(qk, v)       # Intra-chunk: simple masked attention
o_inter = tl.dot(q, kv)       # Inter-chunk: query × accumulated state
o = o_intra + o_inter
```

**With Decay:**
```python
o_intra = tl.dot(qk, v)                    # Intra-chunk: decay-weighted attention
o_inter = tl.dot(q, kv) * q_decay          # Inter-chunk: with query position decay
o = o_intra + o_inter
```

### 3.5 State Accumulation

**No Decay** (Simple accumulation):
```python
kv += tl.dot(k_trans, v)
```

**With Decay** (Decayed accumulation):
```python
kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
```

---

## 4. Backward Pass Implementation Differences

### 4.1 Intra-Chunk Backward Kernel

**No Decay (`_bwd_intra_kernel`):**
- Uses simple binary causal mask
- No decay factors in gradient computation

```python
index = array[:, None] - array[None, :]
# Simple mask for backward
```

**With Decay (`_bwd_intra_kernel`):**
- Loads decay parameter `s`
- Computes `diag_decay` and its transpose
- Applies decay to gradient computations

```python
s = tl.load(S_block_ptr)
s_index = s * index
s_index = tl.where(index >= 0, -s_index, float("-inf"))
diag_decay = tl.exp(s_index)
diag_decay_trans = tl.trans(diag_decay)

dqk = tl.dot(do, v_trans) * diag_decay
dq_intra = tl.dot(dqk, k)
dk_intra_trans = tl.dot(q_trans, dqk)
qk_trans = tl.dot(k, q_trans) * diag_decay_trans
dv_intra = tl.dot(qk_trans, do)
```

### 4.2 Inter-Chunk Backward Kernel

**No Decay:**
```python
# Simple state accumulation for gradients
kv_trans = tl.zeros([e, d], dtype=tl.float32)
dkv_current += tl.dot(q_trans, do)
dkv += dkv_current
```

**With Decay:**
```python
# Decay-aware gradient computation
block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
q_decay = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[:, None]))
q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None])))

dq_inter = tl.dot(do, kv_trans) * q_decay
dkv_current += tl.dot(q_trans * q_decay_trans, do)
dkv = block_decay * dkv + dkv_current
```

---

## 5. Key Implementation Patterns

### 5.1 Shared Patterns

| Pattern | Both Implementations |
|---------|---------------------|
| **Block size** | `BLOCK = 64` |
| **Compute block** | `CBLOCK = 32`, `NUM_CBLOCK = BLOCK // CBLOCK` |
| **Parallelization** | Grid over `(b * h, num_e_blocks)` for forward |
| **Memory layout** | `[batch, heads, seq_len, dim]` |
| **Precision** | FP32 accumulation with mixed-precision loads |

### 5.2 Decay-Specific Patterns

| Pattern | Description |
|---------|-------------|
| **Per-head decay** | Single `s` value per attention head |
| **Position encoding** | Decay encodes relative position information |
| **Block-level decay** | `block_decay = exp(-s * BLOCK)` for state transitions |
| **Smooth masking** | Replaces hard causal mask with exponential decay |

---

## 6. Performance Implications

### 6.1 Computational Overhead

| Operation | No Decay | With Decay |
|-----------|----------|------------|
| **Memory reads** | Q, K, V | Q, K, V, S |
| **Exp operations** | 0 | 4 per block (q_decay, k_decay, block_decay, diag_decay) |
| **Multiplications** | Standard | Additional decay factor multiplications |
| **State update** | `kv += ...` | `kv = decay * kv + ...` |

### 6.2 Memory Access Pattern

**No Decay:**
- Simpler memory access pattern
- No per-head decay parameter lookup

**With Decay:**
- Additional memory read for `S` tensor
- Decay factors computed on-the-fly (register pressure)

### 6.3 Numerical Properties

| Property | No Decay | With Decay |
|----------|----------|------------|
| **State growth** | Unbounded accumulation | Bounded by decay |
| **Long-range dependency** | Equal weight to all history | Exponentially decaying memory |
| **Numerical stability** | May accumulate large values | Decay provides natural regularization |

---

## 7. Use Case Recommendations

### 7.1 When to Use No-Decay Version

- **Language modeling** without explicit temporal decay requirements
- **Performance-critical** applications where decay overhead is unacceptable
- **Short sequences** where decay doesn't significantly impact quality
- **Baseline experiments** for ablation studies

### 7.2 When to Use Decay Version

- **Long sequence modeling** where recent context should be weighted more heavily
- **Speech/Audio processing** with natural temporal decay
- **Time-series prediction** with recency bias
- **Models requiring learnable decay** as an attention mechanism hyperparameter

---

## 8. Code Structure Summary

```
lightning_attn2_no_decay.py          lightning_attn2.py
├── _fwd_kernel                      ├── _fwd_kernel (+ S param, decay logic)
├── _bwd_intra_kernel               ├── _bwd_intra_kernel (+ decay in gradients)
├── _bwd_inter_kernel               ├── _bwd_inter_kernel (+ decay in gradients)
└── LightningAttention2NoDecay      └── LightningAttention2
    ├── forward(q, k, v)                ├── forward(q, k, v, s)
    └── backward → (dq, dk, dv)         └── backward → (dq, dk, dv, None, None)
```

---

## 9. Conclusion

The two implementations represent a fundamental design choice in linear attention:

| Aspect | No Decay | With Decay |
|--------|----------|------------|
| **Philosophy** | All history equally important | Recent history more important |
| **Complexity** | Simpler, fewer operations | More complex, additional exponentials |
| **Flexibility** | Fixed behavior | Learnable decay rate per head |
| **State dynamics** | Linear accumulation | Exponential moving average |

The decay version (`lightning_attn2.py`) is the **more general formulation**, as setting `s = 0` recovers the no-decay behavior. However, the no-decay version (`lightning_attn2_no_decay.py`) provides a **cleaner, more efficient baseline** when temporal decay is not required.

Both implementations share the same chunkwise algorithm structure, making them drop-in replacements for each other with minimal code changes (adding/removing the `s` parameter).

---

## References

1. [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658)
2. [OpenNLPLab/lightning-attention GitHub Repository](https://github.com/OpenNLPLab/lightning-attention)
3. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
