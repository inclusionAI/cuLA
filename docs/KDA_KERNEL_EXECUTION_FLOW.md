# KDA Kernel Execution Flow

## Overview

This document describes the detailed execution flow of the KDAChunkwise kernel, covering the complete pipeline from input loading to output writeback. The kernel processes attention in 64-token chunks using three specialized warp groups that work concurrently.

**Key Characteristics**:
- **Chunk Size**: 64 tokens per iteration
- **Thread Organization**: 256 threads (8 warps) = 1 Load warp + 6 CUDA warps + 1 MMA warp
- **Execution Model**: Multi-stage pipelined with producer-consumer synchronization
- **Memory Hierarchy**: Global → TMA → SMEM → RMEM/TMEM → Output

---

## Thread Organization

```
CTA (256 threads, 8 warps)
│
├─ Warp 0: Load Warp (32 threads)
│  └─ Role: Asynchronous TMA data movement (G → SMEM)
│
├─ Warps 1-6: CUDA Warpgroup (192 threads)
│  └─ Role: Scalar computations (gate processing, matrix inverse, V correction)
│
├─ Warp 7: MMA Warp (32 threads)
│  └─ Role: Tensor Core matrix multiplications (QK, KK, KV, PV, SQ)
│
└─ (Warp 0 doubles as Epilogue Warp after TMA loads)
   └─ Role: Output writeback (SMEM → G via TMA)
```

**Register Allocation per Warp**:
- Load warp: 248 registers (CUDA config, reused for epilogue)
- CUDA warpgroup: 248 registers (critical for gate processing)
- MMA warp: 64 registers (sufficient for accumulator staging)

---

## Main Loop Structure

The kernel iterates over sequence length in chunks of 64 tokens:

```python
for chunk_start in range(0, S, C):  # S = sequence length, C = 64
    idx = chunk_start // C  # Chunk index: 0, 1, 2, ...
    
    # Three warp groups execute concurrently:
    if warp_idx == LOAD_WARP:
        execute_load_stage(idx)
    elif warp_idx == MMA_WARP:
        execute_mma_stage(idx)
    elif warp_idx in CUDA_WARPS:
        execute_cuda_stage(idx)
```

Each chunk processes:
- Q<sub>i</sub>, K<sub>i</sub>, V<sub>i</sub>: Current chunk's queries, keys, values (64×128 each)
- g<sub>i</sub>: Gate values for current chunk (64×128, FP32)
- β<sub>i</sub>: Beta scaling factors (64×1, FP32)
- S<sub>i</sub>: Recurrent state from previous chunks (128×128, FP32)

---

## Detailed Execution Flow by Warp

### Phase 1: Load Warp - TMA Input Loading

**Timeline**: Runs ahead of compute, prefetches next chunk

**Operations per Chunk**:

1. **Acquire Pipeline Handles**
   ```python
   g_handle = load_g_producer.acquire_and_advance()
   q_handle = load_q_producer.acquire_and_advance()
   k_handle = load_k_producer.acquire_and_advance()
   v_handle = load_v_producer.acquire_and_advance()
   ```

2. **Issue TMA Copies** (Global → SMEM)
   ```python
   # Gate values (64×128 FP32 = 32 KB)
   copy(tma_atom_g, src=tGgG[idx], dst=tGsG[g_handle.index], 
        tma_bar_ptr=g_handle.barrier)
   
   # Queries (64×128 BF16 = 16 KB)
   copy(tma_atom_q, src=tQgQ[idx], dst=tQsQ[q_handle.index],
        tma_bar_ptr=q_handle.barrier)
   
   # Keys (64×128 BF16 = 16 KB)
   copy(tma_atom_k, src=tKgK[idx], dst=tKsK[k_handle.index],
        tma_bar_ptr=k_handle.barrier)
   
   # Values (64×128 BF16 = 16 KB)
   copy(tma_atom_v, src=tVgV[idx], dst=tVsV[v_handle.index],
        tma_bar_ptr=v_handle.barrier)
   ```

3. **Pipeline Advancement**
   - TMA barriers automatically signal completion to consumers
   - Load warp proceeds to next chunk immediately (overlapped execution)

**Memory Access Pattern**:
- Burst transfers: 1024-byte aligned, coalesced reads
- Multi-stage pipelining: 2 stages for Q/K/G, 1 stage for V
- Bandwidth: ~4 TMA operations × 16 KB = 64 KB per chunk

---

### Phase 2: CUDA Warpgroup - Gate Processing

**Timeline**: Waits for TMA completion, processes gates and matrices

**Step 2.1: Load Beta Scaling Factors**
```python
# Each thread loads one beta value (64 threads for 64 tokens)
if local_tidx < 64:
    sBeta[local_tidx] = beta_chunk[local_tidx]
cuda_wg_sync_barrier.arrive_and_wait()
```

**Step 2.2: Load Inputs from SMEM to RMEM**
```python
# Wait for TMA completion
g_handle = load_g_consumer.wait_and_advance()
q_handle = load_q_consumer.wait_and_advance()
k_handle = load_k_consumer.wait_and_advance()

# SMEM → RMEM transfers (coalesced, 32 bytes per thread)
copy(tiled_s2r_g, src=tRS_sG[g_handle.index], dst=tRS_rG)
copy(tiled_s2r_q, src=tRS_sQ[q_handle.index], dst=tRS_rQ)
copy(tiled_s2r_k, src=tRS_sK[k_handle.index], dst=tRS_rK)

# Fence to ensure SMEM reads complete
fence_proxy(async_shared, shared_cta)
```

**Step 2.3: Gate Computation** (Critical Path)
```python
# Load values from RMEM (per-thread data)
g_val = tRS_rG.load()  # BF16
q_val = tRS_rQ.load()  # BF16
k_val = tRS_rK.load()  # BF16

# Convert to FP32 for exponential computation
g_f32 = g_val.to(Float32)
q_f32 = q_val.to(Float32)
k_f32 = k_val.to(Float32)

# Compute gate exponentials
exp_g = exp2(g_f32)         # exp(g_cumsum) for inter-chunk
exp_neg_g = exp2(-g_f32)    # exp(-g_cumsum) for intra-chunk

# Apply gates with scaling
q_gated = q_f32 * exp_g * scale          # Q' = Q·exp(g)·scale
k_inter = k_f32 * exp_g                  # K_inter = K·exp(g)
k_intra = k_f32 * exp_neg_g              # K_intra = K·exp(-g)

# Convert back to BF16
q_gated_bf16 = q_gated.to(BF16)
k_inter_bf16 = k_inter.to(BF16)
k_intra_bf16 = k_intra.to(BF16)

# Store to RMEM tensors
tRS_rQ.store(q_gated_bf16)        # Gated queries
tRS_rK.store(k_inter_bf16)        # K for state update
tRS_rG_bf16.store(k_intra_bf16)   # K for attention
```

**Gate Value Semantics**:
- `g_cumsum`: Cumulative sum of raw gate values within chunk
- `exp(g)`: Forward decay factor (modulates state contributions)
- `exp(-g)`: Intra-chunk normalization (attention scores)

**Step 2.4: Write Gated Values Back to SMEM**
```python
# Ensure SMEM reads completed before overwrite
cuda_wg_sync_barrier.arrive_and_wait()

# Produce gated K for inter-chunk (state update path)
k2_handle = load_k2_producer.acquire_and_advance()
copy(tiled_s2r_k, src=tRS_rK, dst=tRS_sK[k_handle.index])
k2_handle.commit()

# Produce gated K for intra-chunk (attention path)
kt2_handle = load_kt2_producer.acquire_and_advance()
copy(tiled_s2r_g_bf16, src=tRS_rG_bf16, dst=tRS_sG_bf16[kt2_handle.index])
kt2_handle.commit()

# Produce gated Q
q2_handle = load_q2_producer.acquire_and_advance()
copy(tiled_s2r_q, src=tRS_rQ, dst=tRS_sQ[q_handle.index])
q2_handle.commit()

fence_proxy(async_shared, shared_cta)
```

**Step 2.5: Save Last-Row exp(g) for State Decay**
```python
# Critical for inter-chunk recurrence
rG_last = exp_g[63]  # Last token's gate value
sG_last[local_tidx, g_stage_idx] = rG_last
```

**Step 2.6: Load KK from TMEM and Compute M Matrix**
```python
# Wait for MMA warp to complete K·K^T
mma_kk_handle = mma_kk_consumer.wait_and_advance()

# TMEM → RMEM load (8×8 blocks per thread)
copy(tiled_t2r_KK, src=tTR_tKK[mma_kk_handle.index], dst=tTR_rKK)
fence_view_async_tmem_load()

# Compute M = I + StrictTril(β·KK^T) in-place
apply_M_transform(tTR_rKK, sBeta, tTR_cMask, tTR_rKK_f16)
# Result: Lower triangular matrix with identity diagonal

# Store M to SMEM for MMA consumption
smem_kk_handle = smem_kk_producer.acquire_and_advance()
copy(tiled_r2s_KK, src=tRS_rKK, dst=tRS_sKK[smem_kk_handle.index])
fence_proxy(async_shared, shared_cta)
```

**Step 2.7: Matrix Inverse M^{-1}** (64×64 Triangular)
```python
# Block-wise Schur complement algorithm
curr_sM_f16 = sM_f16[smem_kk_handle.index]
compute_matrix_inverse_64x64(curr_sM_f16)
# Stages: 8×8 → 16×16 → 32×32 → 64×64 diagonal blocks

# Scale by beta and convert to BF16
scale_M_inverse_with_beta(local_tidx, sBeta, curr_sM_f16, curr_sM)
smem_kk_handle.commit()
```

**Algorithm Breakdown**:
1. Invert 8 diagonal 8×8 blocks independently (threads 0-63)
2. Combine to 4 diagonal 16×16 blocks using Schur complement
3. Combine to 2 diagonal 32×32 blocks
4. Final 64×64 inversion
5. Scale each row by corresponding β value

**Step 2.8: V Correction** (V' = V - K·State)
```python
# Wait for V from TMA
v_handle = load_v_consumer.wait_and_advance()
copy(tiled_s2r_v, src=tRS_sV[v_handle.index], dst=tRS_rV)

if idx != 0:  # Not first chunk
    # Wait for K·State result from MMA warp
    ks_handle = ks_consumer.wait_and_advance()
    copy(tiled_copy_t2r_sq, src=tTR_tAcc_ks_i, dst=tTR_rAcc_pv)
    fence_view_async_tmem_load()
    
    # Perform correction: V' = V - K·S
    v_corrected = tRS_rV.load().to(Float32)
    ks = tTR_rAcc_ks.load()
    v_corrected -= ks
    tRS_rV.store(v_corrected.to(BF16))
    
    # Write corrected V back to SMEM
    copy(tiled_s2r_v, src=tRS_rV, dst=tRS_sV[v_handle.index])

v2_handle.commit()  # Signal to MMA warp
```

**Step 2.9: Load Pseudo-V and Write to SMEM**
```python
# Wait for M·V_corr result from MMA warp
pseudo_v_handle = pseudo_v_consumer.wait_and_advance()
copy(tiled_copy_t2r_pv, src=tTR_tAcc_pv_i, dst=tTR_rAcc_pv)

# Convert to BF16 and store
tTR_rPseudoV.store(tTR_rAcc_pv.load().to(BF16))
v3_handle = load_v3_producer.acquire_and_advance()
copy(tiled_r2s_pseudo_v, src=tRS_rPseudoV, dst=tRS_sPseudoV[v3_handle.index])
v3_handle.commit()
```

**Step 2.10: Attention Scores (S = Q'·K'^T) and Masking**
```python
# Wait for MMA result
s0_handle = mma_s0_consumer.wait_and_advance()
copy(tiled_t2r_S, src=tTR_tSi, dst=tTR_rS)
fence_view_async_tmem_load()

# Apply causal mask: Tril(S)
apply_mask(tTR_rS, tTR_cMask, tTR_rP)

# Write P to SMEM
p_handle = p_producer.acquire_and_advance()
copy(tiled_r2s_P, src=tRS_rP, dst=tRS_sPi)
p_handle.commit()
```

**Step 2.11: Combine Outputs (O = O_intra + O_inter)**
```python
# Wait for O_intra = P·PseudoV
o_intra_handle = o_intra_consumer.wait_and_advance()
copy(tiled_copy_t2r_pv, src=tTR_tAcc_pv_i, dst=tTR_rAcc_pv)
acc_vec = tTR_rAcc_pv.load()

if idx != 0:
    # Wait for O_inter = State·Q'
    o_inter_handle = o_inter_consumer.wait_and_advance()
    copy(tiled_copy_t2r_sq, src=tTR_tAcc_sq_i, dst=tTR_rAcc_sq)
    acc_vec_inter = tTR_rAcc_sq.load()
    acc_vec += acc_vec_inter

# Convert to BF16 and store to SMEM for writeback
tTR_rO.store(acc_vec.to(BF16))
smem_o_handle = smem_o_producer.acquire_and_advance()
copy(tiled_copy_r2s_o, src=tRS_rO, dst=tRS_sO[smem_o_handle.index])
smem_o_handle.commit()
```

**Step 2.12: State Update** (S_{i+1} = exp(g_last)·S_i + K'^T·PseudoV)
```python
if idx != (S // C - 1):  # Not last chunk
    # Load current state from TMEM
    kv_handle = kv_consumer.wait_and_advance()
    copy(tiled_copy_t2r_kv, src=tTR_tKVi, dst=tTR_rKV)
    
    # Decay state by gate: S' = exp(g_last) * S
    flat = make_tensor(tTR_rKV.iterator, layout=make_layout(D))
    scale_state(flat, sG_last[g_stage_idx])
    # Each element: state[d] *= exp(g_last[d])
    
    # Convert to BF16 for next chunk's MMA operations
    tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(BF16))
    
    # Store decayed state back to TMEM (FP32 for accumulation)
    copy(tmem_store_kv_f32, src=tmem_store_rKV, dst=tmem_store_tKVi)
    fence_view_async_tmem_store()
    
    # Store BF16 state for MMA consumption
    copy(tmem_store_kv, src=tmem_store_rAccKV, dst=tmem_store_tAccKVi)
    kv16_handle.commit()

v_handle.release()
```

**State Update Formula**:
```
S_{i+1} = diag(exp(g_last)) · S_i + K_inter^T · PseudoV
         └─────────┬──────────┘   └────────┬─────────┘
              Decay term              Contribution term
                                    (computed by MMA warp)
```

---

### Phase 3: MMA Warp - Matrix Multiplications

**Timeline**: Waits for gated inputs, computes all GEMM operations

**Step 3.1: K·K^T** (Gram Matrix)
```python
k_handle = load_k2_consumer.wait_and_advance()
kt_handle = load_kt2_consumer.wait_and_advance()
mma_kk_handle = mma_kk_producer.acquire_and_advance()

# K_inter · K_intra^T → (64×128) @ (128×64) = (64×64)
exec_mma(
    tiled_mma=kk_tiled_mma,
    tCtAcc=tCtAccKK,
    tCrA=tCrKG[k_handle.index],        # K·exp(g)
    tCrB=tCrKNegG[kt_handle.index],    # K·exp(-g)
    acc_stage_idx=mma_kk_handle.index
)
mma_kk_handle.commit()
```

**Output**: Used for M matrix construction (I + β·Tril(KK^T))

**Step 3.2: K·State** (for V correction, only if idx > 0)
```python
if idx != 0:
    kv16_handle = kv16_consumer.wait_and_advance()
    ks_handle = ks_producer.acquire_and_advance()
    
    # State^T · K_inter^T → (128×128) @ (128×64) = (128×64)
    exec_mma(
        tiled_mma=ks_tiled_mma,
        tCtAcc=tCtAccKS,  # Reuses TMEM of PV
        tCrA=tCrState_KS[kv16_handle.index],
        tCrB=tCrK_KS[k_handle.index],
        acc_stage_idx=ks_handle.index
    )
    ks_handle.commit()
```

**Output**: K·S for V correction (V' = V - K·S)

**Step 3.3: Q'·K'^T** (Attention Scores)
```python
q_handle = load_q2_consumer.wait_and_advance()
s0_handle = mma_s0_producer.acquire_and_advance()

# Q' · K_intra^T → (64×128) @ (128×64) = (64×64)
exec_mma(
    tiled_mma=qk_tiled_mma,
    tCtAcc=tCtAccQK,
    tCrA=tCrQ[q_handle.index],         # Q·exp(g)
    tCrB=tCrK[kt_handle.index],        # K·exp(-g)
    acc_stage_idx=s0_handle.index
)
s0_handle.commit()
```

**Output**: Raw attention scores (before softmax/masking)

**Step 3.4: State·Q'** (Inter-chunk Output, only if idx > 0)
```python
if idx != 0:
    o_inter_handle = o_inter_producer.acquire_and_advance()
    
    # State · Q' → (128×128) @ (128×64) = (128×64)
    exec_mma(
        tiled_mma=sq_tiled_mma,
        tCtAcc=tCtAccSQ,
        tCrA=tCrState[0],
        tCrB=tCrQ_sq[q_handle.index],
        acc_stage_idx=0
    )
    o_inter_handle.commit()
```

**Output**: O_inter (contribution from previous chunks)

**Step 3.5: M·V_corr** (Pseudo-V Generation)
```python
kk_handle = smem_kk_consumer.wait_and_advance()
v2_handle = load_v2_consumer.wait_and_advance()
pseudo_v_handle = pseudo_v_producer.acquire_and_advance()

# V_corr · M^T → (64×128) @ (128×64) = (64×128)
exec_mma(
    tiled_mma=mv_tiled_mma,
    tCtAcc=tCtAccMV,  # Reuses TMEM of PV
    tCrA=tCrV_corr[v2_handle.index],
    tCrB=tCrM[kk_handle.index],
    acc_stage_idx=pseudo_v_handle.index
)
pseudo_v_handle.commit()
```

**Output**: Pseudo-V = M^{-1}·(V - K·State), normalized values for intra-chunk attention

**Step 3.6: P·PseudoV** (Intra-chunk Output)
```python
v3_handle = load_v3_consumer.wait_and_advance()
p_handle = p_consumer.wait_and_advance()
o_intra_handle = o_intra_producer.acquire_and_advance()

# P · PseudoV → (64×64) @ (64×128) = (64×128)
exec_mma(
    tiled_mma=vp_tiled_mma,
    tCtAcc=tCtAccPV,
    tCrA=tCrV[v3_handle.index],        # PseudoV
    tCrB=tCrP[p_handle.index],         # Masked attention
    acc_stage_idx=o_intra_handle.index
)
o_intra_handle.commit()
```

**Output**: O_intra (contribution from current chunk)

**Step 3.7: K'^T·PseudoV** (State Update, only if not last chunk)
```python
if idx != (S // C - 1):
    kv_handle = kv_producer.acquire_and_advance()
    
    # PseudoV^T · K_inter → (128×64) @ (64×128) = (128×128)
    exec_mma(
        tiled_mma=kv_tiled_mma,
        tCtAcc=tCtAccKV,
        tCrA=tCrV[v3_handle.index],
        tCrB=tCrK_kv[k_handle.index],
        acc_stage_idx=0,
        always_acc=True  # Accumulate with previous state
    )
    kv_handle.commit()
```

**Output**: New state contribution (K^T·V), to be decayed and accumulated by CUDA warp

---

### Phase 4: Epilogue Warp - Output Writeback

**Timeline**: Runs after CUDA warp produces combined output

**Operations per Chunk**:
```python
for chunk_start in range(0, S, C):
    idx = chunk_start // C
    
    # Wait for output in SMEM
    smem_o_handle = smem_o_consumer.wait_and_advance()
    
    # TMA Store: SMEM → Global
    copy(tma_atom_o,
         src=bSG_sO[smem_o_handle.index],
         dst=bSG_gO[idx])
    
    # Commit and wait for TMA completion
    cp_async_bulk_commit_group()
    cp_async_bulk_wait_group(0, read=True)
    
    smem_o_handle.release()
```

**Memory Pattern**:
- Output size: 64×128 BF16 = 16 KB per chunk
- TMA burst write: Coalesced, 1024-byte aligned
- No format conversion (direct BF16 writeback)

---

## Pipeline Synchronization

### Producer-Consumer Handles

The kernel uses 13 pipeline instances for fine-grained synchronization:

| Pipeline | Producer | Consumer | Depth | Purpose |
|----------|----------|----------|-------|---------|
| `load_g` | Load warp | CUDA warp | 2 | Gate values (G) |
| `load_q` | Load warp | CUDA warp | 2 | Queries (Q) |
| `load_k` | Load warp | CUDA warp | 2 | Keys (K) |
| `load_v` | Load warp | CUDA warp | 1 | Values (V) |
| `load_q2` | CUDA warp | MMA warp | 2 | Gated Q' |
| `load_k2` | CUDA warp | MMA warp | 2 | Gated K_inter |
| `load_kt2` | CUDA warp | MMA warp | 2 | Gated K_intra |
| `mma_kk` | MMA warp | CUDA warp | 2 | KK^T GEMM result |
| `smem_kk` | CUDA warp | MMA warp | 2 | M^{-1} matrix |
| `mma_s0` | MMA warp | CUDA warp | 2 | QK^T scores |
| `pseudo_v` | MMA warp | CUDA warp | 2 | M·V_corr result |
| `kv` | MMA warp | CUDA warp | 1 | K^T·V state update |
| `o_intra` | MMA warp | CUDA warp | 2 | P·PseudoV output |
| `o_inter` | MMA warp | CUDA warp | 1 | State·Q' output |
| `smem_o` | CUDA warp | Epilogue | 2 | Final output |

### Barrier Usage

**4 Named Barriers** coordinate critical sections:

1. **TMA Completion Barrier** (Barrier 0)
   - Signals: Load warp after TMA copy
   - Waits: CUDA warp before SMEM reads

2. **TMEM Deallocation Barrier** (Barrier 1)
   - Signals: All warps at kernel end
   - Waits: Before `tmem.free()`

3. **CUDA Warpgroup Sync Barrier** (Barrier 2)
   - Signals: All CUDA warps after gate processing
   - Waits: Before M^{-1} computation, V correction

4. **MMA Sync Barrier** (Barrier 3)
   - Signals: MMA warp after GEMM completion
   - Waits: Before TMEM load of accumulator results

---

## Memory Access Patterns

### SMEM Layout (216 KB Total)

```
Offset  Size    Purpose              Stages  Reuse
------  ------  -------------------  ------  ---------------------------------
0 KB    32 KB   Q (gated)            2       Overwritten in-place by CUDA warp
32 KB   32 KB   K (inter/intra)      2       Overwritten in-place by CUDA warp
64 KB   16 KB   V / PseudoV          1       Overwritten after correction
80 KB   64 KB   G → K_intra          2       Gate → K·exp(-g) (reused buffer)
144 KB  32 KB   O (output staging)   2       For TMA writeback
176 KB  8 KB    P (attention probs)  2       Masked QK^T scores
184 KB  8 KB    M (matrix inverse)   2       M^{-1} for pseudo-V
192 KB  512 B   G_last               2       Last-row exp(g) for state decay
~193 KB 256 B   Beta (β)             1       Row-wise scaling factors
```

**Key Optimizations**:
- **Buffer Reuse**: G buffer overwritten with K_intra after gate consumed
- **In-place Updates**: Q, K updated in SMEM without additional staging
- **Multi-stage Pipelining**: 2 stages for Q/K/G, 1 stage for V

### TMEM Layout (512 Columns, 75% Utilized)

```
Purpose                   Shape        Type    Columns  Stages
------------------------  -----------  ------  -------  ------
QK accumulator            64×64        FP32    64       2
KK accumulator            64×64        FP32    64       2
PV accumulator            64×128       FP32    128      2
KV accumulator (state)    128×128      FP32    128      1
KV16 (state as BF16)      128×128      BF16    64       1
SQ accumulator            64×128       FP32    128      1
Total:                                         384/512
```

**Reuse Strategy**:
- PV TMEM reused for: KS (K·State), MV (M·V_corr), O_intra
- Reduces TMEM pressure while maintaining throughput

---

## Data Dependencies

### Critical Path (Chunk i)

```
Load Q/K/V/G (TMA)
    ↓
Gate Processing (CUDA): exp(g), Q', K_inter, K_intra
    ↓ (Q', K_intra)
    ├─→ QK MMA → S → Mask → P
    │       ↓ (P, PseudoV)
    │       └─→ PV MMA → O_intra
    ↓ (K_inter, K_intra)
    └─→ KK MMA → M → M^{-1} (CUDA)
            ↓ (M^{-1}, V_corr)
            └─→ MV MMA → PseudoV (reuse above)

(O_intra + O_inter) → Combined Output → TMA Store
```

### Inter-Chunk Dependencies

```
Chunk i-1:                Chunk i:
  KV MMA (State update)     
    ↓                     
  Decay State (CUDA)
    ↓
    └──────────────────→ KS MMA (V correction)
                            ↓
                          V' = V - K·State (CUDA)
                            ↓
                          MV MMA (PseudoV)
                        
  State (FP32 in TMEM) ──→ SQ MMA (O_inter)
```

**Recurrence Bottleneck**: State update must complete before next chunk's V correction

---

## Performance Characteristics

### Latency Breakdown (Estimated, per Chunk)

| Stage | Operation | Time (μs) | Bottleneck |
|-------|-----------|-----------|------------|
| 1 | TMA Load Q/K/V/G | 2-3 | Memory bandwidth |
| 2 | Gate Processing (exp, mul) | 5-8 | Register spills, CUDA ALU |
| 3 | KK MMA | 1-2 | Tensor Core |
| 4 | M^{-1} Computation | 10-15 | Schur complement (64×64) |
| 5 | V Correction | 2-3 | TMEM load (KS), arithmetic |
| 6 | QK MMA + Masking | 2-3 | Tensor Core + CUDA |
| 7 | MV MMA (PseudoV) | 2-3 | Tensor Core |
| 8 | PV MMA (O_intra) | 2-3 | Tensor Core |
| 9 | SQ MMA (O_inter) | 2-3 | Tensor Core |
| 10 | Output Combine + Store | 2-3 | CUDA arithmetic, TMA |
| **Total** | **Per-Chunk** | **~30-45 μs** | **Matrix Inverse** |

**Critical Bottleneck**: Matrix inverse (Step 4) dominates at 30-35% of total time

### Throughput Analysis

**For B=2, H=64, S=4096, D=128**:
- Total chunks: 4096 / 64 = 64 chunks
- Expected time: 64 chunks × 35 μs = 2.24 ms
- Measured time: **0.876 ms** (2.6× faster due to overlap)

**Overlap Factor**: ~2.6× speedup from concurrent warp execution

### Memory Bandwidth

**Per Chunk**:
- TMA Load: 64 KB (Q+K+V) + 32 KB (G) = 96 KB
- TMA Store: 16 KB (O)
- TMEM Traffic: ~384 KB read + 256 KB write = 640 KB
- **Total**: ~750 KB per chunk

**For S=4096**:
- 64 chunks × 750 KB = 48 MB total traffic
- 0.876 ms → **55 GB/s effective bandwidth** (vs ~300 GB/s peak)

**Observation**: Compute-bound on matrix inverse, not memory-bound

---

## Optimization Opportunities

### 1. Matrix Inverse Acceleration (🔥 High Impact)
**Current Bottleneck**: 64×64 Schur complement takes 10-15 μs

**Potential Solutions**:
- **Hierarchical Blocking**: 4×32×32 sub-blocks (20% compute overhead, 50% register reduction)
- **TMEM-backed Pivots**: Store intermediate results in TMEM (5% slowdown, 40 register reduction)
- **Mixed Precision**: FP16 intermediates, FP32 accumulation (50% register reduction, stability risk)

**Expected Gain**: 20-30% overall speedup if inverse time reduced by 50%

---

### 2. Pipeline Depth Tuning (🟡 Medium Impact)
**Current**: 2-stage Q/K/G, 1-stage V

**Proposal**: Reduce to 1-stage across the board
- **Pro**: 50% fewer pipeline registers
- **Con**: ~15% throughput loss from exposed memory latency

**Expected Gain**: 10-15% register spill reduction, 5-10% net slowdown

---

### 3. Gate Processing Reordering (🔥 High Impact)
**Current Issue**: exp_g and exp_neg_g both live simultaneously

**Optimization**:
```python
# Before (2 tensors live):
exp_g = exp2(g_f32)
exp_neg_g = exp2(-g_f32)

# After (1 tensor live):
exp_g = exp2(g_f32)
k_inter_bf16 = (k_f32 * exp_g).to(BF16)  # Consume immediately
exp_neg_g = exp2(-g_f32)  # Now compute second tensor
k_intra_bf16 = (k_f32 * exp_neg_g).to(BF16)
```

**Expected Gain**: 15-20% reduction in gate processing register pressure

---

### 4. V Correction Fusion (🟡 Medium Impact)
**Current**: Separate TMEM load → subtract → SMEM write

**Proposal**: Fused kernel combining KS MMA epilogue with V correction
- Eliminates TMEM → RMEM → SMEM roundtrip
- Requires custom MMA epilogue (TiledMMA extension)

**Expected Gain**: 5-10% speedup for chunks with inter-dependencies

---

### 5. Warp Specialization Refinement (🟢 Low Risk)
**Current**: CUDA warpgroup handles gate, inverse, V-correction

**Proposal**: Split into 3 sub-groups (2 warps each)
- Warps 1-2: Gate processing only
- Warps 3-4: Matrix inverse only
- Warps 5-6: V correction and output combine

**Expected Gain**: Better instruction cache locality, 5-8% speedup

---

## Future Enhancements

### 1. Variable Sequence Length Support
**Goal**: Handle non-multiple-of-64 sequences

**Approach**:
- Predicate TMA loads for last partial chunk
- Masked MMAs for partial tiles
- Conditional state updates

**Complexity**: Medium (requires extensive testing)

---

### 2. Flash-Decoding Style Parallelism
**Goal**: Parallelize across chunks for batch-1 scenarios

**Approach**:
- Split chunks across thread blocks
- Hierarchical state reduction (similar to FlashDecoding)
- Final combine in separate kernel

**Expected Gain**: 2-3× speedup for B=1, no change for B≥4

---

### 3. FP8 Mixed Precision (Hopper+)
**Goal**: Reduce SMEM/TMEM footprint by 2×

**Approach**:
- FP8 for Q/K/V storage and MMA inputs
- FP32 for accumulators (unchanged)
- Dynamic rescaling for gate values

**Expected Gain**: 40% SMEM reduction, 1.2-1.5× speedup from higher occupancy

---

### 4. Initial/Final State Support
**Goal**: Enable stateful inference (streaming scenarios)

**Approach**:
- Load initial state from global memory on first chunk
- Store final state to global memory on last chunk
- Predicate state updates based on flags

**Complexity**: Low (straightforward extension)

---

## Debugging and Validation

### Key Debug Points

1. **Gate Values**: Check `sG_flat` after cumsum
   ```cpp
   if (tidx == 0 && idx == 0) {
       cute::printf("sG after cumsum:\n");
       cute::print_tensor(sG_flat[None, None, 0]);
   }
   ```

2. **Gated Inputs**: Verify Q', K_inter, K_intra
   ```cpp
   cuda_wg_sync_barrier.arrive_and_wait();
   if (should_debug) {
       cute::printf("Q' (gated):\n");
       cute::print_tensor(sQ_flat[None, None, q_stage_idx]);
   }
   ```

3. **Matrix Inverse**: Check M^{-1} correctness
   ```cpp
   if (tidx == 0) {
       cute::printf("M after inverse:\n");
       cute::print_tensor(curr_sM);
   }
   ```

4. **State Decay**: Validate exp(g_last) application
   ```cpp
   if (should_debug_f) {
       cute::printf("State before decay:\n");
       cute::print_tensor(tTR_rKV);
       // ... scale_state() ...
       cute::printf("State after decay:\n");
       cute::print_tensor(tTR_rKV);
   }
   ```

### Common Issues

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| NaN in output | Gate overflow (exp(g) → ∞) | Clamp g_cumsum to [-20, 20] |
| Wrong attention | Missing causal mask | Verify `apply_mask()` predicate |
| State divergence | Incorrect exp(g_last) indexing | Check `sG_last[local_tidx, g_stage_idx]` |
| Slow performance | Register spills | Reorder gate computation, reduce live ranges |
| TMEM errors | Capacity overflow | Verify `_plan_tmem_offsets()` totals < 512 |

---

## References

- **Source Code**: [flashla/kda.py](../flashla/kda.py)
- **Implementation Design**: [KDA_IMPL.md](KDA_IMPL.md)
- **Performance Analysis**: [benchmark/bench_kda_performance.py](../benchmark/bench_kda_performance.py)

---

*Last Updated: 2026-01-26*  
*Kernel Version: Optimized (MMA=64, CUDA=248, commit e162933)*
