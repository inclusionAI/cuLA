# KDA (Kimi Delta Attention) on Blackwell SM100 实现设计

## 1. 执行摘要

本文档详细设计如何在 NVIDIA Blackwell SM100 架构上实现 Kimi Delta Attention (KDA)，从当前的 decay-based linear attention 出发。重点关注：
1. SM100 存储层次的优化利用
2. KDA 特有的 WY-representation (M, W, U matrices)
3. Gate-based causal attention 的高效实现
4. TMEM capacity 限制下的数据布局
5. 保证正确性优先，性能其次

---

## 2. SM100 存储层次回顾

### 2.1 存储层次结构

```
GMEM (Global Memory)
  ↓ TMA (Tensor Memory Accelerator)
SMEM (Shared Memory) - 227KB per SM
  ├→ UTCMMA (operand A/B) → TMEM (accumulator) - 256KB per SM
  │                           ↓ 
  │                      MMA Accumulator
  │
  └→ cp.async / ldmatrix → RMEM (Register Memory) - 256KB register file per SM
                            ↓
                       CUDA Cores (element-wise ops)
```

**关键数据路径**：
- **UTCMMA 路径**: SMEM → UTCMMA → TMEM (直接，无需 RMEM)
  - Operand A: 可以从 SMEM 或 TMEM 读取
  - Operand B: 从 SMEM 读取
  - Output: 直接写入 TMEM accumulator
- **CUDA Cores 路径**: TMEM → RMEM → CUDA cores → RMEM (element-wise 操作)
  - 用于 mask application, decay, gating 等操作

### 2.2 容量和特性

| 存储类型 | 容量 | 延迟 | 带宽 | 适用场景 |
|---------|------|------|------|---------|
| **GMEM** | ~数GB | 200-400 cycles | ~2TB/s | 主存储 |
| **SMEM** | 227KB/SM | 20-30 cycles | ~15TB/s | 线程间共享，软件管理 |
| **RMEM** | 256KB/SM | 1 cycle | ~200TB/s | 线程私有，编译器管理 |
| **TMEM** | 256KB/SM (128×512 cols) | 1-2 cycles | ~200TB/s | MMA 专用，accumulator 缓存 |

### 2.3 TMEM 关键特性

- **专用于 MMA**: TMEM 是 Blackwell 新增的，专门为 UTCMMA 设计的 accumulator 存储
- **容量组织**: 128 lanes × 512 columns × 32 bits/element = 256KB per SM
- **高速访问**: 直接连接 MMA 单元，无需经过 RMEM
- **容量充足**: 512 columns 的列约束，但总容量 256KB 相对充足
- **数据类型**: 主要存储 FP32 accumulator，也可以存储 BF16/FP16 (占用一半 lanes)

---

## 3. 当前 Decay Linear Attention 的 TMEM 使用

### 3.1 当前 TMEM 分配

根据代码分析，当前实现的 TMEM 布局：

```python
# TMEM allocation offsets (in columns)
tmem_qk_cols_offset      = 0              # QK accumulator (FP32)
tmem_pv_cols_offset      = [offset]       # PV accumulator (FP32)
tmem_kv_cols_offset      = [offset]       # KV accumulator (FP32) for state update
tmem_kv16_cols_offset    = [offset]       # KV16 (BF16) for state storage
tmem_sq_cols_offset      = [offset]       # SQ accumulator (FP32) for inter-chunk
```

**实际容量使用**（从代码中的 `_plan_tmem_offsets`）:
- `QK acc`: ~64-128 columns (depending on tile size and stages)
- `PV acc`: ~64-128 columns
- `KV acc`: ~64-128 columns (FP32 state computation)
- `KV16`: ~32-64 columns (BF16 state storage - 压缩版本)
- `SQ acc`: ~64-128 columns

**总计**: ~288-512 columns (占用 512 columns 约 56-100%，总容量 256KB 中的 ~140-250KB)

### 3.2 当前的数据流

```
Phase 1: Intra-chunk QK
  SMEM[Q] (operand A), SMEM[K^T] (operand B) → UTCMMA → TMEM[QK_acc]
  
Phase 2: Decay mask application
  TMEM[QK_acc] → RMEM → Apply exp(-s*(i-j)) (CUDA cores) → SMEM[P]
  
Phase 3: PV (intra-chunk output)
  SMEM[P] (operand A), SMEM[V] (operand B) → UTCMMA → TMEM[PV_acc]
  
Phase 4: KV (state update)
  SMEM[K^T] (operand A), SMEM[V] (operand B) → UTCMMA → TMEM[KV_acc]
  TMEM[KV_acc] → RMEM → Apply block_decay (CUDA cores) → TMEM[KV16]
  
Phase 5: SQ (inter-chunk output)
  TMEM[KV16] (operand A), SMEM[Q] (operand B) → UTCMMA → TMEM[SQ_acc]
  
Phase 6: Output combination
  TMEM[PV_acc], TMEM[SQ_acc] → RMEM → Apply query_decay + combine (CUDA cores) → SMEM[O] → GMEM[O]
```

**数据流说明**：
- **UTCMMA 输入**：直接从 SMEM 读取 operand A/B，或 operand A 可以在 TMEM 中
- **UTCMMA 输出**：直接写入 TMEM accumulator
- **CUDA cores 计算**：需要从 TMEM → RMEM 进行 element-wise 操作

---

## 4. KDA 算法核心差异分析

### 4.1 KDA 新增的计算

KDA 相比 decay linear attention 新增：

1. **Gate `g`** (per-token per-dimension):
   - Shape: `[B, T, H, K]`
   - 用途: Element-wise gating for state decay
   - 计算: `S = exp(g[-1]) * S + ...`

2. **Beta `β`** (per-token scalar):
   - Shape: `[B, T, H, 1]`
   - 用途: Scaling factor for new KV contribution
   - 计算: `... + (K * beta) @ V`

3. **Chunkwise WY-representation**:
   - 公式: `W = M @ (K * beta * exp(g))`, `U = M @ (V * beta)`
   - 用途: 用于高效计算 intra-chunk attention 和 state update
   - `M`: Intra-chunk attention matrix with gate-based causal mask

### 4.2 KDA 递归形式伪代码（回顾）

```python
S = zeros([B, H, K, V])  # State: [B, H, K, V]

for i in range(T):
    q_i = q[:, i]      # [B, H, K]
    k_i = k[:, i]      # [B, H, K]
    v_i = v[:, i]      # [B, H, V]
    g_i = g[:, i]      # [B, H, K] - NEW
    beta_i = beta[:, i]  # [B, H, 1] - NEW
    
    # State update with gating - MODIFIED
    # S = exp(g_i) * S + beta_i * k_i^T @ v_i
    S = exp(g_i)[..., None] * S + beta_i[..., None] * (k_i[..., None] * v_i[:, None])
    
    # Output computation with gating
    o[:, i] = einsum('bhk, bhkv -> bhv', q_i * exp(g_i), S)
```

### 4.3 KDA Chunkwise 形式关键步骤

对于块 `c`：

1. **Prologue - Prepare g_cumsum and beta**:
   ```python
   g_cumsum_c = cumsum(g[:, c*C:(c+1)*C], dim=1)  # Cumulative gate [B, C, H, K]
   beta_c = beta[:, c*C:(c+1)*C]                   # [B, C, H, 1]
   ```

2. **计算 Intra-chunk attention matrix M (Akk)**:
   ```python
   # M[i, j] = exp(g_cumsum[i] - g_cumsum[j]) * (Q[i] @ K[j]^T) for i >= j
   # 这是一个下三角矩阵，带 gate-based causal mask
   
   # 对于每个 subchunk：
   for i in range(0, C, BC):  # BC = subchunk size (e.g., 16)
       for j in range(0, i+BC, BC):
           # Compute Q[i:i+BC] @ K[j:j+BC]^T
           QK = Q[i:i+BC] @ K[j:j+BC].T
           # Apply gate decay: exp(g_cumsum[i] - g_cumsum[j])
           M[i:i+BC, j:j+BC] = QK * exp(g_cumsum[i:i+BC, None, :] - g_cumsum[None, j:j+BC, :])
   
   # Solve for M^{-1} using forward substitution (因为 M 是下三角)
   M_inv = solve_triangular_lower(M)
   ```

3. **计算 W 和 U matrices (WY representation)**:
   ```python
   # W = M_inv @ (K * beta * exp(g_cumsum))
   W = M_inv @ (K * beta[:, None] * exp(g_cumsum))  # [B, C, H, K]
   
   # U = M_inv @ (V * beta)
   U = M_inv @ (V * beta[:, None])                   # [B, C, H, V]
   ```

4. **State update with W and U**:
   ```python
   # 新的 state 贡献
   delta_S = W.transpose(-2, -1) @ U  # [B, H, K, V]
   
   # State update with gate decay
   g_last = g_cumsum[:, -1]  # 最后一个 token 的 g_cumsum
   S_new = exp(g_last)[..., None] * S + delta_S
   ```

5. **Output computation**:
   ```python
   # Intra-chunk: O_intra = M @ U
   O_intra = M @ U
   
   # Inter-chunk: O_inter = (Q * exp(g_cumsum)) @ S
   O_inter = (Q * exp(g_cumsum)) @ S
   
   # Combined output
   O = O_intra + O_inter
   ```

---

## 5. KDA on Blackwell: 核心挑战

### 5.1 存储容量挑战

**新增数据**:
1. `g_cumsum`: `[B, T, H, K]` - 与 K 同样大小
2. `beta`: `[B, T, H, 1]` - 较小
3. `M` (Akk) matrix: `[B, H, C, C]` - Intra-chunk attention
4. `W` matrix: `[B, H, C, K]` - 用于 state update
5. `U` matrix: `[B, H, C, V]` - 用于 output

**TMEM 容量**:
- 当前使用约 288-512 columns (占用 512 columns 约 56-100%)
- TMEM 总容量为 256KB (128 lanes × 512 columns)，相对充足
- KDA 需要额外空间存储 M, W, U matrices
- 通过 aggressive reuse 和 staging，TMEM 总容量基本足够

### 5.2 计算复杂度挑战

**新增操作**:
1. Element-wise gating: `exp(g) ⊙ Q/K/S` (K 次 exp 和乘法)
2. Intra-chunk M computation: gate-based causal mask `exp(g_cumsum[i] - g_cumsum[j])` 
3. Matrix inversion: `M^{-1}` using triangular solve (下三角矩阵，前向替代)
4. W/U computation: `M^{-1} @ (K * beta * exp(g))` 和 `M^{-1} @ (V * beta)`
5. State update: `W^T @ U` for delta state

### 5.3 数据依赖挑战

- M matrix 需要先计算 off-diagonal blocks，再进行 triangular solve
- W/U 计算依赖 M^{-1}，需要 solve 完成后才能开始
- State update 需要 W 和 U 都准备好

---

## 6. KDA Blackwell 实现方案设计

### 6.1 总体策略

**优先级**:
1. **正确性优先**: 先实现功能正确的版本
2. **简化版本**: 先实现递归形式，再考虑 chunkwise
3. **逐步优化**: 分阶段优化 TMEM 和计算效率

**设计原则**:
- **Reuse SMEM**: gate 和 beta 存储在 SMEM，避免占用 TMEM
- **Minimize TMEM**: 只在 TMEM 存储必须的 accumulator
- **Stage carefully**: 精心设计 pipeline stage，避免死锁

### 6.2 方案 A: 简化 Chunkwise KDA (推荐初版)

#### 6.2.1 算法简化

**简化假设**:
1. **忽略块内递归展开**: 先使用简化的 intra-chunk attention
2. **简化 delta correction**: 每步只做一次 `v - K@S`，不做完整的 unrolling
3. **Element-wise gating**: 使用 `g` 作为 element-wise gate，而非复杂的 cumulative 形式

**简化后的伪代码**:
## 6.2 简化版 chunkwise KDA 设计

### 6.2.1 核心优化：Prologue Elementwise Fusion

**关键设计决策**：
1. **输入变更**: 输入 `g_cumsum` 而非 `g`（预计算在 host/前序 kernel）
2. **Prologue Fusion**: 在加载阶段完成所有 g_cumsum 的 elementwise 乘法
3. **SMEM 复用**: 复用 g_cumsum buffer 存储 K^T'，节省 16KB

**Prologue Elementwise Fusion 伪代码**：
```python
# Phase 0: Load inputs to SMEM
TMA_load_async(sGCumsum, g_cumsum)  # [C, K] 从 GMEM 加载
TMA_load_async(sQ, Q)                # [C, K]
TMA_load_async(sK, K)                # [C, K]  
TMA_load_async(sV, V)                # [V, C]
TMA_load_async(sBeta, beta)          # [C, 1]
barrier_arrive_and_wait()

# Phase 0.5: Elementwise Fusion in Registers
for warp_id in cuda_core_warps:
    # Load SMEM → Registers
    rGCumsum = load_smem(sGCumsum, warp_tile)  # [warp_C, warp_K]
    rQ = load_smem(sQ, warp_tile)
    rK = load_smem(sK, warp_tile)
    
    # Elementwise multiply (在寄存器中)
    rQ_gated = rQ * rGCumsum       # Q' = Q ⊙ g_cumsum
    rK_gated = rK * rGCumsum       # K' = K ⊙ g_cumsum
    
    # Transpose K' for M computation
    rKT_gated = transpose(rK_gated) # K^T' = (K ⊙ g_cumsum)^T
    
    # Store back to SMEM (原地复用)
    store_smem(sQ, rQ_gated)         # sQ ← Q'
    store_smem(sK, rK_gated)         # sK ← K'
    store_smem(sGCumsum, rKT_gated)  # 复用 buffer: sGCumsum ← K^T'
    
    # Extract last row of g_cumsum and store to RMEM
    # (用于 state 更新时的 gating: S_new = S * g_cumsum[-1])
    if is_last_row(warp_tile):
        rGCumsumLast = rGCumsum[-1, :]  # [warp_K]
        # 保存在 RMEM，每个 CUDA warp 持有一部分

barrier_arrive_and_wait()

# 后续所有阶段使用处理后的数据:
# - sQ 现在是 Q' (已 gated)
# - sK 现在是 K' (已 gated)
# - sGCumsum 现在是 K^T' (已 gated & transposed)
# - rGCumsumLast 在 RMEM 中保存最后一行
```

### 6.2.2 算法流程（基于 M, W, U）

```python
for chunk_c:
    # Phase 0: Prologue - Load inputs
    # 输入: g_cumsum (预计算), Q, K, V, beta
    TMA_load(g_cumsum, Q, K, V, beta → SMEM)
    
    # Phase 1: 计算 M (intra-chunk attention with gate-based mask)
    # M = Akk, 带 exp(g_cumsum[i] - g_cumsum[j]) 的 gate decay
    # 这是一个下三角矩阵
    
    # Phase 1a: Compute diagonal blocks of M
    for subchunk_i in range(0, C, BC):  # BC = 16
        # M_diag[i,i] = (Q[i] * exp(g_cumsum[i])) @ (K[i]^T * exp(g_cumsum[i]))
        M_diag[i, i] = UTCMMA(Q[i] * exp(g_cumsum[i]), 
                              K[i]^T * exp(g_cumsum[i]))  → TMEM
    
    # Phase 1b: Compute off-diagonal blocks of M
    for subchunk_i in range(BC, C, BC):
        for subchunk_j in range(0, subchunk_i, BC):
            # M[i,j] = (Q[i] * exp(g_cumsum[i])) @ (K[j]^T * exp(g_cumsum[j]))
            # 注意: exp(g_cumsum[i] - g_cumsum[j]) 已经被分解到 Q 和 K 上
            M_offdiag[i, j] = UTCMMA(Q[i] * exp(g_cumsum[i]), 
                                     K[j]^T * exp(g_cumsum[j]))  → TMEM
    
    # Phase 2: Solve for M^{-1} (triangular solve)
    # 因为 M 是下三角，使用 forward substitution
    M_inv = triangular_solve_forward(M)  # In-place in TMEM
    
    # Phase 3: 计算 W = M^{-1} @ (K * beta * exp(g_cumsum))
    # 先准备 K_beta_g = K * beta * exp(g_cumsum)
    for i in cuda_warps:
        K_beta_g[i] = K[i] * beta[i] * exp(g_cumsum[i])  → SMEM
    
    # W = M^{-1} @ K_beta_g
    W = UTCMMA(M_inv, K_beta_g)  # [C, K] → TMEM
    
    # Phase 4: 计算 U = M^{-1} @ (V * beta)
    # 先准备 V_beta = V * beta
    for i in cuda_warps:
        V_beta[i] = V[i] * beta[i]  → SMEM
    
    # U = M^{-1} @ V_beta
    U = UTCMMA(M_inv, V_beta)  # [C, V] → TMEM
    
    # Phase 5: Inter-chunk output (Q @ S)
    # o_inter = (Q * exp(g_cumsum)) @ S
    o_inter = UTCMMA(Q * exp(g_cumsum), S)  # [C, V] → TMEM
    
    # Phase 6: Intra-chunk output (直接使用 U)
    # o_intra = M @ U = U (因为我们已经有 U = M^{-1}^{-1} @ (V * beta) = M @ ...)
    # 实际上 U 就是 intra-chunk 的输出
    o_intra = U
    
    # Phase 7: Combine outputs
    O_c = o_intra + o_inter
    
    # Phase 8: State update
    # delta_S = W^T @ U
    delta_S = UTCMMA(W^T, U)  # [K, V] → TMEM
    
    # S_new = exp(g_cumsum[-1]) * S + delta_S
    for i in cuda_warps:
        S_new[i] = exp(g_cumsum[-1]) * S[i] + delta_S[i]
    
    # Phase 9: Store output
    TMA_store(O_c → GMEM)
    
    # Phase 1-3: Intra-chunk attention
    # M = Q' @ K^T' (注意已经是 gated 版本)
    M = UTCMMA(sQ, sGCumsum)  # [C, C] in TMEM tAccQK
    
    # Apply causal mask (简化版，不需要复杂的 recursive unroll)
    # P[i,j] = M[i,j] if i >= j else 0
    # P[i,j] *= beta[i]  (broadcast)
    P = apply_causal_mask_and_beta(M, sBeta)
    
    # Phase 4-7: Delta correction
    # ks = K' @ S (注意 K' 已经是 gated)
    ks = UTCMMA(sK, tKV)  # [C, V] → sKS (SMEM)
    
    # v_corrected = V - K'@S
    v_corrected = elementwise_sub(sV, sKS)  # → sVCorr (SMEM)
    
    # Phase 8-9: Intra-chunk output
    o_intra = UTCMMA(P, sVCorr)  # P @ v_corrected → tAccPV
    
    # Phase 10-11: Inter-chunk output
    # o_inter = Q' @ S (注意 Q' 已经是 gated)
    o_inter = UTCMMA(sQ, tKV)  # [C, V] → tSQ
    
    # Combine outputs
    O_c = o_intra + o_inter
    
    # Phase 12: State update with gating
    # S_new = S * g_cumsum[-1] + beta_avg * K'^T @ v_corrected
    # (注意: K'^T 存储在 sGCumsum buffer，已经是 gated)
    S = elementwise_mul(S, rGCumsumLast)  # S * g_cumsum[-1]
    kv_new = UTCMMA(sGCumsum, sVCorr)      # K'^T @ v_corrected
    S += beta_scale(kv_new, sBeta)
```

### 6.2.3 存储决策

### 6.2.3 存储决策

#### 6.2.3.1 GMEM 输入
- `Q, K, V`: 与现有相同 `[B, T, H, D]`
- `g_cumsum`: `[B, T, H, K]` - **新增**，预计算的累积和（与 K 相同 layout）
- `beta`: `[B, T, H, 1]` - **新增**，可以 broadcast

#### 6.2.3.2 SMEM 分配（重新设计 - M 存储在 SMEM）

```python
# Blackwell SMEM capacity: 227KB per SM

# Working buffers
sQ: [C, K, 2 stages]       # 32KB
sK: [C, K, 2 stages]       # 32KB
sV: [V, C, 2 stages]       # 32KB
sP: [C, C, 2 stages]       # 16KB
sO: [V, C, 2 stages]       # 32KB
# Subtotal: 144KB

# KDA-specific buffers
sM (Akk): [C, C, 2 stages]      # 16KB (NEW - intra-chunk attention matrix, double-buffered)
sGCumsum: [C, K, 1 stage]       # 16KB (g_cumsum input, 后续可复用)
sKT_exp_neg_g: [K, C, 2 stages] # 32KB (K^T ⊙ exp(-g), 用于 inter-chunk 计算, double-buffered)
sBeta: [C, 1, 2 stages]         # 0.25KB
# Subtotal: 64.25KB

# Total: 144KB + 64.25KB = 208.25KB < 227KB ✓
# 剩余: 227KB - 208.25KB = 18.75KB
```

**关键设计决策**:
- ✅ **M 存储在 SMEM**: 16KB (2 stages)，用于 intra-chunk attention 和 triangular solve
- ✅ **K^T ⊙ exp(-g) 专用 buffer**: 32KB (2 stages)，用于 inter-chunk 计算
- ✅ **W/U 存储在 TMEM**: W [C, K] 和 U [C, V] 各 16KB，直接存储在 TMEM，节省 SMEM
- ✅ **去除 sKS/sVCorr**: 不再需要 delta correction buffers
- ✅ **总容量**: 208.25KB，使用 92% SMEM，仍有余量### 6.2.2.1 SMEM 容量优化方案

**问题**: 初步分配超出 SMEM 容量 13KB

**解决方案: 单 staging G buffer (推荐)**
- **优化**: sG: [C, K, 1 stage] 而非 2 stages
- **节省**: 16KB → 总量降至 224KB < 227KB ✓
- **理由**: Gate 在 prologue 阶段一次性加载，计算完 g_cumsum 后该 buffer 可被复用
- **实现**: 
  1. Prologue: 加载 g → sG
  2. 计算 g_cumsum → sGCumsum (保留)
  3. 后续 chunk 不再需要原始 g 值

```python
# Optimized SMEM allocation
sG: [C, K, 1 stage]       # 16KB (single stage, prologue only)
sGCumsum: [C, K, 1 stage] # 16KB (computed once, reused across phases)
sBeta: [C, 1, 2 stages]   # 0.25KB
sKS: [C, V, 1 stage]      # 16KB
sVCorr: [C, V, 1 stage]   # 16KB
# New subtotal: 64.25KB
# Grand total: 160KB + 64.25KB = 224.25KB < 227KB ✓
```
```

**关键决策**:
- ✅ **输入 g_cumsum 而非 g**: 预计算，避免 kernel 内 cumsum 开销
- ✅ **Prologue fusion**: 一次性完成所有 elementwise 乘法，消除 sG buffer
- ✅ **SMEM 复用**: sGCumsum buffer 在 prologue 后存储 K^T'
- ✅ **RMEM 保存 g_cumsum[-1]**: 256 bytes/warp，用于 state gating
- ✅ **Beta 存储在 SMEM**: 数据量小，broadcast 容易  
- ✅ **K'@S 结果存储在 SMEM**: 临时结果，用于 delta correction

#### 6.2.3.3 TMEM 分配

```python
# Reused accumulators (timeline-based)
tAccQK: [C, C, fp32]      # 256 cols, Phase 1-3 (M = Q'@K^T')
  → reused as tAccKS      # 256 cols, Phase 4-5 (K'@S)

tAccPV: [V, C, fp32]      # 256 cols, Phase 8-9 (O_intra = P@V_corr)
  → reused as tSQ         # 256 cols, Phase 10-11 (O_inter = Q'@S)

# Persistent state
tKV: [K, V, bf16]         # 256 cols, updated in Phase 12

# Total concurrent usage:
# Phase 1-5: 256 (QK/KS) + 256 (KV) = 512 cols
# Phase 8-11: 256 (PV/SQ) + 256 (KV) = 512 cols
# = 512 cols (exact fit ✓)
```

**TMEM 优化关键**:
1. **Aggressive reuse**: QK→KS, PV→SQ 两对累加器复用
2. **Single KV state**: 消除 tKV16，通过 pipeline 重叠避免双份
3. **g_cumsum[-1] 移至 RMEM**: 避免占用 TMEM 宝贵空间

**容量验证**:
- Total columns: 512 / 512 (100% utilization)
- Total size: 512 × 128 lanes × 4 bytes (avg) = 256KB / 256KB ✓

#### 6.2.3 数据流设计

##### Prologue (Load Warp)

```python
# Standard TMA loads (existing)
TMA_LOAD Q_c: GMEM → SMEM[sQ]
TMA_LOAD K_c: GMEM → SMEM[sK]
TMA_LOAD V_c: GMEM → SMEM[sV]

# New TMA loads
TMA_LOAD G_c: GMEM → SMEM[sG]     # NEW - gate values
TMA_LOAD Beta_c: GMEM → SMEM[sBeta]  # NEW - beta values

# Pipeline: Double buffering for all inputs
# Stage 0: Load chunk i
# Stage 1: Load chunk i+1 while processing chunk i
```

##### Phase 1: Intra-chunk QK (Compute Warp)

```python
# Same as existing
UTCMMA: SMEM[Q] × SMEM[K]^T → TMEM[QK_acc]

# Pipeline: Wait for Q, K loads
```

##### Phase 2: Apply Gate-based Mask (CUDA Warps)

```python
# Modified from existing decay mask
TMEM[QK_acc] → RMEM[qk]
SMEM[G] → RMEM[g]       # NEW - load gate values
SMEM[Beta] → RMEM[beta] # NEW - load beta values

# For each (i, j):
#   g_cumsum_i = cumulative_sum(g[0:i+1])  # Precomputed or on-the-fly
#   g_cumsum_j = cumulative_sum(g[0:j+1])
#   decay = exp(g_cumsum_i - g_cumsum_j)  # Element-wise decay
#   if i >= j:
#       P[i,j] = qk[i,j] * decay * beta[i]
#   else:
#       P[i,j] = 0

RMEM[p] → SMEM[P]

# Pipeline: Produce P for next phase
```

**优化考虑**:
- `g_cumsum` 可以在 prologue 预计算并存储在 SMEM
- 或者使用 parallel scan 在 CUDA cores 上计算

##### Phase 3: Delta Correction - K@S (Compute Warp)

```python
# NEW PHASE - Delta correction

# GEMM: K_c @ State → KS
UTCMMA: SMEM[K] × TMEM[State_BF16] → TMEM[KS_acc]

# Convert and store KS to SMEM
TMEM[KS_acc] → RMEM → SMEM[sKS]

# Element-wise subtraction (CUDA Warps)
SMEM[V] - SMEM[sKS] → SMEM[sVCorr]

# Pipeline:
# - Wait for State from previous chunk (if idx > 0)
# - Produce sVCorr for PV phase
```

**关键点**:
- K@S 是一个新的 GEMM，需要 TMEM accumulator
- 结果转移到 SMEM 进行 element-wise 减法
- 可能成为性能瓶颈（新增 GEMM）

##### Phase 4: PV with Corrected Values (Compute Warp)

```python
# Modified: Use sVCorr instead of sV
UTCMMA: SMEM[P] × SMEM[sVCorr] → TMEM[PV_acc]

# Pipeline: Wait for P and sVCorr
```

##### Phase 5: KV State Update (Compute Warp)

```python
# Modified: Use sVCorr and apply beta weighting

# Option A: Apply beta in SMEM before GEMM
# SMEM[sVCorr] * SMEM[sBeta] → SMEM[sVCorr_weighted]
# UTCMMA: SMEM[K]^T × SMEM[sVCorr_weighted] → TMEM[KV_acc]

# Option B: Apply beta after GEMM (in RMEM)
# UTCMMA: SMEM[K]^T × SMEM[sVCorr] → TMEM[KV_acc]
# TMEM[KV_acc] → RMEM → Apply beta → TMEM

# Apply element-wise gating (CUDA Warps)
# TMEM[State_prev] → RMEM[state]
# SMEM[G] → RMEM[g]
# g_final = exp(cumsum(g))  # Final gate value
# state_new = state * g_final  # Element-wise
# state_new += kv_acc_value
# RMEM → TMEM[State_BF16]

# Pipeline: Produce new state for next chunk
```

**关键点**:
- Element-wise gating 需要在 RMEM 中进行
- 可能需要从 TMEM 读取 state，应用 gate，再写回
- Beta weighting 可以在 SMEM 预处理或 RMEM 后处理

##### Phase 6: SQ Inter-chunk (Compute Warp)

```python
# Modified: Apply g-based query decay

# GEMM: State × Q
UTCMMA: TMEM[State_BF16] × SMEM[Q] → TMEM[SQ_acc]

# Apply query decay (CUDA Warps)
# TMEM[SQ_acc] → RMEM[sq]
# SMEM[G] → RMEM[g]
# query_decay = exp(cumsum(g) + chunk_offset)  # Query position decay
# sq *= query_decay  # Element-wise
# RMEM → TMEM[SQ_acc]

# Pipeline: Produce SQ for output phase
```

##### Epilogue: Output Combination

```python
# Same as existing
TMEM[PV_acc] + TMEM[SQ_acc] → RMEM → SMEM[O] → GMEM[O]

# TMA store to GMEM
```

### 6.3 详细 CuTe DSL 实现流程

本节提供接近实际代码的详细 CuTe DSL 实现流程，展开每个阶段的具体操作。

#### 6.3.1 Warp 分工和线程组织

```python
# Blackwell CTA 配置
NUM_WARPS_TOTAL = 7
NUM_THREADS_PER_WARP = 32
NUM_THREADS_PER_CTA = 224

# Warp specialization
LOAD_WARP_ID = 5          # TMA load warp
MMA_WARP_ID = 4           # UTCMMA compute warp
CUDA_WARP_IDS = [0,1,2,3] # CUDA core processing warps (mask, correction, etc.)
EPILOGUE_WARP_ID = 6      # TMA store warp

# Thread groups for pipeline
load_group = threads[160:192]      # 1 warp for TMA
mma_group = threads[128:160]       # 1 warp for UTCMMA
cuda_group = threads[0:128]        # 4 warps for CUDA cores
epilogue_group = threads[192:224]  # 1 warp for epilogue
```

#### 6.3.2 Kernel 主体结构

```python
@cute.jit
def kda_chunkwise_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    state_ptr,  # Persistent state across chunks
    problem_size: Tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
):
    """KDA Chunkwise Kernel with CuTe DSL"""
    
    B, S, H, D = problem_size
    C = 64  # Chunk size
    num_chunks = S // C
    
    # Get block and thread indices
    tidx = cute.arch.thread_idx()
    _, hidx, bidx = cute.arch.block_idx()
    warp_idx = tidx // 32
    lane_idx = tidx % 32
    
    # ============================================
    # SECTION 1: Storage Allocation
    # ============================================
    
    # Allocate shared memory storage
    storage = allocate_smem_storage()
    
    # Allocate TMEM
    tmem = allocate_tmem_storage()
    
    # Setup TMA descriptors
    tma_q, tma_k, tma_v, tma_g, tma_beta, tma_o = setup_tma_descriptors(
        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
        B, S, H, D, hidx, bidx
    )
    
    # ============================================
    # SECTION 2: Tiled MMA Setup
    # ============================================
    
    # QK MMA: (C, C, D) -> Q @ K^T
    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        dtype_q=BF16, dtype_k=BF16, dtype_acc=FP32,
        tile_shape=(C, C)
    )
    
    # KS MMA: (C, V, D) -> K @ State
    ks_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        dtype_k=BF16, dtype_state=BF16, dtype_acc=FP32,
        tile_shape=(C, V)
    )
    
    # PV MMA: (D, C, C) -> P @ V_corr
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        dtype_p=BF16, dtype_v=BF16, dtype_acc=FP32,
        tile_shape=(D, C)
    )
    
    # KV MMA: (D, D, C) -> K^T @ V_corr
    kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        dtype_k=BF16, dtype_v=BF16, dtype_acc=FP32,
        tile_shape=(D, D)
    )
    
    # SQ MMA: (D, C, D) -> State @ Q
    sq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        dtype_state=BF16, dtype_q=BF16, dtype_acc=FP32,
        tile_shape=(D, C)
    )
    
    # ============================================
    # SECTION 3: Pipeline Setup
    # ============================================
    
    # Create pipeline stages
    q_pipeline = create_pipeline(num_stages=2, producer=load_group, consumer=mma_group)
    k_pipeline = create_pipeline(num_stages=2, producer=load_group, consumer=mma_group)
    v_pipeline = create_pipeline(num_stages=2, producer=load_group, consumer=mma_group)
    g_pipeline = create_pipeline(num_stages=2, producer=load_group, consumer=cuda_group)
    beta_pipeline = create_pipeline(num_stages=2, producer=load_group, consumer=cuda_group)
    
    qk_pipeline = create_pipeline(num_stages=2, producer=mma_group, consumer=cuda_group)
    p_pipeline = create_pipeline(num_stages=2, producer=cuda_group, consumer=mma_group)
    ks_pipeline = create_pipeline(num_stages=1, producer=mma_group, consumer=cuda_group)
    vcorr_pipeline = create_pipeline(num_stages=1, producer=cuda_group, consumer=mma_group)
    kv_pipeline = create_pipeline(num_stages=1, producer=mma_group, consumer=cuda_group)
    state_pipeline = create_pipeline(num_stages=1, producer=cuda_group, consumer=mma_group)
    
    pv_pipeline = create_pipeline(num_stages=2, producer=mma_group, consumer=cuda_group)
    sq_pipeline = create_pipeline(num_stages=1, producer=mma_group, consumer=cuda_group)
    o_pipeline = create_pipeline(num_stages=2, producer=cuda_group, consumer=epilogue_group)
    
    # ============================================
    # SECTION 4: Chunk Loop
    # ============================================
    
    for chunk_idx in cutlass.range(0, num_chunks, unroll=0):
        
        # ========================================
        # PHASE 0: Load Input Data (LOAD WARP)
        # ========================================
        if warp_idx == LOAD_WARP_ID:
            # Load g_cumsum (预计算), Q, K, V, beta
            load_phase_with_gcumsum(chunk_idx, tma_q, tma_k, tma_v, 
                                   tma_gcumsum, tma_beta,
                                   storage, q_pipeline, k_pipeline, v_pipeline, 
                                   gcumsum_pipeline, beta_pipeline)
        
        # Barrier: Wait for all loads
        cute.arch.syncthreads()
        
        # ========================================
        # PHASE 0.5: Prologue Elementwise Fusion (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS:
            # Perform g_cumsum ⊙ Q, g_cumsum ⊙ K, transpose to K^T
            # Store results back to sQ, sK, sGCumsum (复用 as K^T')
            # Extract and save g_cumsum[-1] to RMEM
            
            prologue_elementwise_fusion(
                storage.sGCumsum,  # input: g_cumsum [C, K]
                storage.sQ,         # input/output: Q -> Q'
                storage.sK,         # input/output: K -> K'
                warp_idx, CUDA_WARP_IDS, tidx
            )
            # After this:
            # - sQ contains Q' = Q ⊙ g_cumsum
            # - sK contains K' = K ⊙ g_cumsum  
            # - sGCumsum contains K^T' = (K ⊙ g_cumsum)^T
            # - rGCumsumLast (RMEM) contains g_cumsum[-1, :]
        
        # Barrier: Ensure fusion is complete
        cute.arch.syncthreads()
        
        # ========================================
        # PHASE 1: QK GEMM (MMA WARP)
        # ========================================
        # Note: sQ and sGCumsum are now Q', K^T' (already gated)
        if warp_idx == MMA_WARP_ID:
            # Acquire QK accumulator buffer
            qk_handle = qk_pipeline.producer.acquire()
            
            # Execute QK GEMM: Q' @ K^T' -> TMEM[QK]
            # sGCumsum now holds K^T' (复用 buffer)
            execute_qk_gemm(qk_tiled_mma, storage.sQ, storage.sGCumsum,
                          tmem.tAccQK, qk_handle.index)
            
            # Commit QK result
            qk_handle.commit()
        
        # ========================================
        # PHASE 2: Apply Causal Mask + Beta Scaling (CUDA WARPS)
        # ========================================
        # Note: No need for gate-based decay mask, already applied in prologue
        elif warp_idx in CUDA_WARP_IDS:
            # Wait for QK accumulator
            qk_handle = qk_pipeline.consumer.wait()
            
            # Acquire P buffer
            p_handle = p_pipeline.producer.acquire()
            
            # Apply causal mask and beta scaling (simplified)
            # P[i,j] = M[i,j] if i >= j else 0
            # P[i,j] *= beta[i]
            apply_causal_mask_and_beta(tmem.tAccQK, storage.sBeta,
                                      storage.sP, qk_handle.index, p_handle.index,
                                      warp_idx, CUDA_WARP_IDS, tidx)
            
            # Release QK, commit P
            qk_handle.release()
            p_handle.commit()
        
        # ========================================
        # PHASE 3: Delta Correction - K'@S (MMA WARP)
        # ========================================
        # Note: sK is now K' (already gated)
        if warp_idx == MMA_WARP_ID and chunk_idx > 0:
            # Reuse tAccQK as tAccKS
            ks_handle = ks_pipeline.producer.acquire()
            
            # Execute K'@S GEMM: K' @ State -> TMEM[KS] (reused QK buffer)
            execute_ks_gemm(ks_tiled_mma, storage.sK, tmem.tKV,
                          tmem.tAccQK,  # Reused as tAccKS
                          ks_handle.index)
            
            ks_handle.commit()
        
        # ========================================
        # PHASE 4: V Correction (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS:
            # Wait for V
            v_handle = v_pipeline.consumer.wait()
            
            if chunk_idx > 0:
                # Wait for K@S result
                ks_handle = ks_pipeline.consumer.wait()
                
                # Acquire V_corr buffer
                vcorr_handle = vcorr_pipeline.producer.acquire()
                
                # Compute V_corr = V - K@S
                compute_v_correction(tmem.tAccKS, storage.sV, storage.sVCorr,
                                   ks_handle.index, vcorr_handle.index,
                                   warp_idx, CUDA_WARP_IDS, tidx)
                
                ks_handle.release()
                vcorr_handle.commit()
            else:
                # First chunk: V_corr = V (no correction)
                vcorr_handle = vcorr_pipeline.producer.acquire()
                copy_v_to_vcorr(storage.sV, storage.sVCorr,
                              v_handle.index, vcorr_handle.index,
                              warp_idx, CUDA_WARP_IDS, tidx)
                vcorr_handle.commit()
            
            v_handle.release()
        
        # ========================================
        # PHASE 6: Apply Beta Weighting (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS:
            # Wait for V_corr and beta
            vcorr_handle = vcorr_pipeline.consumer.wait()
            beta_handle = beta_pipeline.consumer.wait()
            
            # Apply beta weighting: V_corr *= beta (broadcast)
            apply_beta_weighting(storage.sVCorr, storage.sBeta,
                               vcorr_handle.index, beta_handle.index,
                               warp_idx, CUDA_WARP_IDS, tidx)
            
            vcorr_handle.release()
            beta_handle.release()
        
        # ========================================
        # PHASE 7: PV GEMM - Intra-chunk Output (MMA WARP)
        # ========================================
        if warp_idx == MMA_WARP_ID:
            # Wait for P and V_corr
            p_handle = p_pipeline.consumer.wait()
            vcorr_handle = vcorr_pipeline.consumer.wait()
            
            # Acquire PV accumulator buffer
            pv_handle = pv_pipeline.producer.acquire()
            
            # Execute PV GEMM: P @ V_corr -> TMEM[PV]
            execute_pv_gemm(pv_tiled_mma, storage.sP, storage.sVCorr,
                          tmem.tAccPV, p_handle.index, vcorr_handle.index,
                          pv_handle.index)
            
            p_handle.release()
            vcorr_handle.release()
            pv_handle.commit()
        
        # ========================================
        # PHASE 8: KV GEMM - State Update (MMA WARP)
        # ========================================
        if warp_idx == MMA_WARP_ID:
            # Wait for K and V_corr
            k_handle = k_pipeline.consumer.wait()
            vcorr_handle = vcorr_pipeline.consumer.wait()
            
            # Acquire KV accumulator buffer
            kv_handle = kv_pipeline.producer.acquire()
            
            # Execute KV GEMM: K^T @ V_corr -> TMEM[KV]
            execute_kv_gemm(kv_tiled_mma, storage.sK, storage.sVCorr,
                          tmem.tAccKV, k_handle.index, vcorr_handle.index,
                          kv_handle.index)
            
            k_handle.release()
            vcorr_handle.release()
            kv_handle.commit()
        
        # ========================================
        # PHASE 9: State Gating and Update (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS:
            # Wait for KV result and gate cumsum
            kv_handle = kv_pipeline.consumer.wait()
            
            if chunk_idx > 0:
                state_prev_handle = state_pipeline.consumer.wait()
            
            # Acquire new state buffer
            state_new_handle = state_pipeline.producer.acquire()
            
            # Apply element-wise gating and update
            # S_new = g_final * S_prev + KV
            update_state_with_gating(
                tmem.tAccKV if chunk_idx == 0 else tmem.tState,
                tmem.tAccKV, storage.sGCumsum,
                tmem.tStateNew, kv_handle.index,
                state_prev_handle.index if chunk_idx > 0 else None,
                state_new_handle.index,
                warp_idx, CUDA_WARP_IDS, tidx, chunk_idx
            )
            
            kv_handle.release()
            if chunk_idx > 0:
                state_prev_handle.release()
            state_new_handle.commit()
        
        # ========================================
        # PHASE 10: SQ GEMM - Inter-chunk Output (MMA WARP)
        # ========================================
        if warp_idx == MMA_WARP_ID and chunk_idx > 0:
            # Wait for state and Q
            state_handle = state_pipeline.consumer.wait()
            q_handle = q_pipeline.consumer.wait()
            
            # Acquire SQ accumulator buffer
            sq_handle = sq_pipeline.producer.acquire()
            
            # Execute SQ GEMM: State @ Q -> TMEM[SQ]
            execute_sq_gemm(sq_tiled_mma, tmem.tState, storage.sQ,
                          tmem.tAccSQ, state_handle.index, q_handle.index,
                          sq_handle.index)
            
            state_handle.release()
            q_handle.release()
            sq_handle.commit()
        
        # ========================================
        # PHASE 11: Apply Query Decay (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS and chunk_idx > 0:
            # Wait for SQ result
            sq_handle = sq_pipeline.consumer.wait()
            
            # Apply query position decay: SQ *= exp(-g_cumsum * chunk_offset)
            apply_query_decay(tmem.tAccSQ, storage.sGCumsum,
                            sq_handle.index, chunk_idx,
                            warp_idx, CUDA_WARP_IDS, tidx)
            
            sq_handle.release()
        
        # ========================================
        # PHASE 12: Output Combination (CUDA WARPS)
        # ========================================
        elif warp_idx in CUDA_WARP_IDS:
            # Wait for PV (intra-chunk)
            pv_handle = pv_pipeline.consumer.wait()
            
            # Acquire output buffer
            o_handle = o_pipeline.producer.acquire()
            
            if chunk_idx > 0:
                # Wait for SQ (inter-chunk)
                sq_handle = sq_pipeline.consumer.wait()
                
                # Combine: O = PV + SQ
                combine_outputs(tmem.tAccPV, tmem.tAccSQ, storage.sO,
                              pv_handle.index, sq_handle.index, o_handle.index,
                              warp_idx, CUDA_WARP_IDS, tidx)
                
                sq_handle.release()
            else:
                # First chunk: O = PV only
                copy_pv_to_output(tmem.tAccPV, storage.sO,
                                pv_handle.index, o_handle.index,
                                warp_idx, CUDA_WARP_IDS, tidx)
            
            pv_handle.release()
            o_handle.commit()
        
        # ========================================
        # PHASE 13: Store Output (EPILOGUE WARP)
        # ========================================
        if warp_idx == EPILOGUE_WARP_ID:
            # Wait for output
            o_handle = o_pipeline.consumer.wait()
            
            # TMA store: SMEM[O] -> GMEM[O]
            store_output(tma_o, storage.sO, o_handle.index, chunk_idx)
            
            o_handle.release()
    
    # End of chunk loop
    
    # Final cleanup
    tmem.deallocate()
    cute.arch.syncthreads()
```

#### 6.3.3 关键函数详细实现

##### 6.3.3.1 Prologue Elementwise Fusion

```python
@cute.jit
def prologue_elementwise_fusion(
    sGCumsum: cute.Tensor,  # input: [C, K] - g_cumsum from GMEM
    sQ: cute.Tensor,         # input/output: [C, K, 2] - Q -> Q'
    sK: cute.Tensor,         # input/output: [C, K, 2] - K -> K'
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
):
    """
    Prologue fusion: Apply g_cumsum elementwise to Q, K
    Store results back to sQ, sK, and reuse sGCumsum for K^T'
    
    After execution:
    - sQ contains Q' = Q ⊙ g_cumsum
    - sK contains K' = K ⊙ g_cumsum
    - sGCumsum contains K^T' = (K ⊙ g_cumsum)^T
    - rGCumsumLast (RMEM) contains g_cumsum[-1, :] for state gating
    """
    C = 64
    K = 128
    
    # Warp assignment: 4 CUDA warps split work
    warps_count = len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    
    # Each warp handles C/4 rows
    c_per_warp = C // warps_count
    c_start = local_warp_idx * c_per_warp
    c_end = c_start + c_per_warp
    
    lane_idx = tidx % 32
    
    # Allocate RMEM for storing g_cumsum last row (thread-local)
    rGCumsumLast = cute.Tensor(shape=(K // 32,), dtype=BF16)  # Each thread handles K/32 elements
    
    # Process assigned rows
    for c in range(c_start, c_end):
        for k in range(lane_idx, K, 32):  # Stride by warp size
            # Load from SMEM to registers
            g_val = sGCumsum[c, k]
            q_val = sQ[c, k, 0]  # Stage 0
            k_val = sK[c, k, 0]
            
            # Elementwise multiply
            q_gated = q_val * g_val  # Q' = Q ⊙ g_cumsum
            k_gated = k_val * g_val  # K' = K ⊙ g_cumsum
            
            # Store back (Q', K' in-place)
            sQ[c, k, 0] = q_gated
            sK[c, k, 0] = k_gated
            
            # Prepare K^T': transpose indices for storage
            # sGCumsum[k, c] = K^T'[k, c] = K'[c, k]
            # Note: We store transposed version directly
            sGCumsum[k, c] = k_gated  # Reuse buffer for K^T'
            
            # Save g_cumsum[-1, k] to RMEM (last row)
            if c == C - 1:
                k_local = k // 32
                rGCumsumLast[k_local] = g_val
    
    # Ensure all writes complete
    cute.arch.warpgroup_fence()
    
    # Note: rGCumsumLast stays in RMEM for Phase 12 (state update)
```

##### 6.3.3.2 Gate Cumsum 计算 (Legacy - 如使用 g 输入则需要)

```python
@cute.jit
def compute_gate_cumsum(
    sG: cute.Tensor,        # [C, K, stage]
    sGCumsum: cute.Tensor,  # [C, K, stage]
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
):
):
    """
    Compute cumulative sum of gates across sequence dimension.
    Each warp processes K/4 dimensions.
    
    Parallel scan algorithm (Kogge-Stone) for efficiency.
    """
    C = 64
    K = 128
    
    # Warp assignment: 4 warps handle 128 dimensions
    k_per_warp = K // len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    k_start = local_warp_idx * k_per_warp
    k_end = k_start + k_per_warp
    
    # Each thread handles multiple K dimensions
    lane_idx = tidx % 32
    k_per_thread = k_per_warp // 32
    
    for k_offset in range(k_per_thread):
        k = k_start + lane_idx * k_per_thread + k_offset
        
        if k < K:
            # Initialize: cumsum[0, k] = g[0, k]
            sGCumsum[0, k, 0] = sG[0, k, 0]
            
            # Sequential cumsum (could be optimized with parallel scan)
            for c in range(1, C):
                sGCumsum[c, k, 0] = sGCumsum[c-1, k, 0] + sG[c, k, 0]
    
    # Ensure all threads finish
    cute.arch.warpgroup_fence()
```

##### 6.3.3.3 Causal Mask + Beta Scaling (Simplified)

```python
@cute.jit
def apply_causal_mask_and_beta(
    tAccQK: cute.Tensor,    # TMEM QK accumulator [C, C]
    sBeta: cute.Tensor,     # SMEM beta [C, 1]
    sP: cute.Tensor,        # SMEM P output [C, C]
    qk_stage: int,
    p_stage: int,
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
):
    """
    Apply causal mask and beta scaling (simplified KDA).
    
    P[i,j] = M[i,j] if i >= j else 0
    P[i,j] *= beta[i]
    
    Note: g_cumsum已经在 prologue 融合到 Q', K' 中，
          这里只需要 causal mask 和 beta scaling
    """
    C = 64
    
    # Partition work among CUDA warps
    total_elements = C * C
    elements_per_warp = total_elements // len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    elem_start = local_warp_idx * elements_per_warp
    elem_end = elem_start + elements_per_warp
    
    # Each thread handles multiple elements
    lane_idx = tidx % 32
    elems_per_thread = elements_per_warp // 32
    
    for elem_offset in range(elems_per_thread):
        elem_idx = elem_start + lane_idx * elems_per_thread + elem_offset
        
        if elem_idx < total_elements:
            i = elem_idx // C
            j = elem_idx % C
            
            # Causal mask
            if i >= j:
                # Load QK value from TMEM
                qk_val = tAccQK[i, j, qk_stage]
                
                # Apply beta scaling
                beta_val = sBeta[i, 0, 0]
                
                # Compute P (简化版，不需要额外的 gate decay)
                p_val = qk_val * beta_val
                sP[i, j, p_stage] = p_val.to(cute.BFloat16)
            else:
                # Masked position
                sP[i, j, p_stage] = cute.BFloat16(0.0)
    
    # Fence to ensure SMEM writes are visible
    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)
```

##### 6.3.3.4 Delta Correction - K'@S GEMM

```python
@cute.jit
def execute_ks_gemm(
    ks_tiled_mma: cute.TiledMma,
    sK: cute.Tensor,        # SMEM K' [C, K, stage] (already gated)
    tKV: cute.Tensor,       # TMEM State [K, V]
    tAccKS: cute.Tensor,    # TMEM KS accumulator [C, V] (reused QK buffer)
    ks_stage: int,
):
    """Execute K @ State GEMM for delta correction."""
    # Partition tensors for MMA
    thr_mma = ks_tiled_mma.get_slice(0)
    
    # Fragment K from SMEM
    tCrK = thr_mma.partition_A(sK[None, None, None, k_stage])
    
    # Fragment State from TMEM (already partitioned)
    tCrState = tState[None, None, None, state_stage]
    
    # Accumulator in TMEM
    tCtAcc = tAccKS[None, None, None, ks_stage]
    
    # Execute GEMM: K @ State
    # Clear accumulator first
    ks_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(False))
    
    for k_phase in cutlass.range(cute.size(tCrK, mode=[2]), unroll_full=True):
        if k_phase > 0:
            ks_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
        
        cute.gemm(
            ks_tiled_mma,
            tCtAcc,
            tCrK[None, None, k_phase, 0],
            tCrState[None, None, k_phase, 0],
            tCtAcc,
        )
    
    # Fence TMEM writes
    cute.arch.fence_view_async_tmem_store()

@cute.jit
def compute_v_correction(
    tAccKS: cute.Tensor,    # TMEM KS [C, V]
    sV: cute.Tensor,        # SMEM V [V, C, stage]
    sVCorr: cute.Tensor,    # SMEM V_corr [C, V]
    ks_stage: int,
    v_stage: int,
    vcorr_stage: int,
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
):
    """
    Compute V_corr = V - K@S (delta correction).
    
    V is [V, C] (column-major), need to compute element-wise subtraction.
    """
    C = 64
    V = 128
    
    # Partition work among warps
    total_elements = C * V
    elements_per_warp = total_elements // len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    elem_start = local_warp_idx * elements_per_warp
    elem_end = elem_start + elements_per_warp
    
    # Each thread handles multiple elements
    lane_idx = tidx % 32
    elems_per_thread = elements_per_warp // 32
    
    # Create copy atom for TMEM to RMEM
    copy_t2r = sm100_utils.get_tmem_load_op(...)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_t2r, tAccKS)
    thr_copy = tiled_copy_t2r.get_slice(tidx)
    
    # Partition TMEM tensor
    tTR_tKS = thr_copy.partition_S(tAccKS[None, None, None, ks_stage])
    
    # Register storage for KS
    tTR_rKS = cute.make_fragment_like(tTR_tKS)
    
    # Load K@S from TMEM to RMEM
    cute.copy(tiled_copy_t2r, tTR_tKS, tTR_rKS)
    cute.arch.fence_view_async_tmem_load()
    
    # Compute V_corr = V - KS
    for elem_offset in range(elems_per_thread):
        elem_idx = elem_start + lane_idx * elems_per_thread + elem_offset
        
        if elem_idx < total_elements:
            c = elem_idx // V
            v = elem_idx % V
            
            # Load V from SMEM (note: V is [V, C])
            v_val = sV[v, c, v_stage]
            
            # Load KS from register
            ks_val = tTR_rKS.flat_index(elem_idx)
            
            # Compute correction
            v_corr = v_val - ks_val.to(cute.BFloat16)
            
            # Store to SMEM (row-major for next GEMM)
            sVCorr[c, v, vcorr_stage] = v_corr
    
    # Fence SMEM writes
    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)
```

##### 6.3.3.4 State Update with Gating

```python
@cute.jit
def update_state_with_gating(
    tKVPrev: cute.Tensor,      # TMEM previous state [K, V]
    tAccKV: cute.Tensor,       # TMEM KV accumulator [K, V]
    rGCumsumLast: cute.Tensor, # RMEM g_cumsum[-1, :] (thread-local, [K//32])
    tKVNew: cute.Tensor,       # TMEM new state [K, V]
    kv_stage: int,
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
    chunk_idx: int,
):
    """
    Update state with element-wise gating:
    S_new = g_cumsum[-1] * S_prev + KV_new
    
    where g_cumsum[-1] is from RMEM (stored in prologue fusion).
    
    Note: rGCumsumLast is thread-local, each thread has K/32 elements
    """
    K = 128
    V = 128
    
    # Partition work among warps
    total_elements = K * V
    elements_per_warp = total_elements // len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    elem_start = local_warp_idx * elements_per_warp
    elem_end = elem_start + elements_per_warp
    
    # Each thread handles multiple elements
    lane_idx = tidx % 32
    elems_per_thread = elements_per_warp // 32
    
    # Setup TMEM copy operations
    copy_t2r_state = sm100_utils.get_tmem_load_op(...)
    copy_t2r_kv = sm100_utils.get_tmem_load_op(...)
    copy_r2t_state = sm100_utils.get_tmem_store_op(...)
    
    for elem_offset in range(elems_per_thread):
        elem_idx = elem_start + lane_idx * elems_per_thread + elem_offset
        
        if elem_idx < total_elements:
            k = elem_idx // V
            v = elem_idx % V
            
            # Get g_cumsum[-1, k] from RMEM (thread-local)
            k_local = k // 32
            k_thread = k % 32
            
            # Broadcast within warp if needed
            g_last = cute.shfl_sync(rGCumsumLast[k_local], k_thread, 32)
            
            # Load KV from TMEM
            kv_val = load_tmem_element(tAccKV, k, v, kv_stage)
            
            if chunk_idx > 0:
                # Load previous state from TMEM
                state_prev_val = load_tmem_element(tKVPrev, k, v)
                
                # Apply gating and update
                # S_new[k, v] = g_cumsum[-1, k] * S_prev[k, v] + KV_new[k, v]
                state_new_val = g_last * state_prev_val + kv_val
            else:
                # First chunk: no previous state
                state_new_val = kv_val
            
            # Store new state to TMEM (convert to BF16)
            store_tmem_element(tStateNew, k, v, state_new_stage, 
                             state_new_val.to(cute.BFloat16))
    
    # Fence TMEM writes
    cute.arch.fence_view_async_tmem_store()
```

##### 6.3.3.5 Output Combination

```python
@cute.jit
def combine_outputs(
    tAccPV: cute.Tensor,    # TMEM intra-chunk output [D, C]
    tAccSQ: cute.Tensor,    # TMEM inter-chunk output [D, C]
    sO: cute.Tensor,        # SMEM output [D, C, stage]
    pv_stage: int,
    sq_stage: int,
    o_stage: int,
    warp_idx: int,
    cuda_warp_ids: List[int],
    tidx: int,
):
    """
    Combine intra-chunk and inter-chunk outputs:
    O = PV + SQ
    """
    D = 128
    C = 64
    
    # Partition work among warps
    total_elements = D * C
    elements_per_warp = total_elements // len(cuda_warp_ids)
    local_warp_idx = cuda_warp_ids.index(warp_idx)
    elem_start = local_warp_idx * elements_per_warp
    elem_end = elem_start + elements_per_warp
    
    # Each thread handles multiple elements
    lane_idx = tidx % 32
    elems_per_thread = elements_per_warp // 32
    
    # Setup TMEM to RMEM copy
    copy_t2r = sm100_utils.get_tmem_load_op(...)
    
    for elem_offset in range(elems_per_thread):
        elem_idx = elem_start + lane_idx * elems_per_thread + elem_offset
        
        if elem_idx < total_elements:
            d = elem_idx // C
            c = elem_idx % C
            
            # Load PV from TMEM
            pv_val = load_tmem_element(tAccPV, d, c, pv_stage)
            
            # Load SQ from TMEM
            sq_val = load_tmem_element(tAccSQ, d, c, sq_stage)
            
            # Combine
            o_val = pv_val + sq_val
            
            # Store to SMEM (column-major for TMA)
            sO[d, c, o_stage] = o_val.to(cute.BFloat16)
    
    # Fence SMEM writes
    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)
```

#### 6.3.4 Pipeline 和 Barrier 优化

```python
class PipelineConfig:
    """Pipeline configuration for KDA kernel"""
    
    # Input data pipelines (double buffering)
    Q_STAGES = 2
    K_STAGES = 2
    V_STAGES = 2
    G_STAGES = 2
    BETA_STAGES = 2
    
    # Intermediate pipelines
    QK_STAGES = 2    # QK can overlap with mask computation
    P_STAGES = 2     # P can overlap with PV GEMM
    KS_STAGES = 1    # KS immediately consumed by correction
    VCORR_STAGES = 1 # V_corr immediately used in GEMMs
    
    # State pipeline
    STATE_STAGES = 1  # Single state buffer (updated in-place)
    
    # Output pipelines
    PV_STAGES = 2    # PV can overlap with combination
    SQ_STAGES = 1    # SQ immediately combined
    O_STAGES = 2     # O can overlap with TMA store
    
    @staticmethod
    def create_barriers():
        """Create named barriers for fine-grained synchronization"""
        return {
            'tmem_alloc': pipeline.NamedBarrier(barrier_id=1, num_threads=224),
            'tmem_dealloc': pipeline.NamedBarrier(barrier_id=2, num_threads=224),
            'g_cumsum_ready': pipeline.NamedBarrier(barrier_id=3, num_threads=128),
            'state_ready': pipeline.NamedBarrier(barrier_id=4, num_threads=160),
        }
```

#### 6.3.5 性能优化技巧

**1. Warp Specialization 优化**:
```python
# Load warp: 仅负责 TMA，寄存器压力小
cute.arch.warpgroup_reg_alloc(24)  # 少量寄存器

# MMA warp: UTCMMA 操作，中等寄存器
cute.arch.warpgroup_reg_dealloc(24)  # 默认配额

# CUDA warps: element-wise 操作，寄存器压力大
cute.arch.warpgroup_reg_alloc(160)  # 更多寄存器用于缓存
```

**2. TMEM 访问优化**:
```python
# 使用 warpgroup fence 减少同步开销
cute.arch.warpgroup_fence()  # 仅同步 warpgroup 内

# 批量 TMEM 操作
for k in range(K // tile_k):
    # Load tile
    cute.copy(tiled_t2r, tmem_tile, rmem_tile)
    # Process tile
    # Store tile
cute.arch.fence_view_async_tmem_load()  # 一次性 fence
```

**3. SMEM Bank Conflict 避免**:
```python
# 使用 swizzle mode 避免 bank conflict
swizzle_mode = cute.nvgpu.warpgroup.make_smem_layout_atom(
    cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
    dtype=BF16
)
```

### 6.4 方案 B: 完整 Chunkwise KDA (进阶版本)

完整版本需要实现递归展开 (recursive unrolling)，这会显著增加复杂度。

**关键差异**:
1. **A matrix 构建**: 需要 `O(C²)` 的注意力矩阵
2. **递归展开**: `for i in range(1, C): A[i, :i] += ...`
3. **w 和 u 计算**: 需要额外的 GEMM 和存储
4. **TMEM 容量**: columns 限制可能需要 aliasing 优化

**建议**: 先实现方案 A (6.3)，验证正确性后再考虑方案 B。

---

## 7. 实现细节和优化

### 7.1 Gate 累积和计算

**方案 1: Prologue 预计算 (推荐)**

```python
# In Load Warp or CUDA Warps during prologue
# Compute cumulative sum of gate: g_cumsum[i] = sum(g[0:i+1])

# SMEM layout: sG_cumsum: [C, K]
# For each position i, for each dimension k:
#   sG_cumsum[i, k] = sG_cumsum[i-1, k] + sG[i, k]

# Use parallel scan (e.g., Kogge-Stone) for efficiency
```

**优势**:
- 避免重复计算 `cumsum`
- Mask application 时直接查表

**代价**:
- 额外的 SMEM 空间 `[C, K]` ≈ 16KB
- 需要 scan 算法实现

**方案 2: On-the-fly 计算**

```python
# During mask application
# For each (i, j), compute cumsum on-the-fly
# More compute, less memory
```

### 7.2 Beta 加权处理

**方案 1: SMEM 预处理 (推荐)**

```python
# Before VP GEMM, apply beta weighting to V_corr
# sVCorr_weighted[i, :] = sVCorr[i, :] * sBeta[i]

# Advantage: Single broadcast operation in SMEM
# Use CUDA cores for element-wise multiplication
```

**方案 2: TMEM/RMEM 后处理**

```python
# After KV GEMM, scale each row of accumulator
# Less efficient due to TMEM → RMEM → TMEM overhead
```

### 7.3 Element-wise Gating 优化

**挑战**: `S = g ⊙ S` 是 K×V 次 element-wise 乘法

**方案 1: RMEM Vector Operations**

```python
# Load state from TMEM to RMEM in chunks
# Apply g element-wise in RMEM (vectorized)
# Store back to TMEM

# Pseudocode for CUDA Warps
for kv_tile in range(num_kv_tiles):
    # Load tile of state
    state_tile = TMEM[State + offset]  # [tile_k, tile_v]
    
    # Load corresponding g values
    g_tile = SMEM[G + k_offset]  # [tile_k]
    g_final_tile = exp(cumsum(g_tile))
    
    # Element-wise multiply (broadcast g across V dimension)
    for v in range(tile_v):
        for k in range(tile_k):
            state_tile[k, v] *= g_final_tile[k]
    
    # Store back
    TMEM[State + offset] = state_tile
```

**优化**:
- 使用 SIMD 指令进行向量化
- Tile 大小选择以最大化寄存器利用率
- 循环展开提高 ILP

### 7.4 Delta Correction 优化

**关键**: K@S 是一个新的 GEMM，需要高效实现

**方案 1: 完整 GEMM with UTCMMA**

```python
# Use UTCMMA for K @ State
# K: [C, K] in SMEM
# State: [K, V] in TMEM (as operand A)
# Output: [C, V] in TMEM

# Advantage: Maximum throughput
# Disadvantage: Requires TMEM space for accumulator
```

**方案 2: 分块 GEMM with CUDA cores**

```python
# If TMEM capacity is insufficient
# Use CUDA cores for K@S, store result in SMEM directly
# Slower but saves TMEM
```

### 7.5 Pipeline 优化

**Double/Triple Buffering**:

```python
# Stage 0: Load chunk i inputs (Q, K, V, G, Beta)
# Stage 1: Process chunk i (all phases)
# Stage 2: Load chunk i+1 inputs

# Overlap load and compute for maximum throughput
```

**Barrier Synchronization**:
- 使用 named barriers 进行细粒度同步
- 避免全局 `__syncthreads()` 导致的性能下降

---

## 8. TMEM 容量分析和优化

### 8.1 当前方案 A 的 TMEM 使用

| Component | Size (columns) | Data Type | Stages | Notes |
|-----------|---------------|-----------|--------|-------|
| QK_acc | 64-128 | FP32 | 2 | Intra-chunk attention |
| PV_acc | 64-128 | FP32 | 2 | Intra-chunk output |
| KV_acc | 64-128 | FP32 | 1 | State computation |
| KV16 | 32-64 | BF16 | 1 | State storage (compressed) |
| SQ_acc | 64-128 | FP32 | 1 | Inter-chunk output |
| **KS_acc** | **64-128** | **FP32** | **1** | **NEW: K@S for delta correction** |
| **Total** | **352-512** | - | - | **~69-100% usage** |

### 8.2 容量优化策略

**优化 1: 复用 TMEM 空间**

```python
# Observation: Some phases are non-overlapping
# - KS_acc only used during delta correction phase
# - QK_acc can be freed after P is computed
# - PV_acc can be freed after output is combined

# Strategy: Alias TMEM regions for non-concurrent use
TMEM_QK_KS = same region  # QK used first, then KS
TMEM_PV_SQ = separate (concurrent in later phases)
```

**Aliasing 示例**:
```python
# Phase 1: QK GEMM → TMEM[region_A]
# Phase 2: Mask → SMEM[P], Free region_A
# Phase 3: KS GEMM → TMEM[region_A] (reuse!)
# Phase 3: V correction → SMEM
# Phase 4: PV GEMM → TMEM[region_B]
# Phase 5: KV GEMM → TMEM[region_C]
# Phase 6: SQ GEMM → TMEM[region_D]
# Phase 7: Output combine from regions B, D
```

**优化后的 TMEM 使用**:
```python
TMEM_QK_KS (aliased):  128 cols  # Reuse same space
TMEM_PV:               128 cols
TMEM_KV:               128 cols
TMEM_KV16:              64 cols
TMEM_SQ:               128 cols
Total:                 448 cols  # ~87% of 512 cols, ~224KB of 256KB
```

**优化 2: 降低 Stages**

```python
# Reduce stages where pipeline pressure is low
# E.g., KV accumulator doesn't need multiple stages if not pipelined
```

**优化 3: 压缩数据类型**

```python
# Use BF16 for intermediate results where precision loss is acceptable
# E.g., KS_acc could be BF16 if V correction doesn't require FP32
```

### 8.3 Spilling 到 SMEM (一般不需要)

注意：TMEM 总容量为 256KB，相对充足。512 columns 的限制主要影响并发性而非绝对容量。只有在极端情况下才需要 spill 到 SMEM：

```python
# Example: Store KS result directly to SMEM (only if column limit exceeded)
# UTCMMA: K @ State → TMEM[KS_acc_temp]
# Immediately convert and copy: TMEM → RMEM → SMEM[sKS]
# Free TMEM[KS_acc_temp]

# Trade-off: Slower but saves TMEM
```

---

## 9. 正确性验证策略

### 9.1 单元测试

**测试 1: 单个组件**
```python
# Test delta correction in isolation
# Input: K, V, State
# Expected: v_corrected = V - K @ State
# Compare with PyTorch reference

# Test gate application
# Input: State, g
# Expected: State_new = g * State
# Compare with PyTorch reference
```

**测试 2: 完整 Chunk**
```python
# Test one chunk end-to-end
# Input: Q, K, V, G, Beta, State_prev
# Expected: O, State_new
# Compare with naive PyTorch KDA implementation
```

**测试 3: 多 Chunk**
```python
# Test state propagation across chunks
# Input: Full sequence
# Expected: Full output
# Compare with PyTorch reference
```

### 9.2 数值精度分析

```python
# Check different error sources:
# 1. FP32 vs BF16 conversion error
# 2. TMEM accumulator rounding error
# 3. Gate cumsum accumulation error
# 4. Delta correction cancellation error

# Use double precision reference for comparison
```

### 9.3 边界情况测试

```python
# Test edge cases:
# - First chunk (no previous state)
# - Last chunk (partial chunk)
# - g = 0 (no gating)
# - beta = 0 (no contribution)
# - g → inf (full decay)
```

---

## 10. 实现路线图

### 10.1 阶段 1: 基础实现 (2-3 周)

**目标**: 实现简化版 KDA，保证功能正确

**任务**:
1. ✅ 添加 G 和 Beta 的 TMA load
2. ✅ 实现 gate cumsum (选择方案 1 或 2)
3. ✅ 修改 mask application，加入 g-based decay
4. ✅ 实现 K@S GEMM (delta correction)
5. ✅ 实现 V - K@S element-wise 减法
6. ✅ 修改 PV GEMM 使用 v_corrected
7. ✅ 实现 element-wise gating 在 state update
8. ✅ 修改 SQ 的 query decay
9. ✅ 单元测试和集成测试

**里程碑**: 通过 single chunk 正确性测试

### 10.2 阶段 2: 多块和 Pipeline (2 周)

**目标**: 实现多 chunk 处理和 pipeline 优化

**任务**:
1. ✅ 实现 state 在 chunks 间的传递
2. ✅ 设置 double buffering for all inputs
3. ✅ Synchronization 和 barrier 优化
4. ✅ 验证 state 累积正确性

**里程碑**: 通过 full sequence 正确性测试

### 10.3 阶段 3: 性能优化 (2-3 周)

**目标**: 优化性能，达到实用水平

**任务**:
1. ✅ TMEM aliasing 和 capacity 优化
2. ✅ Gate cumsum 优化 (parallel scan)
3. ✅ Element-wise 操作向量化
4. ✅ Pipeline tuning (stage 数量调优)
5. ✅ Register allocation 优化
6. ✅ Benchmark 和 profiling

**里程碑**: 性能达到 decay attention 的 80-90%

### 10.4 阶段 4: 完整版本 (可选, 3-4 周)

**目标**: 实现完整的 recursive unrolling

**任务**:
1. ✅ 实现 A matrix 构建
2. ✅ 实现递归展开循环
3. ✅ 实现 w 和 u 矩阵计算
4. ✅ 复杂 delta correction 集成
5. ✅ 正确性和性能验证

**里程碑**: 完整 KDA 实现，精度匹配 PyTorch reference

---

## 11. 性能预估

### 11.1 理论分析

**额外计算量**:
- K@S GEMM: `C × K × V` FLOPs (新增 ~1 GEMM)
- Element-wise gating: `K × V` 乘法 (相对较小)
- Gate cumsum: `C × K` 累加 (可忽略)
- Element-wise ops: 多次 broadcast 和减法 (小)

**总计**: 相比 decay attention，计算量增加约 **20-30%**

### 11.2 TMEM 带宽分析

**额外 TMEM 访问**:
- K@S accumulator: 1 次写入 + 1 次读出
- State gating: 1 次读 + 1 次写 (in-place 可能)

**影响**: TMEM 带宽增加约 **15-20%**

### 11.3 预期性能

假设 decay attention 达到 **X TFLOPs**:

- **简化版 KDA**: **0.7-0.8X TFLOPs** (初版)
- **优化后 KDA**: **0.8-0.9X TFLOPs**
- **完整版 KDA**: **0.6-0.7X TFLOPs** (due to recursive unrolling)

**瓶颈预测**:
1. **K@S GEMM**: 新增的 GEMM 会成为主要开销
2. **Element-wise ops**: 如果不向量化，CUDA core 利用率低
3. **TMEM capacity**: 如果需要 spilling，性能下降显著

---

## 12. 风险和缓解

### 12.1 风险 1: TMEM 容量不足

**风险等级**: 高

**表现**: 编译错误或运行时 TMEM allocation 失败

**缓解**:
- 使用 TMEM aliasing 减少使用
- 降低 accumulator stages
- Spill 到 SMEM (性能换空间)

### 12.2 风险 2: 数值精度问题

**风险等级**: 中

**表现**: 与 PyTorch reference 误差超过阈值

**缓解**:
- 保持关键路径使用 FP32
- 仔细选择 BF16 转换点
- 使用 Kahan summation 减少累积误差

### 12.3 风险 3: 性能不达预期

**风险等级**: 中

**表现**: KDA 比 decay attention 慢超过 50%

**缓解**:
- Profile 找到瓶颈
- 优化 element-wise 操作
- 考虑简化算法（trade-off 精度）

### 12.4 风险 4: 复杂度导致 bug

**风险等级**: 中

**表现**: 难以调试的正确性问题

**缓解**:
- 分阶段实现和测试
- 大量单元测试
- 详细的 logging 和 debug print

---

## 13. 总结和建议

### 13.1 核心设计要点

1. **存储决策**:
   - Gate (g) 和 Beta → **SMEM** (element-wise 操作)
   - K@S accumulator → **TMEM** (GEMM 需要)
   - v_corrected → **SMEM** (临时结果)
   - State → **TMEM (BF16)** (跨 chunk 传递)

2. **计算流程**:
   - Prologue: 预计算 g_cumsum
   - Phase 1: QK GEMM (现有)
   - Phase 2: Gate-based mask (修改)
   - Phase 3: K@S GEMM (新增)
   - Phase 4: Delta correction (新增)
   - Phase 5: PV GEMM (修改)
   - Phase 6: State update with gating (修改)
   - Phase 7: SQ GEMM (修改)
   - Epilogue: Output (现有)

3. **TMEM 优化**:
   - Alias QK 和 KS accumulators
   - 降低非关键 stages
   - 考虑 BF16 压缩

### 13.2 实现优先级

**P0 (必须)**:
- ✅ 简化版 KDA (without recursive unrolling)
- ✅ K@S delta correction
- ✅ Element-wise gating
- ✅ 正确性验证

**P1 (重要)**:
- ✅ TMEM capacity 优化
- ✅ Pipeline and double buffering
- ✅ 基本性能优化

**P2 (可选)**:
- 完整 recursive unrolling
- 高级性能优化
- 多 CTA 支持

### 13.3 关键建议

1. **从简单开始**: 先实现简化版，验证核心逻辑
2. **增量开发**: 每个新功能都要单独测试
3. **监控 TMEM**: 随时检查 capacity 使用情况
4. **性能 baseline**: 与 decay attention 对比性能
5. **文档详细**: 记录所有设计决策和 trade-offs

### 13.4 后续工作

- 实现并测试本方案
- 根据实际结果调整设计
- 考虑算法层面的简化（如近似 delta correction）
- 探索混合精度策略（FP8/INT8）

---

## 附录 A: 代码框架示例

### A.1 SMEM 分配伪代码（优化后）

```python
# Blackwell SMEM capacity: 227KB per SM
class KDAStorage:
    # Existing (Q, K 现在存储 gated 版本)
    sQ: SmemTensor   # [C, K, 2] - Q' = Q ⊙ g_cumsum (after prologue)
    sK: SmemTensor   # [C, K, 2] - K' = K ⊙ g_cumsum (after prologue)
    sV: SmemTensor   # [V, C, 2] - V (unchanged)
    sP: SmemTensor   # [C, C, 2] - P (masked attention)
    sO: SmemTensor   # [V, C, 2] - Output
    
    # New for KDA (优化后)
    sGCumsum: SmemTensor # [C, K, 1] - 用于 prologue 输入，后复用为 K^T'
    sBeta: SmemTensor    # [C, 1, 2] - beta scaling
    sKS: SmemTensor      # [C, V, 1] - K'@S result (delta correction)
    sVCorr: SmemTensor   # [C, V, 1] - V - K'@S (corrected values)
    
    # Barriers
    beta_mbar: BarrierStorage
    ks_mbar: BarrierStorage
    
    # Note: sG removed (不再需要，输入直接是 g_cumsum)
    
# RMEM per thread (for g_cumsum last row)
rGCumsumLast: Register[K // 32]  # 每个线程 4 个 BF16 值 = 8 bytes
```

### A.2 TMEM 分配伪代码

```python
# Compute offsets
qk_offset = 0
pv_offset = qk_offset + qk_cols
kv_offset = pv_offset + pv_cols
kv16_offset = kv_offset + kv_cols
sq_offset = kv16_offset + kv16_cols
ks_offset = qk_offset  # ALIAS with QK (non-concurrent)

# Allocate TMEM
tmem.allocate(total_cols)
tmem_ptr = tmem.retrieve_ptr(dtype)

# Create TMEM tensors
tAccQK = make_tmem_tensor(tmem_ptr + qk_offset, ...)
tAccPV = make_tmem_tensor(tmem_ptr + pv_offset, ...)
tAccKV = make_tmem_tensor(tmem_ptr + kv_offset, ...)
tAccKV16 = make_tmem_tensor(tmem_ptr + kv16_offset, ...)
tAccSQ = make_tmem_tensor(tmem_ptr + sq_offset, ...)
tAccKS = make_tmem_tensor(tmem_ptr + ks_offset, ...)  # NEW
```

### A.3 Delta Correction 伪代码

```python
@cute.jit
def compute_delta_correction(
    tiled_mma_ks,
    sK,        # SMEM K: [C, K]
    tState,    # TMEM State: [K, V]
    tAccKS,    # TMEM KS accumulator: [C, V]
    sKS,       # SMEM KS: [C, V]
    sV,        # SMEM V: [V, C]
    sVCorr,    # SMEM V corrected: [V, C]
):
    """Compute delta correction: v_corrected = V - K@S"""
    
    # Step 1: GEMM K @ S → TMEM[KS]
    exec_mma(tiled_mma_ks, tAccKS, sK, tState)
    
    # Step 2: Convert and copy TMEM → SMEM
    copy_tmem_to_smem(tAccKS, sKS)
    
    # Step 3: Element-wise subtraction: V - KS
    for i in range(C):
        for j in range(V):
            sVCorr[i, j] = sV[j, i] - sKS[i, j]
    
    # Barrier to ensure sVCorr is ready
    sync_barrier()
```

### A.4 Gate Application 伪代码

```python
@cute.jit
def apply_gate_based_mask(
    tAccQK,      # TMEM QK accumulator
    sGCumsum,    # SMEM gate cumsum: [C, K]
    sBeta,       # SMEM beta: [C, 1]
    sP,          # SMEM P (output): [C, C]
    tidx,        # Thread index
):
    """Apply gate-based decay mask: P[i,j] = QK[i,j] * exp(g[i]-g[j]) * beta[i]"""
    
    # Load QK from TMEM to RMEM
    rQK = load_from_tmem(tAccQK, tidx)
    
    # Load gate cumsum
    rGCumsum = load_from_smem(sGCumsum, tidx)
    
    # Load beta
    rBeta = load_from_smem(sBeta, tidx)
    
    # For each element assigned to this thread
    for idx in thread_elements(tidx):
        i, j = decode_index(idx)
        
        if i < j:  # Causal mask
            rP[idx] = 0
        else:
            # Compute decay: exp(sum(g[0:i]) - sum(g[0:j]))
            g_diff = rGCumsum[i] - rGCumsum[j]  # [K] element-wise
            decay = exp(g_diff)  # [K] element-wise
            
            # Average decay across K dimensions (or use specific strategy)
            decay_avg = mean(decay)
            
            # Apply decay and beta
            rP[idx] = rQK[idx] * decay_avg * rBeta[i]
    
    # Store P to SMEM
    store_to_smem(rP, sP, tidx)
```

---
