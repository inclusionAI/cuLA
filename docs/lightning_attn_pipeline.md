# LinearAttentionChunkwiseDecay (Lightning Attention) Pipeline

> 文件: `cula/ops/lightning_attn.py`  
> 类名: `LinearAttentionChunkwiseDecay`

## 计算公式

分块线性注意力，带 per-head 指数衰减:

**Intra-chunk**:
$$S = Q @ K^T, \quad P = \text{mask}(S), \quad O_{intra} = P @ V$$

**Inter-chunk** (线性状态递推):
$$\text{State}_{new} = \lambda^C \cdot \text{State} + K_{weighted}^T @ V$$
$$O_{inter} = \text{State} @ Q^T$$

**衰减**:
$$K_{weighted}[i] = K[i] \cdot e^{-s_h \cdot (C - i - 1)} \quad \text{(per-position intra-chunk)}$$
$$\lambda^C = e^{-s_h \cdot C} \quad \text{(block decay for state)}$$

**最终输出**: $O = O_{intra} + O_{inter}$ (加上 inter-chunk position decay)

## 线程布局

| Warp | 角色 | 职责 |
|------|------|------|
| 0-3 | CUDA Core (128 threads) | K decay weighting、causal mask、P 转换、O_intra+O_inter 合并、inter-chunk decay |
| 4 | MMA | 4 个 MMA: QK、VP、KV、SQ |
| 5 | Load | TMA G2S 加载 Q、K、V; 动态调度 (persistent varlen) |
| 6 | Epilogue | TMA S2G 写 O |
| 7 | Empty | 空闲 (预留扩展) |

**总线程**: 256 (8 warps)  
**寄存器分配**: CUDA=208, Others=168

## MMA 操作

| MMA | Tiler (M,N,K) | A 操作数 | B 操作数 | 输出 | 用途 |
|-----|---------------|----------|----------|------|------|
| QK | (64, 64, 128) | Q — SMEM | K — SMEM | S (TMEM FP32) | Attention scores |
| VP | (128, 64, 64) | V — SMEM MN-major | P — SMEM K-major | O_intra (TMEM FP32) | Intra-chunk output |
| KV | (128, 128, 64) | K_weighted — SMEM MN-major | V — SMEM MN-major | State (TMEM FP32) | 状态递推 |
| SQ | (128, 64, 128) | State — **TMEM** K-major | Q — SMEM K-major | O_inter (TMEM FP32) | Inter-chunk output |

### MMA 执行顺序 (per chunk)

1. **SQ**: State @ Q → O_inter (如果 chunk > 0 或有 initial_state)
2. **QK**: Q @ K^T → S
3. **KV**: K_weighted^T @ V → State (累加, 含衰减)
4. **VP**: P @ V → O_intra

## Pipeline 阶段

```
   Load Warp (5)       CUDA Warps (0-3)           MMA Warp (4)         Epilogue (6)
       │                     │                         │                     │
  Q→sQ ┼──load_q_mbar──────>│                         │                     │
  K→sK ┼──load_k_mbar──────>│                         │                     │
       │                     │ K decay weighting:      │                     │
       │                     │ K[i]*=exp(-s*(C-i-1))   │                     │
       │                     │ K_weighted → sK_w       │                     │
       │                     ├──k_weighted_mbar───────────────────────────>│
       │                     │                         │                     │
       │                     │                         │ SQ: State@Q→O_inter │
       │                     │                    <────┼──load_q (consumer) │
       │                     │                         │ QK: Q@K^T→S        │
       │                     │<──s_mbar───────────────┤                     │
       │                     │ causal mask S           │                     │
       │                     │ S→BF16 P → sP          │                     │
       │                     ├──p_mbar────────────────>│                     │
  V→sV ┼──load_v_mbar──────────────────────────────>│ KV: Kw^T@V→State  │
       │                     │                         │ VP: P@V→O_intra    │
       │                     │<──o_intra_mbar─────────┤                     │
       │                     │<──o_inter_mbar─────────┤                     │
       │                     │ T2R O_intra, O_inter   │                     │
       │                     │ apply inter-chunk decay │                     │
       │                     │ O = O_intra + O_inter   │                     │
       │                     │ R2S → sO               │                     │
       │                     ├──smem_o_mbar──────────────────────────────>│
       │                     │                         │           TMA sO→GMEM
```

### Pipeline 深度

| Pipeline | Stages | 方向 | 用途 |
|----------|--------|------|------|
| load_q | 2 | Load→MMA | Q 到 SMEM |
| load_k | 2 | Load→CUDA/MMA | K 到 SMEM |
| load_v | 2 | Load→MMA | V 到 SMEM |
| s (QK acc) | 2 | MMA→CUDA | QK scores 就绪 |
| kv | 1 | MMA→CUDA | KV state 就绪 |
| kv16 | 1 | CUDA→MMA | State BF16 就绪 |
| p | 2 | CUDA→MMA | P 矩阵就绪 |
| o_intra | 1 | MMA→CUDA | VP 结果就绪 |
| o_inter | 1 | MMA→CUDA | SQ 结果就绪 |
| smem_o | 2 | CUDA→Epilogue | O 就绪 |
| k_weighted | 1 | CUDA→MMA | 衰减加权 K 就绪 |
| QK acc | 2 | — | QK 累加器双缓冲 |
| VP acc | 2 | — | VP 累加器双缓冲 |

## 内存分配

### TMEM (~450/512 列, ~88% 利用率)

| 区域 | 大小 | 用途 |
|------|------|------|
| QK acc | ~100 cols × 2 stages | QK MMA 累加器 |
| PV acc | ~100 cols × 2 stages | VP MMA 累加器 |
| KV FP32 acc | ~100 cols × 1 stage | State 累加器 (不双缓冲节省 TMEM) |
| KV16 BF16 | ~50 cols | State 转 BF16 做 SQ MMA A 操作数 |
| SQ acc | ~100 cols | SQ MMA 累加器 |

### SMEM

| 缓冲 | Stages | 用途 |
|-------|--------|------|
| sQ | 2 | Query |
| sK | 2 | Key |
| sV | 2 | Value |
| sK_weighted | 1 | 衰减加权后的 K |
| sP | 2 | Attention P 矩阵 |
| sO | 2 | Output |

## 主循环流程

### Persistent Varlen 模式

```
PERSISTENT LOOP {
    Load warp: atomicAdd → work_idx
    Decode: hidx = work_idx % H, bidx = work_idx // H
    → seq_len = cu_seqlens[bidx+1] - cu_seqlens[bidx]

    for chunk_idx = 0 to ceil(seq_len / C) - 1:
        1. Load warp: TMA Q[chunk], K[chunk], V[chunk]
        2. CUDA warps: K decay weighting → sK_weighted
        3. MMA warp: SQ (if chunk>0) → QK → KV → VP
        4. CUDA warps: mask S, fuse O, apply decay
        5. Epilogue: TMA store O

    CTA barrier sync → 下一 WU
}
```

### 衰减计算 (CUDA warps)

```python
# Per-position intra-chunk decay (K weighting)
for i in range(C):
    K[i, :] *= exp(-s_h * (C - i - 1))

# Block decay (state accumulation)
block_decay = exp(-s_h * C)
State = block_decay * State_prev + K_weighted^T @ V

# Per-position inter-chunk decay (O_inter)
for i in range(C):
    O_inter[:, i] *= exp(-s_h * (chunk_idx * C + (C - i - 1)))
```

### State 管理 (TMEM)

- **位置**: TMEM offset `tmem_kv_cols_offset` (FP32)
- **形状**: (D, D) FP32 累加器
- **生命周期**: 跨 chunk 持久存在 (WU 内)
- **双视图**:
  - `tCtAccKV`: FP32 状态累加
  - `tCtAccKV16`: 重解释为 BF16，做 SQ MMA A 操作数

## Persistent / Varlen 支持

### Persistent Varlen
- Grid = SM_count
- 动态调度: Load warp `atomicAdd` → sWorkIdx → 双缓冲 `sched_mbar`
- 所有 warp wait → 读 sWorkIdx → arrive `sched_consumed_mbar` (反压防 ABA)
- Phase cycling 双缓冲

### Non-Varlen
- Grid 按常规 (B, H) 分配，chunk 串行循环

### Initial State / Final State
- 支持 `h0` (initial state) 和 `ht` (final state) 输入/输出
- h0 写入 TMEM State 区域 (FP32)
- ht 在最后一个 chunk 后从 State 读出

## 关键优化

- **4 MMA 最大化重叠**: SQ、QK、KV、VP 四个 MMA 通过 pipeline 最大化 Load/CUDA/MMA 重叠
- **TMEM State**: D×D 线性状态直接在 TMEM 做 SQ MMA A 操作数，避免 SMEM roundtrip
- **K Decay Weighting 与 MMA 重叠**: CUDA warps 计算 K decay 同时 MMA 可执行上一 chunk 的尾部 MMA
- **Per-Position Decay**: 精确的逐位置指数衰减，而非简化近似
- **Persistent Varlen**: 支持变长序列的 persistent kernel，atomic 动态调度 + 双缓冲 mbarrier
- **KV 单 stage**: KV 累加器不双缓冲，节省宝贵的 TMEM 列空间 (FP32 占列多)

## 与 LinearAttentionChunkwise 的对比

| 特性 | Lightning Attention (本文件) | Linear Attention |
|------|-----|------|
| 衰减 | 有 (per-head exponential decay) | 无 |
| K weighting | `K[i] *= exp(-s*(C-i-1))` | 直接使用 K |
| State 更新 | `λ^C * S + K_w^T @ V` | `S + K^T @ V` |
| Inter-chunk O | 带 position decay | 直接加 |
| Varlen persistent | 支持 | 不支持 |
| 动态调度 | 有 | 无 |
