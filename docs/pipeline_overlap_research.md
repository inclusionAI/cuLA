# Lightning Attention Pipeline 重叠优化研究报告

> 平台: NVIDIA Blackwell SM100a (B200), 152 SMs  
> 配置: B=1, T=4096, H=16, D=128, C=64, BF16 I/O, FP32 累加  
> Kernel Duration: 177.82 μs  
> NCU 报告: Compute throughput 3.99%, Memory throughput 3.52% — **严重受 latency 限制**

---

## 1. 当前 Pipeline 架构总览

### 1.1 Warp 角色分配 (12 warps / 384 threads)

| Warp ID | 角色 | 职责 |
|---------|------|------|
| W9  | Load Warp | TMA 异步加载 Q, K, V 到 SMEM |
| W8  | MMA Warp  | 执行 QK, SQ, KV, VP 四个 GEMM |
| W4-7 | Prologue CUDA Warps (×4) | K decay 加权 (S2R→LUT→R2S), QK decay mask → P |
| W0-3 | Epilogue CUDA Warps (×4) | KV state 类型转换/衰减, O 组合输出 |
| W10 | Store Warp | TMA 或 CopyUniversal 将 O 写回 GMEM |
| W11 | Empty Warp | 空闲 |

### 1.2 MMA 计算量分析 (每 chunk)

| GEMM | Tile (M×N×K) | FLOPs | 占比 | 用途 |
|------|-------------|-------|------|------|
| QK | (64, 64, 128) | 1,048,576 | 16.7% | 注意力分数 Q·K^T |
| SQ | (128, 64, 128) | 2,097,152 | 33.3% | 跨 chunk 贡献 State·Q |
| KV | (128, 128, 64) | 2,097,152 | 33.3% | 状态更新 K_w^T·V |
| VP | (128, 64, 64) | 1,048,576 | 16.7% | 块内注意力 P·V |
| **Total** | | **6,291,456** | 100% | |

### 1.3 资源使用

**SMEM (166.91 KB / 228 KB capacity):**

| Buffer | Size | Stages | Bytes |
|--------|------|--------|-------|
| Q (C×D) | 64×128 | 2 | 32,768 |
| K (C×D) | 64×128 | 2 | 32,768 |
| V (C×D) | 64×128 | 2 | 32,768 |
| P (C×C) | 64×64 | 2 | 16,384 |
| K_weighted (C×D) | 64×128 | 1 | 16,384 |
| O (D×C) | 128×64 | 2 | 32,768 |
| DecayLUT | 65 | 1 | 260 |
| **Total** | | | **~164 KB + padding** |

**TMEM (512 / 512 cols — 满载):**

| Accumulator | Shape | Dtype | Stages | Cols |
|-------------|-------|-------|--------|------|
| QK acc | 64×64 | FP32 | 2 | 128 |
| PV acc | 128×64 | FP32 | 2 | 128 |
| KV acc | 128×128 | FP32 | 1 | 128 |
| KV16 | 128×128 | BF16 | 1 | 64 |
| SQ acc | 128×64 | FP32 | 1 | 64 |
| **Total** | | | | **512 (100%)** |

**寄存器:** 168 / thread → 限制 occupancy 为 1 block/SM

---

## 2. Pipeline 依赖关系分析

### 2.1 每 chunk 内的依赖图

```
                    Load Warp (TMA)
                   ┌──Q──┐  ┌──K──┐  ┌──V──┐
                   │     │  │     │  │     │
                   ▼     ▼  ▼     ▼  ▼     ▼
MMA Warp:     ┌────QK GEMM────┐
              │   (Q, K)      │
              └───────┬───────┘
                      │
              ┌───────┼────────────────────────────────┐
              │       │ commit s0                       │ commit k_ready
              │       ▼                                 ▼
Prologue:     │  ┌──T2R QK──┐                   ┌──S2R K──┐
              │  │apply mask│                   │  LUT    │
              │  │→ R2S P   │                   │→ R2S Kw │
              │  └────┬─────┘                   └────┬────┘
              │       │ commit P                     │ commit k_weighted
              │       ▼                              ▼
MMA Warp:     │  ┌──VP GEMM──┐              ┌───KV GEMM───┐
              │  │  (P, V)   │              │ (V, K_w)     │
              │  └─────┬─────┘              │ wait sr(i-1) │
              │        │                    └──────┬───────┘
              │        │ commit o_intra            │ commit kv
              │        ▼                           ▼
Epilogue:     │   ┌──T2R PV──┐            ┌──T2R KV state──┐
              │   │  + SQ    │            │  → BF16 → kv16 │
              │   │→ scale   │            │  → decay → sr  │
              │   │→ R2S O   │            └────────────────┘
              │   └────┬─────┘
              │        │ commit smem_o        ↑ 跨 chunk 依赖
Store Warp:   │   ┌──TMA──┐                  ↑
              │   │Store O│                  ↑
              │   └───────┘                  ↑
              │                              │
MMA Warp:     └──── SQ GEMM ─── wait kv16 ──┘
                    (State, Q)   wait sr ────┘
```

> 注: SQ GEMM 在 QK 之后、KV 之前执行。上图简化展示了依赖关系。

### 2.2 MMA Warp 的精确执行顺序

每个 chunk 内，MMA warp 按以下严格顺序执行:

```python
# Chunk i (MMA Warp W8):
1. wait Q, K         # TMA load 完成
2. QK GEMM           # 注意力分数
3. commit s0         # 通知 prologue 读 QK 结果
4. commit k_ready    # 通知 prologue 读 K SMEM

5. wait kv16(i-1)    # ⚠ 跨 chunk 阻塞点 1: 等 epilogue 的 BF16 state
6. SQ GEMM           # O_inter = State_bf16 · Q
7. commit o_inter
8. release Q

9. wait k_weighted   # 等 prologue 的 K 加权完成
10. wait V           # TMA load V 完成
11. wait sr(i-1)     # ⚠ 跨 chunk 阻塞点 2: 等 epilogue 的 F32 decayed state
12. KV GEMM          # State += K_w^T · V (带累加)
13. commit kv
14. release K, k_weighted

15. wait P           # 等 prologue 的 decay mask
16. VP GEMM          # O_intra = P · V
17. commit o_intra
18. release V
```

### 2.3 跨 chunk 关键阻塞点

**阻塞点 A: kv16 (SQ GEMM 依赖)**
```
产生: Epilogue(i-1) 收到 KV commit → T2R(128×128 FP32) → BF16 转换 → TMEM store kv16 → commit
消费: MMA(i) 在 QK 完成后立即需要 kv16 来执行 SQ GEMM
间隔: 仅 QK GEMM 的执行时间 (~1M FLOPs)
```

**阻塞点 B: state_ready (KV GEMM 依赖)**
```
产生: Epilogue(i-1) 在 kv16 commit 之后 → F32 decay 乘法 → TMEM store F32 → commit sr
消费: MMA(i) 在 SQ 完成后需要 sr 来执行 KV GEMM (带累加)
间隔: QK + SQ GEMM 的执行时间 (~3M FLOPs)
```

> **关键观察**: state_ready 在 kv16 之后产生, 而 MMA 的消费顺序恰好是先 kv16(for SQ) 后 sr(for KV)。这个顺序是**最优**的 — 先等早到的信号，后等晚到的信号。

### 2.4 Epilogue 处理 KV State 的串行链路

```
KV commit ──→ T2R (128×128 FP32 从 TMEM 读到 RMEM)
          ──→ BF16 转换 (128×128 FP32 → BF16)
          ──→ TMEM store kv16 (写回 TMEM BF16 区域)
          ──→ fence + commit kv16
          ──→ F32 乘 block_decay (128×128 × scalar)
          ──→ TMEM store F32 (写回 TMEM FP32 区域覆盖)
          ──→ fence + commit state_ready
```

这条链路涉及 **128×128 = 16384 个元素** 的 T2R + 类型转换 + 两次 TMEM store + 标量乘法。
处理延迟是跨 chunk 阻塞的核心来源。

---

## 3. 当前 Pipeline 中的改进机会

### 3.1 已有的良好设计

1. **MMA 顺序最优**: QK → SQ(wait kv16) → KV(wait sr) 的顺序匹配了 epilogue 的产出顺序
2. **双缓冲**: TMA load (Q/K/V), MMA accumulators (QK/PV), P, O 都使用 2 stages
3. **K 提前释放**: K 在 KV GEMM 之后立即释放，允许下一 chunk 的 K 加载
4. **Load 与 Compute 重叠**: TMA load warp 与 MMA/CUDA 完全异步

### 3.2 潜在的改进方向

**机会 1: Prologue P 生成与 VP GEMM 的重叠**

当前 VP GEMM 等待 P 完成 (step 15)。P 的生成需要:
1. wait s0 (QK commit)
2. T2R QK (64×64 FP32)
3. apply_decay_mask (逐元素)
4. BF16 转换
5. R2S → SMEM P

这条链路与 SQ, KV GEMM 并行执行。如果 SQ+KV 总耗时 > P 生成耗时, 则 P 在 VP 开始前已就绪, 不构成瓶颈。

**评估**: SQ(2M FLOPs) + KV(2M FLOPs) 共 4M FLOPs 的 MMA 时间, 远大于 P 生成的 CUDA core 操作时间。**P 大概率不是瓶颈**。

**机会 2: K_weighted 相对于 KV 的时序**

K_weighted 需要 prologue 完成 K 加权:
1. wait k_ready
2. S2R K (64×128 BF16)
3. apply decay LUT (64×128 逐元素)
4. R2S K_weighted

MMA 在 SQ GEMM 之后才需要 k_weighted (step 9)。而 prologue 在 QK GEMM commit 后立即开始。
SQ GEMM 2M FLOPs 为 prologue 提供了充足的时间。**k_weighted 大概率不是瓶颈**。

**机会 3: Epilogue O 组合的串行性**

Epilogue 需要同时等待 o_intra (VP) 和 o_inter (SQ) 才能组合 O:
```
wait o_intra → T2R PV result
wait o_inter → T2R SQ result  (仅 chunk > 0)
→ apply_inter_chunk_decay
→ O = (O_intra + O_inter_decayed) * scale
→ BF16 convert → R2S → SMEM O → commit
```

o_inter 在 SQ commit (early) 产生, o_intra 在 VP commit (late) 产生。
所以 epilogue 等待 o_intra 是后阻塞。但这不影响关键路径 — O 组合只影响 store warp, 不影响下一 chunk 的 MMA。

### 3.3 真正的瓶颈: SM 利用率

NCU 诊断的最大性能问题:

```
Grid Size:       (1, 16, 1) = 16 blocks
SMs Available:   152
Waves per SM:    0.11
SM Active Ratio: ~10.5%
Est. Speedup:    89.47% (如果 grid 能填满 SM)
```

**16 个 CTA 占 152 个 SM, 136 个 SM 完全空闲**。这是压倒性的性能瓶颈。

任何 pipeline 微调的收益都远小于提高 SM 利用率。

---

## 4. V 维度切分方案分析

### 4.1 核心思路

将 V (value) 的 D=128 维度切分为两半 (D_v=64), 独立处理, 以实现:
1. **Grid 层级**: 增加 CTA 数量, 提高 SM 利用率
2. **CTA 层级**: 缩小 KV state, 加速跨 chunk epilogue 处理, 减少阻塞

### 4.2 数学验证

Lightning Attention 的核心计算 (per chunk i):

$$O_i = \underbrace{(P_i \cdot V_i)}_{\text{intra-chunk}} + \underbrace{(S_{i-1} \cdot Q_i \cdot \Lambda_i)}_{\text{inter-chunk}}$$
$$S_i = \gamma^C \cdot S_{i-1} + K_{w,i}^T \cdot V_i$$

其中 $S \in \mathbb{R}^{D \times D}$, $V_i \in \mathbb{R}^{C \times D}$, $O_i \in \mathbb{R}^{D \times C}$.

V 维度切分的关键观察: **S 和 O 的"行"维度对应 V 的列维度, 可以独立切分**。

设 $V = [V_0 | V_1]$, $V_0 \in \mathbb{R}^{C \times D/2}$, $V_1 \in \mathbb{R}^{C \times D/2}$:

$$S_0 = \gamma^C \cdot S_{0,i-1} + K_w^T \cdot V_0 \quad \in \mathbb{R}^{D \times D/2}$$
$$S_1 = \gamma^C \cdot S_{1,i-1} + K_w^T \cdot V_1 \quad \in \mathbb{R}^{D \times D/2}$$

$$O_0 = P \cdot V_0 + S_0 \cdot Q \cdot \Lambda \quad \in \mathbb{R}^{D/2 \times C}$$
$$O_1 = P \cdot V_1 + S_1 \cdot Q \cdot \Lambda \quad \in \mathbb{R}^{D/2 \times C}$$

$$O = [O_0; O_1]$$

**Q, K, P, decay 等都是共享的, 只有 V, State, O 需要切分。** 切分完全合法, 不改变最终结果。

### 4.3 方案 A: Grid 层级 V 切分 (推荐)

**设计**: 将 V 切分作为 grid 的一个维度, 每个 CTA 处理 D/2 的 V 切片。

```
Grid: (V_SPLIT=2, H, B)  而不是  (1, H, B)
每个 CTA: V[:, v_id*D/2 : (v_id+1)*D/2]
```

#### 资源变化

| 资源 | 当前 | Grid V-split | 变化 |
|------|------|-------------|------|
| **Grid 大小** | 16 | 32 | +100% |
| **Waves / SM** | 0.11 | 0.21 | +100% |
| **SM 空闲率** | 89.5% | 78.9% | -10.6pp |

**per-CTA MMA FLOPs:**

| GEMM | 当前 | V-split | 说明 |
|------|------|---------|------|
| QK (C×C×D) | 1,048,576 | 1,048,576 | **不变** (冗余计算) |
| SQ (D/2×C×D) | 2,097,152 | 1,048,576 | **减半** |
| KV (D×D/2×C) | 2,097,152 | 1,048,576 | **减半** |
| VP (D/2×C×C) | 1,048,576 | 524,288 | **减半** |
| **Total** | **6,291,456** | **3,670,016** | **-41.7%** |

2× CTA 总 FLOPs: 7,340,032 (多 16.7% 冗余 QK)

> 由于 compute throughput 仅 3.99%, 16.7% 的冗余 QK 计算对实际延迟几乎无影响。

**SMEM 变化:**

| Buffer | 当前 | V-split | 变化 |
|--------|------|---------|------|
| Q (C×D) | 32 KB | 32 KB | 不变 |
| K (C×D) | 32 KB | 32 KB | 不变 |
| V (C×D/2) | 32 KB | **16 KB** | -50% |
| P (C×C) | 16 KB | 16 KB | 不变 |
| K_weighted (C×D) | 16 KB | 16 KB | 不变 |
| O (D/2×C) | 32 KB | **16 KB** | -50% |
| DecayLUT | 0.26 KB | 0.26 KB | 不变 |
| **Total** | **~167 KB** | **~129 KB** | **-23%** |

**TMEM 变化:**

| Accumulator | 当前 Cols | V-split Cols | 变化 |
|-------------|----------|-------------|------|
| QK acc (64×64, 2s) | 128 | 128 | 不变 |
| PV acc (64×64, 2s) | 128 | 128 | 不变 |
| KV acc (128×64, 1s) | 128 | **64** | -50% |
| KV16 (128×64, 1s) | 64 | **32** | -50% |
| SQ acc (64×64, 1s) | 64 | 64 | 不变 |
| **Total** | **512 (100%)** | **416 (81.2%)** | **-18.8%** |

**关键收益: TMEM 从 100% 满载降到 81.2%, 释放出 96 cols, 可用于增加 pipeline 深度或其他 staging。**

#### 寄存器压力

KV state 相关的 RMEM tensor 缩小一半 (128×128 → 128×64):
- Epilogue T2R 读取: 数据量减半
- BF16 转换: 元素减半
- Decay 乘法: 元素减半
- TMEM store: 数据量减半

预期寄存器使用从 168 降低, 有可能降至 128 以下, 从而允许 2 blocks/SM (occupancy 翻倍)。

#### 跨 chunk 阻塞改善

Epilogue 处理 KV state 的关键链路缩短约 50%:
```
原始: T2R(128×128) → BF16(16K elems) → TMEM store → decay(16K elems) → TMEM store
V-split: T2R(128×64) → BF16(8K elems) → TMEM store → decay(8K elems) → TMEM store
```

kv16 和 state_ready 的产出延迟大约减半, 直接减少 MMA warp 的等待时间。

#### 预期性能收益

1. **per-CTA MMA 减少 42%** → MMA 执行时间缩短
2. **Epilogue 处理减半** → 跨 chunk 阻塞减少 ~50%
3. **SMEM 减少 23%** → 更多余量, 可考虑增加 pipeline stages
4. **TMEM 使用率 81%** → 不再满载, 有空间调优
5. **潜在的寄存器降低** → 可能允许 2 blocks/SM

**保守估计: per-CTA 延迟降低 30-40%, 整体 kernel 延迟降低到 ~110-125 μs (相比当前 177.82 μs)。**

#### 实现要点

1. Grid 增加 V-split 维度: `grid = (2, H, B)`
2. 每个 CTA 根据 `v_split_idx = blockIdx.x` 选择 V 的上半或下半
3. TMA load V: 偏移 `v_split_idx * D/2`, 加载 C×D/2 的 tile
4. TMA store O: 偏移 `v_split_idx * D/2`, 写入 D/2×C 的 tile
5. QK, P, K_weighted 计算不变 (完全共享)
6. KV state, SQ, VP MMA tiler 调整为 D/2
7. h0/ht (initial/final state) 需要按 V 维度切分读写

#### 风险与注意事项

- **QK 冗余**: 两个 V-split CTA 独立计算相同的 QK, P → 浪费 ~16.7% FLOPs。但 kernel 是 latency-bound 不是 compute-bound, 所以影响极小。
- **Prologue 不变**: K_weighting 和 QK masking 完全相同, prologue CUDA warps 做冗余工作。在 latency-bound 下可接受。
- **h0/ht 切分**: initial_state 和 final_state 是 (D×D) 矩阵, 需按 V-split dim 切分读写。需确保内存访问对齐。
- **varlen 兼容**: V-split 是 head-dim 切分, 与序列维度正交, 对 varlen 支持无影响。

### 4.4 方案 B: CTA 内部 V 切分 (备选)

**设计**: 在单个 CTA 内, MMA warp 顺序处理 V 的两半, 与 epilogue pipeline 交错。

```python
# MMA Warp per chunk:
QK → SQ0(wait kv16_0) → KV0(wait sr_0) → VP0 → SQ1(wait kv16_1) → KV1(wait sr_1) → VP1

# Epilogue per chunk:
after KV0 commit: process state0 → kv16_0 → sr_0  (与 MMA 的 VP0→SQ1→KV1→VP1 重叠)
after KV1 commit: process state1 → kv16_1 → sr_1
after VP1 commit: combine O = [O0; O1]
```

#### 重叠分析

**核心优势**: KV0 commit 后, epilogue 可以开始处理 state0 (仅 128×64)。同时 MMA warp 继续执行 VP0 → SQ1 → KV1 → VP1, 提供约 0.5+1+1+0.5 = 3 个 GEMM 单元的并行时间。

```
时间线 (GEMM unit 为计算单元):

Chunk i:
MMA:  |QK(1)| SQ0(1) | KV0(1) | VP0(0.5) | SQ1(1) | KV1(1) | VP1(0.5) |
Epi:                  |-----state0 process (0.5-1)-----| |--state1--|
                                                                         |--O combine--|

Chunk i+1:
MMA:  |QK(1)| SQ0(1) wait kv16_0... | ...
```

**对比当前 (无 V-split):**
```
Chunk i:
MMA:  |QK(1)| SQ(2)  | KV(2)        | VP(1)  |
Epi:                  |----state process (2)----| |--O combine--|
```

**CTA-level V-split 的好处:**
1. State0 的 kv16_0/sr_0 更早可用 (数据量减半, 且有更多 MMA 重叠时间)
2. 每个 half-state 的 epilogue 处理更短

**缺点:**
1. MMA warp 执行 7 个 GEMM 而非 4 个 (更多 pipeline barrier overhead)
2. 不增加 grid 大小, SM 利用率不变
3. 更多 TMEM load/store 操作 (两次 T2R, 两次 TMEM store)
4. 实现复杂度大幅增加

### 4.5 方案对比

| 指标 | 当前 | 方案 A (Grid V-split) | 方案 B (CTA V-split) |
|------|------|----------------------|---------------------|
| Grid 大小 | 16 | **32** | 16 |
| SM 利用率 | 10.5% | **21.1%** | 10.5% |
| per-CTA FLOPs | 6.29M | **3.67M** | 6.29M (相同总量) |
| Epilogue 状态处理 | 16K elems | **8K elems** | 8K × 2 |
| 跨 chunk 阻塞 | 基准 | **~50% reduction** | ~30% reduction |
| SMEM 用量 | 167 KB | **129 KB** | ~167 KB |
| TMEM 用量 | 100% | **81.2%** | 100% (两半复用) |
| 冗余计算 | 0% | 16.7% (QK/Prologue) | 0% |
| 实现难度 | — | **中等** | 高 |
| 预期加速 | — | **30-40%** | 10-20% |

**推荐方案 A (Grid V-split)**, 原因:
1. 直接解决最大瓶颈 (SM 利用率), 收益远大于 pipeline 微调
2. 实现较简洁: 只需在 grid 增加一个维度, 调整 V/O/State 的偏移
3. 资源全面改善: SMEM -23%, TMEM -19%, 潜在 register 降低
4. 可以与方案 B 叠加, 先做 A 验证收益, 再考虑 B 进一步优化

---

## 5. 其他可探索的优化方向

### 5.1 多 chunk 分组 (T 维度切分)

对于 B=1, T=4096 的场景:
- 当前: 每个 CTA 处理完整的 4096/64 = 64 个 chunks
- 潜在: 将 T 维度分组, 每组 n 个 chunks, 再做 state merge
- 问题: state 有严格的因果依赖 ($S_i$ 依赖 $S_{i-1}$), 不能简单并行化
- 需要 scan-style 并行 (parallel prefix sum), 复杂度高

### 5.2 增加 Pipeline Stages

当前 KV state 和 KV16 只有 1 stage, 如果 TMEM/SMEM 有余量 (方案 A 释放的), 可以增加到 2 stages:
- KV acc 2 stages: 允许 MMA 在写当前 chunk 的 state 同时, epilogue 读上一 chunk 的 state
- 但 TMEM 从 416 cols 增加到 416+128+32 = 576 cols > 512 → **超出 TMEM 容量**
- 结论: 即使 V-split, KV state 2-stage 仍受 TMEM 限制

### 5.3 寄存器优化 → 提升 Occupancy

如果通过 V-split 将 register 需求降到 128/thread:
- Occupancy: 2 blocks/SM (37.5%)
- 32 CTAs 需要 16 个 SM (每 SM 2 blocks)
- 两个 block 可以交替执行, 隐藏 pipeline stall
- 但需要将 num_regs_prologue / num_regs_other 也相应调低

---

## 6. 结论与建议

### 当前 Pipeline 状态

1. **MMA 顺序 (QK→SQ→KV→VP) 已最优**: 匹配 epilogue 的 kv16/sr 产出顺序
2. **Prologue K_weighted / P 不是瓶颈**: 有足够的 MMA 时间重叠
3. **跨 chunk 阻塞 (kv16, sr) 是可改善的**: 缩小 state 可直接减少阻塞
4. **SM 利用率极低 (10.5%) 是压倒性瓶颈**: 任何 pipeline 微调都远不如提高 SM 利用率

### 推荐行动

| 优先级 | 行动 | 预期收益 | 难度 |
|--------|------|---------|------|
| P0 | Grid V-split (方案 A) | 30-40% latency 降低 | 中 |
| P1 | 寄存器压力降低 → 2 blocks/SM | 进一步改善 occupancy | 中 |
| P2 | CTA 内 V-split (方案 B) | 额外 10-20% latency 降低 | 高 |
| P3 | Pipeline stage 调优 | 5-10% 改善 | 低 |

**首要推荐实现 Grid V-split (方案 A)**, 以最小的实现成本获得最大的性能收益。
