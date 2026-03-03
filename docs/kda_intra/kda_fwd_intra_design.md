# KDA Forward Intra-Chunk SM100 Kernel 设计

## 1. 计算目标

对于一个 tile（T_TILE=64 行，K_SIZE=128 列），计算两个 [64×64] 下三角注意力矩阵：

```
Aqk[i,j] = scale * (q[i] * exp2(g[i] - gn)) @ (k[j] * exp2(gn - g[j]))^T    (j <= i)
Akk[i,j] = beta[i] * (k[i] * exp2(g[i] - gn)) @ (k[j] * exp2(gn - g[j]))^T  (j < i)
```

其中 `gn` 是每个 sub-chunk 对的参考 gate 值（与 backward 相同的分解技巧）。

safe_gate=True 时无需 forward substitution，直接输出 Aqk 和 Akk。

---

## 2. 与 Backward Intra 的关键差异

| 维度 | Backward Intra | Forward Intra |
|------|---------------|---------------|
| MMA 输出形状 | [64 × 32]（dQ，per k_idx） | [64 × 64]（Aqk/Akk，跨 k_idx 累加） |
| MMA 累加维度 | sub-chunk 行数（16/32/48） | K_TILE=32（迭代 4 次） |
| A 矩阵来源 | dA（从 global load，mask 后写入 TMEM） | q/k * exp2(g-gn)（CE 计算后写入 TMEM） |
| B 矩阵来源 | exp2(gn-g)*k（CE 计算，写入 SMEM） | (k * exp2(gn-g))^T（CE 计算，写入 SMEM） |
| 输出数量 | 1 个（dQ） | 2 个（Aqk + Akk） |
| Epilogue | scale + combine + output dQ/dK/dG | 仅 mask + store（safe_gate 下） |

核心变化：backward 的 A 矩阵是从 global load 的 dA，forward 的 A 矩阵需要 CE 实时计算并写入 TMEM。这意味着 **prologue 阶段更重**，但 epilogue 更轻。

---

## 3. Sub-chunk 分解（与 Backward 相同）

64 行分为 4 个 sub-chunk（BC=16）：S0=[0,16), S1=[16,32), S2=[32,48), S3=[48,64)

**Off-diagonal（intra）**：j < i 的 sub-chunk 对，gn = g[Si_start]
```
Aqk[Si,Sj] = (q[Si] * exp2(g[Si] - g[Si_start])) @ (k[Sj] * exp2(g[Si_start] - g[Sj]))^T
```

**Diagonal（inter）**：同一 sub-chunk 内，gn_half = g[Si*16+8]
```
Aqk[Si,Si] = (q[Si] * exp2(g[Si] - gn_half)) @ (k[Si] * exp2(gn_half - g[Si]))^T
```

---

## 4. TMEM 布局

Forward 的 A 矩阵由 CE 计算后写入 TMEM，MMA 从 TMEM 读取。
输出 C 矩阵（Aqk/Akk）也在 TMEM 中累加。

```
TMEM 地址分配（512 columns × 64 rows）：

A 矩阵区域（每个 k_idx 覆写）：
┌──────────────────────────────────────────────────────┐
│ A_q  [0, 32)     : q * exp2(g - gn)     [64×32]     │
│ A_k  [32, 64)    : k * exp2(g - gn) * β [64×32]     │
│   + lane offset 变体用于 intra/inter 不同 gn         │
└──────────────────────────────────────────────────────┘

C 累加器区域（跨 k_idx 累加）：
┌──────────────────────────────────────────────────────┐
│ Aqk_intra [64, 128)  : off-diagonal Aqk  [64×64]    │
│ Aqk_inter [128, 192) : diagonal Aqk      [64×64]    │
│ Akk_intra [192, 256) : off-diagonal Akk  [64×64]    │
│ Akk_inter [256, 320) : diagonal Akk      [64×64]    │
└──────────────────────────────────────────────────────┘
```

关键点：A_q 和 A_k 每个 k_idx 都会被 CE 覆写，而 C 累加器跨 4 个 k_idx 持续累加。

---

## 5. SMEM 布局

B 矩阵由 CE 计算后写入 SMEM，MMA 从 SMEM 读取。

```
B 矩阵（single-buffered，每个 k_idx 覆写）：
┌──────────────────────────────────────────────────────┐
│ B_intra[6] : (k[Sj] * exp2(gn_i - g[Sj]))^T        │
│              SmemLayoutMatBTF32Tranposed<1>           │
│              每个 [K_TILE=32 × BC=16]                 │
│              6 个 slot = 下三角 6 对 (Si,Sj)          │
├──────────────────────────────────────────────────────┤
│ B_inter[4] : (k[Si] * exp2(gn_half - g[Si]))^T      │
│              SmemLayoutMatBTF32Tranposed<1>           │
│              每个 [K_TILE=32 × BC=16]                 │
│              4 个 slot = 4 个对角块                    │
└──────────────────────────────────────────────────────┘

输入缓冲（double-buffered）：
┌──────────────────────────────────────────────────────┐
│ q[2]  : BF16 [64×32]                                 │
│ k[2]  : BF16 [64×32]                                 │
│ g[2]  : FP32 [64×32]                                 │
└──────────────────────────────────────────────────────┘
```

B_intra 和 B_inter 的 slot 分配与 backward 完全相同（见 kg_computation 文档第 3.2 节）。

---

## 6. 流水线设计：两阶段异步重叠

每个 k_idx 迭代分为两个 MMA 阶段，每个阶段内 CE 和 MMA 异步重叠。

### 6.1 阶段划分

```
阶段 1 (Aqk phase)：
  MMA 计算 Aqk += A_q @ B
  CE 同时准备 A_k（写入 TMEM 不同区域）

阶段 2 (Akk phase)：
  MMA 计算 Akk += A_k @ B（复用同一 B 矩阵）
  CE 同时准备下一个 k_idx 的 B 矩阵
```

### 6.2 单个 k_idx 迭代时间线

```
时间 →

CE (256 threads, CUDA core)              MMA (1 elected thread, Tensor Core)
═══════════════════════════              ═══════════════════════════════════

[1] Wait bar_load_qkg_ready
    Compute A_q = q * exp2(g-gn)
    → tmem_st to TMEM A_q region
    Compute B_intra[6] + B_inter[4]
    → store to SMEM
    ──arrive bar_A_q_ready──→
    ──arrive bar_B_ready──→
                                         [2] Wait bar_A_q_ready + bar_B_ready
                                             Aqk_intra: 3× MMA (A_q @ B_intra)
                                             Aqk_inter: 4× MMA (A_q @ B_inter)
[3] Compute A_k = k*exp2(g-gn)*β              ──arrive bar_Aqk_phase_done──→
    → tmem_st to TMEM A_k region
    ──arrive bar_A_k_ready──→
                                         [4] Wait bar_A_k_ready
                                             Akk_intra: 3× MMA (A_k @ B_intra)
                                             Akk_inter: 4× MMA (A_k @ B_inter)
[5] (如果非最后 k_idx)                        ──arrive bar_Akk_phase_done──→
    Wait bar_load_qkg_ready[next]
    开始准备下一个 k_idx 的 B 矩阵
    ──arrive bar_B_ready[next]──→
```

### 6.3 异步重叠分析

三处关键重叠：

**重叠 1：CE 准备 A_k ‖ MMA 执行 Aqk phase（7 次 MMA）**

CE arrive `bar_B_ready` 后，MMA 开始 7 次 Aqk MMA 调用。
CE 不等待 MMA，立即计算 `A_k = k * exp2(g-gn) * beta` 并写入 TMEM 的 A_k 区域。
A_q 和 A_k 在 TMEM 中占不同地址，不冲突。

**重叠 2：CE 准备下一 k_idx 的 B ‖ MMA 执行 Akk phase（7 次 MMA）**

MMA 开始 Akk phase 后，CE 可以开始准备下一个 k_idx 的 B 矩阵。
B 矩阵是 single-buffered，需要等 MMA 读完当前 B 才能覆写。
这里有两个选择：
- (a) B 也做 double-buffer，CE 写入另一份 → SMEM 开销翻倍但完全重叠
- (b) CE 等 Akk phase 结束后再写 B → 无额外 SMEM 但重叠减少

推荐 (a)，因为 B 矩阵总共 10 个 slot × [32×16] × 4B = 20KB，double-buffer 也只 40KB。

**重叠 3：TMA Load ‖ CE compute（与 backward 相同）**

Load warp 预取下一个 k_idx 的 q/k/g，与 CE 的 B 矩阵计算重叠。

---

## 7. MMA 调用计划

### 7.1 Aqk Phase：A_q @ B → Aqk 累加器

与 backward 的 kg phase 结构完全相同，使用 MASK02/MASK13 + lane offset。

**Intra（off-diagonal，3 次 MMA）：**

```
Call 1: MASK02, A=A_q[lane+16], B=B_intra[0][16cols], C=Aqk_intra[lane+16], clear(k0)/accum
  → Aqk[S1,S0] 和 Aqk[S3,S0]

Call 2: MASK13, A=A_q[lane+0],  B=B_intra[1..2][32cols], C=Aqk_intra[lane+0], clear(k0)/accum
  → Aqk[S1,S0:S1] 和 Aqk[S3,S0:S1]

Call 3: MASK13, A=A_q[lane+16], B=B_intra[3..5][48cols], C=Aqk_intra[lane+16], accum
  → Aqk[S2,S0:S2] 和 (S0 masked by lower-tri)
```

**Inter（diagonal，4 次 MMA）：**

```
Call 4: MASK02, A=A_q[lane+0],  B=B_inter[0][16cols], C=Aqk_inter[lane+0], clear(k0)/accum
  → Aqk[S0,S0] 和 Aqk[S2,S2]

Call 5: MASK02, A=A_q[lane+16], B=B_inter[1][16cols], C=Aqk_inter[lane+16], clear(k0)/accum
  → Aqk[S1,S1] 和 Aqk[S3,S3]

Call 6: MASK13, A=A_q[lane+0],  B=B_inter[2][16cols], C=Aqk_inter[lane+0], accum
  → Aqk[S1,S1'] 和 Aqk[S3,S3']

Call 7: MASK13, A=A_q[lane+16], B=B_inter[3][16cols], C=Aqk_inter[lane+16], accum
  → Aqk[S2,S2'] 和 (S0 masked)
```

注意：k_idx=0 时 clear accumulator，后续 k_idx 累加。

### 7.2 Akk Phase：A_k @ B → Akk 累加器

结构与 Aqk phase 完全相同，只是 A 矩阵换成 A_k，C 累加器换成 Akk 区域。
B 矩阵完全复用（同一份 SMEM 数据），无需重新计算。

```
Akk_intra: 3× MMA (Call 8-10)，与 Aqk_intra 相同的 MASK/lane/B 组合
Akk_inter: 4× MMA (Call 11-14)，与 Aqk_inter 相同的 MASK/lane/B 组合
```

每个 k_idx 共 14 次 MMA 调用（Aqk 7 + Akk 7）。
