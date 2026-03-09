# KDA Forward Intra - B-Matrix Prologue 详解

## 概述

本文档详细分析 Forward Intra kernel 中 ComputeEpilogue (CE) warpgroup 对 **B-matrix (KG)** 的处理逻辑。B-matrix 是为后续 UMMA (Tensor Core) 准备的 SMEM 数据，对应 MMA 的 B 操作数。

核心设计思路：**按列处理 (column-based)**，每列只加载一次 K 和 G 数据，融合产出该列的所有输出，最大化 SMEM 带宽复用。

## 1. 数学公式

### 1.1 下三角 4×4 subchunk 矩阵

T_TILE = 64，划分为 4 个 SUB_T_TILE = 16 的 subchunk。下三角共 10 个非零块 (i ≥ j)：

```
         j=0         j=1         j=2         j=3
  i=0  intra[0]
  i=1  inter[0]   intra[1]
  i=2  inter[1]   inter[2]   intra[2]
  i=3  inter[3]   inter[4]   inter[5]   intra[3]
```

### 1.2 B-matrix 公式

对于块 (i, j)，其中 i ≥ j：

- **Inter (i > j)**:  `B(i,j) = exp2(g_first_i - g_j[x]) * K_j[x]`
  - `g_first_i = g[i*16]`：第 i 个 subchunk 的**首行** gate 值
  - `g_j[x]`、`K_j[x]`：第 j 个 subchunk 内第 x 行的 gate / key 数据
  - 含义：跨 subchunk 的注意力权重

- **Intra (i == j)**: `B(i,i) = exp2(g_half_i - g_i[x]) * K_i[x]`
  - `g_half_i = g[i*16 + 8]`：第 i 个 subchunk 的**中间行** (第 8 行) gate 值
  - 含义：subchunk 内部的注意力权重

### 1.3 为什么 inter 用 `g_first` 而 intra 用 `g_half`

这与 MMA 的 A-matrix 侧配对有关：
- A-matrix inter: `exp2(g[row] - g_first_i)`，与 B-matrix inter 的 `exp2(g_first_i - g_j[x])` 配对
  - 两者相乘: `exp2(g[row] - g_j[x])`，即完整的 gate 差
- A-matrix intra: `exp2(g[row] - g_half_i)`，与 B-matrix intra 的 `exp2(g_half_i - g_i[x])` 配对
  - 两者相乘: `exp2(g[row] - g_i[x])`，同样是完整的 gate 差
- `g_first` / `g_half` 的选择是为了保证指数不爆炸（数值稳定性）

## 2. 数据布局

### 2.1 输入 Tensor (SMEM, TMA 加载)

| Tensor | 形状 | 数据类型 | 布局 |
|--------|------|---------|------|
| `sK` | `[T_TILE=64, K_TILE=32]` | `bf16` | `Layout_K_SW64_Atom` (swizzle) |
| `sG` | `[T_TILE=64, K_TILE=32]` | `float` | `Layout_K_SW128_Atom` (swizzle) |

- 双缓冲 (double-buffered)，由 TMA Load warp (warp 13) 异步填充
- K_ITERATION = K_SIZE / K_TILE = 128 / 32 = 4 次迭代

### 2.2 输出 Tensor (SMEM, 单缓冲)

| Tensor | 形状 | 数量 | 数据类型 | 布局 |
|--------|------|------|---------|------|
| `sKG_inter` | `[SUB_T_TILE=16, K_TILE=32]` | 6 个 | `tf32` | `Layout_K_SW128_Atom` |
| `sKG_intra` | `[SUB_T_TILE=16, K_TILE=32]` | 4 个 | `tf32` | `Layout_K_SW128_Atom` |

### 2.3 Forward vs Backward B-matrix 布局差异

Forward 计算 `Q/K @ K^T`，Backward 计算 `dAqk/dAkk @ K`。
两者 B-matrix 分别是 K^T 和 K，导致 MMA shape 和 SMEM 布局不同：

| | Forward (B = K^T) | Backward (B = K) |
|--|-------------------|-------------------|
| 计算 | `Q/K @ K^T` | `dAqk/dAkk @ K` |
| MMA 形状 (M×N×K) | 64 × X × 32，reduce head dim (K=32) | 64 × 32 × X，reduce chunk dim (K=SUB_T_TILE) |
| B-matrix 内容 | K^T (转置) | K (非转置) |
| B-matrix 形状 | (N × K) = (SUB_T_TILE × K_TILE) = (16 × 32) | (K × N) = (SUB_T_TILE × K_TILE) = (16 × 32) |
| SMEM 布局 | K-major `Layout_K_SW128_Atom` | MN-major `Layout_MN_SW128_32B_Atom` |
| 存储顺序 | `sKG(x_local, y)` | `sKG(y, x_local)` |

注：两者 B-matrix 的物理形状都是 `(16 × 32)` 即 `(SUB_T_TILE × K_TILE)`，
但 forward 的 16 对应 MMA 的 N 维度（chunk），32 对应 K 维度（head dim reduce）；
backward 的 16 对应 MMA 的 K 维度（chunk reduce），32 对应 N 维度（head dim）。
因此同样的数据需要不同的 SMEM 布局（K-major vs MN-major）来匹配 UMMA 的读取模式。

### 2.4 Buffer 索引映射

```
inter[0] → (i=1, j=0)    inter[3] → (i=3, j=0)
inter[1] → (i=2, j=0)    inter[4] → (i=3, j=1)
inter[2] → (i=2, j=1)    inter[5] → (i=3, j=2)

intra[0] → (i=0, j=0)    intra[2] → (i=2, j=2)
intra[1] → (i=1, j=1)    intra[3] → (i=3, j=3)
```

输出地址计算：`&sKG_inter(x_local, y) + KG_OFFSET * index`
- `KG_OFFSET = SUB_T_TILE * K_TILE = 16 * 32 = 512` (tf32 元素间距)
- `index` 即上面的 inter[0..5] / intra[0..3] 编号

## 3. 线程映射

### 3.1 CE Warpgroup 结构

CE 由 2 个 warpgroup 组成，共 256 线程：
- **WG0**: `threadIdx.x ∈ [0, 128)` → `wg_idx = 0`
- **WG1**: `threadIdx.x ∈ [128, 256)` → `wg_idx = 1`

每个 WG 内：
```cpp
int idx_in_warpgroup = threadIdx.x % 128;  // 0..127
int wg_idx = threadIdx.x / 128;            // 0 or 1
```

### 3.2 每个线程的数据负责范围

128 个线程覆盖一个 `[16, 32]` 的 sub_tile：

```cpp
int x_local = idx_in_warpgroup / 8;    // 行 0..15 (sub_tile 内)
int y       = idx_in_warpgroup % 8 * 4; // 列 0, 4, 8, 12, 16, 20, 24, 28
```

- 每个线程写 4 个连续 tf32 值 (128 bits)
- 16 行 × 8 列组 = 128 个线程，恰好覆盖 `16 × 32` sub_tile
- 使用 `store_128b` 写入 SMEM

### 3.3 线程映射示意

```
idx_in_warpgroup:  0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | ... | 120..127
x_local (row):     0  0  0  0  0  0  0  0 | 1  1  1  1  1  1  1  1 | ... |  15..15
y (col group):     0  4  8 12 16 20 24 28 | 0  4  8 12 16 20 24 28 | ... |   0..28
写入位置:        [0,0:3] [0,4:7] ...      [1,0:3] [1,4:7] ...            [15,24:27]
```

### 3.4 SMEM 加载模式

每个 helper 函数中，线程加载以下数据：

| 数据 | 加载指令 | 大小 | 说明 |
|------|---------|------|------|
| `g[x, y..y+3]` | `float4` load | 128 bits | 当前行 gate 值 |
| `k[x, y..y+3]` | `nvbf16x4` load | 64 bits | 当前行 key 值 (4 个 bf16) |
| `g_ref[ref_row, y..y+3]` | `float4` load | 128 bits | 参考 gate 值 (g_first 或 g_half) |

## 4. 按列处理 (Column-Based) 设计

### 4.1 核心思想

同一列 j 的所有块 `(i, j)` (i ≥ j) **共享相同的 K 数据行** (`K_j[x]`, x ∈ [j*16, j*16+15])。
按列处理意味着：只加载一次 `K_j + G`，产出该列的全部输出。

```
Column j=0: 加载 K_0+G → intra(0,0), inter(1,0), inter(2,0), inter(3,0)  [4 个输出]
Column j=1: 加载 K_1+G → intra(1,1), inter(2,1), inter(3,1)              [3 个输出]
Column j=2: 加载 K_2+G → intra(2,2), inter(3,2)                          [2 个输出]
Column j=3: 加载 K_3+G → intra(3,3)                                      [1 个输出]
```

### 4.2 为什么按列而不按行

**按行处理**的问题：行 i 的多个块 `(i,0), (i,1), ..., (i,i-1)` 使用**不同**的 K 数据 (K_0, K_1, ...)，无法复用 SMEM 加载。

**按列处理**的优势：列 j 的多个块 `(j,j), (j+1,j), ..., (3,j)` 使用**相同**的 K_j 数据，区别仅在 `g_ref` (g_first_i 或 g_half_j)。g_ref 是标量广播读取（所有线程读同一行），开销很小。

### 4.3 SMEM 带宽节省

| 设计 | K+G 加载次数 | SMEM 读取 (K data) |
|------|-------------|-------------------|
| 朴素 (10 个独立 helper) | 10 次 | 每输出 1 次 |
| 按列融合 (4 个 helper) | **4 次** | 每列仅 1 次 |
| **节省** | **60%** | |

## 5. Work 分配策略

### 5.1 目标

将 10 个输出 (4 列) 分配给 2 个 WG，满足：
1. **负载均衡**：每个 WG 处理相同数量的输出
2. **数据复用**：同列输出由同一 WG 处理（不拆分列）
3. **无空闲**：没有 WG 在等待

### 5.2 分配方案

```
WG0: col0 (4 outputs) + col3 (1 output) = 5 outputs
WG1: col1 (3 outputs) + col2 (2 outputs) = 5 outputs
```

| WG | 处理的列 | 输出数 | 调用的 Helper |
|----|---------|--------|--------------|
| WG0 | col0, col3 | 4 + 1 = **5** | `fwd_setup_kg_col0_4out`, `fwd_setup_kg_col3_1out` |
| WG1 | col1, col2 | 3 + 2 = **5** | `fwd_setup_kg_col1_3out`, `fwd_setup_kg_col2_2out` |

**完美均衡**: 5:5 输出比例。

### 5.3 为什么是 {col0, col3} / {col1, col2} 而非其他组合

所有合法的 2 分组方案（不拆列，每组 5 个输出）：

| 分组 | WG0 | WG1 | 均衡? |
|------|-----|-----|-------|
| {0,3} / {1,2} | 4+1=5 | 3+2=5 | ✅ |
| {0} / {1,2,3} | 4 | 3+2+1=6 | ❌ (4:6) |
| {0,2} / {1,3} | 4+2=6 | 3+1=4 | ❌ (6:4) |
| {0,1} / {2,3} | 4+3=7 | 2+1=3 | ❌ (7:3) |

**唯一**的完美均衡方案就是 `{col0, col3} / {col1, col2}`。

### 5.4 与 Backward 分配的对比

Backward intra 使用**按行**分配 (因为 backward 的 B-matrix 公式中，同行块共享 g_ref)：
- WG0: row 0 + row 3 = 1 + 4 = 5 outputs
- WG1: row 1 + row 2 = 2 + 3 = 5 outputs

Forward intra 使用**按列**分配 (同列块共享 K 数据)：
- WG0: col 0 + col 3 = 4 + 1 = 5 outputs
- WG1: col 1 + col 2 = 3 + 2 = 5 outputs

两者都是 `{0,3}/{1,2}` 的分组模式，但方向不同（行 vs 列）。

## 6. Helper 函数详解

### 6.1 `fwd_setup_kg_col0_4out` — 列 j=0, 4 个输出

**功能**: 加载 K_0 + G 数据一次，产出 intra(0,0) + inter(1,0) + inter(2,0) + inter(3,0)。

```
输出:
  intra(0,0) → sKG_intra + KG_OFFSET * 0  (g_ref = g_half_0 = g[0*16+8])
  inter(1,0) → sKG_inter + KG_OFFSET * 0  (g_ref = g_first_1 = g[1*16])
  inter(2,0) → sKG_inter + KG_OFFSET * 1  (g_ref = g_first_2 = g[2*16])
  inter(3,0) → sKG_inter + KG_OFFSET * 3  (g_ref = g_first_3 = g[3*16])
```

**执行流程**:

```
Step 1: 加载 4 个 g_ref 值 (SMEM → 寄存器)
  g_half_0  = sG[min(8, sub_seq_len-1), y..y+3]     // 4 floats
  g_first_1 = sG[min(16, sub_seq_len-1), y..y+3]    // 4 floats
  g_first_2 = sG[min(32, sub_seq_len-1), y..y+3]    // 4 floats
  g_first_3 = sG[min(48, sub_seq_len-1), y..y+3]    // 4 floats

Step 2: 加载本线程负责行的 K+G 数据 (SMEM → 寄存器)
  x = x_local + 0*16   // 行 0..15 (column j=0 的 K 数据)
  g   = sG[x, y..y+3]  // float4: 当前行 gate (4 floats)
  k   = sK[x, y..y+3]  // nvbf16x4: 当前行 key (4 bf16 → 4 floats)
  kf_a = bf16_to_f32(k.a), kf_b = bf16_to_f32(k.b)

Step 3: 对每个 g_ref 计算 exp2(g_ref - g) * K 并写 SMEM
  // 输出 1: intra(0,0)
  s = exp2(g_half_0 - g)    // 4 个 exp2f
  res = s * kf              // 4 个 float mul
  store_128b → sKG_intra + KG_OFFSET * 0

  // 输出 2: inter(1,0)
  s = exp2(g_first_1 - g)
  res = s * kf
  store_128b → sKG_inter + KG_OFFSET * 0

  // 输出 3: inter(2,0)
  s = exp2(g_first_2 - g)
  res = s * kf
  store_128b → sKG_inter + KG_OFFSET * 1

  // 输出 4: inter(3,0)
  s = exp2(g_first_3 - g)
  res = s * kf
  store_128b → sKG_inter + KG_OFFSET * 3
```

**寄存器开销 (live registers)**:
- 4 个 g_ref × 4 floats = 16 floats
- kf_a, kf_b = 4 floats
- g_a, g_b = 4 floats
- 临时 s1, s2, res = ~8 floats
- **总计 ~32 floats ≈ 32 registers**

### 6.2 `fwd_setup_kg_col1_3out` — 列 j=1, 3 个输出

**功能**: 加载 K_1 + G 数据一次，产出 intra(1,1) + inter(2,1) + inter(3,1)。

```
输出:
  intra(1,1) → sKG_intra + KG_OFFSET * 1  (g_ref = g_half_1 = g[1*16+8])
  inter(2,1) → sKG_inter + KG_OFFSET * 2  (g_ref = g_first_2 = g[2*16])
  inter(3,1) → sKG_inter + KG_OFFSET * 4  (g_ref = g_first_3 = g[3*16])
```

**SMEM 加载**: K data 行 x ∈ [16, 31]，3 个 g_ref 值。  
**寄存器开销**: ~28 floats ≈ 28 registers.

### 6.3 `fwd_setup_kg_col2_2out` — 列 j=2, 2 个输出

**功能**: 加载 K_2 + G 数据一次，产出 intra(2,2) + inter(3,2)。

```
输出:
  intra(2,2) → sKG_intra + KG_OFFSET * 2  (g_ref = g_half_2 = g[2*16+8])
  inter(3,2) → sKG_inter + KG_OFFSET * 5  (g_ref = g_first_3 = g[3*16])
```

**SMEM 加载**: K data 行 x ∈ [32, 47]，2 个 g_ref 值。  
**寄存器开销**: ~24 floats ≈ 24 registers.

### 6.4 `fwd_setup_kg_col3_1out` — 列 j=3, 1 个输出

**功能**: 加载 K_3 + G 数据一次，产出 intra(3,3)。

```
输出:
  intra(3,3) → sKG_intra + KG_OFFSET * 3  (g_ref = g_half_3 = g[3*16+8])
```

**SMEM 加载**: K data 行 x ∈ [48, 63]，1 个 g_ref 值。  
**寄存器开销**: ~16 floats ≈ 16 registers.

## 7. 边界处理

### 7.1 sub_seq_len 短序列处理

当序列长度不是 T_TILE=64 的整数倍时，`sub_seq_len < 64`：

- **g_ref 加载**: `min(ref_row, sub_seq_len - 1)` 确保不越界
- **K data 行**: `if (x < sub_seq_len)` 条件保护
  - 有效行：正常计算 `exp2(g_ref - g) * K`
  - 越界行：写零 (`float4 z = {0, 0, 0, 0}`)

### 7.2 越界写零的必要性

即使行越界，也必须将对应 SMEM 位置清零，因为 MMA 会读取整个 `[16, 32]` 的 B-matrix sub_tile。未清零的陈旧数据会导致计算错误。

## 8. 完整执行时序

```
每个 tile (batch, head, tile_idx):
  for k_idx in 0..3:                      // K_ITERATION = 4
    [CE] Wait TMA: K, G, Q ready          // Pipeline consumer_wait
    [CE] Create SMEM tensor views          // sK, sG, sKG_inter, sKG_intra

    [CE WG0] col0_4out:                   // 加载 K_0+G, 产出 4 个输出
      Load g_half_0, g_first_1, g_first_2, g_first_3
      Load K_0[row] + G[row]
      Compute & store intra(0,0), inter(1,0), inter(2,0), inter(3,0)
    [CE WG0] col3_1out:                   // 加载 K_3+G, 产出 1 个输出
      Load g_half_3
      Load K_3[row] + G[row]
      Compute & store intra(3,3)

    [CE WG1] col1_3out:                   // (并行于 WG0)
      Load g_half_1, g_first_2, g_first_3
      Load K_1[row] + G[row]
      Compute & store intra(1,1), inter(2,1), inter(3,1)
    [CE WG1] col2_2out:
      Load g_half_2, g_first_3
      Load K_2[row] + G[row]
      Compute & store intra(2,2), inter(3,2)

    [CE] fence_view_async_shared           // 确保 SMEM 写入可见
    [CE] Signal MMA: qkg_inter ready       // Pipeline producer_commit
    [CE] Signal MMA: qkg_intra ready

    [CE] Release K, G, Q buffers           // Pipeline consumer_release
```

## 9. 寄存器压力分析

### 9.1 每个 WG 的最大 live register

| WG | Helper | g_ref regs | data regs | temp regs | 总计 |
|----|--------|-----------|-----------|-----------|------|
| WG0 | col0_4out | 16 (4 g_ref) | 8 (g+kf) | 8 | **~32** |
| WG0 | col3_1out | 4 (1 g_ref) | 8 | 8 | ~20 |
| WG1 | col1_3out | 12 (3 g_ref) | 8 | 8 | **~28** |
| WG1 | col2_2out | 8 (2 g_ref) | 8 | 8 | ~24 |

- 两个 helper 串行执行，不叠加寄存器
- WG0 峰值 ~32 registers，WG1 峰值 ~28 registers
- SM100 每线程可用 ~255 registers，远低于上限

### 9.2 TODO: 寄存器压力优化

`col0_4out` 是寄存器最重的 helper（4 个 g_ref = 16 floats live）。如果后续与 A-matrix prologue 叠加后寄存器溢出，可以考虑：
1. 将 col0 拆成 col0_2out + col0_2out_rest (各 2 个输出)
2. 将 g_ref 中的 `g_first_2` / `g_first_3` 延迟加载（在写完 intra(0,0) 和 inter(1,0) 后再读）

目前作为 TODO，等性能跑出来后根据实际 register spill 情况决定。

## 10. 与 Backward B-matrix 的对比

| 维度 | Forward | Backward |
|------|---------|----------|
| 计算 | `Q/K @ K^T` | `dAqk/dAkk @ K` |
| MMA 形状 (M×N×K) | 64 × X × 32，reduce head dim | 64 × 32 × X，reduce chunk dim |
| B-matrix 内容 | K^T (转置) | K (非转置) |
| B-matrix 公式 | `exp2(g_ref - g[x]) * K[x]` | `exp2(g[x] - g_ref) * K[x]` (neg_exp) |
| 处理方向 | 按列 (column j) | 按行 (row i) |
| 共享数据 | K_j data (同列共享) | g_ref (同行共享 gn_half) |
| 分组 | {col0,col3}/{col1,col2} | {row0,row3}/{row1,row2} |
| 输出均衡 | 5:5 | 5:5 |
| SMEM 布局 | K-major `(x_local, y)` | MN-major `(y, x_local)` |
| 额外输出 | 无 | sBkExp + sBkNegExp |

## 11. 文件索引

| 文件 | 内容 |
|------|------|
| `csrc/kda_bwd/fwd_util_func.h` | 4 个 column-based helper 函数定义 |
| `csrc/kda_bwd/kda_fwd_intra_mainloop_sm100.hpp` | CE loop 中 Step 3 调用 helper |
| `csrc/kda_bwd/util_func.h` | Backward 对应的 helper 函数 (对比参考) |
| `csrc/kda_bwd/kda_bwd_intra_sm100.cu` | Backward mainloop (对比参考) |
