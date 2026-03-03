# KDA Backward Intra: KG Intra/Inter 计算逻辑详解

## 1. 数学背景

backward intra kernel 的核心目标是计算 `dQ`（对 Q 的梯度）。公式为：

```
dQ[i] = sum_j { dA[i,j] * exp2f(g[i] - g[j]) * K[j] }    (j <= i, 同一 tile 内)
```

一个 tile 有 T_TILE=64 行，被分成 4 个 sub-chunk（每个 SUB_T_TILE=16 行）：

| Sub-chunk | 行范围 |
|-----------|--------|
| S0 | [0, 16) |
| S1 | [16, 32) |
| S2 | [32, 48) |
| S3 | [48, 64) |

对于 sub-chunk `Si` 中的行 `i`，其 dQ 贡献来自两部分：

- **kg_intra**（下三角，跨 sub-chunk）：`j` 在 `S0..S(i-1)` 中的所有行
- **kg_inter**（块对角）：`j` 在同一个 sub-chunk `Si` 中且 `j <= i`

两者分别对应不同的 B 矩阵准备方式和 MMA 调用。

---

## 2. 指数分解技巧

直接计算 `exp2f(g[i] - g[j])` 需要对每个 `(i,j)` 对单独求指数，无法矩阵化。
核心技巧是引入参考点 `gn` 将指数拆成两部分：

```
exp2f(g[i] - g[j]) = exp2f(g[i] - gn) * exp2f(gn - g[j])
```

- `exp2f(gn - g[j]) * K[j]` → 预计算为 B 矩阵（CE warp 准备）
- `exp2f(g[i] - gn)` → 后处理 scale（Epilogue 阶段应用）

**kg_intra 的参考点**：`gn = g[Si 的起始行]`，即消费者 sub-chunk 的第一行的 gate 值。
**kg_inter 的参考点**：`gn_half = g[Si*16 + 8]`，即 sub-chunk 的中间行的 gate 值。

---

## 3. KG Intra：CE 侧 B 矩阵准备

### 3.1 公式

对于 sub-chunk `Si` 从 sub-chunk `Sj`（j < i）读取的贡献：

```
B_intra[Sj→Si] = exp2f(g[Si_start] - g[rows_of_Sj]) * K[rows_of_Sj]
```

其中 `Si_start` 是消费者 sub-chunk 的起始行号（16, 32, 48）。

### 3.2 下三角结构

kg_intra 只处理跨 sub-chunk 的下三角部分，共 6 个 (消费者, 生产者) 对：

```
         S0    S1    S2    S3
    S0 [inter]  -     -     -
    S1 [ ✓ ] [inter]  -     -
    S2 [ ✓ ] [ ✓ ] [inter]  -
    S3 [ ✓ ] [ ✓ ] [ ✓ ] [inter]
```

对应 `kg_all.intra[6]` 的 6 个 slot：

| Index | 消费者→生产者 | gn | 行范围 | 由谁计算 |
|-------|-------------|-----|--------|---------|
| 0 | S1←S0 | g[16] | rows [0,16) | WG1: `setup_kg_intra_2gn` |
| 1 | S2←S0 | g[32] | rows [0,16) | WG1: `setup_kg_intra_2gn` |
| 2 | S2←S1 | g[32] | rows [16,32) | WG1: `setup_kg_intra` |
| 3 | S3←S0 | g[48] | rows [0,16) | WG0: `setup_kg_intra` |
| 4 | S3←S1 | g[48] | rows [16,32) | WG0: `setup_intra_fused` |
| 5 | S3←S2 | g[48] | rows [32,48) | WG0: `setup_intra_fused` |

### 3.3 CE 代码映射

**WG1** 负责 index 0, 1, 2（消费者为 S1, S2）：

```cpp
// WG1:
float4 gn1 = sG(16, y);  // g[16] = S1 起始
float4 gn2 = sG(32, y);  // g[32] = S2 起始

// tile_j=0, 两个 gn 共享同一行数据 → 一次 load 两次输出
setup_kg_intra_2gn(tile_j=0, gn1, gn2, index1=0, index2=1);
//   index 0: exp2f(g[16] - g[0..15]) * k[0..15]
//   index 1: exp2f(g[32] - g[0..15]) * k[0..15]

// tile_j=1
setup_kg_intra(tile_j=1, gn2, index=2);
//   index 2: exp2f(g[32] - g[16..31]) * k[16..31]
```

**WG0** 负责 index 3, 4, 5（消费者为 S3）：

```cpp
// WG0:
float4 gn3 = sG(48, y);  // g[48] = S3 起始

// tile_j=0, 仅 kg（无对应 qkg）
setup_kg_intra(tile_j=0, gn3, index=3);
//   index 3: exp2f(g[48] - g[0..15]) * k[0..15]

// tile_j=1, fused kg+qkg（共享 sK/sG load）
setup_intra_fused(tile_j=1, gn_kg=gn3, ..., kg_index=4, ...);
//   index 4: exp2f(g[48] - g[16..31]) * k[16..31]

// tile_j=2, fused kg+qkg
setup_intra_fused(tile_j=2, gn_kg=gn3, ..., kg_index=5, ...);
//   index 5: exp2f(g[48] - g[32..47]) * k[32..47]
```

### 3.4 SMEM 布局

每个 `kg_all.intra[i]` 是 `SmemLayoutMatBTF32Tranposed<1>`，即 `[K_TILE=32 × SUB_T_TILE=16]` 的转置 TF32 矩阵。
存储方式为列主序（MMA 的 B 矩阵要求 Major::MN）。

`setup_kg_intra` 中的写入模式：
```cpp
// 每个线程处理 1 行（x = idx_in_warpgroup/8 + tile_j*16）
// 写入 4 个 float（y = idx_in_warpgroup%8 * 4）
store_128b(&sKG_intra(y, idx_in_warpgroup/8) + KG_OFFSET * index, res);
```

128 个线程覆盖 16 行 × 32 列（每线程 4 列）。

---

## 4. KG Intra：MMA 侧矩阵乘法

### 4.1 MMA 操作概述

MMA warp 执行 `dq = dA @ kg_intra`，其中：
- A 矩阵 = `dAqk`（已 mask 的注意力梯度），存储在 TMEM，形状 [64 × 64]
- B 矩阵 = `kg_all.intra[...]`，存储在 SMEM
- C 矩阵 = `dq`（输出），存储在 TMEM，形状 [64 × 32]

### 4.2 TMEM Lane Offset 机制

SM100 TMEM 地址编码：`addr = col_offset + lane * 65536`

通过 lane offset 实现行循环移位：
- `dAqk_02 = dAqk`：MMA row r 读取 dA[r]
- `dAqk_13 = dAqk + 16*65536`：MMA row r 读取 dA[(r+16) % 64]

输出同理：
- `tDQ_02 = dq`：MMA row r 写入 dq[r]
- `tDQ_13 = dq + 16*65536`：MMA row r 写入 dq[(r+16) % 64]

### 4.3 MASK 语义

SM100 tcgen05 MMA 支持 disable-output-lane mask，控制 64 行中哪些 16 行 chunk 被写入：

| MASK | 启用的输出行 | 对应 sub-chunk |
|------|------------|---------------|
| MASK02 | [0,16) + [32,48) | S0 + S2 |
| MASK13 | [16,32) + [48,64) | S1 + S3 |

### 4.4 三次 MMA 调用详解

```
kg_intra MMA Call 1:
  utcmma_ts(MASK02, A=dAqk_13, B=intra[0][16cols], C=tDQ_13, clear=true)
```

| 启用行 | A 读取 (lane+16) | C 写入 (lane+16) | B 列数 | 效果 |
|--------|-----------------|-----------------|--------|------|
| [0,16) | dA[16..31] | dq[16..31] | 16 (S0) | dq[S1] = dA[S1,S0] @ kg[0] |
| [32,48) | dA[48..63] | dq[48..63] | 16 (S0) | dq[S3] = dA[S3,S0] @ kg[0] |

注意：S1 和 S3 共用同一个 B 矩阵（gn=g[16]），S3 的 gn 不匹配问题由 Epilogue scale 修正（见第 6 节）。

```
kg_intra MMA Call 2:
  utcmma_ts(MASK13, A=dAqk_02, B=intra[1..2][32cols], C=tDQ_02, clear=true)
```

| 启用行 | A 读取 (lane+0) | C 写入 (lane+0) | B 列数 | 效果 |
|--------|----------------|----------------|--------|------|
| [16,32) | dA[16..31] | dq[16..31] | 32 (S0+S1) | dq[S1] += dA[S1,S0:S1] @ kg[1..2] |
| [48,64) | dA[48..63] | dq[48..63] | 32 (S0+S1) | dq[S3] += dA[S3,S0:S1] @ kg[1..2] |

注意：B 矩阵 `intra[1..2]` 是 32 列（`SmemLayoutMatBTF32Tranposed<2>`），包含 S0 和 S1 的数据，gn=g[32]。

```
kg_intra MMA Call 3:
  utcmma_ts(MASK13, A=dAqk_13, B=intra[3..5][48cols], C=tDQ_13, accumulate)
```

| 启用行 | A 读取 (lane+16) | C 写入 (lane+16) | B 列数 | 效果 |
|--------|-----------------|-----------------|--------|------|
| [16,32) | dA[32..47] | dq[32..47] | 48 (S0+S1+S2) | dq[S2] = dA[S2,S0:S2] @ kg[3..5] |
| [48,64) | dA[0..15] | dq[0..15] | 48 (S0+S1+S2) | (masked out by dA=0) |

注意：Call 3 对 tDQ_13 的 [16,32) 行是首次写入（Call 1 写的是 [0,16) 和 [32,48)），所以 clear=true 不会覆盖 Call 1 的结果。行 [48,64) 映射到 dq[0..15]（S0），但 dA[S0, S0:S2] 的下三角 mask 使得这些值为 0，不影响结果。

### 4.5 汇总：kg_intra MMA 后各 sub-chunk 的 dq 值

经过 3 次 MMA 调用后，TMEM 中 dq 的内容（未经 Epilogue scale）：

| Sub-chunk | TMEM 位置 | 累积内容 |
|-----------|----------|---------|
| S0 (rows 0-15) | — | 无 kg_intra 贡献（S0 没有更低的 sub-chunk） |
| S1 (rows 16-31) | tDQ_13[0:16] + tDQ_02[16:32] | Call1: dA[S1,S0]@kg[0](gn=16) + Call2: dA[S1,S0:S1]@kg[1..2](gn=32) |
| S2 (rows 32-47) | tDQ_13[16:32] | Call3: dA[S2,S0:S2]@kg[3..5](gn=48) |
| S3 (rows 48-63) | tDQ_13[32:48] + tDQ_02[48:64] | Call1: dA[S3,S0]@kg[0](gn=16) + Call2: dA[S3,S0:S1]@kg[1..2](gn=32) |

注意 S1 和 S3 各有两次 MMA 写入到不同的 TMEM 地址（tDQ_02 和 tDQ_13），Epilogue 分别读取并合并。

---

## 5. KG Inter：块对角部分

### 5.1 公式

kg_inter 处理同一 sub-chunk 内的贡献（块对角），参考点为 sub-chunk 中间行 `gn_half = g[Si*16 + 8]`：

```
B_inter[Si] = exp2f(-(g[rows_of_Si] - gn_half)) * K[rows_of_Si]
            = exp2f(gn_half - g[rows_of_Si]) * K[rows_of_Si]
```

注意符号：inter 使用 `neg_exp`（负指数），因为参考点在 sub-chunk 中间，行可能在参考点之前或之后。

### 5.2 CE 侧准备

`kg_all.inter[4]` 有 4 个 slot，每个 sub-chunk 一个：

| Index | Sub-chunk | gn_half | 由谁计算 |
|-------|-----------|---------|---------|
| 0 | S0 | g[8] | WG0: `setup_inter_fused` |
| 1 | S1 | g[24] | WG1: `setup_inter_fused` |
| 2 | S2 | g[40] | WG1: `setup_inter_fused` |
| 3 | S3 | g[56] | WG0: `setup_inter_fused` |

`setup_inter_fused` 同时计算 kg_inter 和 qkg_inter（共享 sK/sG/sQ load）：

```cpp
// setup_inter_fused 核心逻辑：
gn_half = sG(min(sub_tile_i * 16 + 8, sub_seq_len - 1), y);
float2 sub = g[x] - gn_half;

// kg_inter: neg_exp * k
res_neg_exp = exp2f(-sub);
res_kg = res_neg_exp * k;
store → sKG_inter

// qkg_inter: exp * q  和  exp * k * beta
res_exp = exp2f(sub);
res_q = res_exp * q;
store → sQKG_inter (q part)
res_kbeta = res_exp * k * beta;
store → sQKG_inter (k part)
```

### 5.3 WG 分工

```cpp
// WG0 处理 S0 和 S3:
setup_inter_fused(sub_tile_i=0, beta[0], gn_half_0);  // S0
setup_inter_fused(sub_tile_i=3, beta[3], gn_half_3);  // S3

// WG1 处理 S1 和 S2:
setup_inter_fused(sub_tile_i=1, beta[1], gn_half_1);  // S1
setup_inter_fused(sub_tile_i=2, beta[2], gn_half_2);  // S2
```

### 5.4 MMA 侧：4 次调用

kg_inter 的 MMA 输出写入 `tDQ2`（与 kg_intra 的 `tDQ` 分开），后续在 Epilogue 合并。

```
kg_inter MMA Call 1:
  utcmma_ts(MASK02, A=dAqk_02, B=inter[0][16cols], C=tDQ2_02, clear=true)
```

| 启用行 | A 读取 | C 写入 | 效果 |
|--------|--------|--------|------|
| [0,16) | dA[0..15] | dq2[0..15] | dq2[S0] = dA[S0,S0] @ kg_inter[0] |
| [32,48) | dA[32..47] | dq2[32..47] | dq2[S2] = dA[S2,S2] @ kg_inter[0] |

注意：S2 使用了 S0 的 B 矩阵（gn_half=g[8]），gn 不匹配由 Epilogue scale 修正。

```
kg_inter MMA Call 2:
  utcmma_ts(MASK02, A=dAqk_13+16, B=inter[1][16cols], C=tDQ2_13, clear=true)
```

A 地址 `dAqk_13 + 16` 表示 lane+16 再加 16 列偏移，读取 dA 的不同列区域。

| 启用行 | A 读取 (lane+16, col+16) | C 写入 (lane+16) | 效果 |
|--------|------------------------|-----------------|------|
| [0,16) | dA[16..31, 16..31] | dq2[16..31] | dq2[S1] = dA[S1,S1] @ kg_inter[1] |
| [32,48) | dA[48..63, 16..31] | dq2[48..63] | dq2[S3] = dA[S3,S3] @ kg_inter[1] |

```
kg_inter MMA Call 3:
  utcmma_ts(MASK13, A=dAqk_02+32, B=inter[2][16cols], C=tDQ2_02, accumulate)
```

| 启用行 | A 读取 (col+32) | C 写入 | 效果 |
|--------|----------------|--------|------|
| [16,32) | dA[16..31, 32..47] | dq2[16..31] | dq2[S1] += dA[S1,S1'] @ kg_inter[2] |
| [48,64) | dA[48..63, 32..47] | dq2[48..63] | dq2[S3] += dA[S3,S3'] @ kg_inter[2] |

```
kg_inter MMA Call 4:
  utcmma_ts(MASK13, A=dAqk_13+48, B=inter[3][16cols], C=tDQ2_13, accumulate)
```

| 启用行 | A 读取 (lane+16, col+48) | C 写入 (lane+16) | 效果 |
|--------|------------------------|-----------------|------|
| [16,32) | dA[32..47, 48..63] | dq2[32..47] | dq2[S2] += dA[S2,S2'] @ kg_inter[3] |
| [48,64) | dA[0..15, 48..63] | dq2[0..15] | (dA masked → 0) |

### 5.5 汇总：kg_inter MMA 后各 sub-chunk 的 dq2 值

| Sub-chunk | TMEM 位置 | 累积内容 |
|-----------|----------|---------|
| S0 | tDQ2_02[0:16] | Call1: dA[S0,S0] @ kg_inter[0] (gn_half=g[8]) |
| S1 | tDQ2_13[0:16] + tDQ2_02[16:32] | Call2 + Call3 |
| S2 | tDQ2_02[32:48] + tDQ2_13[16:32] | Call1 + Call4 |
| S3 | tDQ2_13[32:48] + tDQ2_02[48:64] | Call2 + Call3 |

同样，S1/S3 和 S0/S2 各有两个 TMEM 地址的结果需要在 Epilogue 合并。

---

## 6. Epilogue：Scale 修正与结果合并

### 6.1 kg_intra 的 scale 修正

MMA 输出的 dq 使用了 B 矩阵中的 gn（消费者 sub-chunk 起始行），但实际需要的是 `exp2f(g[i] - g[j])`。

Epilogue 计算 `intra_scale = exp2f(g[i] - g[block_start])`，其中 `block_start = (i/16)*16`：

```cpp
epilogue_compute_intra_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, scale);
```

然后将 MMA 结果乘以 scale：

```cpp
// 读取 tDQ（kg_intra 结果）并乘以 intra_scale
epilogue_apply_dq_intra<HALF_K>(idx_in_warpgroup, tmem_addr::dq + K_OFF, res, scale);
```

数学验证（以 S1 行 i 为例）：
```
result = intra_scale * MMA_output
       = exp2f(g[i] - g[16]) * dA[i, 0..15] @ (exp2f(g[16] - g[0..15]) * K[0..15])
       = dA[i, 0..15] @ (exp2f(g[i] - g[0..15]) * K[0..15])  ✓
```

### 6.2 kg_inter 的 scale 修正

kg_inter 使用 `gn_half = g[Si*16+8]` 作为参考点。Epilogue 计算：

```
inter_scale = exp2f(g[i] - gn_half)
```

```cpp
// 从 sG 直接计算 inter scale
int g_half_row = min(row / 16 * 16 + 8, sub_seq_len - 1);
for (...) {
    scale[i] = exp2f(sG(row, ...) - sG(g_half_row, ...));
}
```

然后合并到 intra 结果中：

```cpp
// 读取 tDQ2（kg_inter 结果），乘以 inter_scale，加到 res 上
epilogue_combine_dq_inter<HALF_K>(tmem_addr::dq2 + K_OFF, res, scale);
// res += dq2 * inter_scale
```

### 6.3 双 TMEM 地址合并

对于 S1 和 S3，kg_intra 的结果分布在 tDQ_02 和 tDQ_13 两个 TMEM 地址。
Epilogue 分两步读取：

1. `epilogue_apply_dq_intra` 读取 `tmem_addr::dq`（对应 tDQ_02 或 tDQ_13 取决于 buf_idx_value）
2. `epilogue_combine_dq_inter` 读取 `tmem_addr::dq2`（对应 tDQ2_02 或 tDQ2_13）

两者使用不同的 scale 分别处理后相加，得到最终的 dQ。

---

## 7. 完整数据流图

```
CE Warp (256 threads)                    MMA Warp (1 elected thread)
========================                 ========================

[1] Wait bar_load_kg_ready
    Load sK, sG from SMEM

[2] Compute kg_intra B-matrices
    WG0: intra[3,4,5] (gn=g[48])
    WG1: intra[0,1,2] (gn=g[16],g[32])
                                         [3] Wait bar_kg_all_ready
[2'] Wait bar_load_qb                       kg_intra: 3× MMA → tDQ
     Load sQ from SMEM
                                             kg_inter: 4× MMA → tDQ2
[4] Compute kg_inter B-matrices
    WG0: inter[0,3]                          Signal bar_dq_done
    WG1: inter[1,2]                      ─────────────────────────

    Signal bar_kg_all_ready
    ─────────────────────

[5] Compute intra_scale from sG

[6] Wait bar_dq_done
    Read tDQ from TMEM
    Apply intra_scale → res

[7] Compute inter_scale from sG
    Read tDQ2 from TMEM
    res += tDQ2 * inter_scale

[8] Output dQ / accumulate dB
```

---

## 8. 优化技巧总结

1. **指数分解**：将 `exp2f(g[i]-g[j])` 拆为 B 矩阵预计算 + Epilogue scale，使得核心计算可以用 MMA 加速。

2. **MASK 复用**：MASK02/MASK13 每次写入两个 sub-chunk，一次 MMA 调用服务两个消费者（如 S1+S3 共用 B 矩阵），减少 MMA 调用次数。

3. **Lane offset**：通过 TMEM lane offset 实现 A 矩阵的行移位，无需实际搬运数据即可让 MMA 读取 dA 的不同行区域。

4. **Fused compute**：`setup_intra_fused` 和 `setup_inter_fused` 将 kg 和 qkg 的 B 矩阵准备融合，共享 sK/sG/sQ 的 SMEM load，减少带宽压力。

5. **双 TMEM 累加器**：tDQ/tDQ2 分别存储 intra/inter 结果，避免 MMA 之间的依赖，允许 kg_intra 和 kg_inter 的 MMA 连续发射。

---

## 9. 异步重叠：MMA 与 CUDA Core 的流水线分析

### 9.1 SM100 vs Ampere 的根本区别

Ampere 上 MMA 是 warp 级同步指令（`mma.sync`），发射 MMA 的 warp 必须等结果返回才能继续。
同一个 warp 无法在 MMA 执行期间做其他事，只能靠多 warp 交错来隐藏延迟，但 warp 数量受寄存器压力限制。

SM100 (Blackwell) 的 `tcgen05` MMA 是根本不同的执行模型：
- MMA 由**单独的 elected thread** 发射，发射后立即返回（fire-and-forget）
- MMA 结果写入 **TMEM**（独立于寄存器文件），通过 mbarrier 通知完成
- CE warp 使用完全独立的 CUDA core 执行单元

本质上是**两套硬件在并行工作**：

```
硬件资源      │ 使用者         │ 职责
─────────────┼───────────────┼──────────────────────────
Tensor Core  │ MMA warp      │ 7+7 次矩阵乘法
CUDA Core    │ CE warps(256) │ exp2f, scale, output
TMA Engine   │ Load warp     │ 预取下一批数据
SMEM         │ 共享           │ barrier 保护的生产者-消费者
TMEM         │ MMA 写入       │ CE 读取
```

### 9.2 单个 k_idx 迭代内的时间线

```
时间 →

CE (256 threads, CUDA core)              MMA (1 elected thread, Tensor Core)
═══════════════════════════              ═══════════════════════════════════

[A] wait bar_load_kg_ready
    compute kg_intra B-matrices          (idle, waiting bar_kg_all_ready)
    wait bar_load_qb
    compute kg_inter + qkg_inter
    ──arrive bar_kg_all_ready──→
                                         [B] wait bar_kg_all_ready ✓
[C] compute mask_At (k_idx==0)               kg_intra: 3× MMA  ─┐
    compute intra_scale                      kg_inter: 4× MMA   │ 7 MMA calls
    compute qkg_intra                        ──arrive bar_dq_done──→
    ──arrive bar_qkg_all_ready──→
                                         [D] wait bar_qkg_all_ready ✓
[E] wait bar_dq_done ✓                      qkg_intra: 3× MMA  ─┐
    read TMEM dq, apply scale                qkg_inter: 4× MMA   │ 7 MMA calls
    read TMEM dq2, combine                   ──arrive bar_dkt_done──→
    output dQ / accumulate dB
    wait bar_dkt_done ✓
    read TMEM dkt, apply scale
    exchange dkt (WG0↔WG1)
    output dK / dG
    ──arrive bar_dvalue_free──→
```

### 9.3 重叠 1：CE 准备 qkg 数据 ‖ MMA 执行 kg phase

这是最大的收益点。

CE arrive `bar_kg_all_ready` 后，MMA 开始执行 kg_intra (3次) + kg_inter (4次) = 7 次 MMA 调用。
与此同时，CE **不等待 MMA**，立即开始做：

1. `mask_At_tensor`（第一个 k_idx 时，将 dA 转置写入 TMEM）
2. `epilogue_compute_intra_scale`（纯 CUDA core 计算 exp2f）
3. `setup_qkg_intra` 系列（准备下一阶段 qkg 的 B 矩阵）

代码中的注释明确标注了这一点：
```cpp
// === EPILOGUE: compute intra scale (can overlap with MMA kg phase) ===
float scale[HALF_K];
epilogue_compute_intra_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, scale);
```

这段 CE 工作量不小——每个线程要做多次 exp2f + 乘法 + SMEM store，全部被 MMA 的 7 次矩阵乘法延迟所隐藏。

### 9.4 重叠 2：CE 处理 dq 结果 ‖ MMA 执行 qkg phase

MMA arrive `bar_dq_done` 后，CE 开始处理 dq 结果（TMEM 读取 + scale + 输出）。
与此同时，MMA 继续执行 qkg_intra (3次) + qkg_inter (4次) = 又 7 次 MMA 调用。

CE 的 dq epilogue 工作包括：
- `epilogue_apply_dq_intra`：从 TMEM 读取 dq，乘以 intra_scale
- 计算 inter_scale（exp2f）
- `epilogue_combine_dq_inter`：从 TMEM 读取 dq2，乘以 inter_scale 并累加
- `epilogue_output_dq`：写出 dQ（lower half）
- `epilogue_accumulate_db`：累加 dB + beta scaling（upper half）
- DB reduce：WG0 → smem → WG1 跨 warpgroup 归约

这些全部与 qkg 的 7 次 MMA 调用重叠执行。

### 9.5 重叠 3：CE 处理 dkt 结果 ‖ Load warp 预取下一个 k_idx

CE 在处理 dkt 输出（dK/dG）的同时，Load warp 已经在发射下一个 k_idx 的 TMA load。
双缓冲（`NUM_BUF_VALUE=2`）保证 Load 和 CE 不会争抢同一块 SMEM。

```cpp
// CE 侧：处理完当前 k_idx 后释放 buffer
cute::arrive_barrier(shared_plan->bar_dvalue_free[buf_idx_value]);

// Load 侧：等待 buffer 释放后立即发射下一批 TMA
cute::wait_barrier(shared_plan->bar_dvalue_free[buf_idx_value], local_phase^1);
launch_tma_copy(tma_k, gK, sK, bar_load_kg_ready);
launch_tma_copy(tma_g, gG, sG, bar_load_kg_ready);
```

### 9.6 关键路径分析

每个 k_idx 迭代中，如果串行执行（如 Ampere），关键路径为：

```
串行: CE_prepare_kg + 7×MMA_kg + CE_prepare_qkg + 7×MMA_qkg + CE_epilogue
```

而本 kernel 通过异步重叠，关键路径缩短为：

```
异步: CE_prepare_kg + max(7×MMA_kg, CE_prepare_qkg)
                    + max(7×MMA_qkg, CE_epilogue_dq)
                    + CE_epilogue_dkt
```

即 14 次 MMA 调用的延迟几乎被 CE 的 B 矩阵准备 + epilogue 处理完全覆盖。
关键路径大致为 `max(MMA_latency, CE_compute_latency)`，两者高度重叠。
