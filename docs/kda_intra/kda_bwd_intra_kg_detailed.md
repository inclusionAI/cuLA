# KDA Backward Intra - KG 部分详细解读

## 概述

本文档详细分析 `kda_bwd_intra_sm100.cu` 中 Epilogue Warpgroup 对 **KG (K×Gate)** 部分的 CUDA core 处理逻辑。KG 计算是为后续 Tensor Core MMA 准备 B 矩阵数据的关键步骤。

## 1. KG 计算的数学含义

### 1.1 KG Intra 计算
**公式**: `KG_intra[i,j] = exp2(gn - g[i,j]) * k[i,j]`

- `gn`: 参考 gate 值（从特定行读取）
- `g[i,j]`: 当前位置的 gate 值
- `k[i,j]`: 当前位置的 key 值
- 计算的是 intra-chunk 的注意力权重与 key 的乘积

### 1.2 KG Inter 计算
**公式**: `KG_inter[i,j] = exp2(-(g[i,j] - gn_half)) * k[i,j]`

- `gn_half`: 子 tile 中间位置的 gate 值（第 8 行）
- 计算的是 inter-chunk 的注意力权重与 key 的乘积
- 注意符号相反（负指数）

## 2. 数据布局与线程映射

### 2.1 Tensor 维度
```cpp
sG: [T_TILE=64, K_TILE=32]  // Gate tensor (float)
sK: [T_TILE=64, K_TILE=32]  // Key tensor (bf16)
sKG_intra: [K_TILE=32, SUB_T_TILE=16] × 6  // 输出 (tf32)
sKG_inter: [K_TILE=32, SUB_T_TILE=16] × 4  // 输出 (tf32)
```

### 2.2 线程到数据的映射
每个 Warpgroup 有 128 个线程：
```cpp
int x = idx_in_warpgroup / 8 + tile_j * 16;  // 行索引 (0-15 per tile_j)
int y = idx_in_warpgroup % 8 * 4;             // 列索引 (0, 4, 8, ..., 28)
```

- 每个线程处理 4 个连续的 K 维度元素（float4）
- 16 个线程（idx % 8 = 0..7, idx / 8 = 0..1）覆盖一个 16×32 的 sub-tile
- 输出地址：`sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index`
  - `y`: 列坐标（K 维度）
  - `idx_in_warpgroup / 8`: 行坐标（T 维度，0-15）
  - `KG_OFFSET * index`: 不同 tile_j 的偏移

### 2.3 输出布局（转置）
输出矩阵是转置的：
- 输入 sK/sG: `[T_TILE, K_TILE]` = `[64, 32]`
- 输出 sKG: `[K_TILE, SUB_T_TILE]` = `[32, 16]`
- 这样的布局适配后续 MMA 的 B 矩阵要求（Major::MN）

## 3. KG Intra 核心函数详解

### 3.1 setup_kg_intra - 单 gn 值计算

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void setup_kg_intra(
    G_TENSOR &sG, K_TENSOR &sK, KG_TENSOR &sKG_intra,
    int tile_j, int idx_in_warpgroup, float4 &gn, int index)
```

**执行流程**:

#### Step 1: 计算访问坐标
```cpp
int x = idx_in_warpgroup / 8 + tile_j * 16;  // 行: 0-15 (相对于 tile_j)
int y = idx_in_warpgroup % 8 * 4;             // 列: 0,4,8,...,28
```

#### Step 2: 加载数据（向量化）
```cpp
float4 tmp = *reinterpret_cast<float4*>(&sG(x, y));      // 加载 4 个 gate 值
nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(x, y)); // 加载 4 个 key 值 (bf16)
```
- 使用 128-bit 向量化加载（float4 = 16 bytes）
- `nvbf16x4` 是自定义结构，包含 2 个 `__nv_bfloat162` (共 4 个 bf16)

#### Step 3: 计算 gate 差值
```cpp
float2 sub1, sub2;
sub1 = float2_sub(reinterpret_cast<float2*>(&gn)[0], reinterpret_cast<float2*>(&tmp)[0]);
sub2 = float2_sub(reinterpret_cast<float2*>(&gn)[1], reinterpret_cast<float2*>(&tmp)[1]);
```
- 将 float4 拆分为 2 个 float2 处理
- 计算 `gn - g[x,y:y+4]`（4 个元素）

#### Step 4: 计算指数
```cpp
sub1.x = exp2f(sub1.x);
sub1.y = exp2f(sub1.y);
sub2.x = exp2f(sub2.x);
sub2.y = exp2f(sub2.y);
```
- 使用 `exp2f` 计算 2 的幂次（比 `expf` 更快）
- 4 次独立的指数计算

#### Step 5: 乘以 key 值
```cpp
reinterpret_cast<float2*>(&res)[0] = float2_mul(sub1, __bfloat1622float2(tmp_k.a));
reinterpret_cast<float2*>(&res)[1] = float2_mul(sub2, __bfloat1622float2(tmp_k.b));
```
- `__bfloat1622float2`: 将 bf16×2 转换为 float2
- `float2_mul`: 向量化乘法（2 个元素并行）

#### Step 6: 存储结果（转置）
```cpp
store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index, res);
```
- 输出地址：`[y, idx/8] + offset*index`
- 实现了转置：输入 `[x, y]` → 输出 `[y, x']`

### 3.2 setup_kg_intra_2gn - 双 gn 值融合计算

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void setup_kg_intra_2gn(
    G_TENSOR &sG, K_TENSOR &sK, KG_TENSOR &sKG_intra,
    int tile_j, int idx_in_warpgroup,
    float4 &gn1, float4 &gn2, int index1, int index2)
```

**优化策略**: 对于同一行 `[x, y]`，使用两个不同的 `gn` 值计算两个输出，共享 sG 和 sK 的加载。

**执行流程**:

#### Step 1: 加载数据（只加载一次）
```cpp
float4 g = *reinterpret_cast<float4*>(&sG(x, y));
nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
float2 kf_a = __bfloat1622float2(k.a);
float2 kf_b = __bfloat1622float2(k.b);
```

#### Step 2: 计算第一个输出（使用 gn1）
```cpp
float2 s1a = float2_sub(reinterpret_cast<float2*>(&gn1)[0], reinterpret_cast<float2*>(&g)[0]);
float2 s1b = float2_sub(reinterpret_cast<float2*>(&gn1)[1], reinterpret_cast<float2*>(&g)[1]);
s1a.x = exp2f(s1a.x); s1a.y = exp2f(s1a.y);
s1b.x = exp2f(s1b.x); s1b.y = exp2f(s1b.y);
float4 res1;
reinterpret_cast<float2*>(&res1)[0] = float2_mul(s1a, kf_a);
reinterpret_cast<float2*>(&res1)[1] = float2_mul(s1b, kf_b);
store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index1, res1);
```

#### Step 3: 计算第二个输出（使用 gn2）
```cpp
float2 s2a = float2_sub(reinterpret_cast<float2*>(&gn2)[0], reinterpret_cast<float2*>(&g)[0]);
float2 s2b = float2_sub(reinterpret_cast<float2*>(&gn2)[1], reinterpret_cast<float2*>(&g)[1]);
s2a.x = exp2f(s2a.x); s2a.y = exp2f(s2a.y);
s2b.x = exp2f(s2b.x); s2b.y = exp2f(s2b.y);
float4 res2;
reinterpret_cast<float2*>(&res2)[0] = float2_mul(s2a, kf_a);
reinterpret_cast<float2*>(&res2)[1] = float2_mul(s2b, kf_b);
store_128b(&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index2, res2);
```

**性能优势**:
- 减少 50% 的 smem 加载（sG 和 sK 只加载一次）
- 寄存器复用（kf_a, kf_b）
- 适用于同一行需要多个 gn 值的场景

## 4. KG Inter 核心函数详解

### 4.1 setup_kg_inter - 独立计算（已废弃）

原始版本会计算并存储 `sBkExp` 和 `sBkNegExp`，但在融合版本中已移除。

### 4.2 setup_inter_fused - KG+QKG 融合计算

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename Q_TENSOR,
          typename KG_TENSOR, typename QKG_TENSOR, int KG_OFFSET, int QKG_OFFSET>
__forceinline__ __device__ void setup_inter_fused(
    G_TENSOR &sG, K_TENSOR &sK, Q_TENSOR &sQ,
    KG_TENSOR &sKG_inter, QKG_TENSOR &sQKG_inter,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    float2 &beta, float4 &gn_half)
```

**执行流程**:

#### Step 1: 读取参考 gate 值
```cpp
int y = idx_in_warpgroup % 8 * 4;
gn_half = *reinterpret_cast<float4*>(&sG(min(sub_tile_i * 16 + 8, sub_seq_len - 1), y));
```
- `gn_half`: 子 tile 中间位置（第 8 行）的 gate 值
- 用于 inter-chunk 的归一化

#### Step 2: 加载当前位置数据
```cpp
int x = idx_in_warpgroup / 8 + sub_tile_i * 16;
if (x < sub_seq_len) {
    float4 tmp = *reinterpret_cast<float4*>(&sG(x, y));
    nvbf16x4 tmp_k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
    nvbf16x4 tmp_q = *reinterpret_cast<nvbf16x4*>(&sQ(x, y));
```

#### Step 3: 计算 gate 差值和指数
```cpp
float2 sub1 = float2_sub(reinterpret_cast<float2*>(&tmp)[0], reinterpret_cast<float2*>(&gn_half)[0]);
float2 sub2 = float2_sub(reinterpret_cast<float2*>(&tmp)[1], reinterpret_cast<float2*>(&gn_half)[1]);
float4 res_exp, res_neg_exp;
res_exp.x = exp2f(sub1.x);
res_exp.y = exp2f(sub1.y);
res_exp.z = exp2f(sub2.x);
res_exp.w = exp2f(sub2.y);
res_neg_exp.x = exp2f(-sub1.x);
res_neg_exp.y = exp2f(-sub1.y);
res_neg_exp.z = exp2f(-sub2.x);
res_neg_exp.w = exp2f(-sub2.y);
```
- 同时计算正指数和负指数
- `res_exp`: 用于 QKG inter
- `res_neg_exp`: 用于 KG inter

#### Step 4: 计算 KG inter 输出
```cpp
float4 res_kg;
reinterpret_cast<float2*>(&res_kg)[0] = float2_mul(reinterpret_cast<float2*>(&res_neg_exp)[0], __bfloat1622float2(tmp_k.a));
reinterpret_cast<float2*>(&res_kg)[1] = float2_mul(reinterpret_cast<float2*>(&res_neg_exp)[1], __bfloat1622float2(tmp_k.b));
store_128b(&sKG_inter(y, idx_in_warpgroup / 8) + KG_OFFSET * sub_tile_i, res_kg);
```
- 公式: `KG_inter = exp2(-(g - gn_half)) * k`
- 存储到转置后的位置

#### Step 5: 计算 QKG inter 输出（Q 部分）
```cpp
float4 res_q;
reinterpret_cast<float2*>(&res_q)[0] = float2_mul(__bfloat1622float2(tmp_q.a), reinterpret_cast<float2*>(&res_exp)[0]);
reinterpret_cast<float2*>(&res_q)[1] = float2_mul(__bfloat1622float2(tmp_q.b), reinterpret_cast<float2*>(&res_exp)[1]);
store_128b(&sQKG_inter(y, idx_in_warpgroup / 8) + QKG_OFFSET * sub_tile_i, res_q);
```

#### Step 6: 计算 QKG inter 输出（K 部分）
```cpp
float4 res_kbeta;
reinterpret_cast<float2*>(&res_kbeta)[0] = float2_mul(__bfloat1622float2(tmp_k.a), reinterpret_cast<float2*>(&res_exp)[0]);
reinterpret_cast<float2*>(&res_kbeta)[1] = float2_mul(__bfloat1622float2(tmp_k.b), reinterpret_cast<float2*>(&res_exp)[1]);
reinterpret_cast<float2*>(&res_kbeta)[0] = float2_mul(reinterpret_cast<float2*>(&res_kbeta)[0], beta);
reinterpret_cast<float2*>(&res_kbeta)[1] = float2_mul(reinterpret_cast<float2*>(&res_kbeta)[1], beta);
store_128b(&sQKG_inter(y, idx_in_warpgroup / 8 + 16) + QKG_OFFSET * sub_tile_i, res_kbeta);
```

**融合优势**:
- 一次加载 sG, sK, sQ，产生 3 个输出（KG inter + QKG inter 两部分）
- 指数计算复用（exp 和 neg_exp 同时计算）
- 减少内存带宽消耗

## 5. 调用流程与 Warpgroup 分工

### 5.1 WG0 的 KG 计算任务

#### 阶段 1: KG Intra（Q 加载前）
```cpp
float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
setup_kg_intra<kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn3, 3);
```
- tile_j=0, 使用 gn3 (G[48, :])
- 输出到 `sKG_intra[..] + kg_offset*3`

#### 阶段 2: KG+QKG Intra 融合（Q 加载后）
```cpp
float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
float2 beta[4];
for (int j = 1; j <= 2; ++j) {
    int x = idx_in_warpgroup / 8 + j * 16;
    if (x < sub_seq_len)
        beta[j] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x]));
}
// tile_j=1
setup_intra_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_intra, sQKG_intra,
                                         1, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[1], 4, 0);
// tile_j=2
setup_intra_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_intra, sQKG_intra,
                                         2, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[2], 5, 1);
```
- tile_j=1,2 融合计算
- KG 使用 gn3, QKG 使用 gn1

#### 阶段 3: KG+QKG Inter 融合
```cpp
beta[0] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][idx_in_warpgroup / 8]));
int x3 = idx_in_warpgroup / 8 + 48;
if (x3 < sub_seq_len)
    beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
float4 gn_half_0, gn_half_3;
setup_inter_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_inter, sQKG_inter,
                                         0, idx_in_warpgroup, sub_seq_len, beta[0], gn_half_0);
setup_inter_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_inter, sQKG_inter,
                                         3, idx_in_warpgroup, sub_seq_len, beta[3], gn_half_3);
```
- sub_tile=0,3
- 每个 sub_tile 产生 KG inter + QKG inter

### 5.2 WG1 的 KG 计算任务

#### 阶段 1: KG Intra（Q 加载前）
```cpp
float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
// tile_j=0: 融合两个 gn
setup_kg_intra_2gn<kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn1, gn2, 0, 1);
// tile_j=1: 单独计算
setup_kg_intra<kg_offset>(sG, sK, sKG_intra, 1, idx_in_warpgroup, gn2, 2);
```
- tile_j=0 使用 2gn 优化（gn1 和 gn2）
- tile_j=1 使用 gn2

#### 阶段 2: KG+QKG Inter 融合
```cpp
int x1 = idx_in_warpgroup / 8 + 16;
if (x1 < sub_seq_len)
    beta[1] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x1]));
int x2 = idx_in_warpgroup / 8 + 32;
if (x2 < sub_seq_len)
    beta[2] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x2]));
float4 gn_half_1, gn_half_2;
setup_inter_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_inter, sQKG_inter,
                                         1, idx_in_warpgroup, sub_seq_len, beta[1], gn_half_1);
setup_inter_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_inter, sQKG_inter,
                                         2, idx_in_warpgroup, sub_seq_len, beta[2], gn_half_2);
```
- sub_tile=1,2

### 5.3 KG 输出矩阵布局

#### KG Intra 输出
```
sKG_intra[0..5]: 6 个 [32, 16] 矩阵
- WG0 负责: index 3, 4, 5
- WG1 负责: index 0, 1, 2
```

#### KG Inter 输出
```
sKG_inter[0..3]: 4 个 [32, 16] 矩阵
- WG0 负责: sub_tile 0, 3
- WG1 负责: sub_tile 1, 2
```

## 6. 性能优化技巧

### 6.1 向量化访问
- 使用 `float4` (128-bit) 加载/存储
- 使用 `float2` 进行 SIMD 运算
- 减少内存事务数量

### 6.2 数据复用
- `setup_kg_intra_2gn`: 一次加载，两次计算
- `setup_intra_fused`: 一次加载，产生 KG + QKG 两个输出
- `setup_inter_fused`: 一次加载，产生 3 个输出

### 6.3 寄存器优化
- 预先转换 bf16 → float2，避免重复转换
- 复用指数计算结果（exp 和 neg_exp）

### 6.4 内存访问模式
- 合并访问：8 个线程访问连续的 32 字节（float4）
- 转置输出：适配 MMA B 矩阵布局，避免后续转置

### 6.5 流水线重叠
- KG Intra 在 Q 加载前开始计算
- KG/QKG 融合计算与 MMA 并行执行

## 7. 关键常量与地址计算

### 7.1 偏移量
```cpp
constexpr int kg_offset = SUB_T_TILE * K_TILE = 16 * 32 = 512;
```
- 每个 tile_j 或 sub_tile 的输出占用 512 个 tf32 元素
- 不同 index 的输出通过 `KG_OFFSET * index` 偏移

### 7.2 输出地址计算
```cpp
&sKG_intra(y, idx_in_warpgroup / 8) + KG_OFFSET * index
```
- `y`: K 维度坐标（0, 4, 8, ..., 28）
- `idx_in_warpgroup / 8`: T 维度坐标（0-15）
- `KG_OFFSET * index`: tile 偏移

### 7.3 gn 读取位置
```cpp
WG0:
- gn1: G[16, y]  // 第 16 行
- gn3: G[48, y]  // 第 48 行

WG1:
- gn1: G[16, y]
- gn2: G[32, y]  // 第 32 行
```

## 8. 边界处理

### 8.1 序列长度检查
```cpp
if (x < sub_seq_len) {
    // 正常计算
} else {
    float4 res_zero = {0.0f, 0.0f, 0.0f, 0.0f};
    store_128b(..., res_zero);
}
```
- 超出有效序列长度的位置填充 0

### 8.2 gn_half 边界保护
```cpp
gn_half = *reinterpret_cast<float4*>(&sG(min(sub_tile_i * 16 + 8, sub_seq_len - 1), y));
```
- 确保不会访问越界

## 9. 总结

### 9.1 KG 计算的核心思想
1. **分块计算**: 将 64×32 的矩阵分解为多个 16×32 的 sub-tile
2. **融合优化**: KG 和 QKG 共享数据加载，减少内存访问
3. **转置输出**: 直接生成 MMA 所需的 B 矩阵布局
4. **向量化**: 充分利用 SIMD 指令和向量化内存访问

### 9.2 两个 Warpgroup 的协作
- WG0 和 WG1 处理不同的 K 维度范围（前 16 和后 16）
- 处理不同的 tile_j 和 sub_tile，避免冲突
- 最终合并成完整的 KG 矩阵供 MMA 使用

### 9.3 性能关键点
- **内存带宽**: 通过融合计算减少 smem 访问
- **计算吞吐**: 向量化指数和乘法运算
- **流水线**: 与 TMA 加载和 MMA 计算重叠
