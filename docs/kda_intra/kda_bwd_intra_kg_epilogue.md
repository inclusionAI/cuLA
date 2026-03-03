# KDA Backward Intra KG Epilogue 分析

## 概述

在 `kda_bwd_intra_sm100.cu` 中，Epilogue Warpgroup (ComputeEpilogue) 负责使用 CUDA core 计算 KG 和 QKG 相关的中间矩阵，为后续的 MMA (Tensor Core) 计算做准备。整个处理分为两个 Warpgroup (WG0 和 WG1)，每个处理不同的 K 维度范围。

## 架构设计

### Warpgroup 划分
- **WG0**: 处理 K 维度的前半部分 (K_OFF = 0, HALF_K = 16)
- **WG1**: 处理 K 维度的后半部分 (K_OFF = 16, HALF_K = 16)
- 每个 Warpgroup 有 128 个线程 (idx_in_warpgroup: 0-127)

### 常量定义
```cpp
constexpr int SUB_T_TILE = 16;      // 子tile大小
constexpr int T_TILE = 64;          // 时间维度tile大小
constexpr int K_TILE = 32;          // K维度tile大小
constexpr int kg_offset = SUB_T_TILE * K_TILE = 512;
constexpr int qkg_offset = SUB_T_TILE * K_TILE * 2 = 1024;
```

## KG 处理流程

### 1. KG Intra 计算 (非重叠行)

**目的**: 在等待 Q 数据加载时，先计算不需要 Q 的 KG intra 部分

#### WG0 处理 (tile_j=0)
```cpp
float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
setup_kg_intra<kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn3, 3);
```
- 读取 G[48, y:y+4] 作为 gn3
- 计算 tile_j=0 的 KG intra (使用 gn3)
- 输出到 `sKG_intra[y, idx/8] + kg_offset*3`

#### WG1 处理 (tile_j=0, 1)
```cpp
float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
// tile_j=0: 融合两个 gn 值的计算
setup_kg_intra_2gn<kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn1, gn2, 0, 1);
// tile_j=1: 单独计算
setup_kg_intra<kg_offset>(sG, sK, sKG_intra, 1, idx_in_warpgroup, gn2, 2);
```
- 读取 G[16, y:y+4] 作为 gn1, G[32, y:y+4] 作为 gn2
- tile_j=0 融合计算 (使用 gn1 和 gn2)
- tile_j=1 单独计算 (使用 gn2)

### 2. 等待 Q 数据
```cpp
cute::wait_barrier(shared_plan->bar_load_qb[buf_idx_value], local_phase);
Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_idx_value].data()), SmemLayoutInputBF16{});
```

### 3. KG Intra + QKG Intra 融合计算 (重叠行)

**优化策略**: 对于同时需要计算 KG 和 QKG 的行，融合计算以节省 sK 和 sG 的重复加载

#### WG0 处理 (tile_j=1, 2)
```cpp
float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
float2 beta[4];
for (int j = 1; j <= 2; ++j) {
    int x = idx_in_warpgroup / 8 + j * 16;
    if (x < sub_seq_len)
        beta[j] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x]));
}
// tile_j=1: kg 使用 gn3, qkg 使用 gn1
setup_intra_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_intra, sQKG_intra,
                                         1, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[1], 4, 0);
// tile_j=2: kg 使用 gn3, qkg 使用 gn1
setup_intra_fused<kg_offset, qkg_offset>(sG, sK, sQ, sKG_intra, sQKG_intra,
                                         2, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[2], 5, 1);
```

### 4. KG Inter + QKG Inter 融合计算

**目的**: 计算 inter-chunk 的 KG 和 QKG 矩阵

#### WG0 处理 (sub_tile=0, 3)
```cpp
Tensor sKG_inter = make_tensor(make_smem_ptr(shared_plan->kg_all.inter[0].data()),
                               SmemLayoutMatBTF32Tranposed<1>{});
Tensor sQKG_inter = make_tensor(make_smem_ptr(shared_plan->qkg_all.inter[0].data()),
                                SmemLayoutMatBTF32Tranposed<2>{});
float2 beta[4];
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

#### WG1 处理 (sub_tile=1, 2)
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

### 5. 同步并通知 MMA
```cpp
fence_view_async_shared();
cute::arrive_barrier(shared_plan->bar_kg_all_ready); // kg_all complete (intra + inter)
```

### 6. QKG Intra 计算 (非重叠行)

**时机**: 在 MMA 处理 KG 阶段时并行计算

#### WG0 处理 (tile_j=3)
```cpp
float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
int x3 = idx_in_warpgroup / 8 + 48;
if (x3 < sub_seq_len)
    beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
setup_qkg_intra<qkg_offset>(sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn1, 2);
```

#### WG1 处理 (tile_j=2, 3)
```cpp
float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
int x2 = idx_in_warpgroup / 8 + 32;
if (x2 < sub_seq_len)
    beta[2] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x2]));
int x3 = idx_in_warpgroup / 8 + 48;
if (x3 < sub_seq_len)
    beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));

setup_qkg_intra<qkg_offset>(sG, sQ, sK, sQKG_intra, 2, idx_in_warpgroup, sub_seq_len, beta[2], gn2, 3);
setup_qkg_intra_2gn<qkg_offset>(sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn2, gn3, 4, 5);
```

### 7. 最终同步
```cpp
fence_view_async_shared();
cute::arrive_barrier(shared_plan->bar_qkg_all_ready); // all qkg data ready for MMA
```

## 核心函数详解

### setup_kg_intra
**功能**: 计算 `exp2f(gn - g[j]) * k[j]`

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void setup_kg_intra(
    G_TENSOR &sG, K_TENSOR &sK, KG_TENSOR &sKG_intra,
    int tile_j, int idx_in_warpgroup, float4 &gn, int index)
```

**计算流程**:
1. 计算线程坐标: `x = idx_in_warpgroup / 8 + tile_j * 16`, `y = idx_in_warpgroup % 8 * 4`
2. 读取 `g[x, y:y+4]` 和 `k[x, y:y+4]`
3. 计算 `sub = gn - g[x, y:y+4]` (float2 向量化)
4. 计算 `exp2f(sub)` (4个元素)
5. 计算 `res = exp2f(sub) * k[x, y:y+4]` (float2 向量化乘法)
6. 存储到 `sKG_intra[y, idx/8] + KG_OFFSET * index`

### setup_kg_intra_2gn
**功能**: 融合计算两个不同 gn 值的 kg_intra

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void setup_kg_intra_2gn(
    G_TENSOR &sG, K_TENSOR &sK, KG_TENSOR &sKG_intra,
    int tile_j, int idx_in_warpgroup,
    float4 &gn1, float4 &gn2, int index1, int index2)
```

**优化**: 一次读取 g 和 k，分别用 gn1 和 gn2 计算两个结果，减少内存访问

### setup_intra_fused
**功能**: 融合计算 kg_intra 和 qkg_intra

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename Q_TENSOR,
          typename KG_TENSOR, typename QKG_TENSOR, int KG_OFFSET, int QKG_OFFSET>
__forceinline__ __device__ void setup_intra_fused(
    G_TENSOR &sG, K_TENSOR &sK, Q_TENSOR &sQ,
    KG_TENSOR &sKG_intra, QKG_TENSOR &sQKG_intra,
    int tile_j, int idx_in_warpgroup, int sub_seq_len,
    float4 &gn_kg, float4 &gn_qkg, float2 &beta,
    int kg_index, int qkg_index)
```

**计算**:
- KG 部分: `exp2f(gn_kg - g[x, y]) * k[x, y]`
- QKG 部分: `exp2f(gn_qkg - g[x, y]) * q[x, y] * k[x, y] * beta`

### setup_inter_fused
**功能**: 融合计算 kg_inter 和 qkg_inter

```cpp
template <typename G_TENSOR, typename K_TENSOR, typename Q_TENSOR,
          typename KG_TENSOR, typename QKG_TENSOR, int KG_OFFSET, int QKG_OFFSET>
__forceinline__ __device__ void setup_inter_fused(
    G_TENSOR &sG, K_TENSOR &sK, Q_TENSOR &sQ,
    KG_TENSOR &sKG_inter, QKG_TENSOR &sQKG_inter,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    float2 &beta, float4 &gn_half)
```

**特点**:
- 计算 `gn_half = g[min(sub_tile_i * 16 + 8, sub_seq_len - 1), y]` (中间行)
- KG 部分: `exp2f(-(g[x, y] - gn_half)) * k[x, y]`
- QKG 部分: `exp2f(gn_half - g[x, y]) * q[x, y] * k[x, y] * beta`
- 同时输出 `sBkExp` 和 `sBkNegExp` 用于后续 epilogue

### setup_qkg_intra
**功能**: 单独计算 qkg_intra

```cpp
template <typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR,
          typename QKG_TENSOR, int QKG_OFFSET>
__forceinline__ __device__ void setup_qkg_intra(
    G_TENSOR &sG, Q_TENSOR &sQ, K_TENSOR &sK,
    QKG_TENSOR &sQKG_intra,
    int tile_j, int idx_in_warpgroup, int sub_seq_len,
    float2 &beta, float4 &gn, int index)
```

**计算**: `exp2f(gn - g[x, y]) * q[x, y] * k[x, y] * beta`

## 数据布局

### 输入数据
- **sG**: `[T_TILE=64, K_TILE=32]` float32, SmemLayoutInputFP32
- **sK**: `[T_TILE=64, K_TILE=32]` bf16, SmemLayoutInputBF16
- **sQ**: `[T_TILE=64, K_TILE=32]` bf16, SmemLayoutInputBF16
- **beta_smem**: `[T_TILE=64]` bf16, 双缓冲

### 输出数据
- **kg_all.intra**: 6个子矩阵, 每个 `[K_TILE=32, SUB_T_TILE=16]` tf32
- **kg_all.inter**: 4个子矩阵, 每个 `[K_TILE=32, SUB_T_TILE=16]` tf32
- **qkg_all.intra**: 6个子矩阵, 每个 `[K_TILE=32, SUB_T_TILE*2=32]` tf32
- **qkg_all.inter**: 4个子矩阵, 每个 `[K_TILE=32, SUB_T_TILE*2=32]` tf32

## 线程映射

每个 Warpgroup (128 threads) 的线程映射:
- `x = idx_in_warpgroup / 8`: 行索引 (0-15)
- `y = idx_in_warpgroup % 8 * 4`: 列索引 (0, 4, 8, ..., 28)
- 每个线程处理 4 个连续的 K 维度元素 (float4)

## 性能优化策略

1. **融合计算**: 对于需要同时计算 KG 和 QKG 的行，融合计算以减少内存访问
2. **向量化**: 使用 float4/float2 向量化操作
3. **流水线**: KG intra 计算与 Q 数据加载重叠
4. **双缓冲**: beta_smem 使用双缓冲避免冲突
5. **Warpgroup 分工**: WG0 和 WG1 处理不同的 K 维度范围，提高并行度
6. **2gn 优化**: setup_kg_intra_2gn 一次读取数据计算两个结果

## 同步机制

1. `bar_load_kg_ready`: Load warp 通知 K/G 数据就绪
2. `bar_load_qb`: Load warp 通知 Q/beta 数据就绪
3. `bar_kg_all_ready`: Epilogue 通知 MMA kg_all 数据就绪
4. `bar_qkg_all_ready`: Epilogue 通知 MMA qkg_all 数据就绪

## 总结

Epilogue WG 的 KG 处理逻辑通过精心设计的融合计算、向量化操作和流水线重叠，高效地完成了 KG 和 QKG 矩阵的准备工作，为后续的 Tensor Core MMA 计算提供了优化的输入数据。整个设计充分利用了 CUDA core 的计算能力，并通过 Warpgroup 分工实现了良好的并行性。
