# 64x64 FP16 矩阵求逆 kernel 文档

## 概述

`flashla/inv.py` 实现了一个独立的 CuTe kernel，用于计算 64x64 FP16 下三角矩阵的逆矩阵。该 kernel 采用分块 Schur 补算法，分 4 个阶段逐步构建矩阵的逆。

## 算法原理

### 分块 Schur 补方法

对于下三角矩阵的逆矩阵计算，使用分块 Schur 补公式：

```
对于分块矩阵 M = [A 0  ]，其逆矩阵为：
            [C D]

inv(M) = [inv(A)      0        ]
         [-inv(D)CA⁻¹ inv(D)]
```

其中：
- A: 左上角对角块（需要逆矩阵）
- C: 左下角块
- D: 右下角对角块（需要逆矩阵）

### 4 阶段计算流程

#### 第 1 阶段：8x8 对角块求逆
- 将 64x64 矩阵分成 8 个 8x8 的对角块
- 每个线程处理一个 8x8 块的逆
- 使用 warp-level 操作进行 8x8 矩阵求逆
- 数据类型：FP16 → 计算 → FP32 → 结果 FP16

#### 第 2 阶段：8x8 → 16x16
- 从已求逆的 8x8 块构建 16x16 对角块
- 对每个 16x16 块应用 Schur 补算法
- 使用 16x8x8 MMA（矩阵乘法-累积）操作
- 计算：`DC = -D * C`，`O = DC * inv(A)`

#### 第 3 阶段：16x16 → 32x32
- 从已求逆的 16x16 块构建 32x32 对角块
- 对每个 32x32 块应用 Schur 补算法
- 类似第 2 阶段但规模更大

#### 第 4 阶段：32x32 → 64x64
- 从两个 32x32 块构建完整 64x64 逆矩阵
- 使用多个 warp（4 个 warp）进行并行计算
- 最终输出存储回 SMEM

## 关键类和方法

### MatrixInverse64x64 类

主要接口类，提供矩阵求逆功能。

#### 初始化
```python
inv_kernel = MatrixInverse64x64(
    acc_dtype=cutlass.Float32,      # 中间计算精度（默认 FP32）
    cuda_core_threads=128           # CUDA 线程数（默认 128）
)
```

#### 核心方法

##### compute_matrix_inverse_64x64(s_mat)
主入口函数，计算 64x64 矩阵的逆。
- **输入**：`s_mat` - SMEM 中的 64x64 FP16 矩阵
- **输出**：逆矩阵，存储在原位置的 SMEM
- **同步**：各阶段间进行线程组同步

##### compute_diagonal_inverse_8x8_to_16x16(mat)
从 8x8 块构建 16x16 块的逆矩阵。
- 使用 16x8x8 MMA 操作
- 应用 Schur 补算法

##### compute_diagonal_inverse_16x16_to_32x32(mat)
从 16x16 块构建 32x32 块的逆矩阵。
- 使用 16x8x16 MMA 操作
- 类似 8x8→16x16 的流程

##### compute_diagonal_inverse_32x32_to_64x64(mat)
@cute.jit 装饰的最终阶段，从 32x32 块构建完整 64x64 逆矩阵。
- 使用多个 warp 进行分布式计算
- warp 坐标分布用于不同块的处理
- 全线程组同步

### 辅助方法

#### load_row_mat8x8(mat, idx)
从 SMEM 加载 8x8 矩阵的一行，并进行 FP16→FP32 转换。

#### store_row_mat8x8(mat, row, idx)
将计算结果（FP32）存储回 SMEM，并进行 FP32→FP16 转换。

#### canonical_lane_id()
获取 warp 内的 lane ID（0-31）。

#### convert_layout_c_to_a(c_layout, tiled_mma)
将 MMA 累积器布局转换为操作数 A 的布局。

#### make_acc_as_a(acc, tiled_mma, dtype)
将 MMA 累积器转换为操作数 A 的格式。

## 数据布局与内存

### 共享内存（SMEM）布局
```
[64 x 64] FP16 矩阵
分块视图：
  8x8 块 (8 行 x 8 列)
  16x16 块 (2x2 = 4 个 8x8)
  32x32 块 (2x2 = 4 个 16x16)
  64x64 块 (2x2 = 4 个 32x32)
```

### 数据类型流
```
输入 (SMEM)     FP16
  ↓
计算 (RMEM)     FP32  ← 使用 FP32 保证精度
  ↓
转换            FP32 → FP16
  ↓
输出 (SMEM)     FP16
```

## 线程协调

### 线程分布
- 总线程数：128（4 个 warp × 32 线程）
- 每个 warp 处理一个区域或进行 MMA 操作

### 同步机制
- `cuda_wg_sync_barrier`：NamedBarrier，ID=3
- 各阶段之间进行 `arrive_and_wait()` 同步
- 确保所有线程完成当前阶段再进入下一阶段

### Warp 分布（阶段 4）
```
warp_id_wg % 4 = 0-3
x = warp_id_wg // 2  (0-1)  → 列分布
y = warp_id_wg % 2   (0-1)  → 行分布

用于处理 2x2 分块的 32x32 矩阵
```

## 使用示例

### 基本使用
```python
import torch
import cutlass.cute as cute
from flashla.inv import MatrixInverse64x64

# 创建 kernel 实例
inv_kernel = MatrixInverse64x64()

# 准备 64x64 FP16 矩阵（在 SMEM 中）
# 注意：实际使用中这需要在 CUDA kernel 中执行
matrix_64x64 = torch.randn(64, 64, dtype=torch.float16, device="cuda")

# 在 kernel 中调用
# inv_kernel.compute_matrix_inverse_64x64(s_mat)
```

### 测试验证
```bash
cd /ossfs/workspace/flashla
python -m pytest tests/test_matrix_inverse_64x64.py -v
```

## 关键 CuTe 操作

### MMA（Matrix Multiply-Accumulate）
```python
mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=dtype,
    acc_dtype=Float32,
    shape_mnk=mma_atom_shape,
)
```

### 复制操作
- **LdMatrix**：从 SMEM 加载 8x8 块到 RMEM
- **StMatrix**：从 RMEM 存储 8x8 块到 SMEM
- 支持转置操作（transpose=True/False）

### 张量分割（Tiling）
```python
cute.flat_divide(mat, (8, 8))   # 分成 8x8 块
cute.logical_divide(...)        # 逻辑分割
cute.partition_S/D/C(...)       # 按线程分区
```

## 精度考量

### FP16 精度限制
- 取值范围：6.1e-5 到 6.55e4
- 约 3.3 位十进制有效数字

### FP32 中间计算优势
- 减少累积舍入误差
- 改进条件数差的矩阵的精度
- 最后转换回 FP16 满足输出要求

### 数值稳定性
- Schur 补算法对对角线主导矩阵稳定
- 推荐矩阵条件数 < 1e4

## 性能特性

### 计算复杂度
- 64x64 矩阵求逆：O(64³/3) 浮点操作
- 预计：~280k 次 FP32 操作

### 内存使用
- SMEM：64×64×2B (FP16) = 8 KB
- RMEM：每个 thread 需要存储 MMA fragments

### 并行性
- 利用 4 个 warp 的并行性
- 逐阶段递进计算
- 阶段间进行同步

## 限制和未来改进

### 当前限制
1. 仅支持 64x64 大小
2. 仅支持 FP16 输入/输出
3. 要求矩阵为下三角形式
4. Blackwell GPU 特定（使用 BF16 MMA）

### 可能的扩展
1. 支持更大矩阵（128x128, 256x256）
2. 支持其他数据类型（BF16, TF32）
3. 上三角矩阵支持
4. 泛型矩阵求逆

## 调试信息

### 打印日志
kernel 中包含多个 `print()` 语句用于调试：
- 张量形状和布局
- 片段配置（fragment shapes）
- 分区结果（partition results）

### 验证检查
测试脚本包括：
- 接口方法存在性检查
- FP16 精度验证
- 矩阵结构验证（下三角）

## 文件结构

```
flashla/
├── inv.py                              # 主实现文件
├── __init__.py                         # 导出 MatrixInverse64x64
└── ../tests/
    └── test_matrix_inverse_64x64.py   # 测试套件
```

## 相关文件参考

- [flashla/kda.py](flashla/kda.py) - 原始 KDA kernel（包含初始求逆实现）
- [benchmark/bench_kda.py](benchmark/bench_kda.py) - KDA 基准测试
- [docs/OPTIMIZATION_IMPLEMENTATION.md](docs/OPTIMIZATION_IMPLEMENTATION.md) - 优化实现细节
