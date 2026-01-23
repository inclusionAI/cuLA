# 矩阵求逆 Kernel (64x64 FP16) - Launch 指南

## 概述

`MatrixInverse64x64` 是一个独立的 CuTe GPU kernel，用于计算 64x64 FP16 下三角矩阵的逆矩阵。

## Kernel 启动方式

### 方式1: 通过 `__call__` 方法（推荐）

```python
import torch
from flashla.inv import MatrixInverse64x64
import cutlass

# 创建 kernel 实例
inv_kernel = MatrixInverse64x64(
    acc_dtype=cutlass.Float32,      # 中间计算精度
    cuda_core_threads=128            # CUDA 线程数
)

# 创建/准备 64x64 FP16 矩阵
mat = torch.randn(64, 64, dtype=torch.float16, device='cuda')
mat = torch.tril(mat)  # 下三角化
mat.diagonal().add_(1.0)  # 确保对角线非零

# 获取 CUDA 流
stream = torch.cuda.current_stream()

# 执行 kernel（结果就地存储在 mat 中）
inv_kernel(mat.data_ptr(), stream=stream)
```

### 方式2: 在 CuTe Kernel 中直接调用

```python
@cute.kernel
def my_kernel(...):
    inv_kernel = MatrixInverse64x64()
    # 从全局内存加载到共享内存
    s_mat = ...  # 64x64 FP16 张量在 SMEM 中
    # 执行求逆
    inv_kernel.compute_matrix_inverse_64x64(s_mat)
```

### 方式3: 调用主计算方法

```python
inv_kernel = MatrixInverse64x64()
# 在 SMEM 中的张量
s_mat = ...  # 64x64 张量
# 直接计算
inv_kernel.compute_matrix_inverse_64x64(s_mat)
```

## Grid/Block 配置

| 参数 | 值 | 说明 |
|------|---|------|
| Grid dimensions | (1, 1, 1) | 单个 CTA（合作线程数组） |
| Block dimensions | (128, 1, 1) | 128 个线程（4 个 Warp） |
| Cluster shape | (1, 1, 1) | 单个集群 |
| min_blocks_per_mp | 1 | 最小块数/多处理器 |

## 共享内存配置

| 参数 | 值 |
|------|-----|
| 矩阵大小 | 64×64 = 4096 元素 |
| 数据类型 | FP16（2 字节/元素） |
| 总大小 | 8192 字节 ≈ 8 KB |
| 对齐要求 | 1024 字节 |

## 类配置常量

```python
class MatrixInverse64x64:
    MATRIX_SIZE = 64              # 矩阵维度
    THREADS_PER_CTA = 128         # 每 CTA 线程数
    GRID_SIZE = 1                 # Grid 大小
    SMEM_ALIGN_BYTES = 1024       # 共享内存对齐
```

## 计算流程

矩阵求逆使用 4 阶段块状 Schur 补法：

```
┌─────────────────────────────────────┐
│ 输入: 64×64 FP16 下三角矩阵         │
├─────────────────────────────────────┤
│ Stage 1: 8×8 对角块求逆             │
│          (128 个线程并行)            │
│          ↓ [Barrier] ↓              │
│ Stage 2: 8×8 → 16×16 块              │
│          (使用 Schur 补)             │
│          ↓ [Barrier] ↓              │
│ Stage 3: 16×16 → 32×32 块            │
│          (使用 Schur 补)             │
│          ↓ [Barrier] ↓              │
│ Stage 4: 32×32 → 64×64 矩阵          │
│          (最终完整逆矩阵)            │
├─────────────────────────────────────┤
│ 输出: 64×64 FP16 逆矩阵（就地）     │
└─────────────────────────────────────┘
```

## 数据类型

| 部分 | 数据类型 | 用途 |
|------|---------|------|
| 输入 | Float16 | 矩阵元素 |
| 中间计算 | Float32 | 提高精度 |
| 输出 | Float16 | 逆矩阵 |

## 线程同步

- **同步原语**: `NamedBarrier`
- **Barrier ID**: 3
- **同步线程数**: 128
- **同步点**:
  - Stage 1 完成后
  - Stage 2 完成后
  - Stage 3 完成后
  - Stage 4 完成后

## 主要方法

| 方法 | 功能 |
|------|------|
| `__call__(mat_ptr, stream)` | 启动 kernel 的主入口 |
| `compute_matrix_inverse_64x64(s_mat)` | 4 阶段求逆主计算 |
| `compute_diagonal_inverse_8x8_to_16x16(mat)` | Stage 2 计算 |
| `compute_diagonal_inverse_16x16_to_32x32(mat)` | Stage 3 计算 |
| `compute_diagonal_inverse_32x32_to_64x64(mat)` | Stage 4 计算 |
| `kernel(mat)` | CuTe 内核装饰器方法 |

## 性能特性

- **吞吐量**: 单个 64×64 矩阵求逆
- **延迟**: 4 个同步屏障（阶段间）
- **共享内存**: 8 KB（对齐到 1024 字节）
- **线程利用率**: 128 个线程（4 个 Warp 满负载）
- **目标 GPU**: Blackwell 及以上

## 限制条件

1. 矩阵大小固定为 64×64
2. 只支持下三角矩阵
3. 输入必须为 FP16
4. 矩阵必须非奇异（行列式 ≠ 0）
5. 对角线元素必须非零
6. 结果就地存储（覆盖输入）

## 使用建议

1. **创建实例一次**: 在循环外创建 kernel 实例，避免重复初始化
2. **确保矩阵有效**: 验证输入矩阵非奇异
3. **流同步**: 使用 `torch.cuda.synchronize()` 或流同步确保计算完成
4. **精度考虑**: FP16 精度有限，对于病态矩阵可能需要特殊处理

## 测试

运行完整的测试套件：

```bash
cd /ossfs/workspace/flashla
/ossfs/workspace/venv/bin/python tests/test_matrix_inverse_64x64.py
```

预期输出：14 个测试全部通过
