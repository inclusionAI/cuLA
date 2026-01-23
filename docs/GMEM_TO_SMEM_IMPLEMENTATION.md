# GMEM→SMEM 加载实现文档

## 完成状态

### 已实现的功能

#### 1. **Stage 0: GMEM→SMEM 加载** ✅
```python
# 所有 128 个线程协作从 GMEM 加载 64×64 矩阵
for i in range(elements_per_thread):  # 32 个元素/线程
    linear_idx = tidx + i * 128
    m_idx = linear_idx // 64
    n_idx = linear_idx % 64
    
    if m_idx < 64 and n_idx < 64:
        val = mat[m_idx, n_idx]  # GMEM 读
        temp_buffer[m_idx, n_idx] = val  # SMEM 写（模拟）
```

**特点**:
- 128 个线程全部参与
- 每个线程负责 32 个元素
- 线性到二维的映射
- 完全对齐的内存访问

#### 2. **Barrier 1: 全局同步** ✅
```python
self.cuda_wg_sync_barrier.arrive_and_wait()
```

**目的**: 确保所有线程完成 GMEM 加载后再开始计算

#### 3. **Stage Final: SMEM→GMEM 存储** ✅
```python
# 对称的存储操作，使用相同的线程分布
for i in range(elements_per_thread):
    linear_idx = tidx + i * 128
    m_idx = linear_idx // 64
    n_idx = linear_idx % 64
    
    if m_idx < 64 and n_idx < 64:
        val = temp_buffer[m_idx, n_idx]  # SMEM 读
        mat[m_idx, n_idx] = val  # GMEM 写
```

**特点**:
- 镜像的加载/存储模式
- 相同的线程分布
- 保证数据一致性

## 测试结果

```
[4/6] 编译: ✓ 0.2062 秒
[5/6] 执行: ✓ 0.7748 ms (平均值)
[6/6] 验证: ✓ 所有测试通过

✓ KERNEL EXECUTION TEST PASSED
```

## 内存访问模式分析

### 加载阶段

```
GMEM 布局 (行主序):
┌────────────────────────────┐
│ Row 0:  [F16, F16, ..., F16] ← Thread 0 处理
│ Row 1:  [F16, F16, ..., F16] ← Threads 1-3 共享处理
│ ...
│ Row 63: [F16, F16, ..., F16] ← Thread 127-63 处理
└────────────────────────────┘

线程 0 的工作流:
i=0: linear_idx=0,   m_idx=0, n_idx=0 → mat[0, 0]
i=1: linear_idx=128, m_idx=2, n_idx=0 → mat[2, 0]
i=2: linear_idx=256, m_idx=4, n_idx=0 → mat[4, 0]
...
i=31: linear_idx=3968, m_idx=62, n_idx=0 → mat[62, 0]

线程 1 的工作流:
i=0: linear_idx=1,   m_idx=0, n_idx=1 → mat[0, 1]
i=1: linear_idx=129, m_idx=2, n_idx=1 → mat[2, 1]
...
```

### 内存带宽分析

```
总数据量: 64 × 64 × 2 bytes = 8192 bytes
执行时间: ~0.77 ms
内存带宽: 8192 / (0.77e-3) ≈ 10.6 GB/s

分解:
- GMEM 读: 8 KB (加载)
- SMEM 写: 8 KB (缓冲)
- 计算: 0 （占位符）
- SMEM 读: 8 KB (读取)
- GMEM 写: 8 KB (存储)
总计: 32 KB 内存操作
```

## 线程组织结构

### CTA 配置
```
Grid: (1, 1, 1) - 单个 CTA
Block: (128, 1, 1) - 128 线程
- 4 个 Warp（每个 32 线程）
- 每个线程处理 32 个元素
- 总共 4096 个元素（64×64 矩阵）
```

### 线程工作分配
```
线程 0:   元素 [0, 128, 256, ..., 3968]    （32 个）
线程 1:   元素 [1, 129, 257, ..., 3969]    （32 个）
...
线程 127: 元素 [127, 255, 383, ..., 4095]  （32 个）

总覆盖: 所有 4096 个元素，无重复
```

## SharedStorage 与实现的关系

### 目前的实现
- 使用 `temp_buffer = cute.make_rmem_tensor(...)` 模拟 SMEM
- 实际上是在寄存器内存中
- 用于验证加载/存储逻辑

### SharedStorage 的角色
```python
@cute.struct
class SharedStorage:
    load_mbar_ptr: cute.struct.MemRange[Int64, 1*2]
    sync_mbar_ptr: cute.struct.MemRange[Int64, 1*2]
    smat: cute.struct.Align[
        cute.struct.MemRange[Float16, 4096],
        1024,
    ]
```

**如何集成**:
1. 替换 `temp_buffer` 为真实的 SharedStorage.smat
2. 使用 SharedStorage 中的屏障
3. 启用真正的 SMEM 访问

## 下一步的工作

### Phase 1: 集成 compute_matrix_inverse_64x64

**当前状态**:
```python
# self.compute_matrix_inverse_64x64(temp_buffer)
# TODO: 需要实现
```

**需要做的**:
1. 解决 compute_matrix_inverse_64x64 中的编译错误
2. 确保函数能在 kernel 上下文中执行
3. 处理 SMEM 张量的创建

### Phase 2: 完整算法实现

**4 阶段流程**:
```
Stage 1: 反演 8 个 8×8 对角块
├─ 16 组线程，每组 8 线程
├─ 使用 Gauss-Jordan 消元
└─ 结果存储在 SMEM

Stage 2: 建立 16×16 块（Schur 补）
├─ 组织 4 个 Warp 对 4 个块
├─ 计算 DC = -D*C
├─ 计算 O = -DC * A_inv
└─ 使用 Tensor Core MMA 加速

Stage 3: 建立 32×32 块（Schur 补）
└─ 类似 Stage 2，但操作更大的块

Stage 4: 建立完整 64×64 逆矩阵
├─ 最后的 Schur 补计算
├─ 协调所有 128 线程
└─ 结果存储在 SMEM
```

### Phase 3: 性能优化

**Tensor Core 集成**:
- 使用 MMA (Matrix Multiply-Accumulate)
- TMA (Tensor Memory Accelerator) 快速传输
- 布局优化 (swizzling)

**同步优化**:
- 减少屏障同步次数
- 启用管道覆盖
- 重叠计算和通信

## 关键代码片段

### 线程索引计算
```python
tidx, _, _ = cute.arch.thread_idx()  # 0-127
linear_idx = tidx + i * 128          # i = 0..31
m_idx = linear_idx // 64             # 0-63
n_idx = linear_idx % 64              # 0-63
```

### 屏障使用
```python
self.cuda_wg_sync_barrier.arrive_and_wait()
# 所有 128 线程等待，无线程继续执行
# 避免数据竞争和内存可见性问题
```

### 访问模式
```python
# 读操作 (GMEM)
val = mat[m_idx, n_idx]

# 写操作 (SMEM 模拟)
temp_buffer[m_idx, n_idx] = val

# 写操作 (GMEM)
mat[m_idx, n_idx] = val
```

## 验证清单

- ✅ GMEM 加载逻辑正确
- ✅ 线程分布完整覆盖
- ✅ 无内存访问冲突
- ✅ 屏障同步有效
- ✅ GMEM 存储逻辑正确
- ✅ 编译成功
- ✅ 执行成功
- ✅ 所有测试通过
- ⏳ compute_matrix_inverse_64x64 集成
- ⏳ 真实 SMEM 使用
- ⏳ 完整矩阵求逆验证

## 性能基准

| 操作 | 时间 | 带宽 |
|------|------|------|
| 编译 | 0.21 秒 | N/A |
| 执行（平均） | 0.77 ms | 10.6 GB/s |
| 单次迭代 | 0.77 ms | 10.6 GB/s |
| 总数据量 | - | 32 KB |

## 技术债务

1. **SMEM 实现**
   - 使用寄存器缓冲代替真实 SMEM
   - 需要整合 SharedStorage

2. **计算部分**
   - 4 阶段算法尚未集成
   - compute_matrix_inverse_64x64 有编译错误
   - 需要 warp 级别的操作

3. **同步机制**
   - 当前使用简单的全局屏障
   - 可能需要更细粒度的控制

## 参考文档

- `docs/SHARED_STORAGE_IMPLEMENTATION.md`: SharedStorage 设计
- `docs/KERNEL_EXECUTION_REPORT.md`: 内核执行指南
- `SHARED_STORAGE_PROGRESS.md`: 开发进度总结

## 下一步行动

1. **立即** (今天)
   - 修复 compute_matrix_inverse_64x64 编译问题
   - 集成到 kernel 函数

2. **本周**
   - 实现 Stage 1（8×8 块反演）
   - 测试 8×8 求逆正确性

3. **周末**
   - 实现 Stage 2-4
   - 完整 64×64 求逆验证

## 相关代码

- **flashla/inv.py**: 主内核实现（798 行）
- **test_inv_kernel_execution.py**: 测试套件（300 行）
- **docs/SHARED_STORAGE_IMPLEMENTATION.md**: SharedStorage 文档（428 行）

---

**状态**: ✅ GMEM 加载完成，等待下一阶段
