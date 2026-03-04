# fwd_o (chunk_gla_fwd_o) CuTe DSL Kernel 优化总结

> GPU: NVIDIA GB200 (Blackwell SM100a), 152 SMs, 228KB SMEM/SM
> 算法: O = scale * (q ⊙ 2^g) @ h + tril(Aqk) @ v_new
> 配置: K=V=128, BT=64, bf16 IO, fp32 ACC, fp32 g, 8 warps (256 threads)

---

## 目录

1. [版本演进与性能里程碑](#1-版本演进与性能里程碑)
2. [关键优化详解](#2-关键优化详解)
   - [2.1 occ=2 寄存器优化 (1.07x → 1.43x)](#21-occ2-寄存器优化-107x--143x)
   - [2.2 Persistent Kernel + 双缓冲 TMA (1.03x → 1.59x non-varlen; 1.24x → 1.34x varlen)](#22-persistent-kernel--双缓冲-tma)
   - [2.3 Store Warp 寄存器调优 (varlen +4%)](#23-store-warp-寄存器调优)
   - [2.4 ⭐ 循环不变量外提 + 乘法重排 + 无分支 varlen (varlen 1.28x → 1.63x)](#24-循环不变量外提--乘法重排--无分支-varlen)
3. [最终性能](#3-最终性能)
4. [NCU 指标对比](#4-ncu-指标对比)
5. [核心经验总结](#5-核心经验总结)

---

## 1. 版本演进与性能里程碑

| 阶段 | Non-Varlen | Varlen | 关键改动 |
|------|-----------|--------|--------|
| 初始版本 | 0.85-1.69x | — | 6 warps, TMEM A-op, T2R+R2S epilog |
| occ=2 优化 | **1.36-1.65x** | — | 8 warps, 208 regs, 1-stage pipeline |
| Persistent + 双缓冲 TMA | **1.53-1.59x** | **1.29-1.34x** | grid-stride loop, occ=1, 2-stage TMA, store warp regs 40→168 |
| **外提 + 重排 + 无分支** | **1.54-1.60x** | **1.59-1.63x** | 循环不变量外提, FMA/XU 重排, branchless select |

---

## 2. 关键优化详解

### 2.1 occ=2 寄存器优化 (1.07x → 1.43x)

**提升幅度:** geomean 1.07x → 1.43x (全面 >1.0x)

初始版本仅 6 个 warp，占用率低。参考 chunk_delta_h 的优化模式：

1. **Warp 数量 6→8**：增加 store warp (warp 6) + empty warp (warp 7)，empty warp 是 CUDA warp group 寄存器重分配所必需的
2. **Pipeline 2-stage → 1-stage**：该 kernel 没有 chunk 维度循环，2-stage 只是浪费 ~88KB SMEM 而无法真正 overlap
3. **CUDA warp 寄存器 232→208**：满足 occ=2 预算约束
   ```
   4 × 208 × 32 + 4 × 40 × 32 = 31,744 ≤ 32,768 (occ=2 上限)
   ```
4. **`min_blocks_per_mp=2`**：编译器 hint 确保 occ=2

**经验：** SM100a 上，occ=2 能让两个 CTA 交替执行、互相掩盖延迟，是最基础也是提升最显著的优化。

---

### 2.2 Persistent Kernel + 双缓冲 TMA

**提升幅度：** Non-varlen 1.03x → 1.59x; Varlen 建立 → 1.34x

#### Persistent Kernel（grid-stride loop）

```python
grid = (SM_count, 1, 1)  # 每个 SM 启动 1 个 CTA
for wu_iter in cutlass.range(0, num_iters, unroll=0):
    # 每个 CTA 通过 warp_uniform_idx 解码自己的 work unit
    tile_idx = wu_iter * SM_count + sm_id
    ...
```

优势：
- 消除 CTA launch overhead（varlen 场景下可能有 >8000 个 work unit）
- 所有 pipeline 可以跨 WU 复用，避免反复初始化

#### 双缓冲 TMA（2-stage prefetch）

```python
# occ=1 模式下，SMEM 预算 228KB
# q=2, h=2, v=2, A=2, g=1 stages → ~200KB < 228KB
```

LOAD warp 可以为 WU[i+1] 发起 TMA，同时 CUDA/MMA warps 处理 WU[i]，隐藏 DRAM 延迟。

#### occ=1 vs occ=2 的权衡

| 模式 | Occupancy | SMEM | Stages | 性能 |
|------|-----------|------|--------|------|
| non-persistent | occ=2 | 114KB | 1-stage | 1.03-1.05x |
| **persistent** | **occ=1** | **228KB** | **2-stage** | **1.53-1.59x** |

**经验：** Persistent + double-buffer 远优于 occ=2 + single-buffer。occ=2 的 SM 利用率优势不足以补偿失去 TMA prefetch 的损失。

---

### 2.3 Store Warp 寄存器调优

**提升幅度：** Varlen 1.26x → 1.34x (+~4%)

初始 store warp 只有 40 个寄存器，无法容纳完整 O tile partition（~128 regs for 256 bf16 values），导致严重的 register spill → local memory 访问。

**NCU 发现：** `local stores to DRAM, 1.0/32 bytes per sector utilized`（极低局部性）

**修复策略：**

1. **First pass:** 完全移除 bulk SMEM→REG copy，改为逐行读 SMEM → 写 GMEM
2. **Second pass:** Persistent 模式下 occ=1，寄存器充裕，将 store warp 寄存器 40→168
   ```
   4×32×208 + 4×32×168 = 48,128 ≤ 65,536 (occ=1 上限)
   ```
   恢复 bulk SMEM→REG 预取，让编译器掩盖 SMEM read latency。

**经验：** 不同 warp 角色的寄存器需求差异很大，要根据 occupancy 预算精确分配。NCU local store/load 指标是发现 spill 的关键信号。

---

### 2.4 ⭐ 循环不变量外提 + 乘法重排 + 无分支 varlen

**提升幅度：**
- Varlen: **1.28-1.34x → 1.59-1.63x (+19% ~ +27%)**
- Non-varlen: 1.53-1.59x → 1.54-1.60x (+1%)
- NCU Duration: **403,584 → 328,288 ns (-18.7%)**

这是本轮优化中**提升最大**的一次改动，包含三个正交的优化：

#### 改动 1: T2R/R2T/R2S Setup 代码外提 (最关键)

**问题：** 每次 persistent loop 迭代都重新构建 T2R atom、R2T atom、R2S atom、tiled_copy、identity tensor、坐标映射、register tensor 分配。这些操作仅依赖编译时常量和 `local_tidx`，是 **循环不变量 (loop-invariant)**。

**改动前（循环内）：**
```python
for wu_iter in cutlass.range(0, num_iters, unroll=0):
    # Work unit decode (per-iteration)
    ...
    # ❌ 以下全部在循环内反复执行
    t2r_atom_acc = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
        self.acc_dtype,
    )
    tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
    fake_sQG = cute.make_tensor(...)
    tiled_t2r_acc = tcgen05.make_tmem_copy(t2r_atom_acc, ...)
    thr_t2r_acc = tiled_t2r_acc.get_slice(local_tidx)
    cM_qg = cute.make_identity_tensor(qg_tile)
    tTR_cM_qg = thr_t2r_acc.partition_D(cM_qg)
    # ... R2T QG, R2T AM, R2S 全部类似 ...
    tTR_rQG_fp32 = cute.make_rmem_tensor(...)  # register tensor allocation
    tRT_rQG_bf16 = cute.make_rmem_tensor(...)
    tRT_rAM = cute.make_rmem_tensor(...)
```

**改动后（循环外）：**
```python
# ---- Hoist loop-invariant T2R/R2T/R2S setup ----
# All depend only on compile-time constants and local_tidx.
t2r_atom_acc = cute.make_copy_atom(...)
tiled_t2r_acc = tcgen05.make_tmem_copy(...)
thr_t2r_acc = tiled_t2r_acc.get_slice(local_tidx)
cM_qg = cute.make_identity_tensor(qg_tile)
tTR_cM_qg = thr_t2r_acc.partition_D(cM_qg)
# ... 所有 T2R, R2T, R2S, identity, partition, register tensor 一次性分配 ...
tTR_rQG_fp32 = cute.make_rmem_tensor(...)
tRT_rQG_bf16 = cute.make_rmem_tensor(...)
tRT_rAM = cute.make_rmem_tensor(...)

# ====== Persistent loop (loop body 大幅精简) ======
for wu_iter in cutlass.range(0, num_iters, unroll=0):
    # Work unit decode only
    ...
    # 直接使用预分配的 tensor 和 copy plan
```

**外提的具体对象清单：**

| 类别 | 外提对象 | 说明 |
|------|---------|------|
| **T2R (TMEM→Reg)** | `t2r_atom_acc`, `tiled_t2r_acc`, `thr_t2r_acc` | ACC 读取的 copy atom 和线程分片 |
| **R2T QG (Reg→TMEM)** | `r2t_atom_qg`, `tiled_r2t_qg`, `thr_r2t_qg`, `tRT_tQG` | QG 写入 TMEM 的 copy plan |
| **R2T AM (Reg→TMEM)** | `r2t_atom_am`, `tiled_r2t_am`, `thr_r2t_am`, `tRT_tAM` | AM 写入 TMEM 的 copy plan |
| **R2S (Reg→SMEM)** | `r2s_atom_o`, `tiled_r2s_o`, `thr_r2s_o`, `tRS_sO` | Output epilog 的 R2S copy plan |
| **Identity** | `cM_qg`, `tTR_cM_qg`, `cM_am_r4`, `tRS_cM_am` | QG/AM 的坐标映射 identity tensor |
| **Register Tensor** | `tTR_rQG_fp32`, `tRT_rQG_bf16`, `tRT_rAM` | 计算和数据转换的寄存器 buffer |
| **Epilog** | `tTR_tAcc` | ACC 读取的分片 |

**NCU 效果：**
- 总指令数: 94.36M → **81.49M (-13.6%)**
- 分支指令: 1.97M → **0.78M (-60.2%)**
- No Instruction stall: 0.264 → **0.093 (-65%)** — 指令流水线空泡大幅减少
- Wait stall: 0.772 → **0.502 (-35%)**

**原理：** CuTe DSL 中，`make_copy_atom` / `make_tmem_copy` / `get_slice` / `partition_D` / `make_identity_tensor` 等操作虽然看似是"元编程"，但编译后会生成 ALU 指令（地址计算、stride 操作、索引映射）。在 persistent loop 中反复执行这些不变操作，白白消耗 ALU 管线和指令 cache。外提后，这些代码只执行一次，loop body 只保留真正与数据相关的计算。

---

#### 改动 2: 乘法重排 — FMA 与 XU 管线重叠

**改动前：**
```python
tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val) * scale_f32
```

**改动后：**
```python
qs_val = q_val * scale_f32       # FMA pipe
tTR_rQG_fp32[ei] = qs_val * cute.exp2(g_val)  # XU pipe (exp2) + FMA pipe (multiply)
```

**原理：** `exp2()` 是特殊函数指令 (SFU/XU pipe)，`q * scale` 是普通 FMA。改动前 `q * exp2(g)` 必须等 exp2 完成才能乘；改动后 `q * scale` 可以和 `exp2(g)` 并行发射到不同 pipe。

**NCU 效果：**
- Tensor pipe 利用率: 19.9% → **24.0% (+21%)**
- XU pipe 利用率: 14.0% → **16.7% (+19%)**

---

#### 改动 3: 无分支 Varlen（Branchless Select）

**改动前（条件分支）：**
```python
if bt_coord < remaining:
    q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
    g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
    tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val) * scale_f32
else:
    tTR_rQG_fp32[ei] = Float32(0.0)
```

**改动后（无分支）：**
```python
# 无条件 SMEM load — SMEM 总是包含有效数据（当前或下一序列的）
q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
qs_val = q_val * scale_f32
result = qs_val * cute.exp2(g_val)
tTR_rQG_fp32[ei] = cutlass.select_(bt_coord < remaining, result, Float32(0.0))
```

**AM 部分也做了简化：**
```python
# 改动前: 三重条件
if row >= col and row < remaining and col < remaining:
    tRT_rAM[ei] = sA_epi[(row, col, a_h.index)]

# 改动后: row >= col 是 constexpr，col <= row 意味着 col < remaining 隐含于 row < remaining
if row >= col:  # constexpr, 编译期消除
    a_val = sA_epi[(row, col, a_h.index)]
    tRT_rAM[ei] = cutlass.select_(row < remaining, a_val, Float32(0.0).to(self.io_dtype))
```

**原理：**
- 条件分支 (`if bt_coord < remaining`) 会导致 warp divergence：同一 warp 中不同线程走不同路径，序列化执行
- `cutlass.select_` 编译为 `SETP` + `SEL` 指令（无分支），全 warp 统一执行
- 无条件 SMEM load 是安全的：SMEM 中始终有有效数据（TMA 已填充），out-of-bounds 行的计算结果被 select 置零

#### 额外优化: Register Tensor 复用

Output epilog 中，`tTR_rAcc` 和 `tTR_rQG_fp32` 有相同的 shape 且生命周期不重叠（QG 计算完毕后才进入 epilog），因此：

```python
# 改动前 — 分配新 register tensor
tTR_rAcc = cute.make_rmem_tensor(thr_t2r_acc.partition_D(fake_sQG).shape, self.acc_dtype)
cute.copy(tiled_t2r_acc, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)

# 改动后 — 复用已有的
cute.copy(tiled_t2r_acc, tTR_tAcc[(None, None, None, 0)], tTR_rQG_fp32)
```

---

## 3. 最终性能

### vs FLA Triton (H=64)

| 配置 | CuTe DSL / FLA Triton |
|------|----------------------|
| Non-varlen B=2 T=8K | **1.57x** |
| Non-varlen B=2 T=32K | **1.60x** |
| Non-varlen B=4 T=8K | **1.54x** |
| Non-varlen B=4 T=32K | **1.60x** |
| Varlen 20seqs T=8K | **1.63x** |
| Varlen 25seqs T=8K | **1.62x** |
| Varlen 20seqs T=32K | **1.59x** |
| Varlen 25seqs T=32K | **1.62x** |

### 优化历程 (典型配置 Varlen 25seqs T=8K)

```
初始版本  →  occ=2  →  persistent  →  double-buf  →  store warp  →  hoist+branchless
  n/a        n/a        基线建立        1.24x          1.29x          1.62x
```

---

## 4. NCU 指标对比

基于 non-varlen B=2 T=8192 H=64 配置:

| 指标 | Baseline | Optimized | 变化 |
|------|---------|-----------|------|
| **Duration** | 403,584 ns | **328,288 ns** | **-18.7%** |
| IPC | 1.49 | 1.56 | +4.7% |
| **总指令数** | 94,362,616 | **81,491,736** | **-13.6%** |
| **分支指令** | 1,965,232 | **781,480** | **-60.2%** |
| Tensor pipe | 19.9% | **24.0%** | +21% |
| XU pipe | 14.0% | **16.7%** | +19% |
| ALU pipe | 34.69% | 34.37% | -1% |
| **Wait stall** | 0.772 | **0.502** | **-35%** |
| **No Instruction stall** | 0.264 | **0.093** | **-65%** |
| Long Scoreboard | 2.195 | 2.453 | +12% (trade-off) |
| Registers/thread | 208 | 208 | 不变 |

---

## 5. 核心经验总结

### 原则 1: 循环不变量必须外提

CuTe DSL 的 `make_copy_atom` / `make_tmem_copy` / `get_slice` / `partition_D` / `make_identity_tensor` 等看似"元编程"的操作，编译后都会生成真实的 ALU 指令。在 persistent loop 中这些不变量每次迭代都重新计算，是纯粹的浪费。

**判断标准：** 如果一个变量仅依赖 `local_tidx`、编译期常量、TMEM shape，就应外提。

**本次外提的 7 类对象：** T2R atom/tiler/slice, R2T QG atom/tiler/slice/target, R2T AM atom/tiler/slice/target, R2S atom/tiler/slice, Identity tensors (QG/AM), 坐标映射 partitions, Register tensors (QG/AM)。

### 原则 2: 零寄存器增量优化

SM100a CUDA warp 在 208 regs 时已接近 occ=1 上限。任何新增 register tensor 都可能触发灾难性 spill。成功的优化策略：
- 复用已有 register tensor（如 tTR_rQG_fp32 复用为 ACC readback buffer）
- 减少指令而非增加 buffer
- 不变量外提不增加寄存器，因为 tensor 在外层只分配一次

### 原则 3: 利用管线并行性

SM100a 有独立的 FMA / XU (SFU) / Tensor / LSU 管线。通过调整运算顺序可以让不同管线并行工作：
- `q * scale` (FMA) 与 `exp2(g)` (XU) 并行
- SMEM load (LSU) 与 ALU 计算并行（branchless 做法消除分支，让所有线程统一执行 load）

### 原则 4: 消除 warp divergence

条件分支 (`if bt_coord < remaining`) 在 varlen 中导致 warp divergence。改用 `cutlass.select_` 做 branchless select：
- 全 warp 统一执行，无序列化
- SMEM 无条件 load 是安全的（TMA 保证 SMEM 内容有效）
- 简化控制流，减少分支指令 60%

### 原则 5: Persistent > Occupancy

对于 Blackwell 上 SMEM 密集的 kernel：
- Persistent (occ=1, 228KB SMEM, double-buffer TMA) >> Non-persistent (occ=2, 114KB SMEM, single-buffer)
- TMA prefetch 的 latency hiding 价值远超多 CTA 并发

### 原则 6: NCU 驱动优化

每轮优化都应有 NCU profile 支撑：
- `local store/load` → 检测 register spill
- `branch instructions` → 检测不必要的控制流
- `No Instruction stall` → 检测指令流水线空泡
- `pipe utilization %` → 判断瓶颈管线
- Duration 和总指令数是最终判据

---
