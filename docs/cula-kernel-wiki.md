---
title: cuLA Kernel Wiki — Pitfalls & Accuracy Verification
description: Summarized knowledge from KevinZeng08's cuLA PRs on GPU kernel correctness verification and common pitfalls for Linear Attention on Blackwell (SM100) and Hopper (SM90).
tags: [linear-attention, KDA, warp-specialization, TMA, mbarrier, cross-proxy-fence, UMMA, tcgen05, SM100, SM90, accuracy, determinism]
knowledge_cutoff: 2026-05-25
---

# cuLA Kernel Wiki: 精度验证 & 踩坑指南

> cuLA (CUDA Linear Attention) 是一个高性能 Linear Attention CUDA kernel 库，针对 NVIDIA Blackwell (SM100) 和 Hopper (SM90) GPU 进行了深度优化。本文总结 [KevinZeng08 的 cuLA PRs](https://github.com/inclusionAI/cuLA/pulls?q=author%3AKevinZeng08) 中涉及精度验证方法和可能踩坑点的关键经验。

---

## 一、如何验证 Kernel 精度

### 1.1 多维度精度指标体系

cuLA 在 benchmark 和测试中采用**多指标并行**的验证策略，不依赖单一指标判断正确性：

| 指标 | 含义 | 典型阈值 (bf16) | 来源 PR |
|------|------|----------------|---------|
| `RMSE` / `rel_rmse` | 相对均方根误差 | < 0.0004 | [#27](https://github.com/inclusionAI/cuLA/pull/27), [#73](https://github.com/inclusionAI/cuLA/pull/73) |
| `rel_max` | 相对最大误差 | < 0.005 (通常 < 0.003) | [#54](https://github.com/inclusionAI/cuLA/pull/54) |
| `err_ratio` | 逐元素误差均值/参考值均值 | < 0.002 | [#54](https://github.com/inclusionAI/cuLA/pull/54) |
| `mean_diff` | 均值差异 | 接近 0 | [#73](https://github.com/inclusionAI/cuLA/pull/73) |

**Forward + Backward E2E 验证** (PR [#54](https://github.com/inclusionAI/cuLA/pull/54)) 会对**每个输出 tensor 独立报告**精度：

```
o, ht, dq, dk, dv, dg, dbeta, dh0
```

其中 `dg`（gate 梯度）的 `err_ratio` 通常最大（~0.001），因为 gate 的指数运算放大了误差。

### 1.2 Determinism Check（确定性验证）

**这是 cuLA 最核心的验证手段之一**，用于检测概率性 race condition。

来源：PR [#77](https://github.com/inclusionAI/cuLA/pull/77), [#68](https://github.com/inclusionAI/cuLA/pull/68)

```python
def check_determinism(fn, *args, iters=10000):
    """运行 kernel 多次，验证输出 bit-exact 一致"""
    ref = fn(*args)
    for i in range(iters):
        out = fn(*args)
        if not torch.equal(ref, out):
            # 检查 NaN/Inf
            assert not out.isnan().any(), f"NaN at iter {i}"
            assert not out.isinf().any(), f"Inf at iter {i}"
            # 报告不一致的位置和值
            diff_mask = ref != out
            raise AssertionError(f"Non-deterministic at iter {i}, "
                                 f"diff count={diff_mask.sum()}")
```

**关键参数**：
- 迭代次数需要 **≥10K**，某些 timing-sensitive 的 bug 需要 **100K+** 才能复现（如 PR #77 的 cross-proxy fence bug）
- 比较必须使用 `torch.equal`（bit-exact），而非 `allclose`
- 同时检查 NaN/Inf，可以区分"值漂移"和"完全损坏"

**何时需要 determinism check**：
- 涉及 warp specialization 的 pipeline 修改
- 修改 mbarrier arrival count 或 fence
- 修改 TMA pipeline 的 acquire/release 顺序
- 任何 SMEM buffer 复用逻辑

### 1.3 参考实现对比

cuLA 使用 **FLA (Flash Linear Attention)** Triton 实现作为 ground truth：

```python
# 典型测试模式
ref_output = fla_chunk_gla(q, k, v, g)  # FLA Triton 参考
cula_output = cula_chunk_kda(q, k, v, g)  # cuLA CUDA kernel

# 精度比较
rmse = (ref_output - cula_output).pow(2).mean().sqrt()
rel_max = (ref_output - cula_output).abs().max() / ref_output.abs().max()
```

**验证覆盖维度** (来源 PR [#27](https://github.com/inclusionAI/cuLA/pull/27), [#54](https://github.com/inclusionAI/cuLA/pull/54), [#73](https://github.com/inclusionAI/cuLA/pull/73))：

| 维度 | 变化范围 | 目的 |
|------|---------|------|
| B (batch) | 1, 2 | 多 batch 的 stride 正确性 |
| T (seq_len) | 512, 1024, 4096, 8192, 16384 | 覆盖 tail chunk 场景 |
| H / HV | H=64, H=16+HV=64 (GVA) | 覆盖 GQA/GVA 的 head 映射 |
| varlen 分布 | uniform, random, skewed | 序列长度不均匀时的边界处理 |
| 功能 flag | `disable_recompute`, `output_final_state`, `has_init_state` | flag 组合正确性 |

### 1.4 Tail Chunk 边界测试

**绝对不能只测 T % chunk_size == 0 的情况。**

来源：PR [#42](https://github.com/inclusionAI/cuLA/pull/42) (sanity check 重构)

```python
# 必须覆盖的序列长度
T_values = [63, 500, 1000, 512, 1024, 4096]  # 63 = 64-1 (tail chunk = 1 token)
```

`T=63` 是极端 case：整个序列就是一个 tail chunk，chunk 中最后有效 token 在 position 62，position 63 是 padding。

---

## 二、可能踩坑的地方

### 坑 1: Cross-Proxy Fence — TMA 与 CUDA Core 之间的内存序

**严重程度**: ★★★★★ (Silent corruption, 100K+ 迭代才复现)

**PR**: [#77](https://github.com/inclusionAI/cuLA/pull/77) | **架构**: SM100

**现象**：`recomp_wu` kernel 在 100K+ 次迭代后偶发输出不一致。

**根因**：GPU 有两个内存访问 proxy：
- **Generic proxy**：CUDA Core 的 `ld.shared` / `st.shared`
- **Async proxy**：TMA 的 `cp.async.bulk`

`mbarrier.arrive`（`consumer_release` 内部）只保证 generic proxy 内的内存序。当 Prologue Warp Group 通过 `ld.shared` (LDS/S2R) 从 sQ 读取后调用 `consumer_release()`，TMA（async proxy）**看不到** CUDA Core 的读取是否完成。

```
时序竞争:
  Prologue Warp:  ld.shared sQ → consumer_release() [mbarrier.arrive 先完成]
  TMA Warp:       producer_acquire() 成功 → cp.async.bulk 覆写 sQ
  Prologue Warp:  ld.shared 仍在读取 sQ... → 读到脏数据！
```

**修复**：在 `consumer_release()` 前加 `fence_view_async_shared()`：

```cpp
fence_view_async_shared();  // fence.proxy.async.shared::cta
q_pipeline.consumer_release(q_pipe_state_read);
```

**规则**：

> ⚠️ **凡是 CUDA Core 读取或写入了 SMEM，且该 buffer 即将归还给 TMA 使用，必须在 `consumer_release()` 前插入 `fence_view_async_shared()`。`ld.shared` 和 `st.shared` 都需要！**

| 方向 | 是否需要显式 fence | 原因 |
|------|-------------------|------|
| Async → Generic (TMA → Core) | ❌ 不需要 | TMA completion 隐式包含 proxy fence |
| Generic → Async (Core → TMA) | ✅ **需要** | release 不扩展到 async proxy |

**错误尝试**（commit `4cf525e`）：在 `consumer_wait()` 后加 fence（方向错误，冗余但无害）。

---

### 坑 2: Mbarrier Thread Count 与 elect_one_sync

**严重程度**: ★★★★ (间歇性调度错乱)

**PR**: [#69](https://github.com/inclusionAI/cuLA/pull/69) | **架构**: SM100

**现象**：`chunk_delta_h` 动态调度中 `work_idx` 读取不一致。

**根因**：`sched_mbar` 的 arrival count 设为 `1 * num_warps`（使用 `elect_one_sync`，每 warp 只 1 线程 arrive）。但 **consumer 端所有 32 线程都需要读 `work_idx`**。

`elect_one_sync` 使得只有 lane 0 参与 mbarrier wait → 其他 31 个线程可能在 mbarrier 同步完成前就已经读取了过期的 `work_idx`。

**修复**：arrival count 从 `1 * num_warps` 改为 `32 * num_warps`，全线程参与。

**规则**：

> ⚠️ **如果被 mbarrier 保护的数据需要被 warp 中所有线程读取，arrival count 必须 = 消费线程数。不能用 `elect_one` 代替全线程 arrive。**

`elect_one_sync` 只适用于：producer 端（只有一个线程写 `work_idx`）和只需要一个线程执行的操作。

---

### 坑 3: UMMA (tcgen05) 需要专用 Pipeline 变体

**严重程度**: ★★★★ (非确定性错误)

**PR**: [#68](https://github.com/inclusionAI/cuLA/pull/68) | **架构**: SM100

**现象**：SM100 KDA kernel 使用通用 `PipelineAsync` 时出现 UMMA 相关竞争。

**根因**：SM100 的 tcgen05 (UMMA) 的 `umma_arrive` 有特殊的 thread election 语义——只有 elected thread 执行 arrive。通用 pipeline 的 arrival count 和同步逻辑无法正确处理这一点。

**修复**：切换到 CUTLASS UMMA 专用 pipeline：

```cpp
// ❌ 错误：通用 pipeline
using Pipeline = PipelineAsync<...>;

// ✅ 正确：UMMA 专用 pipeline
using Pipeline = PipelineUmmaAsync<...>;
using ConsumerPipeline = PipelineUmmaConsumerAsync<...>;
using TmaPipeline = PipelineTmaUmmaAsync<...>;
```

arrival count 调整为 1（匹配 `umma_arrive` 的 single-thread election）。

**规则**：

> ⚠️ **SM100 上使用 tcgen05/UMMA 时，必须使用 CUTLASS 的 `PipelineUmma*` 系列变体。通用 pipeline 无法正确处理 UMMA 的 arrive 语义。**

---

### 坑 4: TMA Out-of-Bounds 不保证零填充

**严重程度**: ★★★★ (Tail chunk garbage output)

**PR**: [#42](https://github.com/inclusionAI/cuLA/pull/42), 相关 [#38](https://github.com/inclusionAI/cuLA/pull/38) | **架构**: SM100

**现象**：`output_final_state=True` 在 `T % 64 != 0` 时产生 ~1e27 量级的 garbage。

**根因**：非 varlen 的 fused forward 中，tail chunk（如 T=500, chunk_size=64）有两个问题：

1. **`g_last` 选错位置**：始终从 `index_q == 63` 取 gate decay factor，但 tail chunk 最后有效 token 在 `(T % C) - 1`。位置 63 落在 TMA padding 区域，值 **undefined**（不是 0！）

2. **K padding 参与 MMA**：`K^T @ NewV` 累加了 padding 位置的 K 值。TMA 的 out-of-bounds load **不保证零填充**，这些 undefined 值被累加进 KV state。

**规则**：

> ⚠️ **永远不要假设 TMA out-of-bounds 地址是零。Tail chunk 必须：(1) 显式选取正确的 last valid position；(2) 对 padding 区域做显式 mask/清零。**

---

### 坑 5: CUDA 编译器版本导致性能回退

**严重程度**: ★★★ (性能问题，非正确性)

**PR**: [#61](https://github.com/inclusionAI/cuLA/pull/61) | **架构**: SM100

**现象**：CUDA 13.0 的 `recomp_wu` 比 CUDA 12.9 慢约 5%，相同源码不同编译器产出不同性能。

**根因**：不同 CUDA 编译器版本的 **register allocator** 行为不同。CUDA 13.0 在某些场景的寄存器分配策略不如 12.9 最优，导致更多 register spill。

**应对**：

```cpp
// 按 CUDA 版本条件化 register allocation 常量
#if __CUDACC_VER_MAJOR__ >= 13
  static constexpr int kRegCount = 40;  // 13.x 最优
#else
  static constexpr int kRegCount = 32;  // 12.x 最优
#endif
```

**规则**：

> ⚠️ **升级 CUDA 版本后必须重新 benchmark。不同编译器版本的 register allocation 可能导致 5%+ 性能差异。对性能敏感的 kernel 应按版本分离优化策略。**

---

### 坑 6: Warp-Specialized Kernel 的 Pipeline 重排陷阱

**严重程度**: ★★★★ (Silent incorrect output)

**PR**: [#27](https://github.com/inclusionAI/cuLA/pull/27), [#61](https://github.com/inclusionAI/cuLA/pull/61) | **架构**: SM100

**现象**：在 `recomp_wu` 中，当 `B > 1` 时 `v` tensor 未正确 rearrange 导致 kernel 读取错误数据。

**根因**：在 warp-specialized kernel 中，多个 pipeline stage 的数据经过 Python 层的 `rearrange` 后传入 C++ kernel。如果某个 tensor 的 rearrange 逻辑遗漏（如只处理了 `q, k, g` 但忘了 `v`），在 `B=1` 时碰巧 stride 兼容不会报错，`B > 1` 时 stride 错误导致 silent data corruption。

**规则**：

> ⚠️ **所有传入 CUDA kernel 的 tensor 必须在 Python wrapper 中做一致的 layout 转换。用 `B > 1` 测试暴露 stride 不一致问题——`B=1` 往往碰巧正确。**

---

### 坑 7: disable_recompute Flag 的全栈一致性

**严重程度**: ★★★ (功能 bug)

**PR**: [#27](https://github.com/inclusionAI/cuLA/pull/27), [#54](https://github.com/inclusionAI/cuLA/pull/54) | **架构**: SM100

**现象**：`disable_recompute=True` 时 forward 多存了 QG，但 backward 仍然尝试 recompute，导致计算不一致。

**根因**：Forward/Backward 共享配置 flag 时，**必须在两个方向都正确传递并使用**。PR #27 添加 forward 支持后，PR #54 才补齐 backward 的适配。

**规则**：

> ⚠️ **任何影响 forward 输出 tensor 集合的 flag（如 `disable_recompute`, `output_final_state`），必须同时在 forward 和 backward 全链路实现。测试时 fwd+bwd e2e 一起验证。**

---

## 三、性能优化经验

### 3.1 Register Allocation 优化 (PR [#27](https://github.com/inclusionAI/cuLA/pull/27), [#61](https://github.com/inclusionAI/cuLA/pull/61))

**Kernel**: `kda_fwd_recomp_w_u_mainloop_sm100`  
**提升**: 1.29x → 1.43x vs FLA Triton (GB200, T=16384)

| 手段 | 效果 |
|------|------|
| 分离 Q pipeline stage | 减少寄存器 live range，消除 spill |
| Epilogue 改 quarter-tile 增量 | 减少 TMEM load 循环开销 |
| StoreQG flag 条件化常量 | 不同模式不互相占寄存器 |
| CUDA 版本分离策略 | 匹配不同编译器的 RA 行为 |

### 3.2 State Layout 选择 (PR [#33](https://github.com/inclusionAI/cuLA/pull/33))

**Kernel**: KDA Hopper fused forward (SM90)  
**Trade-off**: Prefill 约 -2%，Decode 显著提升

将 state 从默认 layout 改为 **VK transposed layout**：prefill 阶段有微小性能损失，但 decode 阶段的 state 访问模式变为 coalesced，整体端到端收益为正。

> **经验**：State layout 选择应以**端到端推理性能**为目标，而非单阶段最优。

### 3.3 Backward 复用 Forward Kernel (PR [#54](https://github.com/inclusionAI/cuLA/pull/54))

**Kernel**: `recompute_wu` + `chunk_delta_h`  
**提升**: E2E forward+backward 约 +8% (1.08x speedup)

Backward 中的 `Δh` 计算和 forward 中的 `recompute_wu` 共享相同的计算模式。直接复用优化后的 forward kernel 到 backward path，避免维护两套实现。

### 3.4 GVA (Grouped Value Attention) 全栈支持 (PR [#73](https://github.com/inclusionAI/cuLA/pull/73))

**架构**: SM100 | **提升**: GVA (HV=64, H=16) 达到 1.45x–1.49x vs FLA

核心优化：将 Q/K 和 V/O/state 的 head 维度彻底分离，避免 `repeat_interleave`：

```
Grid: B × HV (value heads)
Q/K 按 qk_head_idx = v_head_idx / heads_per_group 索引
V/g/beta/O/state 按 v_head_idx 索引
```

`heads_per_group` 在 host 计算一次、存入 `Params`，device 端零开销。

---

## 四、检查清单

### 开发新 Kernel 时

- [ ] 对 FLA 参考实现比较 `rel_rmse < 0.0004`, `rel_max < 0.005`
- [ ] 测试 `T = 63, 500, 1000, 4096, 16384`（覆盖 tail chunk）
- [ ] 测试 `B = 1, 2`（暴露 stride 问题）
- [ ] 测试 varlen（uniform + random + skewed 分布）
- [ ] 所有 flag 组合：`disable_recompute × output_final_state × has_init_state`
- [ ] Determinism check ≥ 10K 次迭代

### 修改 Pipeline / Sync 时

- [ ] Determinism check ≥ 100K 次迭代
- [ ] 确认 CUDA Core → TMA 方向有 `fence_view_async_shared()`
- [ ] 确认 mbarrier arrival count = 实际消费线程数
- [ ] 如涉及 UMMA，使用 `PipelineUmma*` 变体
- [ ] 检查 tail chunk 的 TMA padding 是否显式 mask

### 升级 CUDA 版本后

- [ ] 全量 benchmark 对比（CUDA 12.x vs 13.x）
- [ ] 检查 register spill（`nvcc --ptxas-options=-v`）
- [ ] 如有回退，考虑版本条件化 register allocation 常量

