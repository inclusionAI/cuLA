# ChunkDeltaRuleFwdH Pipeline

> 文件: `cula/ops/chunk_delta_h.py`  
> 类名: `ChunkDeltaRuleFwdH`

## 计算公式

V2 (No GMEM Roundtrip, Register-Carry) 版本:

$$wh = h \times W$$
$$v_{new} = u - wh$$
$$\text{update} = v_{new}^T \times K^T$$
$$h_{new} = 2^{gk} \cdot h + \text{update}$$

核心特征: **h state 在寄存器中跨 chunk 传递**，消除 GMEM roundtrip 开销。

## 线程布局

| Warp | 角色 | 职责 |
|------|------|------|
| 0-3 | CUDA Core (128 threads) | h state 持有 (寄存器)、v_new 计算、gk decay、R2T/T2R/R2S |
| 4 | MMA | 执行 WH MMA 和 KV MMA |
| 5 | Load | TMA 加载 W、K^T、U、gk、h0 |
| 6 | Store | TMA S2G 写 h_out 和 v_new |
| 7 | Empty | 占位 |

**总线程**: 256 (8 warps)  
**寄存器分配**: CUDA=232, Others=40, occ=1 (仅支持)

## MMA 操作

| MMA | Tiler (M,N,K) | A 操作数 | B 操作数 | 输出 |
|-----|---------------|----------|----------|------|
| WH | (64, 64, 128) | h state — TMEM K-major | W — SMEM K-major | acc_wh (FP32) |
| KV | (64, 128, 64) | v_new^T — TMEM K-major | K^T — SMEM MN-major | update (FP32) |

KV MMA 按 K 维度分块累加 (`ACCUMULATE = kp != 0`)。

## Pipeline 阶段

```
     Load Warp           CUDA Warps                MMA Warp              Store Warp
        │                    │                         │                      │
  h0→sH0 ┼──h0_mbar──────>│                         │                      │
        │                    │ h=load(h0)             │                      │
        │                    │                         │                      │
  ┌─── Chunk Loop (per sequence) ───────────────────────────────────────────┐
  │     │                    │                         │                      │
  │ W→sW ┼──load_w_mbar──────────────────────────────>│                      │
  │     │                    │ h(fp32→bf16)→R2T TMEM  │                      │
  │     │                    ├──state_tmem_mbar──────>│                      │
  │     │                    │                         │ WH MMA              │
  │     │                    │                         │ h×W→wh              │
  │     │                    │<──wh_done_mbar─────────┤                      │
  │ U→sU ┼──load_u_mbar───>│                         │                      │
  │     │                    │ T2R wh                  │                      │
  │     │                    │ v_new=u-wh              │                      │
  │     │                    │ v_new→R2T TMEM         │                      │
  │     │                    ├──vnew_smem_mbar───────>│                      │
  │ Kt→sKt┼──load_kt_mbar──────────────────────────>│ KV MMA              │
  │     │                    │                         │ vnew^T×Kt→update    │
  │ gk→sGK┼──load_gk_mbar─>│                         │                      │
  │     │                    │ gk decay (overlap)      │                      │
  │     │                    │ h *= exp2(gk)           │                      │
  │     │                    │<──kv_done_mbar─────────┤                      │
  │     │                    │ T2R update              │                      │
  │     │                    │ h = h + update          │                      │
  │     │                    │ R2S h→sH_epi           │                      │
  │     │                    ├──h_out_mbar───────────────────────────────>│
  │     │                    │ R2S v_new→sVnew        │              TMA h,vnew
  │     │                    ├──vnew_store_mbar──────────────────────────>│
  └─────┴────────────────────┴─────────────────────────┴──────────────────┘
```

### Pipeline 深度

| Pipeline | 类型 | Stages | 方向 |
|----------|------|--------|------|
| load_w | TMA→MMA | 3 | Load→MMA |
| load_kt | TMA→MMA | 3 | Load→MMA |
| load_u | TMA→CUDA | 3 | Load→CUDA |
| load_gk | TMA→CUDA | 3 | Load→CUDA |
| load_h0 | TMA→CUDA | 1 | Load→CUDA |
| state_tmem | CUDA→MMA | 1 | h state TMEM 就绪 |
| wh_done | MMA→CUDA | 2 | WH acc 就绪 |
| vnew_smem | CUDA→MMA | 1 | v_new TMEM 就绪 |
| kv_done | MMA→CUDA | 1 | KV update 就绪 |
| h_out | CUDA→Store | 2 | sH_epi 就绪 |
| vnew_store | CUDA→Store | 2 | sVnew_store 就绪 |

## 内存分配

### TMEM

| 区域 | 大小 | 用途 |
|------|------|------|
| WH ACC | BV×BT FP32 | WH MMA 累加器 |
| State A | BV×BK BF16 | h state 做 MMA A 操作数 |
| Vnew A | BV×BT BF16 | v_new 做 MMA A 操作数 |
| KV ACC | BV×BK FP32 | KV MMA 累加器 |

### SMEM (~228KB, occ=1)

| 缓冲 | 大小 | Stages |
|-------|------|--------|
| sW | MMA-B layout | 3 |
| sKt | MMA-B layout | 3 |
| sU | BV×BT BF16 COL_MAJOR | 3 |
| sGK | BK FP32 | 3 |
| sH_epi | BV×BK BF16 COL_MAJOR | 2 |
| sVnew_store | BV×BT BF16 | 2 |
| sH0 | BK×BV FP32 | 1 |

### RMEM (关键: h state 寄存器跨 chunk 传递)

`tTR_rKV` — 持有完整 h state (BK×BV FP32)，在 CUDA warps 0-3 的寄存器中生存整个 sequence 的所有 chunk 迭代。

## 主循环流程

每个 WU = 一个 BV tile × 一个 sequence (所有 chunks):

### Per-Chunk 迭代:

1. **Phase 1 — 发布 h state**:
   - CUDA warps: h (FP32→BF16) → R2T 写 TMEM + R2S 写 sH_epi
   - Signal `state_tmem` → MMA 开始 WH MMA

2. **Phase 1 (overlap)**: 预加载 U 从 sU 到寄存器

3. **Phase 2 — 计算 v_new**:
   - 等待 WH 完成 → T2R 读 wh
   - 计算 `v_new = u - wh`
   - R2T 写 v_new 到 TMEM → signal MMA 开始 KV MMA

4. **gk decay (与 KV MMA 重叠)**:
   - 等待 sGK → 128 CUDA threads 协作计算 `exp2(gk)` (128 个 scale)
   - 应用 `h *= scale`

5. **Phase 4 — h state 更新**:
   - 等待 KV 完成 → T2R 读 update
   - 计算 `h = h + update`

## Persistent / Varlen 支持

### Persistent (仅 varlen 模式)
- Load warp 用 `atomicAdd` 获取 work_idx → 写 sWorkIdx → arrive `sched_mbar`
- 其他 warp wait `sched_mbar` → 读 sWorkIdx → arrive `sched_consumed_mbar` (反压)
- **双缓冲防 ABA**: `consumed_mbar` count=7 (MMA=1, CC=4, Store=1, Empty=1)

### Varlen
- `domain_offset` + `chunk_offsets` 处理变长序列映射

## 关键优化

- **寄存器 carry h state**: h 在 CUDA warp 寄存器中跨 chunk 传递，消除 GMEM roundtrip — 这是 V2 版本的核心改进
- **零拷贝 TMEM A 操作数**: h state 和 v_new 直接 R2T 到 TMEM
- **3-stage TMA pipeline**: W、K^T、U、gk 均 3 级预取，最大化 TMA 与计算重叠
- **gk decay 与 KV MMA 重叠**: gk 衰减计算发生在 KV MMA 执行期间
- **协作 gk 预计算**: 128 CUDA threads 协作计算 128 个 exp2(gk)，通过 `gk_precompute_bar` (barrier_id=3) 同步
- **双缓冲 mbarrier 反压**: 防止 persistent 模式下的 ABA 死锁
