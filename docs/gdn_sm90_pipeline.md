# Fully Fused GDN Prefill SM90 Pipeline

> 文件: `cula/ops/gdn/sm90/delta_rule.py`
> 类名: `_FullyFusedDeltaRuleSm90`

## 计算公式

内核以 64 token 为一个 chunk，融合 Gated DeltaNet 的块内辅助矩阵、输出和
recurrent state 更新。对当前 chunk 的 $Q,K,V$，核心步骤为：

$$A_{QK} = QK^T, \qquad A_{KK} = KK^T$$

$$T = \operatorname{tril}(I + \Gamma \odot A_{KK}), \qquad
T^{-1} = \operatorname{inverse}(T)$$

$$V_{new} = (V^T - S_{prev}K^T)T^{-1}$$

$$O^T = S_{prev}Q^T + V_{new}A_{QK}^T$$

$$S_{new} = \operatorname{decay}(S_{prev}) + V_{new}K$$

其中 $\Gamma$ 由 `alpha` 的 chunk 内前缀积构造，`beta` 在三角逆的前后按行、
按列应用。公开 state 始终采用 `[sequence, output_head, V, K]` 的 FP32 布局。

## 线程布局

每个 CTA 对应一个 `(sequence, output_head)`，按顺序处理该序列的所有 chunk。

| Warp Group / Warp | 角色 | 职责 |
|---|---|---|
| WG0, warp 0 | Store O | TMA S2G 写回输出 |
| WG0, warp 1 | Load QKV | TMA G2S 加载 Q、K、V |
| WG0, warp 2 | Load Beta | 标量加载 beta，尾块补零 |
| WG0, warp 3 | Load Alpha | 标量加载 alpha、前缀积和 scale，尾块补一 |
| WG1 | State Math 0 | 持有一半 FP32 state，执行 O1、SK、NewV、O2 和 state 更新 |
| WG2 | State Math 1 | 持有另一半 FP32 state，与 WG1 按 named barrier 有序协作 |
| WG3 | Auxiliary Math | WGMMA 计算 QK/KK、三角变换和 64×64 collective inverse |

**总线程**: 512（16 warps，4 个 warp group）

**Grid**: `(num_sequences * num_output_heads, 1, 1)`

**寄存器目标**: Load/Store=24，State Math=192，Auxiliary Math=104

**驻留目标**: 每个 SM 1 个 CTA

## MMA 操作

| MMA | 逻辑 Tiler (M,N,K) | A 操作数 | B 操作数 | 输出 | 用途 |
|---|---|---|---|---|---|
| QK | (64, 64, 128) | Q — SMEM | K — SMEM | FP32 acc | 块内 query-key 分数 |
| KK | (64, 64, 128) | K — SMEM | K — SMEM | FP32 acc | 构造下三角传递矩阵 |
| O1 | (128, 64, 128) | State — RMEM/BF16 | Q — SMEM | FP32 acc | 历史 state 对输出的贡献 |
| SK | (128, 64, 128) | State — RMEM/BF16 | K — SMEM | FP32 acc | 历史 state 对 V 的修正 |
| NewV | (128, 64, 64) | V-SK — RMEM/BF16 | inverse(KK) — SMEM | FP32 acc | 生成 chunk 更新值 |
| O2 | (128, 64, 64) | NewV — RMEM/BF16 | QK — SMEM | FP32 acc | chunk 内输出贡献 |
| KV | (128, 128, 64) | NewV — RMEM/BF16 | K — SMEM | FP32 acc | 更新 recurrent state |

WG1/WG2 的完整 FP32 state 常驻寄存器并跨 chunk 传递；需要作为 WGMMA 操作数时
才转换为 BF16。两个 state warp group 使用 named barrier 4/5 保证对共享数学阶段
的访问顺序。

## Pipeline 阶段

```text
 WG0 Load/Store             WG3 Auxiliary Math             WG1/WG2 State Math
       │                            │                               │
 K,Q,V ├── TMA G2S ────────────────┼──────────────────────────────>│
 alpha ├── prefix products ───────>│                               │
 beta  ├──────────────────────────>│                               │
       │                            │ QK / KK WGMMA                  │
       │                            │ triangular epilogue            │
       │                            │ collective inverse             │
       │                            ├── sQK / sKK_inv ─────────────>│
       │                            │                               │ O1 = State@Q
       │                            │                               │ SK = State@K
       │                            │                               │ NewV=(V-SK)@inv
       │                            │                               │ O2=NewV@QK
       │                            │                               │ update State
       │<───────────────────────────┼──────────── sO ───────────────┤
       └── TMA S2G O               │                               │
```

### Pipeline 深度

| Pipeline | Stages | 方向 | 用途 |
|---|---:|---|---|
| Q | 2 | Load → Aux/State | Query TMA G2S |
| K | 3 | Load → Aux/State | Key TMA G2S；同一物理缓冲提供 `(T,D)` / `(D,T)` 两种视图 |
| V | 2 | Load → State | Value TMA G2S |
| O | 2 | State → Store | 输出写回前的 SMEM staging |
| QK | 2 | Aux → State | QK BF16 发布 |
| KK inverse | 2 | Aux → State | 三角逆 BF16 发布 |
| Alpha | 5 | Load → Aux/State | 前缀积、缩放和 state decay |
| Beta | 5 | Load → Aux | 三角矩阵 beta 修正 |

## 内存分配

### RMEM

| 区域 | 持有者 | 用途 |
|---|---|---|
| State tile 0 | WG1 | 完整 FP32 state 的一半，跨所有 chunk 保持 |
| State tile 1 | WG2 | 完整 FP32 state 的另一半，跨所有 chunk 保持 |
| QK / KK acc | WG3 | 辅助 WGMMA 的 FP32 累加器 |
| O / SK / NewV / KV acc | WG1/WG2 | 输出和 state 更新的 FP32 累加器 |

### SMEM（约 185,728 bytes）

| 缓冲 | Stages | 用途 |
|---|---:|---|
| `sQ` | 2 | Query |
| `sK` | 3 | Key；一个物理分配、两个逻辑视图 |
| `sV` | 2 | Value |
| `sO` | 2 | Output staging |
| `sQK` | 2 | QK 的 BF16 发布缓冲 |
| `sKK_inv` / `sKK_opd` | 2 | 同一物理缓冲的 FP16 inverse 与 BF16 operand 视图 |
| `sAlpha` | 5 | alpha 前缀积、累计积和 scale 通道 |
| `sBeta` | 5 | beta |

## 主循环流程

每个 CTA 固定处理一个 sequence/head work unit，不使用 persistent 动态调度：

1. WG0 按 `K → Q → V` 顺序发起当前 chunk 的 TMA 加载，同时准备 alpha/beta。
2. WG3 执行 QK 和 KK WGMMA，应用 causal/tail mask 与 alpha/beta 传递系数。
3. WG3 对 64×64 下三角矩阵执行 `8 → 16 → 32 → 64` collective inverse，
   再将 QK 和 inverse 以 BF16 发布给 state warp groups。
4. WG1/WG2 从寄存器 state 计算 O1、SK、NewV、O2，并把有效 token 的输出写入
   `sO`。
5. WG1/WG2 对 state 施加 chunk decay，再累加 `NewV @ K`；最后一个 chunk 后按需
   写回 FP32 final state。
6. Store warp 只在绝对 packed `seq_end` 范围内提交 TMA store，等待 bulk group
   完成后退出。

尾块的 alpha padding 为 1、beta padding 为 0，避免 padding token 改变 state。

## Varlen / Head 模式

- 输入采用 packed varlen 布局，`cu_seqlens` 为每个 CTA 给出绝对 token 边界。
- 支持 MHA：`Hq = Hk = Hv`。
- 支持 GQA：`Hq` 是 `Hk = Hv` 的整数倍。
- 支持 GVA：`Hv` 是 `Hq = Hk` 的整数倍。
- output head 映射到对应的 query/key/value head；CTA 之间不共享 recurrent state。

## 关键优化

- **FP32 register-carried state**：避免每个 chunk 将 128×128 state 往返 GMEM/SMEM。
- **四 warp-group 专职分工**：加载/写回、辅助矩阵、两半 state 数学可流水重叠。
- **K 单分配双视图**：QK/SK 使用 `(T,D)`，KV 使用 `(D,T)`，不做物理转置或复制。
- **Collective inverse**：按 8、16、32、64 分级构造三角逆，使用整个 auxiliary WG。
- **明确的尾块语义**：alpha/beta 使用中性 padding，TMA 读写限定在当前 packed
  sequence 的绝对边界内。
- **无 fallback**：公开 API 只调度这个 SM90 CuTe DSL backend。
