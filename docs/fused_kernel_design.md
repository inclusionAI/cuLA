# Fused ChunkDeltaH + FwdO Kernel 设计文档

## 1. 动机

当前 KDA forward 分三步执行:

```
Step 1: chunk_kda_fwd_intra(q,k,v,g,beta) → w, u, kg, Aqk, Akk
Step 2: chunk_delta_h(kg, w, u, gk=g)     → h[NT,H,K,V], v_new[B,T,H,V], ht
Step 3: chunk_gla_fwd_o(q, v_new, g, Aqk, h) → o[B,T,H,V]
```

Step 2 和 Step 3 之间存在 **h (NT×H×K×V = NT×H×128×128 bf16)** 和 **v_new (B×T×H×V bf16)** 的 GMEM 往返。
Fuse 这两步的核心收益:

| 数据         | 当前状态                   | Fuse 后              |
|-------------|--------------------------|----------------------|
| **h**       | chunk_delta_h wr → GMEM → fwd_o rd | 寄存器内消费，**省 2×NT×H×K×V GMEM 带宽** |
| **v_new**   | chunk_delta_h wr → GMEM → fwd_o rd | SMEM 内转发，**省 2×B×T×H×V GMEM 带宽** |
| **h_out**   | 仍写 GMEM（backward 需要）  | 不变                  |
| **ht**      | fp32 reg → GMEM           | 不变                  |

同时，standalone `fwd_o.py` 的 varlen 实现有 bug，通过 fuse 可以自然复用 `chunk_delta_h` 已经验证通过的 persistent varlen scheduler。

---

## 2. 输入 / 输出

### 输入（与当前两个 kernel 的输入之并集）

| Tensor       | Shape                    | Dtype  | 来源           |
|-------------|--------------------------|--------|---------------|
| `k`         | `[B, T, H, K]`           | bf16   | (kg from intra) |
| `w`         | `[B, T, H, K]`           | bf16   | intra output    |
| `u`         | `[B, T, H, V]`           | bf16   | intra output    |
| `gk`        | `[B, T, H, K]`           | bf16   | gate (log2 domain) |
| `q`         | `[B, T, H, K]`           | bf16   | query           |
| `g`         | `[B, T, H, K]`           | bf16   | cumulative gate (for fwd_o qg = q*exp2(g)*scale) |
| `Aqk`       | `[B, T, H, BT]`          | bf16   | intra attention matrix |
| `h0`        | `[B, H, K, V]`           | bf16   | (optional) initial state |
| `cu_seqlens` | `[N+1]`                  | int32  | varlen only     |
| `chunk_offsets` | `[N+1]`              | int32  | varlen only     |
| `scale`     | scalar                   | float  | attention scale |

### 输出

| Tensor       | Shape                    | Dtype  | 用途            |
|-------------|--------------------------|--------|----------------|
| `o`         | `[B, T, H, V]`           | bf16   | 最终注意力输出    |
| `h_out`     | `[NT, H, K, V]`          | bf16   | backward 所需的中间状态 |
| `ht`        | `[B, H, K, V]`           | fp32   | 最终 hidden state |

### 不再需要作为 GMEM tensor 的

| Tensor   | 说明 |
|----------|------|
| `h`      | 原 `[NT,H,K,V]` bf16 → 寄存器中消费，h_out 仍写 GMEM |
| `v_new`  | 原 `[B,T,H,V]` bf16 → SMEM 内转发给 AV MMA |

---

## 3. Grid 与调度

### 3.1 工作分片

延续 `chunk_delta_h` 的 **persistent varlen scheduler**：

```
Grid = (SM_count, 1, 1)             # persistent, SM_count ≈ 152 for GB200
Work unit = (v_tile_idx, hidx, bidx) # same as chunk_delta_h
```

每个 work unit 负责一个 `(v_tile, head, sequence)` 组合的 **全部 NT 个 chunk**。
W 方向（V 维）的 tile 大小 `BV` 是关键设计参数。

### 3.2 BV 选择

**chunk_delta_h 侧**: M=BV, h carried in regs (BV×BK fp32), WH MMA (BV,BT,BK), KV MMA (BV,BK,BT)
**fwd_o 侧**: M=BT=64, QH MMA (BT,BV,BK), AV MMA (BT,BV,BT)

> **核心约束**: chunk_delta_h 的 M 维 = BV，fwd_o 的 N 维 = BV。
> 两个 phase 的 MMA M 维不同：chunk_delta_h 用 BV，fwd_o 用 BT=64。

有两种方案：

#### 方案 A: BV=64（当前 chunk_delta_h 值），fwd_o 适配 BV=64

- fwd_o 的 QH MMA 输出 shape = (BT=64, BV=64) → 适合 M=BT=64
- 意味着 每个 work unit 覆盖 V 的 64 列→ num_v_tiles = 128/64 = 2
- **优势**: SMEM 压力小, 寄存器少 (h = BV×BK = 64×128 fp32 = 8192 个)
- **劣势**: num_v_tiles=2 → total_work_units 翻倍, 但 SM 利用率高

#### 方案 B: BV=128（当前 fwd_o 值），chunk_delta_h 适配 BV=128

- h state 寄存器 = BV×BK = 128×128 fp32 = 16384 个 → **寄存器爆炸**
- 完全不可行

**结论**: BV=64 是唯一可行选择。

---

## 4. 每 chunk 计算流程

对 chunk_idx ∈ [0, NT):

```
 ┌─────────────────── Phase 1: Delta H ───────────────────┐
 │ (与当前 chunk_delta_h 相同)                               │
 │                                                          │
 │ 1.1 h_bf16 = cast(h, bf16)    // h in regs (BV×BK fp32) │
 │ 1.2 R2T state → TMEM          // WH MMA A operand        │
 │ 1.3 R2S h → sH_epi            // for h_out TMA store     │
 │ 1.4 Load U from sU → regs                                │
 │ 1.5 WH MMA: state(TMEM) × W(SMEM) → acc_wh(TMEM)       │
 │ 1.6 T2R acc_wh → regs, v_new = U - WH                   │
 │ 1.7 gk_decay: h *= exp2(gk)                             │
 │ 1.8 R2T v_new → TMEM          // KV MMA A operand        │
 │ 1.9 KV MMA: v_new(TMEM) × K^T(SMEM) → update(TMEM)     │
 │ 1.10 T2R update → regs, h += update                      │
 └──────────────────────────────────────────────────────────┘
          │
          │  v_new 还在 CUDA regs + SMEM (sVnew_store)
          │  h 已在 CUDA regs (phase 1 完成 h 更新)
          │  但 h_bf16 (= phase 1.1 cast 后的旧 h) 已 R2S 到 sH_epi
          ▼
 ┌─────────────────── Phase 2: Fwd O ────────────────────┐
 │                                                        │
 │ 2.1 从 sQ_epi, sG_epi 读 q, g → qg = q * exp2(g) * s │
 │ 2.2 R2T qg → TMEM             // QH MMA A operand      │
 │ 2.3 QH MMA: qg(TMEM) × h(SMEM) → acc(TMEM)           │
 │     ⚠ h(SMEM) = sH_epi? NO! sH_epi 是 COL_MAJOR      │
 │       for BV×BK epilog store, 不是 MMA B 格式！          │
 │       → 需要新 SMEM buffer sH_mma 或复用 h 数据         │
 │                                                        │
 │ 2.4 从 sA_epi 读 A → apply causal mask                  │
 │ 2.5 R2T A_masked → TMEM       // AV MMA A operand      │
 │ 2.6 AV MMA: A_masked(TMEM) × v_new(SMEM) → acc (ACCUM)│
 │     v_new 来自 sVnew_store (已在 SMEM)                   │
 │ 2.7 T2R acc → regs → R2S → sO → TMA store to GMEM      │
 └────────────────────────────────────────────────────────┘
```

---

## 5. TMEM 布局与时分复用

Blackwell SM100 TMEM 容量 = 512 columns。每个 MMA 需要 ACC + A-operand。

### 当前 chunk_delta_h TMEM:

| Region     | Shape      | Type   | Columns |
|-----------|------------|--------|---------|
| WH ACC    | (BV, BT) = (64, 64) | FP32 | ~64   |
| State A   | (BV, BK) = (64, 128) | BF16 | ~64   |
| Vnew A    | (BV, BT) = (64, 64) | BF16 | ~32   |
| KV ACC    | (BV, BK) = (64, 128) | FP32 | ~128  |
| **Total** |            |        | **~288** |

### fwd_o TMEM (standalone):

| Region     | Shape      | Type   | Columns |
|-----------|------------|--------|---------|
| ACC       | (BT, BV) = (64, 64*)| FP32 | ~64   |
| QG A      | (BT, BK) = (64, 128)| BF16 | ~64   |
| AM A      | (BT, BT) = (64, 64) | BF16 | ~32   |
| **Total** |            |        | **~160** |

(*) 注意 fwd_o M=BT=64, N=BV=64 (因为 BV 统一为 64)

### 时分复用策略

Phase 1 (delta-h) 和 Phase 2 (fwd-o) **串行执行**，可以:

1. **ACC 区域复用**: Phase 1 WH ACC 和 KV ACC → Phase 2 FwdO ACC
   - WH ACC (64,64) vs FwdO ACC (64,64): **兼容!** 同样 FP32、同 M=64
   - KV ACC (64,128): Phase 1 结束后不再需要 → 可被 Phase 2 QG/AM 覆盖

2. **A-operand 区域复用**: Phase 1 State_A/Vnew_A → Phase 2 QG_A/AM_A
   - State A (64,128 bf16) 与 QG A (64,128 bf16): **完全相同 shape!** 可复用同一 offset
   - Vnew A (64,64 bf16) 与 AM A (64,64 bf16): **完全相同 shape!** 可复用同一 offset

**结论**: Phase 2 完全复用 Phase 1 的 TMEM layout，**零额外 TMEM**。

```
TMEM Layout (时分复用):
 ┌──────────────────────────────────────┐
 │ Phase 1:                             │
 │   Offset 0:     WH ACC (64,64) FP32  │ ← Phase 2: FwdO ACC (64,64) FP32
 │   Offset +64:   State A (64,128) BF16│ ← Phase 2: QG A (64,128) BF16
 │   Offset +128:  Vnew A (64,64) BF16  │ ← Phase 2: AM A (64,64) BF16
 │   Offset +160:  KV ACC (64,128) FP32 │ ← Phase 2: 闲置（可被 overlap）
 └──────────────────────────────────────┘
 Total: ~288 columns, 不变
```

---

## 6. SMEM 布局

### 6.1 当前 chunk_delta_h SMEM 预算

| Buffer         | Size (bytes) | 用途                      | Fused 后状态    |
|---------------|-------------|--------------------------|----------------|
| sW (3-stage)  | 49,152      | WH MMA B operand          | **保留**        |
| sKt (3-stage) | 49,152      | KV MMA B operand          | **保留**        |
| sH_epi (2-stage)| 32,768   | h_out S2G store           | **保留 + 兼做 QH MMA B** |
| sU (3-stage)  | 24,576      | U TMA load               | **保留**        |
| sVnew_store (2-stage)| 16,384| v_new S2G store       | **兼做 AV MMA B** |
| sGK (3-stage fp32)| 1,536  | gk decay values          | **保留**        |
| barriers + pad| ~1,024      |                          | 保留            |
| **Subtotal**  | **~174,592**|                          |                |

### 6.2 fwd_o 需要的额外 SMEM

| Buffer         | Size (bytes) | 说明                     | 可否复用 |
|---------------|-------------|--------------------------|---------|
| sQ_epi (1-stage)| 16,384    | q TMA load (BT×BK)       | **新增** |
| sG_epi (1-stage)| 16,384    | g TMA load (BT×BK)       | **可与 sQ_epi 合并** ¹ |
| sA_epi (1-stage)| 8,192     | A TMA load (BT×BT)       | **新增** |
| sH_mma (1-stage)| ???       | h for QH MMA B operand    | **复用 sH_epi** ² |
| sV_mma (1-stage)| ???       | v_new for AV MMA B operand| **复用 sVnew_store** ³ |
| sO (1-stage)  | 8,192       | o output R2S → TMA store  | **新增 或复用** ⁴ |

**注释**:
1. sQ_epi 和 sG_epi 如果串行使用（先读 q 再读 g），可共享同一 buffer → **-16,384**
2. sH_epi (COL_MAJOR, BV=64 × BK=128) 用于 h_out 的 TMA S2G 存储。
   QH MMA 需要 h 的 B-operand SMEM (MN-major, BK×BV)。
   **问题**: sH_epi 的 swizzle mode 与 MMA B-operand 不同 → **不能直接复用!**
   **解法**: h 已经在 CUDA regs (tTR_rKV) 中 → 用 R2S 写到一个新的 sH_mma buffer (MMA B-operand format)
   sH_mma = make_smem_layout_b(qh_mma, (BT=64, BV=64, BK=128)) ≈ 16KB (BK×BV=128×64 bf16 × 1 stage)
3. sVnew_store (COL_MAJOR, BV=64 × BT=64) 可能不直接匹配 AV MMA B-operand swizzle。
   但 v_new 也在 regs → 同理 R2S 到 sV_mma (MMA B-operand format)
   sV_mma = make_smem_layout_b(av_mma, (BT=64, BV=64, BT=64)) ≈ 8KB (BT×BV=64×64 bf16)
4. sO (ROW_MAJOR, BT=64 × BV=64) 可复用 sU (3-stage 中的 stage 0)，
   因为 Phase 2 时 U 已经消费完毕。**可行!**

### 6.3 关键问题: sH_epi vs QH MMA B-operand

**现状**: 
- sH_epi 是 COL_MAJOR (BV=64, BK=128) 带 epilog swizzle — 为 TMA S2G store 优化
- QH MMA B-operand 需要 MN-major layout 带 MMA swizzle

**方案 A: 额外分配 sH_mma (16KB)**

```
Phase 1: R2S h → sH_epi (for TMA store h_out)
Phase 2: R2S h → sH_mma (for QH MMA B)  // h 还在 regs, 可以再次 R2S
```
- 但: Phase 2 时 h 已被 gk_decay 修改! Phase 1.3 R2S 的是旧 h，Phase 1.7 做了 gk_decay
- 解法: Phase 2 需要的是 **旧 h** (decay 前的 h, 即 chunk 开始时的 h)
   → 正好就是 Phase 1.1-1.3 R2S 到 sH_epi 的那个 h!
- 但 sH_epi swizzle 不对...

**方案 B: 在 Phase 1 同时 R2S 到 sH_epi (store) + sH_mma (MMA B)**

Phase 1.3 时 h 在 regs → 可以做两次 R2S:
1. R2S → sH_epi (epilog swizzle, for h_out TMA store)
2. R2S → sH_mma (MMA-B swizzle, for QH MMA)

代价: 一次额外 R2S (在 WH MMA compute 期间 overlap)
+ 16KB 新 SMEM buffer

**方案 C: CUDA regs 直写 h 到 TMEM for QH MMA**

QH MMA: qg(BT,BK,TMEM-A) × h(BK,BV,SMEM-B) → acc(BT,BV)
如果 h 不走 SMEM-B, 而是从 CUDA regs → TMEM 作为某种 operand?

问题: tcgen05 MMA B operand **必须来自 SMEM**, 不能来自 TMEM。
且 A operand 位置已被 qg 占用。**不可行。**

**方案 D: 将 QH MMA 改为 M=BV, swapping A/B**

如果 QH MMA 改为: h(BV,BK,TMEM-A) × qg_T(BK,BT,SMEM-B) → acc(BV,BT)
- h 从 TMEM-A → 可以直接从 state_tmem 读 (Phase 1 已写入!)
- qg_T 从 SMEM-B → 需要 qg transposed

**这非常有趣!** Phase 1 写入 state TMEM 的 h 还在那里 (直到被 Phase 1 KV MMA 覆盖)。
但 Phase 1 更新了 TMEM 中的 state (为下一 chunk 准备)，所以**这个 h 是 decay 后的新 h，不是旧 h。**
而 fwd_o 需要的是旧 h (chunk 开始时的 h)。

**正确时序**: 我们需要的 h 是 Phase 1 **开始时** R2T 到 state_tmem 的那份 (Step 1.2)。
但之后 Phase 1 对 TMEM state 位置做了:
- Step 1.5 WH MMA 读了它
- Step 1.7 gk_decay 在 regs 中修改了 h
- Step 1.8 R2T v_new 到 vnew_tmem
- Step 1.10 h += update (regs)

所以 state_tmem 在 MMA 读完后不再被写入! → Step 1.2 写入 TMEM 的旧 h 在 WH MMA 读完后仍然有效!

**但 Phase 1 后面还需要 R2T 新 h 到 state_tmem (下一 chunk 的 Phase 1)...**

实际时序:
```
Chunk i:
  Phase1.2: R2T h_old → state_tmem  (旧 h, for WH MMA)
  Phase1.5: WH MMA reads state_tmem
  Phase1.7-1.10: h regs updated to h_new
  Phase2: need h_old again for QH MMA → state_tmem 中还有 h_old? 
          YES! TMEM 是只读的对 MMA 而言, state_tmem 未被后续写入覆盖
  Phase2: QH MMA = h_old(TMEM-A) × qg_T(SMEM-B) → acc
Chunk i+1:
  Phase1.2: R2T h_new → state_tmem  (覆盖 h_old)
```

**方案 D 可行!** State TMEM 中的旧 h 在 Phase 2 中可以直接复用!

### 6.4 方案 D: QH MMA = h(TMEM) × qg_T(SMEM) 详细分析

QH MMA 重新定义为:
- M = BV = 64 (与 chunk_delta_h 的 WH/KV MMA 共享 M 维)
- N = BT = 64
- K = BK = 128
- A = h(BV, BK) from TMEM ← **直接复用 state_tmem offset!**
- B = qg^T(BK, BT) from SMEM (K-major, BK contiguous)
- C = acc(BV, BT) from TMEM

TMEM layout:
- A-operand (state): 已经在 tmem_state_off, shape (BV=64, BK=128) BF16 → **零成本**
- ACC: 需要 (BV=64, BT=64) FP32 → **与 WH ACC 完全相同 shape!** 复用 tmem_wh_off

AV MMA 重新定义为:
- M = BV = 64
- N = BT = 64
- K = BT = 64
- A = h_for_AV? NO → A = A_masked^T? 
  
**等等!** 原始 fwd_o: `o = scale * (q ⊙ 2^g) @ h + tril(Aqk) @ v_new`
shape: o(BT,BV) = qg(BT,BK) @ h(BK,BV) + A_masked(BT,BT) @ v_new(BT,BV)

如果我们把输出维度 swap 为 (BV, BT):
`o^T(BV,BT) = h^T(BV,BK) @ qg^T(BK,BT) + v_new^T(BV,BT) @ A_masked^T(BT,BT)`

QH MMA: `h^T(BV,BK) @ qg^T(BK,BT) → acc(BV,BT)`
- A = h^T (TMEM, state_tmem) shape (BV,BK) ← **已有 K-major layout**
- B = qg^T (SMEM) shape (BK,BT), K-major = BK contiguous

AV MMA: `v_new^T(BV,BT) @ A_masked^T(BT,BT) → acc(BV,BT) [ACCUMULATE]`
- A = v_new^T (TMEM, vnew_tmem) shape (BV,BT) ← **已有!**
- B = A_masked^T (SMEM) shape (BT,BT)
- ACC = 同一个 acc(BV,BT)

**太完美了!** 
- QH MMA 的 A-operand = state_tmem (R2T 过的旧 h) → 零额外 TMEM
- AV MMA 的 A-operand = vnew_tmem (R2T 过的 v_new) → 零额外 TMEM
- ACC = WH ACC (BV,BT) → 零额外 TMEM

**但 MMA 时序冲突!**

Phase 1 KV MMA 使用 vnew_tmem 作为 A-operand, 且 KV ACC 占用不同 TMEM。
Phase 2 AV MMA 需要 vnew_tmem 作为 A-operand → 但 phase 1 的 KV MMA 和 phase 2 的 AV MMA 不同时执行, 所以 vnew_tmem 可安全复用。

**Phase 2 AV MMA ACC**: 需要 ACCUMULATE 到 QH MMA 的 acc 上 → 使用 WH ACC offset (tmem_wh_off)

### 6.5 最终 SMEM 布局

以方案 D 为基础，fwd_o Phase 2 需要的新 SMEM:

| Buffer         | Size (bytes) | 说明 | 复用可能性 |
|---------------|-------------|------|-----------|
| sQG_T (1-stage)| ~16,384    | qg^T: MMA-B format (BK=128, BT=64 bf16) | 新增 |
| sA_T (1-stage) | ~8,192     | A_masked^T: MMA-B format (BT=64, BT=64 bf16) | 新增 |
| sO (1-stage)   | ~8,192     | o^T output: (BV=64, BT=64 bf16) for R2S | 复用 sU stage 0 |
| sQ_epi (1-stage)| 16,384    | q TMA load epilog (BT×BK) | 新增 |
| sG_epi (1-stage)| 16,384    | g TMA load epilog (BT×BK) | 可与 sQ_epi 共享 |
| sA_epi (1-stage)| 8,192     | A TMA load epilog (BT×BT) | 新增 |

**但要考虑时序**: Phase 2 时, Phase 1 的某些 buffer 已释放：
- sW: WH MMA 完成后不再需要 → 3×16,384 = 49,152 bytes 可释放复用!
- sKt: KV MMA 完成后不再需要 → 3×16,384 = 49,152 bytes 可释放复用!
- sU: U 消费完后不再需要 → 3×8,192 = 24,576 bytes 可释放复用!

**SMEM 不需要同时 live**! Phase 1 和 Phase 2 串行执行。
但 CuTe DSL 的 SMEM 分配是 **静态** 的（SharedStorage struct），无法做运行时 overlay。

**解决方案: SMEM Union**

在 SharedStorage 中用 union 或 reinterpret 让 Phase 1 和 Phase 2 的独占 buffer overlay:

```
Phase 1 独占: sW (49KB, 3-stage), sKt (49KB, 3-stage), sU (25KB, 3-stage)
Phase 2 独占: sQG_T (16KB), sA_T (8KB), sO (8KB), sQ_epi (16KB), sG_epi (16KB), sA_epi (8KB)
共享: sH_epi (33KB, ping-pong), sVnew_store (16KB, ping-pong), sGK (2KB)
```

Phase 1 独占 = 49+49+25 = 123 KB
Phase 2 独占 = 16+8+8+16+16+8 = 72 KB

**Phase 2 完全可以 overlay 在 Phase 1 独占区域内** (72KB < 123KB)!

但 CuTe DSL 可能不支持 SMEM union... 如果不支持, 则:

**保守方案**: 全部静态分配:
- chunk_delta_h: 174 KB (已有)
- fwd_o 新增: sQG_T(16K) + sA_T(8K) + sQ_epi(16K) + sA_epi(8K) + sO alias sU
- 实际新增: 16+8+16+8 = 48 KB
- 总计: 174 + 48 = **222 KB** ← 在 228 KB 限制内! ✅

**sG_epi 合并**: q 和 g 串行读取 → 复用同一 sQG_epi buffer → 省 16KB

最终: 174 + 16(sQG_T) + 8(sA_T) + 16(sQG_epi) + 8(sA_epi) = **222 KB** ✅

> (sO 复用 sU 的物理空间 via reinterpret，不计额外)

---

## 7. Pipeline 设计

### 7.1 Phase 1 Pipelines (不变, 来自 chunk_delta_h)

| Pipeline          | Producer → Consumer | 数据        | Stages |
|-------------------|---------------------|-------------|--------|
| load_w            | Load → MMA          | sW          | 3      |
| load_kt           | Load → MMA          | sKt         | 3      |
| load_u            | Load → CUDA         | sU          | 3      |
| load_gk           | Load → CUDA         | sGK         | 3      |
| state_tmem        | CUDA → MMA          | state TMEM  | 1      |
| wh_done           | MMA → CUDA          | WH ACC done | 1      |
| vnew_smem         | CUDA → MMA          | vnew TMEM   | 1      |
| kv_done           | MMA → CUDA          | KV ACC done | 1      |
| h_out             | CUDA → Store        | sH_epi      | 2      |
| vnew_store        | CUDA → Store        | sVnew_store | 2      |

### 7.2 Phase 2 新增 Pipelines

| Pipeline          | Producer → Consumer | 数据           | Stages |
|-------------------|---------------------|----------------|--------|
| load_qg_epi       | Load → CUDA         | sQG_epi (q/g)  | 1      |
| load_a_epi        | Load → CUDA         | sA_epi          | 1      |
| qg_tmem           | CUDA → MMA          | QG A TMEM ²     | 1      |
| am_tmem           | CUDA → MMA          | AM A TMEM ²     | 1      |
| acc_fwd_done      | MMA → CUDA          | FwdO ACC done   | 1      |
| o_ready           | CUDA → Store        | sO              | 1      |

² 复用 state_tmem / vnew_smem 的 TMEM offset, 但需要新的 pipeline barrier (不同阶段不同 producer/consumer count)。
  或者: 如果 barrier 可复用 (count 相同), 则直接复用。

**barrier count 检查**:
- state_tmem: producer = 128 CUDA threads, consumer = 1 MMA thread → CUDA→MMA
- qg_tmem: producer = 128 CUDA threads, consumer = 1 MMA thread → CUDA→MMA ← **相同!**
- vnew_smem: same counts → am_tmem 也相同 ✅

**可以复用 state_tmem 和 vnew_smem 的 barriers! 但需要确保 Phase 1 和 Phase 2 之间有隐式同步。**

实际上: Phase 2 CUDA warp 在 Phase 1 的 kv_done 之后才开始 Phase 2 → Phase 1 所有管线都已完成 → barrier 可安全复用。

### 7.3 每 chunk 完整时序

```
Load warp:
  ├── TMA load W[chunk_idx] → sW           (Phase 1)
  ├── TMA load Kt[chunk_idx] → sKt         (Phase 1)
  ├── TMA load U[chunk_idx] → sU           (Phase 1)
  ├── TMA load gk[chunk_idx] → sGK         (Phase 1)
  ├── TMA load q[chunk_idx] → sQG_epi      (Phase 2, 可与 Phase 1 TMA overlap)
  ├── TMA load g[chunk_idx] → sQG_epi      (Phase 2, 在 q 被消费后)
  ├── TMA load A[chunk_idx] → sA_epi       (Phase 2)
  └── TMA store sO → GMEM (o)              (Phase 2, Store warp)

MMA warp:
  ├── WH MMA: state(TMEM) × W(SMEM) → WH_acc      (Phase 1)
  ├── KV MMA: vnew(TMEM) × Kt(SMEM) → KV_acc      (Phase 1)
  ├── QH MMA: h_old(TMEM) × qg_T(SMEM) → FwdO_acc (Phase 2)
  └── AV MMA: vnew(TMEM) × A_T(SMEM) → FwdO_acc++ (Phase 2)

CUDA warps:
  ├── Phase 1: R2T h→TMEM, R2S h→sH_epi, preload U, v_new=U-WH, R2T vnew→TMEM
  │            gk_decay, h+=KV_update
  ├── Phase 2: qg=q*exp2(g)*scale → R2S qg^T → sQG_T → pipe → R2T (reuse state_tmem)
  │            A_masked^T → R2S → sA_T → pipe → R2T (reuse vnew_tmem)
  │            T2R FwdO_acc → bf16 → R2S → sO
  └── ...

Store warp:
  ├── TMA store sH_epi → GMEM (h_out)     (Phase 1)
  ├── S2G sVnew_store → GMEM (v_new)      (Phase 1, if needed)
  └── S2G sO → GMEM (o)                    (Phase 2)
```

### 7.4 关键 overlap 分析

**Phase 1 与 Phase 2 的 Load overlap**:
- q, g, A 的 TMA load 可以在 Phase 1 的 latter 部分开始 (pipeline 异步)
- 关键: Load warp 在发出 W/Kt/U/gk TMA 后就空闲 → 可以提前发 q/g/A TMA

**Phase 2 QH MMA 与 Phase 1 store overlap**:
- Store warp 写 h_out TMA 和 Phase 2 QH MMA 可以并行
- 但 sH_epi 被 store warp 读取中 → h_out pipeline 的 release 必须在 QH MMA 之前完成
  (Phase 2 不_使用_ sH_epi, 所以实际无冲突)

---

## 8. 关键问题: qg^T 如何从 epilog SMEM 转到 MMA-B SMEM?

CUDA warps 需要:
1. 从 sQG_epi 读 q[bt, bk] 和 g[bt, bk]
2. 计算 qg = q * exp2(g) * scale (FP32)
3. 写 qg^T[bk, bt] (bf16) 到 sQG_T (MMA-B swizzle format)

**方案**: CUDA regs → R2S to sQG_T

但 R2S 需要使用 tiled_t2r 的布局! 通常 R2S 是从 T2R regs 写到 epilog SMEM。
这里是: **registers → MMA-B format SMEM** → 不是标准的 epilog R2S。

**替代方案**: 直接 R2T (registers → TMEM)

等等! Phase 2 QH MMA 用 **state_tmem** 作为 A (= h_old)。
B operand 必须是 SMEM。所以 qg^T 必须在 SMEM 中。

**最简方案**: 用 identity tensor + scalar write:

```python
for ei in range(size(tTR_rQG)):
    bt_coord, bk_coord = identity[ei]
    sQG_T[bk_coord, bt_coord] = qg_bf16[ei]  # transposed write
```

128 CUDA threads, 每个写 BT×BK/128 = 64×128/128 = 64 个值 → 64 scalar writes per thread。
不需要 coalesced (SMEM 无 bank conflict 如果 swizzle 正确)。

**或者: 如果 QH MMA B-operand 是 K-major (BK contiguous)**:
- qg^T(BK, BT) K-major: BK contiguous → qg[bk, bt] 中 bk 连续
- 原始 qg[bt, bk] = q[bt,bk]*exp2(g[bt,bk])*scale → bk 连续
- 所以 qg "不需要 transpose"! 只是 SMEM layout interpretation 不同:

原始: qg(BT, BK) row-major = BK contiguous = element [bt][bk] at offset bt*BK+bk
Transposed view: qg^T(BK, BT) col-major, 但物理布局相同

如果 MMA B-operand 是 **K-major** (K=BK contiguous):
- qg^T(BK_N, BT_K): MMA B 的 N=BK, K=BT → K-major = BT contiguous
- 实际物理意义: qg^T 中 BT 维连续 → qg(BT,BK) 中 BK 维连续 → 就是 ROW_MAJOR! ✅

**结论**: qg 不需要显式 transpose! 
- 计算 qg[bt, bk] = q*exp2(g)*scale in regs
- R2S 写到 MMA-B SMEM 时, 使用 K-major 的 B-operand layout (BK contiguous)
- MMA 看到的 B(N=BT, K=BK), K-major → BK dim contiguous
- 这与 qg[bt, bk] 的 bk-contiguous 物理布局完全匹配!

**等一下, 需要重新理清**:

QH MMA (transposed version): h^T(BV, BK) × qg^T(BK, BT) → acc(BV, BT)
- M=BV=64, N=BT=64, K=BK=128
- A = h^T, source = TMEM, K-major (BK dim contiguous in "K" direction)
- B = qg^T, source = SMEM
  - B shape from MMA perspective: (N=BT, K=BK) with partitioning
  - If B-operand is K-major: BK contiguous → good, matches qg's bk-contiguous layout
  - If B-operand is MN-major: BT contiguous → need transpose

选择 B-operand K-major 就无需 transpose!

**最终 QH MMA setup**:
```python
qh_tiled_mma = make_trivial_tiled_mma(
    io_dtype=bf16,
    A-major=K,     # TMEM requires K-major
    B-major=K,     # K-major → BK contiguous → matches qg physical layout
    acc_dtype=fp32,
    cta_group=ONE,
    tile_mn=(BV=64, BT=64),  # M=BV, N=BT
    A_source=TMEM,
)
```

类似地, AV MMA:
```python
# acc += v_new^T(BV, BT) × A_masked^T(BT, BT)
# M=BV=64, N=BT=64, K=BT=64
av_tiled_mma = make_trivial_tiled_mma(
    io_dtype=bf16,
    A-major=K,     # TMEM requires K-major  
    B-major=K or MN,  # depends on A_masked layout
    acc_dtype=fp32,
    tile_mn=(BV=64, BT=64),  # M=BV, N=BT
    A_source=TMEM,
)
```

**关键洞察**: QH/AV MMA 的 M=BV=64 与 Phase 1 的 WH/KV MMA M=BV=64 相同!
这意味着:
- TMEM A-operand 的 M 维度相同 → 可以共享 TMEM offset
- TMEM ACC 的 M 维度相同 → WH ACC (BV,BT) 和 QH ACC (BV,BT) 完全兼容
- 4 个 MMA 全部 M=BV=64, 共享 TMEM M 维

---

## 9. SMEM 总预算 (最终方案)

| Buffer              | Size (bytes) | Phase | 复用     |
|---------------------|-------------|-------|---------|
| **保留自 chunk_delta_h** ||||
| sW (3-stage)        | 49,152      | P1    | -       |
| sKt (3-stage)       | 49,152      | P1    | -       |
| sH_epi (2-stage)    | 32,768      | P1    | -       |
| sU (3-stage)        | 24,576      | P1    | sO 复用 ¹|
| sVnew_store (2-stage)| 16,384     | P1    | -       |
| sGK (3-stage fp32)  | 1,536       | P1    | -       |
| **新增 for fwd_o** ||||
| sQG_mma (1-stage)   | 16,384      | P2    | B-op: (BT,BK)=(64,128) bf16 |
| sA_mma (1-stage)    | 8,192       | P2    | B-op: (BT,BT)=(64,64) bf16 |
| sQG_epi (1-stage)   | 16,384      | P2    | Epilog load: (BT,BK) |
| sA_epi (1-stage)    | 8,192       | P2    | Epilog load: (BT,BT) |
| **barriers + pad**  | ~1,500      | -     | -       |
| **Total**           | **~224,220**| -     | **< 228 KB** ✅ |

¹ sO 直接 reinterpret sU stage 0 的 SMEM 空间 (size 匹配: BV×BT=64×64 bf16 = 8KB < sU 一个 stage 8KB)

> **注意**: sQG_epi 和 sG_epi 合并为一个 buffer, q/g 串行读取。

---

## 10. 寄存器预算

CUDA warps (0-3): 各 208 regs (varlen)

Phase 1 关键寄存器:
- h state: tTR_rKV (BV=64 × BK=128 / 128_threads × 某个 partition factor) ≈ 主要消费者
- WH T2R regs, v_new regs, gk temp 等

Phase 2 额外寄存器:
- qg 计算: q_val, g_val, qg_val → ~3 个 fp32 per iteration (不并行)
- A_masked: 类似
- T2R FwdO ACC: 新增 tTR_rAcc regs

**关键**: Phase 2 的寄存器需求 **低于** Phase 1 (不持有 update/vnew 等)。
且 Phase 2 的计算可以复用 Phase 1 释放的 temp regs (tTR_rWH, tTR_rUpdate 等)。
208 regs 应该足够。如果不够, 可以考虑 non-varlen 路径用 232 regs。

---

## 11. 输出 o 的 GMEM 写入

o^T 在 regs 中 shape (BV=64, BT=64), 需要写到 GMEM o[BT, BV] layout。

两种方案:

### 11a. R2S → sO → TMA S2G

- R2S acc(BV,BT) → sO (COL_MAJOR, BV contiguous) → TMA store
- TMA store 需要 o GMEM 是 (BT, BV) row-major → sO 的 BV×BT COL_MAJOR = BT 行中 BV 连续 = 等价 row-major BT×BV ✅
- 可以复用 sU (stage 0) 空间

### 11b. Register → GMEM 直写 (varlen)

类似 v_new 和 ht 的 direct write pattern:
```python
for ei in range(...):
    v_coord, t_coord = identity[ei]
    gO[t_coord + chunk_idx*BT, v_coord + v_tile*BV] = acc_bf16[ei]
```

Varlen 时必须用此方案 (TMA store 边界问题)。

---

## 12. 实现计划

### Step 1: 新建 `fused_chunk_delta_h_fwd_o.py`
- 复制 `chunk_delta_h.py` 为基础
- 新增 fwd_o 相关的输入参数 (q, g, A, o, scale)
- 联合 SMEM SharedStorage (新增 sQG_mma, sA_mma, sQG_epi, sA_epi)

### Step 2: 新增 Phase 2 MMA setup
- QH MMA: M=BV, N=BT, K=BK, A=TMEM(state_tmem), B=SMEM(K-major)
- AV MMA: M=BV, N=BT, K=BT, A=TMEM(vnew_tmem), B=SMEM(K-major or MN-major)
- ACC 复用 WH ACC TMEM offset

### Step 3: Load warp 扩展
- 在 Phase 1 TMA 后, 发 q/g/A 的 TMA load
- q, g 可以 overlap Phase 1 后半段

### Step 4: CUDA warp Phase 2
- Wait Phase 1 kv_done
- 从 sQG_epi 读 q, g → compute qg → R2S to sQG_mma
-  CUDA→MMA: qg_ready (复用 state_tmem barrier)
- 从 sA_epi 读 A → causal mask → R2S to sA_mma  
- CUDA→MMA: am_ready (复用 vnew_smem barrier)
- Wait acc_fwd_done
- T2R FwdO ACC → R2S → sO
- CUDA→Store: o_ready

### Step 5: MMA warp Phase 2
- After Phase 1 KV MMA
- Wait qg_ready → QH MMA (state_tmem × sQG_T → WH_acc)
- Wait am_ready → AV MMA (vnew_tmem × sA_T → WH_acc, ACCUMULATE)
- Signal acc_fwd_done

### Step 6: Store warp 扩展
- Phase 1: h_out, v_new (不变)
- Phase 2: Wait o_ready → TMA store sO / direct GMEM write

### Step 7: 测试与验证
- 复用 chunk_delta_h 的 46 测试, 同时验证 o 输出
- 与 FLA reference 对比精度

---

## 13. 风险与替代方案

### 风险 1: SMEM 超出 228KB

如果实际 swizzle padding 比估算大, 导致 >228KB:
- **降级方案**: sW/sKt stage 从 3 降到 2 → 省 2×16KB = 32KB
- 代价: Phase 1 pipeline depth 降低, 可能影响 WH/KV MMA latency hiding

### 风险 2: 寄存器溢出

Phase 2 增加新的 register 变量:
- **降级方案**: non-varlen 路径使用 232 regs; varlen 路径尝试 max=256 regs (如果 occ=1 允许)

### 风险 3: TMEM state_tmem 被意外覆盖

确保 Phase 1 WH MMA 仅 **读取** state_tmem, 不修改它。
Phase 1 R2T h → state_tmem 在 Phase 1 开始时执行, Phase 2 QH MMA 在 Phase 1 结束后读取。
中间只有 WH MMA 读取过 state_tmem → 安全。

### 风险 4: qg R2S 到 MMA-B format SMEM 的正确性

如果 CUDA warp 不能正确写入 MMA-B swizzle 格式:
- **替代方案**: 用 TMA load qg 的转置 (qg_T) 到 SMEM
  需要把 qg 先写到一个临时 GMEM buffer, 再 TMA load → 增加 GMEM 访问, 不理想
- **更好的替代**: 用 epilog layout (非 MMA-B) 写 qg, 然后 SMEM→SMEM copy (via registers)?
  但 Blackwell 没有 SMEM→SMEM DMA...

最可靠的方案是: 让 CUDA warps 用 scalar write (identity tensor mapping) 写入已知好的 SMEM 地址。
128 个 CUDA thread, 每个写 BK*BT/128 = 64 个 bf16 值到 sQG_T, 完全可行。

---

## 14. 总结

| 维度          | 当前 (分离)             | Fused 方案               |
|--------------|------------------------|--------------------------|
| Kernel 数     | 2                      | 1                        |
| GMEM h 往返   | 2 × NT×H×K×V × 2B     | 0 (TMEM 内消费)           |
| GMEM v_new    | 2 × B×T×H×V × 2B      | 0 (SMEM 内转发) ³         |
| SMEM          | 174KB (delta_h only)   | ~224KB (含 all fwd_o buf) |
| TMEM          | 288 cols               | 288 cols (零增长)         |
| Occupancy     | occ=1 (delta_h), occ=2 (fwd_o) | occ=1 (fused)   |
| Regs/thread   | 208/40                 | 208/40 (不变)             |
| Pipeline depth| Phase 1: 3-stage       | Phase 1: 3-stage, Phase 2: 1-stage |

³ 如果 backward 仍需 v_new, 则保留 v_new GMEM 写入 (Phase 1)

**预期性能提升**: 
- 省掉 h 和 v_new 的 GMEM 往返 = 省 ~4×NT×H×128×128×2B 带宽
- 例如 B=8, T=2048, H=16: NT×H = 256, 省 ~64MB GMEM 带宽
- kernel launch overhead 从 2 次降为 1 次
