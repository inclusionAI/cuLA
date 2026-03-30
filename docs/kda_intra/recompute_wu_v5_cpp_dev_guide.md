# KDA `recompute_w_u_fwd` V5 — C++ 开发指南

> 对应代码：`csrc/kda/kda_fwd_recomp_w_u_mainloop_sm100.hpp` + `kda_fwd_recomp_w_u_kernel_sm100.hpp`
> 参考：V5 设计文档 `recompute_wu_v5_design.md`，V4 实现 `recompute_v4.py`，fwd_o `fwd_o.py`

---

## 0. 核心设计思想

V5 的核心变更：**将 Akk 放入 TMEM 作为 MMA A-operand，K/V 放入 SMEM 作为 MMA B-operand**。

这消除了 V4 的转置痛点，使得 MMA 输出 `C[BT, BK]` 的行维 = time、列维 = key/value dim，**与 GMEM `[BT, BK]` time-major 布局完全一致**，Epilogue 可以直接 coalesced store。

```
MMA 语义:
  w[BT, BK] = Akk[BT, BT] @ K_proc[BK, BT]^T    (K-GEMM)
  u[BT, BV] = Akk[BT, BT] @ V_proc[BV, BT]^T    (V-GEMM)

  输出行维 = time (BT=64), 列维 = head dim slice (TileK=32)
  → 与 GMEM [BT, HeadDim] row-major 一致，无需转置
```

---

## 1. 常量与 Tile 参数

```
TileT     = 64        // chunk size (time dim)
HeadDim   = 128       // 总 head dim (K = V = 128)
TileK     = 32        // head dim 切分粒度 (每次 MMA 处理 32 列)
NumKIters = HeadDim / TileK = 4   // 每个 work-unit 内 head dim 迭代次数

MMA tiler = (M=64, N=32, K=64)
  M = BT  = 64   (time dim, TMEM lanes 0-63)
  N = TileK = 32  (head dim slice)
  K = BT  = 64   (reduction dim, Akk 的列维)
```

每个 work-unit (WU) = 1 个 chunk (BT=64 行)，内部按 head dim 切分为 `NumKIters=4` 次迭代。

---

## 2. 线程布局与 Warp 分工

```
CTA = 384 threads = 12 warps = 3 Warp Groups

┌────────────────────────────────────────────────────────────────────┐
│ WG0: Prologue (warp 0-3, thread 0-127)    — 128 threads          │
│ WG1: Epilogue (warp 4-7, thread 128-255)  — 128 threads          │
│ WG2: Load/MMA/Aux (warp 8-11, thread 256-383) — 128 threads      │
│   ├─ warp 8:   MMA warp (elect_one 执行 UMMA)                     │
│   ├─ warp 9:   Load warp (elect_one 执行 TMA)                     │
│   └─ warp 10-11: Aux warps (beta, g_last 加载)                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.1 寄存器分配

```cpp
NumPrologueRegs = 208;   // WG0: element-wise 计算 + R2T Akk
NumEpilogueRegs = 208;   // WG1: T2R acc + R2G store + kg 计算
NumLoadRegs     = 88;    // WG2: TMA load + MMA + Aux
```

使用 `cutlass::arch::warpgroup_reg_alloc<N>()` / `warpgroup_reg_dealloc<N>()` 控制。

---

## 3. TMEM 布局

TMEM 只使用 **64 lanes**（lanes 0-63，对应 M=BT=64）。

```
TMEM physical layout (128 lanes × N cols):

         col 0 ─────── col ACC_END    col AKK_START ─── col AKK_END
         ┌────────────────────────┐    ┌──────────────────────┐
lane 0   │                        │    │                      │
  ...    │  Accumulator            │    │  A-operand (Akk)     │
lane 63  │  acc[64, 32] fp32       │    │  Akk[64, 64] bf16    │
         │                        │    │                      │
lane 64  │  (unused)              │    │  (unused)            │
  ...    │                        │    │                      │
lane 127 │                        │    │                      │
         └────────────────────────┘    └──────────────────────┘
```

### TMEM 列估算

| 区域 | Shape | dtype | 约占列数 |
|------|-------|-------|---------|
| Accumulator | `[64, 32]` | fp32 | ~32 cols |
| A-operand (Akk) | `[64, 64]` | bf16 | ~32 cols |
| **总计** | | | **~64 cols** (远小于 512 上限) |

### TmemAllocation 枚举

```cpp
enum class TmemAllocation : uint32_t {
    ACC = 0,           // accumulator 起始列
    AKK = ACC + N_acc, // Akk A-operand 起始列 (由 _plan_tmem_offsets 计算)
};
```

用 `_plan_tmem_offsets()` 或 `tcgen05::find_tmem_tensor_col_offset()` 动态计算偏移。

---

## 4. SMEM 布局

### 4.1 Buffer 一览

| Buffer | dtype | Shape | Stages | 大小 | 用途 |
|--------|-------|-------|--------|------|------|
| `sA` (Akk) | bf16 | `[TileT, TileT]` = `[64, 64]` | 1 | 8 KB | TMA 加载 Akk → Prologue S2R → R2T TMEM |
| `sK` | bf16 | `[TileT, TileK]` = `[64, 32]` | 2 | 8 KB | TMA 加载 K (double-buffer) |
| `sV` | bf16 | `[TileT, TileK]` = `[64, 32]` | 2 | 8 KB | TMA 加载 V (double-buffer) |
| `sG` | fp32 | `[TileT, TileK]` = `[64, 32]` | 2 | 16 KB | TMA 加载 G (double-buffer) |
| `sBeta` | fp32 | `[TileT]` = `[64]` | 2 | 512 B | beta 向量 (double-buffer) |
| `sGLast` | fp32 | `[TileK]` = `[32]` | 2 | 256 B | g_last 向量 (double-buffer) |
| `k_mma` | bf16 | `[TileT, TileK]` = `[64, 32]` | 2 | 8 KB | Prologue 写入 K_proc → MMA B-operand (double-buffer) |
| `v_mma` | bf16 | `[TileT, TileK]` = `[64, 32]` | 2 | 8 KB | Prologue 写入 V_proc → MMA B-operand (double-buffer) |
| `sO` | bf16 | `[TileT, TileK]` = `[64, 32]`<br>ROW_MAJOR | 2 | 8 KB | Epilogue 输出 staging (w/u/kg, double-buffer) |
| `tmem_start_addr` | uint32 | 1 | - | 4 B | TMEM 分配基地址 |
| **总计** | | | | **~64 KB** | |

### 4.2 SMEM Layout 类型

```cpp
// K, V (bf16, TMA 加载用): [TileT, TileK] = [64, 32] K-major swizzled
using SmemLayoutInputBF16 = tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<TileT>, Int<TileK>>{},   // (64, 32)
    Step<_1, _2>{}
);

// Akk (bf16): [TileT, TileT] = [64, 64] K-major swizzled
using SmemLayoutInputAkkBF16 = tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<TileT>, Int<TileT>>{},   // (64, 64)
    Step<_1, _2>{}
);

// G (fp32, TMA 加载用): [TileT, TileK] = [64, 32] K-major swizzled
using SmemLayoutInputFP32 = tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<TileT>, Int<TileK>>{},   // (64, 32)
    Step<_1, _2>{}
);

// MMA B-operand (bf16): [N=TileK, K=TileT] = [32, 64] MN-major swizzled
// UMMA 语义: C[M,N] = A[M,K] @ B[N,K]^T
//   A = Akk [M=64, K=64] TMEM K-major
//   B = K_proc/V_proc [N=32, K=64] SMEM MN-major (UMMA 内部转置)
//   C = w/u [M=64, N=32] TMEM accumulator
// reduce dim = chunk dim (K=BT=64), output head dim (N=BK=32)
// 类似 bwd intra 的 SmemLayoutMatBTF32Transposed, 但用 bf16
using SmemLayoutMatBBF16 = tile_to_shape(
    UMMA::Layout_MN_SW64_Atom<bf16>{},
    Shape<Int<TileK>, Int<TileT>>{},   // (N=32, K=64)
    Step<_1, _2>{}
);

// 输出 staging (bf16): [TileT, TileK] = [64, 32] row-major (无 swizzle)
using SmemLayoutOutputBF16 = tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<TileT>, Int<TileK>>{}
);
```

### 4.3 k_mma / v_mma (MMA B-operand) 独立 Double-Buffer

与设计文档不同，实际实现中 K_proc 和 V_proc 使用**独立的 double-buffer SMEM**：
- `k_mma[2]`: Prologue 写 K_proc → signal `k_prologue_ready` → MMA 执行 K-GEMM → release
- `v_mma[2]`: Prologue 写 V_proc → signal `v_prologue_ready` → MMA 执行 V-GEMM → release

这使得 Prologue 可以在 MMA 执行 K-GEMM 时**同时写入下一个 V_proc**，提高 overlap。

---

## 5. Pipeline 设计

### 5.1 Pipeline 一览

```
共 11 条 Pipeline:

Pipeline               Type                  Producer         Consumer            Stage  说明
─────────────────────────────────────────────────────────────────────────────────────────────────
1.  load_A             PipelineTmaAsync      Load(9)          Prologue(0-3)       1     Akk 在 sA 就绪
2.  load_K             PipelineTmaAsync      Load(9)          Pro(0-3)+Epi(4-7)   2     K 在 sK 就绪
3.  load_V             PipelineTmaAsync      Load(9)          Prologue(0-3)       2     V 在 sV 就绪
4.  load_G             PipelineTmaAsync      Load(9)          Pro+Epi+Aux          2     G 在 sG 就绪 (3 consumers)
5.  beta_ready         PipelineAsync         Aux(10-11)       Prologue(0-3)       2     sBeta 就绪
6.  glast_ready        PipelineAsync         Aux(10-11)       Epilogue(4-7)       2     sGLast 就绪 (Aux 从 sG 提取)
7.  a_ready            PipelineAsync         Prologue(0-3)    MMA(8)              1     Akk 已写入 TMEM, MMA 可开始
8.  k_prologue_ready   PipelineAsync         Prologue(0-3)    MMA(8)              1     K_proc 在 k_mma 就绪
9.  v_prologue_ready   PipelineAsync         Prologue(0-3)    MMA(8)              1     V_proc 在 v_mma 就绪
10. w_done             PipelineAsync         MMA(8)           Epilogue(4-7)       1     K-GEMM acc 就绪
11. u_done             PipelineAsync         MMA(8)           Epilogue(4-7)       1     V-GEMM acc 就绪
```

**注意**: 与 V5 设计文档中 `bproc`/`acc_done` 单管线不同，实际实现使用 **分离的 pipeline**：
- `a_ready`（第 7 条）：Prologue 完成 Akk S2R→R2T→TMEM 后 signal，通知 MMA warp Akk 已在 TMEM 就绪。
  每个 WU 只 signal 1 次，MMA 在首次 K-GEMM 前 wait。
- K_proc 和 V_proc 各自有独立的就绪信号 (`k_prologue_ready` / `v_prologue_ready`)，
  各自写入独立的 `k_mma[2]` / `v_mma[2]` double-buffer，不再复用单个 `sB`。
- W 和 U 的 MMA 完成各有独立信号 (`w_done` / `u_done`)。
- 这避免了单 stage-1 pipeline 上 K/V 交替复用的复杂时序。

**g_last 优化**：g_last 不再从 GMEM 独立加载，而是由 Aux warp 作为 `load_G` 的第 3 个 consumer，
等待 TMA 将 G 写入 sG 后直接从 sG 的最后一行（`sG[sub_seq_len-1, :]`）提取。
这避免了冗余 GMEM 访问，且 Aux warp 在提取完成后 release load_G，不会阻塞 Prologue/Epilogue。

### 5.2 Pipeline 与 V4 的区别

| V4 | V5 | 变化说明 |
|----|-----|---------|
| `load_A → MMA` | `load_A → Prologue` | A 不再直接给 MMA，而是 Prologue 做 S2R→R2T |
| 无 a_ready | `a_ready` (Prologue→MMA) | **新增**: Prologue 完成 Akk R2T 后显式 signal MMA |
| `bproc` (Prologue→MMA, 单管线复用 K/V) | `k_prologue_ready` + `v_prologue_ready` (分离) | Prologue 分别 signal K/V 就绪，各自写独立 SMEM buffer |
| `load_kgk` (k+gk 合并 TMA) | `load_K` + `load_G` 分离 | TileK=32，分别 TMA 更灵活 |
| `acc_done` (单管线复用 W/U) | `w_done` + `u_done` (分离) | MMA 分别 signal K-GEMM/V-GEMM 完成 |
| 7 条 | 11 条 | 多了 `load_V`，`a_ready`，K/V prologue ready 和 W/U done 各拆为 2 条 |

### 5.3 Pipeline 类型映射 (C++)

```cpp
// TMA pipelines (Load→Consumer)
using PipelineA     = cutlass::PipelineTmaAsync<1>;       // load_A
using PipelineK     = cutlass::PipelineTmaAsync<2>;       // load_K
using PipelineV     = cutlass::PipelineTmaAsync<2>;       // load_V
using PipelineG     = cutlass::PipelineTmaAsync<2>;       // load_G (3 consumers: Pro+Epi+Aux)

// Async pipelines (thread-signaled)
using PipelineBeta  = cutlass::PipelineAsync<2>;          // beta_ready
using PipelineGLast = cutlass::PipelineAsync<2>;          // glast_ready (Aux extracts from sG)

// Prologue→MMA pipelines
using PipelineAReady           = cutlass::PipelineAsync<1>; // Akk TMEM ready (1×/WU)
using PipelineKPrologueReady   = cutlass::PipelineAsync<1>; // K_proc ready
using PipelineVPrologueReady   = cutlass::PipelineAsync<1>; // V_proc ready
  // TODO: update to PipelineUmma variants after calling tcgen05.mma

// MMA→Epilogue pipelines (分离 W/U)
using PipelineWDone = cutlass::PipelineAsync<1>;          // K-GEMM acc ready → w
using PipelineUDone = cutlass::PipelineAsync<1>;          // V-GEMM acc ready → u
  // TODO: update to PipelineUmma variants after calling tcgen05.mma
```

### 5.4 Pipeline State 类型映射 (C++)

```cpp
using PipelineStateA                = cutlass::PipelineState<PipelineA::Stages>;
using PipelineStateK                = cutlass::PipelineState<PipelineK::Stages>;
using PipelineStateV                = cutlass::PipelineState<PipelineV::Stages>;
using PipelineStateG                = cutlass::PipelineState<PipelineG::Stages>;
using PipelineStateBeta             = cutlass::PipelineState<PipelineBeta::Stages>;
using PipelineStateGLast            = cutlass::PipelineState<PipelineGLast::Stages>;
using PipelineStateAReady           = cutlass::PipelineState<PipelineAReady::Stages>;
using PipelineStateKPrologueReady   = cutlass::PipelineState<PipelineKPrologueReady::Stages>;
using PipelineStateVPrologueReady   = cutlass::PipelineState<PipelineVPrologueReady::Stages>;
using PipelineStateWDone            = cutlass::PipelineState<PipelineWDone::Stages>;
using PipelineStateUDone            = cutlass::PipelineState<PipelineUDone::Stages>;
```

---

## 6. 各 Warp/WarpGroup 详细功能

### 6.1 WG0: Prologue (warp 0-3, 128 threads)

**职责**：Element-wise 计算 + 写入 MMA operands (Akk→TMEM, K_proc/V_proc→SMEM)

#### 每个 Work-Unit 流程：

```
┌─ 1次/WU ─────────────────────────────────────────────────┐
│ wait beta_ready                                           │
│ wait load_A                                               │
│ Akk: sA → S2R (下三角 mask, 上三角填0) → R2T → TMEM      │
│ release load_A                                            │
│ acquire + commit a_ready (Akk in TMEM → MMA)              │
└───────────────────────────────────────────────────────────┘

┌─ NumKIters 次 (i_k = 0..3) ──────────────────────────────┐
│ wait load_K[i_k]  (sK 就绪)                               │
│ wait load_G[i_k]  (sG 就绪)                               │
│ wait load_V[i_k]  (sV 就绪)                               │
│                                                            │
│ ── K_proc 计算 ──                                          │
│ K_proc[t,k] = K[t,k] * beta[t] * exp2(G[t,k])           │
│ 寄存器中计算 → R2S → k_mma (MMA B-op layout)               │
│ → acquire + commit k_prologue_ready (K_proc ready)         │
│                                                            │
│ ── V_proc 计算 ──                                          │
│ V_proc[t,k] = V[t,k] * beta[t]                            │
│ 寄存器中计算 → R2S → v_mma (独立 buffer)                    │
│ → acquire + commit v_prologue_ready (V_proc ready)         │
│                                                            │
│ release load_K, load_V, load_G                             │
└────────────────────────────────────────────────────────────┘
release beta_ready (WU 结束)
```

#### Akk R2T 关键实现：

```cpp
// Akk 是 [BT, BT] = [64, 64] 下三角矩阵
// R2T 只写 lanes 0-63 (M=64), 上三角 & lane>=64 填 0
for (int ei = 0; ei < size(tRT_rAkk); ei++) {
    auto [row, col] = extract_coord(tRS_cM_akk[ei]);
    if (row < TileT && row >= col)
        tRT_rAkk[ei] = sA(row, col);
    else
        tRT_rAkk[ei] = bf16(0);
}
// cute::copy(tiled_r2t_akk, tRT_rAkk, tRT_tAkk);
// fence_view_async_tmem_store();
```

#### Element-wise → R2S 关键实现：

```cpp
// 128 个 Prologue 线程协作:
// 1) 从 sK, sG 读数据到寄存器
// 2) element-wise: k_proc = k * beta * exp2(g)
// 3) 转 bf16，写入 sB (MMA B-operand SMEM layout)
for (int ei = 0; ei < elems_per_thread; ei++) {
    auto [bt, bk] = coord[ei];
    float k_val = sK(bt, bk);
    float g_val = sG(bt, bk);
    float beta_val = sBeta(bt);
    sB(bk, bt) = bf16(k_val * beta_val * exp2f(g_val));  // K-major: sB[N,K]
}
```

---

### 6.2 WG1: Epilogue (warp 4-7, 128 threads)

**职责**：MMA 结果 store + kg element-wise 计算 store

#### 每个 i_k 迭代 (共 3 个 store: w, u, kg)：

```
┌─ NumKIters 次 (i_k = 0..3) ──────────────────────────────┐
│ ── kg 输出 ──                                              │
│ wait glast_ready[i_k]                                      │
│ wait load_K[i_k], load_G[i_k]  (需要 sK, sG)              │
│ kg[t,k] = K[t,k] * exp2(g_last[k] - G[t,k])              │
│ element-wise → R2S → sO → autovec S2G → GMEM kg           │
│ release load_K, load_G, glast_ready                        │
│                                                            │
│ ── w 输出 ──                                               │
│ wait w_done (K-GEMM)                                       │
│ T2R: TMEM acc → fp32 寄存器                                 │
│ fp32 → bf16 → R2S → sO → autovec S2G → GMEM w             │
│ release w_done                                             │
│                                                            │
│ ── u 输出 ──                                               │
│ wait u_done (V-GEMM)                                       │
│ T2R: TMEM acc → fp32 寄存器                                 │
│ fp32 → bf16 → R2S → sO → autovec S2G → GMEM u             │
│ release u_done                                             │
└────────────────────────────────────────────────────────────┘
```

#### T2R + R2S + S2G 输出流程：

```cpp
// 1. T2R: 从 TMEM accumulator 读取到寄存器
cute::copy(tiled_t2r, tTR_tAcc, tTR_rAcc);
fence_view_async_tmem_load();

// 2. fp32 → bf16
tTR_rAcc_bf16 = convert<bf16>(tTR_rAcc);

// 3. R2S: 寄存器 → sO (ROW_MAJOR SMEM)
cute::copy(tiled_r2s, tRS_rO, tRS_sO);
fence_proxy(ProxyKind::async_shared, SharedSpace::shared_cta);

// 4. Autovec S2G: sO → GMEM (coalesced, 128-bit vectorized)
//    使用 GmemTiledCopyO (AutoVectorizingCopy)
for (int m = 0; m < rows; m++) {
    int bt_coord = coord_m[m];
    if (bt_coord < valid_len) {
        cute::copy(gmem_tiled_copy, sO_row, gW_row);  // 128-bit store
    }
}
```

#### kg element-wise 输出：

```cpp
// kg = k * exp2(g_last - g)
// 直接 element-wise 计算并写入 sO, 然后 autovec S2G
for (int ei = 0; ei < elems_per_thread; ei++) {
    auto [bt, bk] = coord[ei];
    float k_val  = sK(bt, bk);
    float g_val  = sG(bt, bk);
    float gn_val = sGLast(bk);
    sO(bt, bk) = bf16(k_val * exp2f(gn_val - g_val));
}
// sync → autovec S2G
```

---

### 6.3 WG2/MMA: MMA Warp (warp 8, elect_one)

**职责**：执行 UMMA 指令，K-GEMM 和 V-GEMM 串行

```
┌─ 每 WU ──────────────────────────────────────────────────┐
│ wait a_ready (Akk in TMEM, 1×/WU)                         │
│ release a_ready                                            │
│                                                            │
│ for i_k = 0..NumKIters-1:                                  │
│   ── K-GEMM ──                                             │
│   wait k_prologue_ready (K_proc in k_mma)                  │
│   acquire w_done                                           │
│   UMMA: acc[64, 32] = Akk[64, 64] @ K_proc[32, 64]^T     │
│   commit w_done                                            │
│   release k_prologue_ready                                 │
│                                                            │
│   ── V-GEMM ──                                             │
│   wait v_prologue_ready (V_proc in v_mma)                  │
│   acquire u_done                                           │
│   UMMA: acc[64, 32] = Akk[64, 64] @ V_proc[32, 64]^T     │
│   commit u_done                                            │
│   release v_prologue_ready                                 │
└────────────────────────────────────────────────────────────┘
```

MMA 指令调用：

```cpp
// cute::gemm(tiled_mma, tCrA_tmem, tCsB, tCtAcc);
// 其中:
//   tCrA_tmem: TMEM A-operand (Akk), 地址由 TmemAllocation::AKK 指定
//   tCsB:      SMEM B-operand (sB 中的 K_proc 或 V_proc)
//   tCtAcc:    TMEM accumulator, 地址由 TmemAllocation::ACC 指定
```

---

### 6.4 WG2/Load: Load Warp (warp 9, elect_one)

**职责**：TMA G2S 加载所有数据

```
┌─ 每 WU ──────────────────────────────────────────────────┐
│ ── Akk 加载 (1次/WU) ──                                   │
│ TMA: Akk[BT, BT] → sA                                     │
│ → commit load_A                                            │
│                                                            │
│ for i_k = 0..NumKIters-1:                                  │
│   ── K 加载 ──                                             │
│   TMA: K[:, i_k*32:(i_k+1)*32] → sK[stage]                │
│   → commit load_K                                          │
│                                                            │
│   ── V 加载 ──                                             │
│   TMA: V[:, i_k*32:(i_k+1)*32] → sV[stage]                │
│   → commit load_V                                          │
│                                                            │
│   ── G 加载 ──                                             │
│   TMA: G[:, i_k*32:(i_k+1)*32] → sG[stage]                │
│   → commit load_G                                          │
└────────────────────────────────────────────────────────────┘
```

TMA descriptor 构建参考：

```cpp
// K/V shape: (total_len, head_dim, num_heads) stride: (H*D, 1, D)
auto tma_K = make_tma_copy(SM90_TMA_LOAD{},
    make_tensor(gmem_ptr<bf16>(k_ptr), make_layout(shape_KVG, stride_KVG)),
    SmemLayoutInputBF16{});

// Akk shape: (total_len, chunk_size, num_heads) stride: (H*BT, 1, BT)
auto tma_Akk = make_tma_copy(SM90_TMA_LOAD{},
    make_tensor(gmem_ptr<bf16>(A_ptr), make_layout(shape_Akk, stride_Akk)),
    SmemLayoutInputAkkBF16{});
```

---

### 6.5 WG2/Aux: Auxiliary Warps (warp 10-11, 64 threads)

**职责**：加载 beta 到 SMEM + 从 sG 提取 g_last 到 SMEM

g_last 不再从 GMEM 直接加载，而是等待 TMA 将 G 写入 sG 后，从 sG 的最后一行提取。
这样 Aux warp 作为 `load_G` pipeline 的第 3 个 consumer，避免了冗余的 GMEM 访问。

```
┌─ 每 WU ──────────────────────────────────────────────────┐
│ ── beta 加载 (1次/WU) ──                                   │
│ 64 线程协作加载 beta[0:BT] → sBeta (fp32, 64 元素)          │
│ → signal beta_ready (→ Prologue)                           │
│                                                            │
│ for i_k = 0..NumKIters-1:                                  │
│   ── g_last 从 sG 提取 ──                                  │
│   wait load_G[i_k]  (sG 就绪, 作为 g_pipeline 的 consumer)  │
│   g_last[k] = sG[sub_seq_len-1, k]  (chunk 最后一行)        │
│   32 线程提取 → sGLast (fp32, 32 元素)                       │
│   → signal glast_ready (→ Epilogue)                        │
│   release load_G[i_k]  (第 3 个 consumer release)           │
└────────────────────────────────────────────────────────────┘
```

---

## 7. 每个 i_k 迭代的完整执行流程

以 `i_k = 0`（head dim `[0:32]`）为例：

```
时间 →

WG2/Load (w9):
  [TMA Akk→sA] → load_A
  [TMA K[:,0:32]→sK] → load_K
  [TMA V[:,0:32]→sV] → load_V
  [TMA G[:,0:32]→sG] → load_G

WG2/Aux (w10-11):
  [beta→sBeta] → beta_ready          (1次/WU)
  wait load_G → [sG[last_row]→sGLast] → glast_ready → release load_G

WG0/Prologue (w0-3):
  wait beta_ready  (1次/WU)
  wait load_A      (1次/WU)
  [Akk: S2R→mask→R2T→TMEM]           (1次/WU)
  release load_A
  → signal a_ready                     (1次/WU)

  wait load_K, load_V, load_G
  [K_proc=K*beta*exp2(G) → R2S→k_mma] → k_prologue_ready
  [V_proc=V*beta → R2S→v_mma]         → v_prologue_ready
  release load_K, load_V, load_G

WG2/MMA (w8):
  wait a_ready (1次/WU)
  release a_ready

  wait k_prologue_ready
  [K-GEMM: acc=Akk@K_proc^T] → w_done
  release k_prologue_ready

  wait v_prologue_ready
  [V-GEMM: acc=Akk@V_proc^T] → u_done
  release v_prologue_ready

WG1/Epilogue (w4-7):
  wait glast_ready, load_K, load_G
  [kg=K*exp2(g_last-G) → sO → S2G]
  release glast_ready, load_K, load_G

  wait w_done
  [w: T2R→bf16→R2S→sO→S2G]
  release w_done

  wait u_done
  [u: T2R→bf16→R2S→sO→S2G]
  release u_done
```

---

## 8. 并发 Overlap 时序

```
时间 →      WU start     i_k=0                                    i_k=1

Load(w9):  [TMA A]─load_A  [TMA K0,V0,G0]─load_K/V/G ── [TMA K1,V1,G1]─load_K/V/G
Aux(w10):  [beta]─beta_rdy ──── wait load_G─[sG→gn0]─glast_rdy ── wait load_G─[sG→gn1]─glast_rdy
Pro(w0-3): wait β+A─[Akk R2T]─a_ready──┐
                      wait K/V/G─[Kp→k_mma]─k_pro_rdy
                                  [Vp→v_mma]───v_pro_rdy  wait K/V/G─[Kp1→k_mma]─k_pro_rdy
                                                                        [Vp1→v_mma]─v_pro_rdy
MMA(w8):        a_ready─┐
                         k_pro_rdy─[K-GEMM]─w_done
                                    v_pro_rdy─[V-GEMM]─u_done
                                                       k_pro_rdy─[K-GEMM]─w_done
                                                                  v_pro_rdy─[V-GEMM]─u_done
Epi(w4-7):               glast+K/G─[kg0 S2G]
                                     w_done─[w0 S2G]
                                              u_done─[u0 S2G]
                                                      glast+K/G─[kg1 S2G]
                                                                  w_done─[w1 S2G]
                                                                           u_done─[u1 S2G]
```

**关键 overlap**：
1. Prologue i_k=1 与 Epilogue i_k=0 **并行**（不同 WG）
2. MMA K-GEMM 期间 Prologue 可以准备 V_proc
3. Epilogue kg 计算与 MMA 无依赖，可提前执行
4. Load warp 的 TMA 与所有计算异步

---

## 9. SharedMemoryPlan (C++ struct)

```cpp
struct SharedMemoryPlan {
    // ── TMA 加载 buffer ──
    array_aligned<bf16,  cosize_v<SmemLayoutInputAkkBF16>>  akk[StagesA];       // Akk [64,64] ×1
    array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>>     k[StagesLoadStore];  // K   [64,32] ×2
    array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>>     v[StagesLoadStore];  // V   [64,32] ×2
    array_aligned<float, cosize_v<SmemLayoutInputFP32>>     g[StagesLoadStore];  // G   [64,32] ×2

    // ── MMA B-operand staging (K_proc / V_proc, 独立 double-buffer) ──
    array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>>     k_mma[StagesLoadStore]; // K_proc [64,32] ×2
    array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>>     v_mma[StagesLoadStore]; // V_proc [64,32] ×2

    // ── Epilogue 输出 staging ──
    array_aligned<bf16, cosize_v<SmemLayoutOutputBF16>>     out[StagesLoadStore];   // sO [64,32] ×2

    // ── 标量/向量 ──
    alignas(16) float beta_smem[2][TileT];                                       // [2][64]
    alignas(16) float glast_smem[2][TileK];                                      // [2][32]

    // ── TMEM ──
    array_aligned<uint32_t, 1>                               tmem_start_addr;

    // ── Pipeline shared storage (11 条) ──
    alignas(16) typename PipelineA::SharedStorage               pipe_a_storage;
    alignas(16) typename PipelineK::SharedStorage               pipe_k_storage;
    alignas(16) typename PipelineV::SharedStorage               pipe_v_storage;
    alignas(16) typename PipelineG::SharedStorage               pipe_g_storage;
    alignas(16) typename PipelineBeta::SharedStorage            pipe_beta_storage;
    alignas(16) typename PipelineGLast::SharedStorage           pipe_glast_storage;
    alignas(16) typename PipelineAReady::SharedStorage          pipe_a_ready_storage;
    alignas(16) typename PipelineKPrologueReady::SharedStorage  pipe_k_prologue_ready_storage;
    alignas(16) typename PipelineVPrologueReady::SharedStorage  pipe_v_prologue_ready_storage;
    alignas(16) typename PipelineWDone::SharedStorage           pipe_w_done_storage;
    alignas(16) typename PipelineUDone::SharedStorage           pipe_u_done_storage;
};
```

---

## 10. Epilogue GMEM Store 机制

与现有 `kda_fwd_intra_mainloop_sm100.hpp` 和 `fwd_o.py` 的 autovec store 模式一致：

```cpp
// 输出 tile: [TileT, TileK] = [64, 32] bf16, row-major
// kGmemElemsPerStore = 128bit / 16bit = 8 elements per store
// kBlockKGmem = min(TileK, 128/sizeof(bf16)) = 32  (TileK=32 时整行一次搬完)
// kGmemThreadsPerRow = 32 / 8 = 4
// 128 threads / 4 threads_per_row = 32 rows per iteration
// → 64 rows 需要 2 iterations (或展开)

using GmemLayoutAtom = Layout<
    Shape<Int<NumEpilogueThreads/kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
    Stride<Int<kGmemThreadsPerRow>, _1>>;
using GmemTiledCopyO = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{},
    GmemLayoutAtom{},
    Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}
);
```

每个 i_k 迭代输出 3 个 tile (w, u, kg)，共 `3 × TileT × TileK × 2B = 3 × 4KB = 12KB` GMEM 写出。

---

## 11. 数学公式汇总

```
输入:
  K[BT, HeadDim]     bf16   — key
  V[BT, HeadDim]     bf16   — value
  G[BT, HeadDim]     fp32   — cumulative gate (log2 域)
  Akk[BT, BT]        bf16   — 下三角注意力矩阵
  beta[BT]            fp32   — 标量因子

输出 (每个 i_k 切片, head dim [i_k*32 : (i_k+1)*32]):
  w[BT, TileK]  = Akk @ (K * beta * exp2(G))       — MMA (K-GEMM)
  u[BT, TileK]  = Akk @ (V * beta)                  — MMA (V-GEMM)
  kg[BT, TileK] = K * exp2(g_last - G)              — element-wise

其中:
  g_last[k] = G[BT-1, k]   (chunk 的最后一行)
  K_proc = K * beta * exp2(G)   — Prologue 计算
  V_proc = V * beta             — Prologue 计算
```

---

## 12. 开发 Checklist

### Phase 1: 基础结构
- [x] 更新 `TmemAllocation` enum (ACC + AKK 偏移)
- [x] 更新 `SharedMemoryPlan` (k_mma/v_mma double-buffer, out, 11 条 pipeline storage)
- [x] 定义 11 条 pipeline types + 11 条 pipeline state types
- [x] 更新 `TmaParams` (增加 TMA_V)
- [x] 更新 Kernel `operator()` 中的 pipeline 构建和 role dispatch

### Phase 2: Load Warp (warp 9)
- [x] `load_loop()`: TMA Akk, K, V, G 加载, 带 double-buffer 索引管理

### Phase 3: Aux Warps (warp 10-11)
- [x] `load_aux_loop()`: beta (1次/WU) GMEM 加载 + g_last (NumKIters次) 从 sG 提取最后一行
  - g_last 作为 `load_G` pipeline 的第 3 个 consumer，wait sG → 提取 last row → signal glast_ready

### Phase 4: Prologue (WG0, warp 0-3)
- [ ] `compute_prologue_loop()`:
  - [ ] Akk S2R → causal mask → R2T → TMEM (1次/WU) → signal a_ready
  - [ ] K_proc element-wise → R2S → k_mma → signal k_prologue_ready
  - [ ] V_proc element-wise → R2S → v_mma → signal v_prologue_ready

### Phase 5: MMA (warp 8)
- [ ] `mma_loop()`:
  - [ ] wait a_ready (1×/WU)
  - [ ] K-GEMM + V-GEMM 串行, 通过 k/v_prologue_ready + w/u_done 同步

### Phase 6: Epilogue (WG1, warp 4-7)
- [ ] `compute_epilogue_loop()`:
  - [ ] kg element-wise → sO → autovec S2G (wait glast_ready + load_K/G)
  - [ ] w: wait w_done → T2R → bf16 → R2S → sO → autovec S2G
  - [ ] u: wait u_done → T2R → bf16 → R2S → sO → autovec S2G

### Phase 7: 测试与集成
- [ ] 正确性验证 (vs PyTorch reference)
- [ ] Varlen 支持 (per-row boundary check in autovec S2G)
- [ ] Persistent mode 支持
- [ ] 性能 profiling (NCU)
