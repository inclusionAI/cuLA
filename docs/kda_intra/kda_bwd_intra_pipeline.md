# KDA Backward Intra-Chunk Kernel Pipeline Analysis

## 1. Overview

`kda_bwd_intra_sm100.cu` implements the backward pass for the intra-chunk portion of the KDA (Key-Data-Attention) attention mechanism on NVIDIA SM100 (Blackwell) architecture. The kernel computes gradients (`dQ`, `dK`, `dG`, `dB`) using a sophisticated pipeline with **sub-chunk MMA** technique.

### Key Dimensions
| Constant | Value | Description |
|----------|-------|-------------|
| `SUB_T_TILE` | 16 | Sub-chunk tile size (each MMA handles 16 rows) |
| `T_TILE` | 64 | Tile size for sequence dimension |
| `K_SIZE` | 128 | Head dimension |
| `K_TILE` | 32 | K-dimension tile for MMA |
| `K_ITERATION` | 4 | Number of K iterations per tile |
| `NUM_BUF_A` | 1 | Single buffer for dA matrices |
| `NUM_BUF_VALUE` | 2 | Double buffer for Q/K/G/dq/dk/dg |
| `NUM_THREADS` | 384 | Total threads (128×3 warpgroups) |

---

## 2. Warp Role Assignment

### Warp Assignment Table
The kernel uses 12 warps (384 threads), assigned via `kWarpAssignment = 0x12'5555'5555ull`:

| warp_idx | Role | Warpgroup | Description |
|----------|------|-----------|-------------|
| 0, 1 | `ComputeEpilogue` | WG0 (128 threads) | Lower half: rows [0,31] of K-dim |
| 2, 3 | `ComputeEpilogue` | WG1 (128 threads) | Upper half: rows [32,63] of K-dim |
| 4 | `Load` | WG1 | TMA data loading |
| 5, 6, 7 | `Empty` | WG1, WG2 | Load beta from global memory |
| 8 | `Mma` | WG2 | Execute MMA operations |
| 9, 10, 11 | `Empty` | WG2 | Unused |

### Thread Indexing
```cpp
int warpgroup_idx = cutlass::canonical_warp_group_idx(); // 0, 1, or 2
int idx_in_warpgroup = threadIdx.x % 128; // 0-127 within warpgroup
```

---

## 3. Shared Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SharedMemoryPlan                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Double-buffered (NUM_BUF_VALUE=2):                                      │
│   q[2]                : BF16 [64×32] each                               │
│   k[2]                : BF16 [64×32] each                               │
│   g[2]                : FP32 [64×32] each                               │
│   dq[2]               : FP32 [64×32] each                               │
│   dk[2]               : FP32 [64×32] each                               │
│   dg[2]               : FP32 [64×32] each                               │
│   b_k_exp[2]          : FP32 [64×32] each                               │
│   b_k_neg_exp[2]      : FP32 [64×32] each                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Single-buffered (for MMA):                                              │
│   kg_all.intra[6]     : TF32 [32×16] transposed each                    │
│   kg_all.inter[4]     : TF32 [32×16] transposed each                    │
│   qkg_all.intra[6]    : TF32 [32×32] transposed each                    │
│   qkg_all.inter[4]    : TF32 [32×32] transposed each                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Single-buffered (NUM_BUF_A=1):                                          │
│   dAqk[1]             : FP32 [64×64]                                    │
│   dAkk[1]             : FP32 [64×64]                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Special:                                                                 │
│   beta_smem[2][64]    : BF16, double-buffered by A_phase                │
│   tile_id[2]          : int, double-buffered by A_phase                 │
│   db_partial[2][64]   : FP32, for intra-WG db reduction                 │
│   tmem_start_addr     : TMEM allocation address (512 elements)          │
├─────────────────────────────────────────────────────────────────────────┤
│ Barriers (mbarrier):                                                     │
│   bar_load_kg_ready[2]   : Load → CE (K/G data ready)                   │
│   bar_load_qb[2]         : Load → CE (Q/DQ data ready)                  │
│   bar_load_dkg_ready[2]  : Load → CE (DK/DG data ready)                 │
│   bar_load_dA_ready[1]   : Load → CE+Empty+MMA (dA TMA complete)        │
│   bar_kg_all_ready       : CE → MMA (kg_all ready, 256 threads arrive)  │
│   bar_qkg_all_ready      : CE → MMA (qkg_all ready, 256 threads arrive) │
│   bar_dA_ready[1]        : CE → MMA (mask_A complete)                   │
│   bar_dAt_ready[1]       : CE → MMA (mask_At complete, first k_idx)     │
│   bar_dA_mask_ready[1]   : Empty → CE (beta_smem ready)                 │
│   bar_dq_done            : MMA → CE (dq/dq2 results ready)              │
│   bar_dkt_done           : MMA → CE (dkt results ready)                 │
│   bar_dvalue_free[2]     : CE → Load (value buffer free for reuse)      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. TMEM Address Layout

The kernel allocates 512 TMEM elements for MMA accumulator and input:

```
TMEM Address Layout (per buf_idx_value):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ dq [0, 32]   │ dq2 [32, 64] │ dkt [64, 96] │ dAqk [96-160]│
│ dq_02        │ dq2_02       │              │ dAqk_02      │
│ dq_13        │ dq2_13       │              │ dAqk_13      │
└──────────────┴──────────────┴──────────────┴──────────────┘

Lane offset: +16*65536 for _13 variants (lane 16-31)

dAqk_t (transposed dA, first k_idx only):
  [352, 368], [384, 400], [416, 432], [448, 464] → Aqk_t
  [368, 384], [400, 416], [432, 448], [464, 480] → Akk_t
```

---

## 5. Pipeline Overview

### 5.1 High-level Pipeline Flow

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    Persistent Tile Loop                      │
                    └─────────────────────────────────────────────────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         │                                     │                                     │
         ▼                                     ▼                                     ▼
   ┌──────────┐                         ┌──────────┐                         ┌──────────┐
   │   Load   │                         │   Empty  │                         │CE (WG0+WG1)│
   │   Warp   │                         │  Warps   │                         │256 threads│
   └────┬─────┘                         └────┬─────┘                         └────┬─────┘
        │                                    │                                    │
        │ TMA load dAqk/dAkk                 │ Load beta                          │
        │ Signal bar_load_dA_ready ─────────┼──────────────────────────────────►│ Wait bar_load_dA_ready
        │                                    │ Signal bar_dA_mask_ready ────────►│ Wait bar_dA_mask_ready
        │                                    │                                    │ mask_A (dA → TMEM)
        │                                    │                                    │ Signal bar_dA_ready ───────────┐
        │                                    │                                    │                                │
        │ ──────── K_ITERATION loop ──────── │ ───────────────────────────────── │ ◄──────────────────────────────┘
        │                                    │                                    │
        │ TMA load K/G                       │                                    │ Wait bar_load_kg_ready
        │ Signal bar_load_kg_ready ─────────┼──────────────────────────────────►│ Compute kg_intra/kg_inter
        │                                    │                                    │ Signal bar_kg_all_ready ──────┐
        │                                    │                                    │                               │
        │ TMA load Q/DQ                      │                                    │ Wait bar_load_qb              │
        │ Signal bar_load_qb ───────────────┼──────────────────────────────────►│ Compute qkg_intra/qkg_inter
        │                                    │                                    │ Signal bar_qkg_all_ready ───┐│
        │                                    │                                    │                             ││
        │                                    │                                    │   ┌─────────────────────────┘│
        │                                    │                                    │   ▼                        │
        │                                    │                                    │ MMA kg phase ◄─────────────┘
        │                                    │                                    │ wait bar_kg_all_ready
        │                                    │                                    │
        │ TMA load DK/DG                     │                                    │ Compute intra_scale
        │ Signal bar_load_dkg_ready ────────┼──────────────────────────────────►│ Wait bar_dq_done
        │                                    │                                    │ Process dq results
        │                                    │                                    │ Wait bar_dkt_done
        │                                    │                                    │ Process dkt results
        │                                    │                                    │ Exchange dkt (WG0↔WG1)
        │                                    │                                    │ Output: dq/dk/dg/db
        │                                    │                                    │ Signal bar_dvalue_free ─────┐
        │                                    │                                    │                             │
        │ ◄──────────────────────────────────┼──────────────────────────────────── Wait bar_dvalue_free ◄─┘
        │                                    │                                    │
        └────────────────────────────────────┴────────────────────────────────────┘
```

### 5.2 Producer-Consumer Relationships

| Producer | Consumer | Data | Barrier |
|----------|----------|------|---------|
| Load | CE, MMA, Empty | dAqk, dAkk, tile_id | `bar_load_dA_ready` |
| Empty | CE | beta_smem | `bar_dA_mask_ready` |
| Load | CE | K, G | `bar_load_kg_ready` |
| Load | CE | Q, DQ | `bar_load_qb` |
| CE | MMA | kg_all (intra+inter) | `bar_kg_all_ready` |
| CE | MMA | qkg_all (intra+inter) | `bar_qkg_all_ready` |
| CE | MMA | dAqk (masked) in TMEM | `bar_dA_ready` |
| CE | MMA | dAqk_t (transposed) | `bar_dAt_ready` |
| MMA | CE | dq, dq2 in TMEM | `bar_dq_done` |
| MMA | CE | dkt in TMEM | `bar_dkt_done` |
| Load | CE | DK, DG | `bar_load_dkg_ready` |
| CE | Load | value buffer free signal | `bar_dvalue_free` |

---

## 6. Detailed Warp Behaviors

### 6.1 Load Warp

**Responsibilities:**
1. Fetch tile ID via `tile_scheduler.get_next_tile_id()`
2. TMA load dAqk/dAkk (per-tile, once)
3. TMA load K/G/Q/DQ/DK/DG (per k_idx)
4. Manage phase tracking for double-buffered barriers

**Key Code Flow:**
```cpp
for (;;) {  // Persistent tile loop
    // 1. Get tile ID and write to smem
    int tid = tile_scheduler.get_next_tile_id();
    shared_plan->tile_id[A_phase] = tid;

    if (tid >= total_tiles) {
        // Sentinel: signal termination
        break;
    }

    // 2. Decode tile coordinates
    auto blk_coord = TileScheduler::decode_tile_coord(tid, ...);

    // 3. Wait for previous tile's dAt to be consumed
    cute::wait_barrier(shared_plan->bar_dAt_ready[buf_idx_A], A_phase ^ 1);

    // 4. TMA load dA
    launch_tma_copy(tma_dAkk, gDakk, sDAkk, bar_load_dA_ready);
    launch_tma_copy(tma_dAqk, gDaqk, sDAqk, bar_load_dA_ready);

    // 5. Per-k_idx TMA loads
    for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
        cute::wait_barrier(shared_plan->bar_dvalue_free[buf_idx_value], local_phase^1);

        // Sequential TMA loads with different barriers
        launch_tma_copy(tma_k, gK, sK, bar_load_kg_ready);
        launch_tma_copy(tma_g, gG, sG, bar_load_kg_ready);

        launch_tma_copy(tma_q, gQ, sQ, bar_load_qb);
        launch_tma_copy(tma_dq, gDQ, sDQ, bar_load_qb);

        launch_tma_copy(tma_dk, gDK, sDK, bar_load_dkg_ready);
        launch_tma_copy(tma_dg, gDG, sDG, bar_load_dkg_ready);

        // Advance double buffer
        buf_idx_value = (buf_idx_value + 1) % 2;
    }
}
```

### 6.2 Empty Warps

**Responsibilities:**
- Load `beta` values from global memory to smem

**Key Code Flow:**
```cpp
for (;;) {  // Persistent tile loop
    // Wait for Load to complete dA TMA
    cute::wait_barrier(shared_plan->bar_load_dA_ready[buf_idx_A], A_phase);

    if (tid >= total_tiles) break;

    // Each thread loads one beta value
    if (empty_idx < T_TILE) {
        shared_plan->beta_smem[A_phase][empty_idx] =
            (empty_idx < sub_seq_len)
            ? params.beta_ptr[(token_offset + tile_idx * T_TILE + empty_idx) * params.h + head_idx]
            : __nv_bfloat16(0);
    }

    // Signal beta ready
    cute::arrive_barrier(shared_plan->bar_dA_mask_ready[0]);
}
```

### 6.3 ComputeEpilogue Warpgroups (WG0 & WG1)

WG0 handles K-offset [0, 16] (lower half), WG1 handles K-offset [16, 32] (upper half).

**Split within each WG:**
- Threads [0, 63]: Lower half → output dQ (WG0) or dG (WG1)
- Threads [64, 127]: Upper half → accumulate dB (WG0/WG1) or output dK (WG1)

**Key Phases in `compute_epilogue_body`:**

```cpp
// === PROLOGUE: mask_A (dA → TMEM) ===
mask_A_tensor<WG_IDX * 32, 32>(dA_ptr, idx_in_warpgroup, sub_seq_len, tmem_addr::dAqk);
cute::arrive_barrier(shared_plan->bar_dA_ready[buf_idx_A]);

// Wait for beta
cute::wait_barrier(shared_plan->bar_dA_mask_ready[0], tile_phase);

for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
    // === Wait for K/G data ===
    cute::wait_barrier(shared_plan->bar_load_kg_ready[buf_idx_value], local_phase);

    // === COMPUTE: kg_intra (non-overlapping rows) ===
    setup_kg_intra(...);

    // === Wait for Q data ===
    cute::wait_barrier(shared_plan->bar_load_qb[buf_idx_value], local_phase);

    // === COMPUTE: fused kg + qkg ===
    setup_intra_fused(...);    // intra-chunk
    setup_inter_fused(...);    // inter-chunk

    cute::arrive_barrier(shared_plan->bar_kg_all_ready);

    // === COMPUTE: mask_At (first k_idx only) ===
    if (k_idx == 0) {
        mask_At_tensor<...>(dAqk, dAkk, ..., tmem_addr::dAqk_t);
        cute::arrive_barrier(shared_plan->bar_dAt_ready[buf_idx_A]);
    }

    // === EPILOGUE: compute intra scale ===
    epilogue_compute_intra_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, scale);

    // === COMPUTE: qkg_intra (remaining rows) ===
    setup_qkg_intra(...);

    cute::arrive_barrier(shared_plan->bar_qkg_all_ready);

    // === EPILOGUE: process dq results ===
    cute::wait_barrier(shared_plan->bar_dq_done, b_phase);
    epilogue_apply_dq_intra<HALF_K>(..., tmem_addr::dq + K_OFF, res, scale);
    epilogue_combine_dq_inter<HALF_K>(tmem_addr::dq2 + K_OFF, res, scale);

    // === EPILOGUE: output dq/db ===
    if (idx_in_warpgroup >= 64) {
        // Upper half: accumulate db, scale by beta
        epilogue_accumulate_db<HALF_K, K_OFF>(sK, ..., res, db, ...);
    } else {
        // Lower half: output dq
        epilogue_output_dq<HALF_K, K_OFF>(sQ, sDQ, ..., res, dq_out_base);
    }

    // === DB reduction: WG0→WG1 ===
    // WG0 writes partial db to smem, WG1 accumulates

    // === EPILOGUE: process dkt ===
    cute::wait_barrier(shared_plan->bar_dkt_done, b_phase);
    epilogue_process_dkt<HALF_K>(..., tmem_addr::dkt + K_OFF, res_dkt, scale);

    // === DKT exchange: WG0↔WG1 ===
    NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);
    epilogue_exchange_dkt<HALF_K, K_OFF>(sDKT_0, sDKT_1, ..., res, res_dkt);
    NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);

    // === EPILOGUE: output dg/dk ===
    cute::wait_barrier(shared_plan->bar_load_dkg_ready[buf_idx_value], local_phase);
    if (idx_in_warpgroup < 64) {
        epilogue_output_dg<HALF_K, K_OFF>(sK, sDG, sDKT_1, ..., res, res_dkt, dg_out_base);
    } else {
        epilogue_output_dk<HALF_K, K_OFF>(sDK, sDKT_0, ..., res, res_dkt, dk_out_base);
    }

    cute::arrive_barrier(shared_plan->bar_dvalue_free[buf_idx_value]);
}
```

### 6.4 Mma Warp

**Responsibilities:**
- Execute MMA operations using tcgen05 (Blackwell Tensor Core)
- Use MASK variants to enable sub-chunk computation

**Sub-chunk MMA Technique:**

The tile is 64×32 (M×N), but logically divided into 4 sub-chunks of 16 rows each. The MMA uses `MASK` variants to disable output lanes for certain rows, enabling separate accumulation per sub-chunk.

**MMA Phases:**

```cpp
for (;;) {  // Persistent tile loop
    cute::wait_barrier(shared_plan->bar_dA_ready[buf_idx_A], A_phase);
    if (tid >= total_tiles) break;

    for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
        // === KG PHASE ===
        cute::wait_barrier(shared_plan->bar_kg_all_ready, b_phase);

        // kg_intra: 3 MMA calls with MASK02/MASK13
        utcmma_ts(tile_mma_dqk_mask02, tAqk_1, sKG_1, tDQ_13, true);  // MASK02
        utcmma_ts(tile_mma_dqk_mask13, tAqk_2, sKG_2, tDQ_02, true);  // MASK13
        utcmma_ts(tile_mma_dqk_mask13, tAqk_3, sKG_3, tDQ_13, true);  // MASK13

        // kg_inter: 4 MMA calls with MASK02/MASK13
        utcmma_ts(tile_mma_dqk_mask02, tAqk_0, sKG_0, tDQ2_02, true);
        utcmma_ts(tile_mma_dqk_mask02, tAqk_1, sKG_1, tDQ2_13, true);
        utcmma_ts(tile_mma_dqk_mask13, tAqk_2, sKG_2, tDQ2_02, true);
        utcmma_ts(tile_mma_dqk_mask13, tAqk_3, sKG_3, tDQ2_13, true);

        umma_arrive_noelect(shared_plan->bar_dq_done);

        // === QKG PHASE ===
        if (k_idx == 0) {
            cute::wait_barrier(shared_plan->bar_dAt_ready[buf_idx_A], A_phase);
        }
        cute::wait_barrier(shared_plan->bar_qkg_all_ready, b_phase);

        // qkg_intra: 3 MMA calls with MASK0/MASK1
        utcmma_ts(tile_mma_dqk_mask0, tAqk_1, sKG_1, tDKT_02, true);  // MASK0
        utcmma_ts(tile_mma_dqk_mask0, tAqk_2, sKG_2, tDKT_13, true);  // MASK0
        utcmma_ts(tile_mma_dqk_mask1, tAqk_3, sKG_3, tDKT_02, true);  // MASK1

        // qkg_inter: 4 MMA calls with MASK2/MASK3
        utcmma_ts(tile_mma_dqk_mask2, tAqk_0, sKG_0, tDKT_02, true);  // MASK2
        utcmma_ts(tile_mma_dqk_mask2, tAqk_1, sKG_1, tDKT_13, true);  // MASK2
        utcmma_ts(tile_mma_dqk_mask3, tAqk_2, sKG_2, tDKT_02, true);  // MASK3
        utcmma_ts(tile_mma_dqk_mask3, tAqk_3, sKG_3, tDKT_13, true);  // MASK3

        umma_arrive_noelect(shared_plan->bar_dkt_done);
    }
}
```

---

## 7. Sub-chunk MMA Mask Mapping

### 7.1 Understanding MASK Variants

SM100's tcgen05 MMA supports `disable-output-lane` masks, which control which 16-row chunks contribute to the output:

```
MASK0:  Enable rows [16, 32)  (chunk 1)
MASK1:  Enable rows [32, 48)  (chunk 2)
MASK2:  Enable rows [0, 16)   (chunk 0)
MASK3:  Enable rows [48, 64)  (chunk 3)
MASK02: Enable rows [0, 16) + [32, 48)  (chunks 0, 2)
MASK13: Enable rows [16, 32) + [48, 64) (chunks 1, 3)
```

### 7.2 Data Flow for Sub-chunks

```
                    Tile M=64 rows
┌─────────────────────────────────────────────────────────────────────────┐
│ Chunk 0: rows [0,16)   │ Chunk 1: rows [16,32) │ ...                   │
└─────────────────────────────────────────────────────────────────────────┘
         │                        │
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MMA with MASK02                                   │
│   A: dAqk rows [0,63)   B: kg_all (TF32 transposed)                     │
│   C: dq (TMEM)          Accumulate only chunks 0,2                      │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MMA with MASK13                                   │
│   Accumulate only chunks 1,3 (adds to same dq accumulator)              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 kg/qkg Matrix Organization

The `kg_all` and `qkg_all` matrices are organized so each MMA call processes the correct sub-chunk:

**kg_all.intra[6]** - For intra-chunk K×G products:
- Index 0: chunk 0 unique (uses gn3)
- Index 1-2: chunks 1-2 shared rows
- Index 3-5: remaining chunks

**qkg_all.intra[6]** - For intra-chunk Q×K×G products:
- Two parts per entry: Q part (rows 0-15) and K×beta part (rows 16-31)

**kg_all.inter[4]** / **qkg_all.inter[4]** - For inter-chunk products:
- 4 entries for 4 sub-chunks

---

## 8. Phase Tracking

### 8.1 Double Buffering with Phases

The kernel uses phase tracking to correctly wait on cyclic barriers:

```cpp
int state_phase = 0;  // Tracks phase for each buffer

// For each k_idx iteration:
int local_phase = (state_phase >> buf_idx_value) & 1;

// Wait with phase
cute::wait_barrier(barrier, local_phase);

// After processing, flip phase
state_phase ^= 1 << buf_idx_value;
buf_idx_value = (buf_idx_value + 1) % NUM_BUF_VALUE;
```

### 8.2 B-matrix Phase (MMA ↔ CE)

The MMA results (dq, dkt) use a simple `b_phase` counter:

```cpp
int b_phase = 0;

// CE waits for MMA result
cute::wait_barrier(shared_plan->bar_dq_done, b_phase);

// MMA signals result ready
umma_arrive_noelect(shared_plan->bar_dq_done);

// After each k_idx
b_phase ^= 1;
```

---

## 9. Data Exchange Between Warpgroups

### 9.1 DKT Exchange (WG0 ↔ WG1)

After computing dkt, the two warpgroups exchange results:

```cpp
// Step 1: Each WG writes its dkt to smem
if (idx_in_warpgroup < 64) {
    // Lower half writes to sDKT_0
    store_128b(&sDKT_0(row, K_OFF + i*4), res_dkt[i*4]);
} else {
    // Upper half computes dk_sub_dkt and writes to sDKT_1
    dk_sub_dkt = res - res_dkt;  // res = beta*dq
    store_128b(&sDKT_1(row, K_OFF + i*4), dk_sub_dkt);
}

// Step 2: Named barrier for exchange (128 threads = 1 WG)
NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);

// Step 3: Read exchanged data
// Lower half reads sDKT_1 (dk_sub_dkt from upper)
// Upper half reads sDKT_0 (dkt_sub from lower)

NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);
```

### 9.2 DB Reduction (WG0 → WG1)

DB is accumulated across warpgroups:

```cpp
// WG0 writes partial sum
if (WG_IDX == 0 && local_idx < sub_seq_len) {
    shared_plan->db_partial[0][local_idx] = db;
}

// Named barrier for exchange
NamedBarrier::arrive_and_wait(128, 1);

// WG1 accumulates
if (WG_IDX == 1 && local_idx < sub_seq_len) {
    db += shared_plan->db_partial[0][local_idx];
}

// WG0 resets (avoids double-counting)
if constexpr (WG_IDX == 0) {
    db = 0.0f;
}
```

---

## 10. Output Summary

### 10.1 Gradient Outputs

| Gradient | Producer | Shape | Notes |
|----------|----------|-------|-------|
| `dQ` | CE (lower half) | [seq, h, d] | BF16 output, accumulated with input dQ |
| `dK` | CE (upper half) | [seq, h, d] | BF16 output, accumulated with input dK |
| `dG` | CE (lower half) | [seq, h, d] | FP32 output, accumulated with input dG |
| `dB` | CE (WG1 upper only) | [seq, h] | FP32 output, fully reduced |

### 10.2 Output Formulas

**dQ (intra + inter):**
```
dQ = dAqk @ kg_intra + dAqk @ kg_inter
   + scale_intra (applied in epilogue)
```

**dK (intra + inter):**
```
dK = beta * dQ + dkt_sub + dkt_exchanged + dK_input
```

**dG:**
```
dG = dq * Q + (dk_sub_dkt - dkt) * K + dG_input
```

**dB:**
```
dB = sum(beta * dq * K)  // accumulated across all k_idx
```

---

## 11. Performance Considerations

### 11.1 Memory Access Patterns
- **TMA**: All global memory loads use TMA for optimal bandwidth
- **Double buffering**: Hides memory latency with compute overlap
- **SMEM swizzling**: Uses `UMMA::Layout_K_SW64_Atom` for bank conflict avoidance

### 11.2 Compute Overlap
- CE computes `kg_all` while Load fetches `Q/DQ`
- CE computes `intra_scale` while MMA executes `kg_phase`
- CE processes `dq` results while MMA executes `qkg_phase`

### 11.3 Register Allocation
- CE: 200 registers per thread (high for compute-heavy)
- MMA: Load uses reduced registers (96 less)
- Trading registers for shared memory to fit larger tiles

---

## 12. References

- `csrc/kda_bwd/kda_bwd_intra_sm100.cu` - Main kernel implementation
- `csrc/kda_bwd/helpers.h` - TMEM load/store, MMA helpers
- `csrc/kda_bwd/util_func.h` - Compute/Epilogue functions
- `csrc/kda_bwd/gemm.h` - MMA tile definitions