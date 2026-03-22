# ChunkGlaFwdO Pipeline

> 文件: `cula/ops/fwd_o.py`  
> 类名: `ChunkGlaFwdO`

## 计算公式

$$o = \text{scale} \cdot (q \odot 2^g) @ h + \text{tril}(A_{qk}) @ v$$

- **inter-chunk**: `scale * qg @ h`，其中 `qg = q * exp2(g)`
- **intra-chunk**: `tril(A) @ v`，A 经因果掩码
- 最终输出: `o = scale * acc_qh + acc_av`

## 线程布局

| Warp | 角色 | 职责 |
|------|------|------|
| 0-3 | CUDA Core (128 threads) | qg = q * exp2(g)、因果掩码 tril(A)、R2T/T2R/R2S、O 合并 |
| 4 | MMA | 执行 QH MMA 和 AV MMA |
| 5 | Load | TMA G2S 加载 q、g、A、h、v |
| 6 | Store | TMA S2G 写出 o |
| 7 | Empty | Warp group 寄存器重分配占位 |

**总线程**: 256 (8 warps)  
**寄存器分配**: persistent 模式 — CUDA=208, Others=168; non-persistent — CUDA=208, Others=40

## MMA 操作

| MMA | Tiler (M,N,K) | A 操作数 | B 操作数 | 输出 |
|-----|---------------|----------|----------|------|
| QH | (64, 128, 128) | qg — TMEM K-major | h — SMEM MN-major | acc_qh (FP32) |
| AV | (64, 128, 64) | am — TMEM K-major | v — SMEM MN-major | acc_av (FP32) |

两个 MMA 使用**独立 TMEM ACC 区域** (Dual-ACC)，可背靠背执行无需等待 CUDA 读取。

## Pipeline 阶段

```
         Load Warp              CUDA Warps              MMA Warp            Store Warp
            │                      │                       │                    │
  TMA q→sQ ─┼──load_q_mbar───────>│                       │                    │
  TMA g→sG ─┼──load_g_mbar───────>│                       │                    │
            │                      │ qg=q*exp2(g)         │                    │
            │                      │ R2T→TMEM             │                    │
            │                      ├──qg_mbar────────────>│                    │
  TMA h→sH ─┼──load_h_mbar──────────────────────────────>│ QH MMA             │
            │                      │                       │ qg×h→acc_qh       │
  TMA A→sA ─┼──load_a_mbar───────>│                       │                    │
            │                      │ tril(A)               │                    │
            │                      │ R2T→TMEM             │                    │
            │                      ├──am_mbar────────────>│                    │
  TMA v→sV ─┼──load_v_mbar──────────────────────────────>│ AV MMA             │
            │                      │                       │ am×v→acc_av        │
            │                      │<──acc_done_mbar──────┤                    │
            │                      │ T2R acc_qh, acc_av   │                    │
            │                      │ o=scale*qh+av        │                    │
            │                      │ R2S→sO               │                    │
            │                      ├──o_ready_mbar───────────────────────────>│
            │                      │                       │          TMA sO→GMEM
```

### Pipeline 深度

| Pipeline | 类型 | Stages (persistent / non-persistent) |
|----------|------|------|
| load_q | TMA→CUDA | 2 / 1 |
| load_g | TMA→CUDA | 1 / 1 (FP32 太大不双缓冲) |
| load_h | TMA→MMA | 2 / 1 |
| load_v | TMA→MMA | 2 / 1 |
| load_a | TMA→CUDA | 2 / 1 |
| qg_ready | CUDA→MMA | 1 |
| am_ready | CUDA→MMA | 1 |
| acc_done | MMA→CUDA | 1 |
| o_ready | CUDA→Store | 1 |

## 内存分配

### TMEM (≤512 列)

| 区域 | 大小 | 用途 |
|------|------|------|
| ACC_QH | BT×BV FP32 | QH MMA 累加器 |
| ACC_AV | BT×BV FP32 | AV MMA 累加器 |
| QG_A | BT×BK BF16 | qg 做 MMA A 操作数 |
| AM_A | BT×BT BF16 | am 做 MMA A 操作数 |

### SMEM (persistent ~192KB, non-persistent ~120KB)

| 缓冲 | 大小 | Stages |
|-------|------|--------|
| sQ | BT×BK BF16 | 2 (persistent) |
| sG | BT×BK FP32 | 1 |
| sA | BT×BT BF16 | 2 (persistent) |
| sH | MMA-B layout | 2 (persistent) |
| sV | MMA-B layout | 2 (persistent) |
| sO | BT×BV BF16 | 1 |

## 主循环流程

每个 Work Unit = 一个 chunk 的一个 V tile:

1. **Load warp**: TMA 加载 q, g → SMEM epilog 区; h → SMEM MMA-B 区; v → SMEM MMA-B 区; A → SMEM epilog 区
2. **CUDA warps**: 等待 q, g 到达 → 计算 `qg = q * exp2(g)` → R2T 写入 QG TMEM → signal MMA
3. **MMA warp**: 等待 qg → QH MMA (qg × h → acc_qh)
4. **CUDA warps**: 等待 A → 读 A 并应用 tril 因果掩码 → R2T 写入 AM TMEM → signal MMA
5. **MMA warp**: 等待 am → AV MMA (am × v → acc_av) → signal acc_done
6. **CUDA warps**: T2R 读 acc_qh 和 acc_av → 计算 `o = scale * qh + av` → 转 BF16 → R2S 写 sO → signal Store
7. **Store warp**: TMA S2G 写 o 到 GMEM

## Persistent / Varlen 支持

### Persistent 模式 (occ=1)
- Grid = SM_count，CTA 按 grid-stride 循环处理多个 WU
- 双缓冲 TMA 预取下一 WU 数据，计算与加载重叠

### Non-Persistent 模式 (occ=2)
- 所有 pipeline 降为 1-stage
- Grid = (num_v_tiles, NT, B×H)

### Varlen
- `domain_offset` 偏移 TMA tensor，使用 `cu_seqlens` 和 `chunk_indices` 解码 work unit
- 尾部 chunk 使用 `CopyUniversal` 带行级边界检查

## 关键优化

- **Dual-ACC TMEM**: QH 和 AV 独立累加器区域，MMA 可无间断背靠背执行
- **TMEM A-operand 零拷贝**: qg 和 am 直接 R2T 到 TMEM 做 MMA A 操作数
- **Scale 延迟乘法**: 不对 qg 预乘 scale，而在后累加合并时乘入，减少 BF16 精度损失
- **Store warp 寄存器利用**: occ=1 时给 store warp 168 regs，可缓存完整 O tile 后批量写 GMEM
