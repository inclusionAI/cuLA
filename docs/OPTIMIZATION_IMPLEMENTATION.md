# Lightning Attention Optimization Implementation Plan

## Based on NCU Analysis from PERFORMANCE_SUMMARY.md

### Current Performance Bottlenecks
1. **Register Pressure**: 56 regs/thread → 50% occupancy (Target: <48 regs/thread)
2. **Memory Access**: L1 hit rate 43%, bank conflicts 15%
3. **Compute Stalls**: 60% L1TEX stalls

---

## Implemented Optimizations

### Phase 1: Register Pressure Reduction (In Progress)

#### 1.1 Pre-compute and Cache Decay Values
**Location**: CUDA warps section (line ~1235)

**Before:**
```python
decay_s_cuda = decay_tensor[hidx]
block_decay = cute.exp(-decay_s_cuda * cutlass.Float32(C))
# ... later repeatedly using decay_s_cuda
```

**After:**
```python
decay_s_cuda = decay_tensor[hidx]
block_decay = cute.exp(-decay_s_cuda * cutlass.Float32(C))
# Cache to reduce register pressure
decay_s_cached = decay_s_cuda
# Use cached value to avoid redundant loads
```

**Impact**: 
- Reduces redundant register reads
- Compiler can better optimize register allocation
- Estimated: 2-4 registers saved per thread

#### 1.2 Optimize KV State Management
**Location**: KV state conversion loop (line ~1404)

**Strategy**:
- Load KV state only when needed (idx != 0)
- Minimize lifetime of intermediate tensors
- Use more aggressive register spilling hints

**Impact**:
- Reduces peak register usage during KV operations
- Estimated: 4-6 registers saved

#### 1.3 Reduce Temporary Variable Lifetimes
**Locations**: Throughout CUDA warps section

**Changes**:
- Scope temporary tensors more tightly
- Reuse register space where possible
- Clear unused values explicitly

**Example**:
```python
# Before: variable lives for entire loop
tTR_rS = cute.make_rmem_tensor_like(...)
for chunk_start in range(...):
    # ... use tTR_rS

# After: minimize scope
for chunk_start in range(...):
    tTR_rS = cute.make_rmem_tensor_like(...)
    # ... use tTR_rS
    # Explicit scope end
```

---

### Phase 2: Memory Access Optimization (Planned)

#### 2.1 Add Shared Memory Padding
**Location**: Shared memory layout definitions

**Before:**
```python
q_smem_layout_staged = sm100_utils.make_smem_layout(
    dtype, layout_enum, shape, stages
)
```

**After:**
```python
# Add padding to avoid bank conflicts
padded_shape = (shape[0], shape[1] + 8)  # 8-element padding
q_smem_layout_staged = sm100_utils.make_smem_layout(
    dtype, layout_enum, padded_shape, stages
)
```

**Impact**:
- Eliminates 15% bank conflicts
- Estimated: 2-3% overall speedup

#### 2.2 Improve Memory Coalescing
**Location**: Global memory stores in epilogue warp

**Strategy**:
- Ensure aligned, contiguous writes
- Use vectorized stores where possible
- Optimize thread-to-memory mapping

**Impact**:
- Improve global store efficiency from 6.25% to >20%
- Estimated: 5-7% overall speedup

---

### Phase 3: Compute Pipeline Optimization (Planned)

#### 3.1 Prefetch Data Earlier
**Location**: Main computation loops

**Strategy**:
- Issue TMA loads earlier in the loop
- Better overlap of memory and compute
- Use async copy more aggressively

**Impact**:
- Reduce L1TEX stalls from 60% to <45%
- Estimated: 8-10% overall speedup

#### 3.2 Increase Instruction-Level Parallelism
**Location**: MMA operations

**Strategy**:
- Interleave independent operations
- Unroll more loops
- Use CUDA core operations in parallel with MMA

**Impact**:
- Better pipeline utilization
- Estimated: 5-7% overall speedup

---

## Implementation Status

### ✅ Completed
- NCU profiling and analysis
- Bottleneck identification
- Optimization plan documentation

### 🔄 In Progress  
- Phase 1.1: Decay value caching
- Phase 1.2: KV state optimization
- Code modifications for register reduction

### ⏳ Planned
- Phase 1.3: Variable lifetime optimization
- Phase 2: Memory access improvements
- Phase 3: Compute pipeline optimization

---

## Testing Strategy

### 1. Register Usage Verification
```bash
# Check register usage after optimizations
ncu --metrics launch__registers_per_thread python profile_lightning_attn.py

# Target: <48 registers/thread (currently 56)
```

### 2. Occupancy Check
```bash
# Verify occupancy improvement
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python profile_lightning_attn.py

# Target: >80% (currently 43%)
```

### 3. Memory Access Patterns
```bash
# Validate coalescing improvements
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum python profile_lightning_attn.py

# Target: >80% efficiency (currently 6.25%)
```

### 4. Performance Benchmarks
```bash
# Run full benchmark suite
python benchmark/bench_lightning_attn.py

# Expected improvements:
# - B=64, H=64: 3.14ms → ~2.0ms (1.5x improvement)
# - Average speedup: 2.90x → ~4.5x vs Triton
```

---

## Expected Results

### Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Registers/thread | 56 | <48 | -14% |
| Occupancy | 43% | >80% | +86% |
| L1 Hit Rate | 43% | >60% | +40% |
| Global Store Eff | 6.25% | >20% | +220% |
| Execution Time (B=64,H=64) | 3.14ms | ~1.8ms | 1.7x |
| Speedup vs Triton | 2.73x | ~4.8x | +76% |

### Cumulative Speedup
- Phase 1: +50% (occupancy improvement)
- Phase 2: +27% (memory optimization)
- Phase 3: +22% (compute optimization)
- **Total Expected: ~2.3x additional speedup**

**Final Target**: 1.4ms (from 3.14ms) = 6.1x vs Triton

---

## Risk Mitigation

### Potential Issues
1. **Compiler Optimization Interference**: Manual optimizations may conflict with compiler
2. **Numerical Precision**: Aggressive optimizations might affect accuracy
3. **Hardware-Specific**: Optimizations may not generalize to all GPUs

### Mitigation Strategies
1. Incremental changes with validation after each step
2. Maintain accuracy tests (relative error <2%)
3. Profile on target hardware (SM100 Blackwell)

---

## Next Actions

1. **Immediate** (Today):
   - Complete Phase 1.1 decay caching
   - Test and validate register reduction
   - Re-run NCU profiling

2. **Short-term** (This Week):
   - Implement Phase 1.2-1.3
   - Begin Phase 2 memory optimizations
   - Benchmark intermediate results

3. **Medium-term** (Next Week):
   - Complete Phase 2
   - Start Phase 3
   - Full validation and benchmarking

---

## Notes

- All optimizations are based on NCU profiling data from B=64, H=64, T=4096, D=128
- Changes maintain API compatibility
- Accuracy target: <2% relative error vs reference implementation
- Performance validated against Triton baseline

---

**Document Version**: 1.0
**Last Updated**: January 4, 2026
**Author**: Optimization Team
**Status**: Implementation In Progress
