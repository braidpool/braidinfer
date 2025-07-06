# Batch Size 1 Optimization Findings

## Root Cause Identified

The 30x slowdown (32ms actual vs 1.4ms theoretical) is caused by excessive CPU-side tensor operations:

### Operation Counts (per forward pass)
- **676 `.to()` calls** - Dtype conversions
- **672 `.view()` calls** - Tensor reshaping  
- **224 `.unsqueeze()` calls** - Dimension manipulation
- **Total: 1,572 tensor operations**

### Breakdown
1. **RMSNorm**: 440+ `.to()` calls (converting between bfloat16 and float32)
2. **Attention layers**: Hundreds of `.view()` operations
3. **Python loop overhead**: Iterating through 28 layers

### Performance Impact
- Each tensor operation: ~0.02-0.1ms CPU time
- Total overhead: ~30ms (matches our observed slowdown)
- GPU is only busy 3.3ms out of 32ms (10% utilization)

## Optimizations Implemented

### 1. Optimized RMSNorm
- Eliminated unnecessary dtype conversions for bfloat16
- Reduced `.to()` calls from 676 to 236 (65% reduction)
- Performance improvement: 30.3 → 32.3 tok/s (7% speedup)

## Remaining Issues

Even with optimizations, we still have:
- 236 `.to()` calls
- 672 `.view()` calls  
- 224 `.unsqueeze()` calls
- Total: 1,132 operations (still too many)

## Solution: CUDA Graphs

CUDA graphs will solve this by:
1. **Capturing the entire execution graph once**
2. **Replaying without CPU involvement**
3. **Eliminating all Python/CPU overhead**
4. **Expected performance: >500 tok/s for batch size 1**

## Implementation Plan

### Phase 1: Fix Immediate Issues
1. ✅ Optimize RMSNorm to reduce dtype conversions
2. ⬜ Investigate remaining `.to()` operations
3. ⬜ Reduce `.view()` operations where possible

### Phase 2: CUDA Graph Implementation
1. ⬜ Fix FlashInfer wrapper state management
2. ⬜ Implement static memory allocation
3. ⬜ Create graph capture for batch size 1
4. ⬜ Handle dynamic shapes properly

### Phase 3: Further Optimizations
1. ⬜ Fused kernels for common patterns
2. ⬜ Torch.compile with appropriate backend
3. ⬜ Custom CUDA kernels for critical paths

## Key Insights

1. **The GPU is fast** - Raw compute takes only 1.4ms
2. **CPU overhead dominates** - 90% of time is CPU operations
3. **Python loops are expensive** - Each layer iteration has overhead
4. **Tensor operations accumulate** - 1500+ ops per forward pass

## Recommendations

For single-user, batch size 1 scenarios:
1. **CUDA graphs are essential** - Will provide 10-20x speedup
2. **Minimize tensor operations** - Each op has CPU overhead
3. **Prefer fused operations** - Reduce kernel launches
4. **Static shapes when possible** - Enables more optimizations

## Next Steps

1. Complete CUDA graph implementation
2. Profile with NVIDIA Nsight for deeper analysis
3. Consider custom kernels for hotspots
4. Investigate torch.compile with inductor backend