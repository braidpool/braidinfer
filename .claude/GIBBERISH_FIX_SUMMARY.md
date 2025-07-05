# Gibberish Output Fix Summary

## Problem
After making optimization changes for CUDA graphs and torch.compile, the model started generating gibberish output again.

## Root Cause
The planning cache optimization I implemented was keeping stale state and not replanning the FlashInfer decode wrapper when needed. This caused the attention mechanism to use incorrect indices for the KV cache.

## Changes That Caused Issues

1. **Planning Cache** (main culprit):
   ```python
   # Cache was preventing replanning when sequences changed
   if needs_replan:
       self.decode_wrapper.plan(...)
   ```

2. **CUDA Graph Pre-allocated Buffers**:
   ```python
   # Pre-allocated positions buffer was initialized to zeros
   self._positions_buffer = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
   ```
   This buffer was being used even when CUDA graphs weren't active.

## Solution
Reverted all the following changes:
1. Removed planning cache - now always replans for decode
2. Removed CUDA graph pre-allocated buffers
3. Removed torch.compile integration
4. Removed CUDA graph runner code
5. Cleaned up unused optimization files

## Current Status
- Model generates coherent output again
- Performance: ~30 tok/s for batch size 1
- All core functionality working correctly

## Lessons Learned
1. **Planning is critical**: FlashInfer's planning step must be called with correct indices for each decode step
2. **State management is tricky**: Caching planning state can lead to subtle bugs
3. **Test incrementally**: Each optimization should be tested separately before combining

## Next Steps
For custom CUDA kernels:
1. Start with simple fused attention for batch size 1
2. Avoid complex state management
3. Test thoroughly at each step
4. Profile to ensure actual performance gains