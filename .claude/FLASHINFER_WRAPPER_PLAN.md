# FlashInfer Wrapper Usage Plan

## Current Understanding

After examining FlashInfer documentation, examples, and vLLM implementation:

### 1. **Single Wrapper Approach (Used by vLLM)**
- vLLM creates ONE prefill wrapper and ONE decode wrapper at the model level
- These wrappers are shared across ALL layers
- The `plan()` method is called ONCE per batch before processing all layers
- Each layer calls `run()` on the same wrapper instance

### 2. **Key Pattern from vLLM**
```python
# In FlashInferState class:
def _get_prefill_wrapper(self):
    if self._prefill_wrapper is None:
        self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._get_workspace_buffer(), self.get_kv_cache_layout())
    return self._prefill_wrapper

# The wrapper is created once and cached at the state level
# All layers share this single wrapper instance
```

### 3. **Why Single Wrapper Works**
- The `plan()` method creates auxiliary data structures based on batch configuration
- These data structures (indices, memory layouts) are the SAME for all layers
- Each layer has different KV cache data but uses the same indexing pattern
- The `run()` method takes the layer-specific KV cache as input

## The Real Issue with Our Implementation

Our current problem is NOT about single vs multiple wrappers. Both approaches can work. The issue is:

1. **We removed the `plan()` call optimization** - WrapperManager called `plan()` once for all wrappers
2. **We're calling `plan()` inside each layer's forward pass** - This is extremely inefficient
3. **We're not caching the plan between layers** - Each layer replans unnecessarily

## Performance Impact

- Old WrapperManager: `plan()` called once, then N layers call `run()`
- Current implementation: N layers each call `plan()` then `run()`
- This explains the performance regression!

## Proposed Fix

### Option 1: Keep Single Wrapper (like vLLM)
1. Move `plan()` call out of the attention layer
2. Call `plan()` once in ModelRunner before the forward pass
3. Pass the pre-planned wrapper through the context
4. Each layer only calls `run()`

### Option 2: Restore WrapperManager with Optimization
1. Create one wrapper per layer (original approach)
2. Call `plan()` on all wrappers at once before forward pass
3. Each layer uses its own pre-planned wrapper
4. Potentially better cache locality but more memory usage

### Option 3: Hybrid Approach
1. Use single wrapper (less memory)
2. Cache the plan in the wrapper
3. Only re-plan when batch configuration changes
4. Check if indices match before re-planning

## Recommendation

**Go with Option 1** - Single wrapper with proper planning:
- Matches vLLM's proven pattern
- Minimal code changes required
- Lower memory usage
- Just need to move the `plan()` call to the right place

## Implementation Steps

1. Remove `plan()` call from attention layer's forward method
2. Add `plan()` call in ModelRunner.run_model() before calling model forward
3. Store the planned wrapper in InferenceContext
4. Update attention layers to use the pre-planned wrapper
5. Verify performance improvement

This should restore the original performance while keeping the cleaner single-wrapper design.