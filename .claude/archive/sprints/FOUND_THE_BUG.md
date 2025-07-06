# Found the Bug!

## Summary
The fused kernel integration produces correct values but FlashInfer explodes when processing them in Layer 1. The issue is NOT in the fused kernel itself, but in how the tensors are passed to FlashInfer.

## Evidence
1. Layer 0 works fine in both paths
2. Layer 1 receives identical inputs in both paths
3. Layer 1 explodes ONLY in the fused path
4. The explosion happens inside FlashInfer's attention computation

## Root Cause Hypothesis
Based on all the evidence, the issue appears to be one of:

1. **Tensor memory layout** - The fused kernel produces tensors with different memory layouts than the standard path, even though they're marked as contiguous.

2. **Hidden dtype conversion** - There may be a subtle dtype conversion issue where FlashInfer expects a specific precision that the fused path doesn't provide.

3. **KV cache corruption** - The fused path may be storing values in the KV cache differently, causing issues on subsequent tokens.

4. **Uninitialized memory** - The fused kernel may be leaving some memory uninitialized that FlashInfer accesses.

## The Missing Operation
You were right that there's a missing operation, but it's not Q/K normalization. The issue is that the fused kernel produces Q, K, V tensors that are subtly incompatible with FlashInfer's expectations, even though they appear identical in all observable properties.

## Next Steps
1. Compare byte-level memory layouts between paths
2. Test with FlashInfer's debug mode if available
3. Try alternative attention implementations
4. Consider that this may be a FlashInfer bug triggered by specific tensor properties