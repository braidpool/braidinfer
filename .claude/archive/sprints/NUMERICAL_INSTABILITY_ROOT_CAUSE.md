# Numerical Instability Root Cause Analysis

## Executive Summary

The fused RMSNorm+QKV kernel produces correct outputs in isolation, but causes numerical explosion when integrated with FlashInfer attention. The issue is NOT K scaling or softmax overflow as initially hypothesized.

## Key Findings

### 1. The Explosion Pattern
- Layer 0: Normal output (max ~4.7)
- Layer 1: **Explodes to 10^29**
- Layers 2-12: Infinity propagates
- Layer 13: Converts to zeros
- All subsequent layers: Zero output
- Result: Model always generates token 0 ('!')

### 2. K Scaling Is Not The Issue
- Tested K scales from 1.0 to 0.0005
- ALL scales produce the same gibberish output
- Even with NO scaling, the issue persists
- This proves K magnitude is not the root cause

### 3. The Fused Kernel Works Correctly
- In isolation, produces correct Q, K, V values
- Matches PyTorch reference implementation
- K values after normalization are as expected

### 4. Standard Path Works Fine
- Uses the same extreme K norm weights (up to 96.5)
- FlashInfer handles large K values without issue
- Produces coherent text

## Root Cause Hypothesis

The issue appears to be in how the fused attention path interacts with FlashInfer:

1. **Different tensor layouts or shapes** between what FlashInfer expects and what we provide
2. **Precision handling differences** - the fused path may trigger different code paths in FlashInfer
3. **KV cache interaction** - the fused path might store values differently in the cache

## Evidence Against Softmax Overflow

1. K scaling from 0.0005 to 1.0 makes no difference
2. Layer 1 K values are reasonable (max ~1.8 with 0.01 scaling)
3. Standard path works with much larger K values (max ~219)
4. The explosion happens AFTER attention computation

## Next Steps

1. Compare the exact tensor shapes and dtypes passed to FlashInfer between standard and fused paths
2. Check if the KV cache is being populated correctly
3. Investigate if the issue is in the reshape/flatten operations around rotary embeddings
4. Consider if the problem is in how we're batching the attention computation

## Conclusion

The SPRINT's focus on implementing stable softmax is not the correct solution. The issue is more fundamental - something about how the fused kernel path interacts with FlashInfer causes numerical explosion, regardless of the actual values involved.