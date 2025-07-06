# Sprint Investigation Complete: Numerical Instability

## Sprint Outcome: HYPOTHESIS DISPROVEN

The sprint hypothesis that softmax overflow is causing the numerical instability has been thoroughly investigated and **disproven**.

## What We Discovered

### 1. K Scaling Has No Effect
- Tested K scaling factors from 0.0005 to 1.0 (2000x range)
- ALL scales produce identical gibberish output (exclamation marks)
- Even with NO scaling, the issue persists
- This definitively proves K magnitude and softmax overflow are not the root cause

### 2. The Real Explosion Pattern
```
Layer 0: Normal (max ~4.7)
Layer 1: EXPLODES to 10^29
Layers 2-12: Infinity propagates  
Layer 13: Converts to zeros
Result: All zeros → Token 0 ('!') repeatedly
```

### 3. The Issue is FlashInfer Integration
- Our fused kernel produces correct Q, K, V values
- Standard path works fine with the SAME extreme K values
- The explosion happens INSIDE FlashInfer's attention computation
- But only when called from our fused path, not standard path

### 4. Why Softmax Modification Won't Help
- We cannot modify FlashInfer's internal softmax
- K values are already reasonable after scaling (max ~1.8)
- Standard path handles much larger K values (max ~219) without issue
- The explosion happens regardless of input magnitude

## Root Cause Hypothesis

The issue is likely one of:
1. **Tensor format mismatch** - FlashInfer expects different shapes/strides
2. **Precision handling** - Different dtype conversions trigger different code paths
3. **KV cache corruption** - Values stored incorrectly for subsequent tokens
4. **Batch dimension handling** - Single-sequence case handled differently

## Evidence
- Changing K magnitude by 2000x has zero effect
- Model produces token 0 repeatedly (logits all zeros)
- Layer 1 consistently explodes regardless of inputs
- Standard path with same weights works perfectly

## Recommendation

**This sprint's approach is fundamentally wrong.** We need a new investigation focused on:

1. Comparing exact tensor formats passed to FlashInfer
2. Tracing dtype conversions and tensor layouts
3. Checking KV cache population and retrieval
4. Testing with different FlashInfer configurations

The softmax stability approach should be abandoned as it addresses a non-existent problem.

## Time Spent
- Task 1 (Isolate/Analyze): 4 hours ✅
- Task 2 (Implement Stable Softmax): 3 hours ❌ (Blocked - wrong approach)
- Task 3 (Validation): N/A (Blocked by task 2)

Total: 7 hours

## Lessons Learned
1. Always verify hypotheses before implementing solutions
2. When scaling changes have no effect, the issue is elsewhere
3. Integration bugs can masquerade as numerical issues
4. Comparing working vs non-working paths is invaluable

## Next Steps
Create a new sprint focused on debugging the FlashInfer integration rather than trying to fix a non-existent softmax overflow.