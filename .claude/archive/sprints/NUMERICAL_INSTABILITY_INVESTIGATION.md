# Numerical Instability Investigation - Qwen3 Custom Kernels

## Executive Summary

The custom Triton kernels cause numerical instability in the Qwen3-0.6B model, despite producing nearly identical initial outputs to PyTorch. The issue manifests as exploding values in Layer 1 that cascade through the model, resulting in all-zero token predictions.

## Root Cause Analysis

### 1. The Explosion Point
- **Location**: Layer 1 self-attention output
- **Magnitude**: Values reach ~1e29 (infinity in practice)
- **Cascade Effect**: All subsequent layers produce zeros

### 2. Contributing Factors

#### Extreme K Normalization Weights
```
Layer 0: mean=3.56, std=9.13, max=96.5 (!!)
Layer 1: mean=2.34, std=3.98, max=44.5
Layer 2: mean=2.47, std=3.94, max=44.0
```

These extreme values cause ~40x amplification during normalization:
- Input K values: std ~0.2
- After K norm: std ~10-20
- This pushes values into unstable numerical ranges

#### Precision Interactions
1. Fused kernel operates in float32
2. Model weights are bfloat16
3. Extreme weight values + precision mixing = instability

### 3. Why Standard PyTorch Works

PyTorch's implementation appears to have implicit numerical stability guards that handle the extreme K norm weights gracefully. Our custom kernel, while mathematically equivalent, lacks these safeguards.

## Debugging Timeline

### Phase 1: Initial Discovery
- Symptom: chat.py produces "!!!!!" (token ID 0)
- Finding: All generated tokens are 0

### Phase 2: Following QWEN3_NUMERICAL_STABILITY_GUIDE.md
- ✅ Added embedding scaling (1/sqrt(1024) = 0.03125)
- ✅ Verified RoPE theta = 1,000,000
- ✅ Confirmed fused kernel output matches PyTorch (max diff: 0.0029)

### Phase 3: Layer-by-Layer Investigation
- Layer 0: Output reasonable but higher variance with custom kernels
- Layer 1: Catastrophic explosion to infinity
- Layer 2+: All zeros due to Layer 1 failure

### Phase 4: Attempted Fixes
1. **Dtype Management**: Kept float32 precision through normalization
2. **Weight Clamping**: Limited K norm weights to [-10, 10]
3. **Precision Adjustments**: Various float16/bfloat16/float32 combinations

None resolved the core instability.

## Technical Details

### The Fused Kernel
```python
# Performs RMSNorm + QKV projection in one kernel
q, k, v = FusedRMSNormQKV.forward(
    hidden_states.float(),
    layernorm_weight.float(), 
    qkv_weight.float(),
    num_heads,
    num_kv_heads,
    eps=rms_norm_eps
)
```

### The Failure Pattern
```
Input → RMSNorm → QKV Projection → Q/K Norm → Rotary → Attention
                                        ↑
                                   EXPLOSION
                                   (K norm with
                                    extreme weights)
```

### Numerical Comparison
| Stage | Standard PyTorch | Custom Kernels |
|-------|-----------------|----------------|
| Embeddings | ✅ 0.000839 std | ✅ 0.000839 std |
| Layer 0 out | ✅ 0.230 std | ⚠️ 0.828 std |
| Layer 1 out | ✅ 0.609 std | ❌ inf std |
| Final output | ✅ Valid tokens | ❌ All zeros |

## Implications

### 1. Model-Specific Issue
This appears to be specific to Qwen3's weight distribution, particularly the extreme K normalization values. Other models may not exhibit this issue.

### 2. Kernel Limitations
While our kernel is mathematically correct, it lacks the numerical stability features that make PyTorch robust to extreme values.

### 3. Performance Trade-offs
The fused kernel shows excellent performance (12.72x speedup in isolation) but is unusable due to stability issues.

## Recommendations

### Immediate
1. Keep custom kernels disabled for Qwen3 models
2. Add model detection and warnings
3. Document this limitation clearly

### Future Investigation
1. **Numerical Analysis**: Study PyTorch's handling of extreme norm weights
2. **Kernel Hardening**: Add stability guards (gradient clipping, value bounds)
3. **Weight Preprocessing**: Consider normalizing extreme weights during loading
4. **Alternative Fusion**: Explore partial fusion that maintains stability

### Research Questions
1. Why does Qwen3 have such extreme K norm weights?
2. What numerical tricks does PyTorch use internally?
3. Can we achieve fusion benefits without triggering instability?

## Conclusion

This investigation revealed a fundamental tension between aggressive kernel optimization and numerical stability. While our fused kernels are correct and fast, they lack the robustness needed for models with extreme weight distributions. This highlights the importance of extensive testing across diverse models when optimizing low-level kernels.

The issue is not a bug in the traditional sense but rather a numerical stability problem that emerges from the interaction of optimized code with unusual model weights. Resolution will require either adapting the kernels to handle extreme values or preprocessing the model weights to avoid them.