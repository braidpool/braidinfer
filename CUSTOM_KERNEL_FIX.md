# Custom Kernel Numerical Stability Fix

## Problem Summary

The custom kernels produce gibberish output due to numerical instability that accumulates through the model layers. While individual layer computations are nearly identical between custom and standard paths, the accumulated differences interact badly with extreme layer normalization weights in the model.

### Key Findings:

1. **Fused kernel works correctly in isolation** - produces outputs within 0.01 of standard computation
2. **Q/K normalization outputs are nearly identical** between custom and standard paths
3. **The issue accumulates through layers** - by layer 27, hidden states have completely different magnitudes
4. **Final RMSNorm has extreme weights** (max 15.3, mean 3.84) that amplify accumulated differences
5. **Different numerical ranges**: Custom kernels keep values in smaller range (0.1-0.5) while standard path has larger values (11.75)

### Root Cause:

The fused kernel computes in float32 throughout, while the standard path mixes dtypes. This causes subtle differences in how values accumulate through residual connections and layer norms. These small differences are catastrophically amplified by extreme normalization weights.

## Solution Options:

### Option 1: Match Standard Path Numerics (Recommended)
Modify the fused kernel to match the exact numerical behavior of the standard path by:
- Computing RMSNorm in the input dtype (bfloat16) instead of float32
- Only using float32 for the variance calculation
- Ensuring all intermediate values match the standard path

### Option 2: Clamp Extreme Weights
Add weight clamping to prevent extreme values:
```python
# In model loading
for layer in model.layers:
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'k_norm'):
        layer.self_attn.k_norm.weight.data.clamp_(-10, 10)
model.norm.weight.data.clamp_(-10, 10)
```

### Option 3: Use Standard Computation for Layers with Extreme Weights
Detect layers with extreme normalization weights and fall back to standard computation for those layers.

## Recommended Implementation:

The most robust solution is Option 1 - modify the kernel to match standard numerics exactly. This ensures compatibility with all models without requiring model-specific workarounds.