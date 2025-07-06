# Qwen3 Architecture Verification Results

## Key Finding: RMSNorm is Correct

The suggestion in SPRINT.md that K normalization should be simple element-wise multiplication is **incorrect**. Our investigation shows:

### 1. HuggingFace Implementation
```python
# From transformers Qwen3Attention forward:
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
```

Both `q_norm` and `k_norm` are `Qwen3RMSNorm` modules with shape `(128,)` matching the head dimension.

### 2. Qwen3RMSNorm Implementation
```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

This is a standard RMSNorm operation, not simple scaling.

### 3. The Real Issue: Extreme Weight Values

The K normalization weights have extreme values:
- Layer 0: max=96.5, std=9.13
- Layer 1: max=44.5, std=3.98
- Layer 2: max=44.0, std=3.94

These cause ~40x amplification of normalized values:
- Input: std ~0.2
- After RMSNorm: std ~1.0
- After weight multiplication: std ~5-10, max values ~100

## The Mystery: Why Does Standard Path Work?

### Standard Path (PyTorch)
- Handles extreme K values gracefully
- Produces coherent output
- No numerical explosion

### Fused Path (Custom Kernels)
- Same mathematical operations
- Layer 1 explodes to ~1e29
- All subsequent outputs become zero

## Hypothesis: The Difference is Subtle

The issue is likely NOT architectural but rather:

1. **Precision handling**: Different float32/bfloat16 conversion points
2. **Operation ordering**: Subtle differences in when conversions happen
3. **Attention mechanism**: How the attention computation handles large K values
4. **Numerical guards**: PyTorch may have implicit stability features

## Next Steps

1. Compare the exact precision and operation order between paths
2. Check if the attention mechanism has different numerical properties
3. Look for implicit clamping or stability features in PyTorch ops
4. Test if keeping everything in float32 fixes the issue

## Conclusion

The architecture is correctly implemented. The issue is a numerical stability problem with how our fused kernels interact with Qwen3's extreme normalization weights. The solution is NOT to change the architecture but to find and fix the numerical instability.