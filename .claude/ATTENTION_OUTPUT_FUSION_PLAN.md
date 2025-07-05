# Attention Output Fusion Plan

## Overview
Fuse the attention output projection with residual addition to reduce kernel launches and memory traffic.

## Current Operations
```python
# In Qwen3Attention.forward():
attn_output = self.attn(q, k, v, context)
output = self.o_proj(attn_output)  # Linear layer: 0.020 ms
return output

# In Qwen3DecoderLayer.forward():
hidden_states = self.self_attn(positions, hidden_states, context)
hidden_states = residual + hidden_states  # Residual add: 0.005 ms
```

Total: 0.025 ms per layer × 24 layers = 0.6 ms

## Proposed Fusion
```python
# Single fused operation
hidden_states = fused_attn_output_residual(
    attn_output,      # [batch_size, seq_len, num_heads * head_dim]
    residual,         # [batch_size, seq_len, hidden_size]
    o_proj_weight,    # [hidden_size, hidden_size]
    residual_weight=1.0
)
```

Expected: ~0.018 ms per layer × 24 layers = 0.432 ms
Savings: 0.168 ms total

## Implementation Details

### Triton Kernel Design
```python
@triton.jit
def fused_o_proj_residual_kernel(
    # Inputs
    attn_ptr,         # attention output
    residual_ptr,     # residual connection
    weight_ptr,       # o_proj weights
    # Output
    output_ptr,
    # Dimensions
    batch_size,
    seq_len,
    hidden_size,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output element
    idx = tl.program_id(0)
    
    # Compute o_proj (GEMV for batch size 1)
    # Then add residual in same kernel
```

### Benefits
1. **Fewer kernel launches**: 2 → 1 per layer (48 → 24 total)
2. **Better memory access**: Read residual once, write output once
3. **Reduced memory bandwidth**: Eliminate intermediate storage
4. **Simple implementation**: Straightforward Triton kernel

### Integration Points
1. Modify `Qwen3AttentionFused` to use fused kernel when enabled
2. Pass residual into attention layer
3. Update `Qwen3DecoderLayer` to handle fused output

## Expected Performance
- Current: 230 tok/s
- With fusion: ~245 tok/s (6% improvement)
- Kernel launch reduction: 48 kernels saved
- Memory bandwidth saved: ~12 MB per token

## Implementation Steps

### Week 1
1. [ ] Create `fused_attn_output.py` kernel
2. [ ] Benchmark kernel vs separate ops
3. [ ] Integrate into Qwen3AttentionFused
4. [ ] Update decoder layer logic
5. [ ] Validate correctness

### Success Criteria
- [ ] <0.001 numerical difference
- [ ] Measurable speedup (>5%)
- [ ] Clean integration
- [ ] Pass all tests

## Code Structure
```
nanovllm/kernels/
  fused_attn_output.py    # New fused kernel
  
nanovllm/models/
  qwen3.py               # Modified attention and decoder layer
```

## Risk Assessment
- **Low risk**: Simple fusion, well-understood operations
- **High confidence**: Similar to successful RMSNorm+QKV fusion
- **Easy rollback**: Flag-controlled like other optimizations