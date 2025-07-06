# Refactoring Implementation Plan

## Task 1: Architectural Review ✓

### Current Fusion Analysis
- **Problem**: Current kernel fuses RMSNorm + QKV projection, causing numerical instability
- **Root Cause**: Precision loss in RMSNorm gets amplified by QKV projection and extreme K norm weights
- **Solution**: Follow llama.cpp approach - separate RMSNorm from QKV fusion

### Key Findings
1. Current FusedRMSNormQKVMinimalF32 tries to do too much in one kernel
2. RMSNorm needs full float32 precision throughout
3. QKV projection can use mixed precision with float32 accumulators
4. RoPE can be fused with QKV projection safely

## Task 2: Create Standalone RMSNorm Kernel

### Design Requirements
- Input: hidden_states [seq_len, hidden_dim] in bfloat16/float16
- Output: normalized_states [seq_len, hidden_dim] in float32
- Full float32 computation throughout
- Support in-place operation to save memory

### Implementation Details
```python
@triton.jit
def rmsnorm_kernel_f32(
    input_ptr,
    output_ptr,
    norm_weight_ptr,
    seq_len,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute variance in float32
    # Normalize in float32
    # Store output in float32
```

## Task 3: Create QKV+RoPE Fused Kernel

### Design Requirements
- Input: normalized_states [seq_len, hidden_dim] in float32
- Output: Q, K, V tensors with RoPE applied
- Fuse QKV projection with RoPE computation
- Use mixed precision (float16/bfloat16 weights, float32 accumulators)

### Implementation Details
```python
@triton.jit
def qkv_rope_kernel(
    normalized_input_ptr,  # Already normalized in float32
    qkv_weight_ptr,       # In bfloat16/float16
    qkv_bias_ptr,         # Optional
    cos_sin_cache_ptr,    # RoPE cache
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    positions_ptr,
    # dimensions...
):
    # QKV projection with float32 accumulator
    # Apply RoPE to Q and K
    # No RoPE for V
```

## Task 4: Refactor Qwen3AttentionFused

### Changes Required
1. Remove layernorm_weight parameter from forward()
2. Add explicit RMSNorm call before QKV+RoPE kernel
3. Update kernel calls to use new separated kernels
4. Handle dtype conversions properly

### New Forward Flow
```python
def forward(self, positions, hidden_states, context=None):
    # Step 1: RMSNorm (separate kernel)
    normalized = rmsnorm_f32(hidden_states, self.input_layernorm.weight)
    
    # Step 2: QKV+RoPE (fused kernel)
    q, k, v = qkv_rope_fused(normalized, self.qkv_proj.weight, positions)
    
    # Step 3: Q/K normalization
    q = self.q_norm(q)
    k = self.k_norm(k)
    
    # Step 4: Attention
    attn_output = self.attn(q, k, v, context)
    
    # Step 5: Output projection
    return self.o_proj(attn_output)
```

## Task 5: Update Qwen3DecoderLayer

### Changes Required
1. Apply RMSNorm before calling attention
2. Remove layernorm_weight passing
3. Ensure residual connections work correctly

## Performance Targets
- RMSNorm kernel: < 5% overhead vs fused
- QKV+RoPE kernel: Similar performance to current
- Overall: Within 10% of current fused kernel
- Correctness: Qwen3-0.6B produces coherent output

## Risk Mitigation
1. Keep old kernels available for fallback
2. Add extensive unit tests for each kernel
3. Compare outputs with PyTorch reference
4. Profile performance at each step