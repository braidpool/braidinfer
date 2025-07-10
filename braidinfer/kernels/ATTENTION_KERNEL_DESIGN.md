# Fused Qwen3 Attention Kernel Design

## Overview
This document outlines the design for a fully fused attention kernel that handles the complete attention pipeline from RMSNorm to output projection, with special handling for extreme K normalization weights.

## Key Challenges

### 1. Extreme K Normalization Weights
- Layer 0: K norm weight = 96.5x
- Layers 1-7: K norm weights > 20x
- This causes attention scores to explode if not handled carefully

### 2. GQA (Grouped Query Attention)
- Dynamic ratio (e.g., 16:8 = 2:1 for Qwen3-0.6B)
- Each KV head serves multiple Q heads
- Must handle arbitrary ratios from config

### 3. Numerical Stability
- Need stable softmax for large attention scores
- Maintain precision while using float16 storage
- Avoid overflow/underflow in exp() operations

## Implementation Strategy

### Phase 1: RMSNorm + QKV Projection
```python
# Step 1: RMSNorm in float32
variance = sum(x^2) / hidden_dim
rms = sqrt(variance + eps)
normed = x / rms * norm_weight

# Step 2: QKV projection
qkv = normed @ qkv_weight.T
```

### Phase 2: Split and Reshape
```python
# Split QKV
q = qkv[:, :num_heads * head_dim]
k = qkv[:, num_heads * head_dim : num_heads * head_dim + num_kv_heads * head_dim]
v = qkv[:, -num_kv_heads * head_dim:]

# Reshape to head format
q = q.view(seq_len, num_heads, head_dim)
k = k.view(seq_len, num_kv_heads, head_dim)
v = v.view(seq_len, num_kv_heads, head_dim)
```

### Phase 3: Q/K Normalization with Special Handling
```python
# Q normalization (standard)
q_normed = layer_norm(q, q_norm_weight)

# K normalization with extreme weight handling
if max(k_norm_weight) > 20:
    # Store unnormalized K for cache
    k_cache = k  # This stays in reasonable float16 range
    
    # Apply normalization carefully
    k_normed = layer_norm(k, k_norm_weight)
    
    # Consider scaling to prevent overflow
    scale_factor = min(1.0, 100.0 / max(k_norm_weight))
    k_normed = k_normed * scale_factor
else:
    k_cache = k
    k_normed = layer_norm(k, k_norm_weight)
```

### Phase 4: RoPE Application
```python
# Apply rotary position embeddings
q_rope = apply_rope(q_normed, positions, cos, sin)
k_rope = apply_rope(k_normed, positions, cos, sin)
```

### Phase 5: GQA Attention Computation
```python
# For each query head, find corresponding KV head
for q_head_idx in range(num_heads):
    kv_head_idx = q_head_idx // (num_heads // num_kv_heads)
    
    # Get Q for this head
    q_head = q_rope[q_head_idx]  # [seq_len, head_dim]
    
    # Get corresponding K and V
    k_head = k_rope[kv_head_idx]  # [seq_len, head_dim]
    v_head = v[kv_head_idx]       # [seq_len, head_dim]
    
    # Compute attention scores with scaling
    # Option 1: Pre-scale Q
    q_scaled = q_head * (1.0 / sqrt(head_dim))
    scores = q_scaled @ k_head.T
    
    # Option 2: Post-scale (may overflow with extreme K)
    # scores = (q_head @ k_head.T) * scale
    
    # Stable softmax
    scores_max = max(scores, dim=-1)
    scores_stable = scores - scores_max
    exp_scores = exp(scores_stable)
    attn_weights = exp_scores / sum(exp_scores, dim=-1)
    
    # Apply attention
    output_head = attn_weights @ v_head
```

### Phase 6: Output Projection
```python
# Concatenate heads
output = concat([output_head for output_head in outputs])

# Project back to hidden_dim
final_output = output @ o_proj_weight.T
```

## Memory Access Patterns

### Tiling Strategy
- Tile over sequence length for cache efficiency
- Keep head dimension in registers
- Use shared memory for QKV tiles

### KV Cache Integration
- Prefill: Compute and store all KV pairs
- Decode: Load cached KV, append new, compute attention
- Store K BEFORE normalization for numerical stability

## Numerical Stability Techniques

### 1. Stable Softmax
```python
# Instead of: softmax(scores)
# Use: softmax(scores - max(scores))
max_score = tl.max(scores, axis=-1)
stable_scores = scores - max_score
exp_scores = tl.exp(stable_scores)
sum_exp = tl.sum(exp_scores, axis=-1)
attn_weights = exp_scores / sum_exp
```

### 2. Adaptive Scaling
```python
# For extreme K norm weights
if k_norm_max > threshold:
    # Scale down before attention
    k_scale = safe_scale_factor(k_norm_max)
    k_normed = k_normed * k_scale
    
    # Adjust attention computation accordingly
    scores = (q @ k_normed.T) * (1.0 / k_scale)
```

### 3. Mixed Precision
- Use float32 accumulators for reductions
- Store in float16 for memory efficiency
- Critical operations (norm, softmax) in float32

## Configuration Handling

All parameters must be passed dynamically:
```python
def forward(
    hidden_states,
    config_params: dict,  # From config.json
    weights: dict,        # Model weights
    ...
):
    num_heads = config_params['num_attention_heads']
    num_kv_heads = config_params['num_key_value_heads']
    head_dim = config_params['head_dim']
    # ... etc
```

## Testing Strategy

1. **Unit Tests**
   - Each phase independently
   - GQA head mapping correctness
   - Numerical stability with extreme values

2. **Integration Tests**
   - Full pipeline with real model weights
   - Compare with reference implementation
   - Check for attention collapse

3. **Stress Tests**
   - Very long sequences
   - Extreme weight values
   - Edge cases (single token, etc.)

## Performance Targets

- Reduce memory traffic by 50%+ through fusion
- Achieve 2x+ speedup over separate operations
- Maintain numerical accuracy within 1e-3 of reference