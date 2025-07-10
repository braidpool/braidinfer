# Custom CUDA Kernels vs CUDA Graphs for Batch Size 1 Optimization

## Executive Summary

Custom CUDA kernels can achieve similar or better performance than CUDA graphs by:
1. **Fusing operations** to reduce kernel launches
2. **Eliminating CPU overhead** through kernel-side logic
3. **Optimizing for specific configurations** (batch size 1)
4. **Maintaining flexibility** while maximizing performance

Expected performance: **500-800 tok/s** (compared to current 31 tok/s)

## Why Custom Kernels Work

### The Problem We're Solving
```
Current: 1,572 CPU tensor operations → 30ms overhead
Target:  Single fused kernel → <1ms overhead
```

### CUDA Graphs vs Custom Kernels

| Aspect | CUDA Graphs | Custom Kernels |
|--------|-------------|----------------|
| **CPU Overhead** | Eliminates after capture | Eliminates by design |
| **Flexibility** | Very limited | Can include conditions |
| **Development Effort** | Low (when it works) | High |
| **Maintenance** | Depends on framework | Full control |
| **Performance** | ~10-20x speedup | ~15-25x speedup |

## Key Optimization Opportunities

### 1. Fused Attention + RMSNorm
Instead of:
```python
# Current: Multiple kernel launches
hidden = RMSNorm(hidden)          # Kernel 1 + CPU overhead
q, k, v = QKV_proj(hidden)        # Kernel 2 + CPU overhead  
q, k = rotary_embed(q, k)         # Kernel 3 + CPU overhead
attn_out = attention(q, k, v)     # Kernel 4 + CPU overhead
hidden = attn_proj(attn_out)      # Kernel 5 + CPU overhead
```

Fused kernel:
```cuda
__global__ void fused_attention_block(
    float* hidden_states,
    float* kv_cache,
    float* output,
    // All weights and parameters
) {
    // Everything in one kernel - no CPU involvement
}
```

### 2. Eliminate Tensor Operations

Current code has:
- 676 `.to()` calls → Do type conversion in kernel
- 672 `.view()` calls → Handle shapes in kernel
- 224 `.unsqueeze()` calls → Implicit in kernel design

### 3. Specialized for Batch Size 1

```cuda
template<int HEAD_DIM = 64>
__global__ void batch1_decode_attention(
    const half* __restrict__ q,        // [1, num_heads, head_dim]
    const half* __restrict__ k_cache,  // [max_seq_len, num_heads, head_dim]
    const half* __restrict__ v_cache,  // [max_seq_len, num_heads, head_dim]
    half* __restrict__ output,         // [1, num_heads, head_dim]
    const int seq_len,
    const float scale
) {
    // Optimized for single token generation
    // No batch loops, direct indexing
}
```

## Implementation Strategies

### 1. Triton (Recommended for Quick Development)

**Pros:**
- Python-like syntax
- Auto-optimization
- Good performance (~80% of hand-tuned CUDA)
- Faster development

**Example:**
```python
import triton
import triton.language as tl

@triton.jit
def batch1_attention_kernel(
    Q, K_cache, V_cache, Out,
    seq_len, scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Fused attention for batch size 1
    # Triton handles optimization
    pass
```

### 2. CUTLASS (For Maximum Performance)

**Pros:**
- Nvidia's optimized templates
- Best possible performance
- Good for production

**Example:**
```cpp
using GemmKernel = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80  // For A100
>;
```

### 3. Raw CUDA (For Full Control)

**Pros:**
- Complete control
- Can optimize for specific GPU
- No dependencies

**Example for Fused Attention Block:**
```cuda
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template<int HIDDEN_DIM, int NUM_HEADS, int HEAD_DIM>
__global__ void fused_transformer_block_batch1(
    const half* __restrict__ input,
    const half* __restrict__ ln_weight,
    const half* __restrict__ ln_bias,
    const half* __restrict__ qkv_weight,
    const half* __restrict__ qkv_bias,
    const half* __restrict__ out_proj_weight,
    const half* __restrict__ out_proj_bias,
    half* __restrict__ kv_cache,
    half* __restrict__ output,
    const int seq_position,
    const float inv_sqrt_head_dim
) {
    // Thread block handles one attention head
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Shared memory for reductions
    __shared__ half smem_q[HEAD_DIM];
    __shared__ half smem_k[HEAD_DIM];
    __shared__ half smem_v[HEAD_DIM];
    __shared__ float smem_scores[1024]; // Max seq len
    
    // Step 1: LayerNorm (fused)
    // All threads collaborate on normalization
    
    // Step 2: QKV projection (fused)
    // Compute Q, K, V for this head
    
    // Step 3: Update KV cache
    // Direct write to cache position
    
    // Step 4: Scaled dot-product attention
    // Optimized for single query token
    
    // Step 5: Output projection
    // Direct write to output
}
```

## Performance Analysis

### Current Bottlenecks (31 tok/s)
```
Forward pass: 32ms total
├── CPU operations: 28.7ms (90%)
├── GPU compute: 1.4ms (4%)  
└── Memory transfers: 1.9ms (6%)
```

### With Custom Kernels (estimated 600-800 tok/s)
```
Forward pass: 1.5-2ms total
├── CPU operations: 0.1ms (5%)
├── GPU compute: 1.3ms (75%)
└── Memory transfers: 0.3ms (20%)
```

## Specific Optimizations for Nano-VLLM

### 1. Fuse Entire Decoder Layer
```cuda
__global__ void qwen_decoder_layer_batch1(
    // Input/output
    half* hidden_states,
    
    // KV cache
    half* k_cache,
    half* v_cache,
    int cache_position,
    
    // All layer weights
    const half* ln1_weight,
    const half* qkv_weight,
    const half* o_proj_weight,
    const half* ln2_weight,
    const half* mlp_weights,
    
    // Config
    const int num_heads,
    const int head_dim
);
```

### 2. Optimize KV Cache Access
```cuda
// Instead of FlashInfer's complex paging
struct SimpleBatch1KVCache {
    half* data;  // [max_seq_len, 2, num_heads, head_dim]
    int current_len;
    
    __device__ void append(int head_idx, const half* k, const half* v) {
        // Direct write, no paging complexity
        int offset = current_len * 2 * num_heads * head_dim;
        // Copy K and V for this head
    }
};
```

### 3. Specialized Attention Pattern
```cuda
// For batch size 1, causal mask is just [0:seq_len]
__device__ float compute_attention_score(
    const half* q,      // [head_dim]
    const half* k_row,  // [head_dim]
    const int position,
    const float scale
) {
    float score = 0.0f;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i += 2) {
        half2 q_val = *reinterpret_cast<const half2*>(&q[i]);
        half2 k_val = *reinterpret_cast<const half2*>(&k_row[i]);
        score += __half2float(q_val.x) * __half2float(k_val.x);
        score += __half2float(q_val.y) * __half2float(k_val.y);
    }
    return score * scale;
}
```

## Implementation Roadmap

### Phase 1: Prototype with Triton (1-2 days)
1. Implement fused attention for batch size 1
2. Benchmark against current implementation
3. Identify remaining bottlenecks

### Phase 2: Critical Kernels (3-5 days)
1. Fused RMSNorm + QKV projection
2. Optimized KV cache append
3. Fused attention + output projection
4. Fused MLP block

### Phase 3: Full Model Optimization (1 week)
1. Fuse entire decoder layers
2. Optimize memory layout
3. Tune for specific GPU (A100/H100)
4. Integration with Braidinfer

## Advantages Over CUDA Graphs

1. **No Framework Limitations**
   - Works with any dynamic behavior
   - No FlashInfer compatibility issues
   - Can include conditional logic

2. **Better Optimization Potential**
   - Fuse more operations
   - Eliminate intermediate memory
   - Optimize for exact use case

3. **Easier to Debug**
   - Traditional profiling tools work
   - Can add printf debugging
   - Predictable behavior

## Example: Triton Fused Attention

```python
@triton.jit
def batch1_causal_attention(
    Q, K_cache, V_cache, Out,
    seq_len_ptr, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single query token attending to all previous tokens
    head_idx = tl.program_id(0)
    seq_len = tl.load(seq_len_ptr)
    
    # Load query vector
    q_offs = head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_offs)
    
    # Compute attention scores
    scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, seq_len, BLOCK_SIZE):
        # Load K block
        k_block = tl.load(K_cache + ...)
        
        # Compute dot products
        scores_block = tl.sum(q[None, :] * k_block, axis=1)
        scores += scores_block * scale
    
    # Softmax (numerically stable)
    scores = tl.softmax(scores, axis=0)
    
    # Apply to values
    output = tl.zeros([HEAD_DIM], dtype=tl.float16)
    
    for block_start in range(0, seq_len, BLOCK_SIZE):
        v_block = tl.load(V_cache + ...)
        output += tl.sum(scores[:, None] * v_block, axis=0)
    
    # Store output
    tl.store(Out + head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM), output)
```

## Performance Expectations

Based on similar optimizations in other projects:

| Implementation | Performance | Development Time |
|----------------|-------------|------------------|
| Current (FlashInfer) | 31 tok/s | - |
| + Custom Attention | 150-200 tok/s | 2-3 days |
| + Fused QKV/Norm | 300-400 tok/s | 3-5 days |
| + Full Layer Fusion | 500-800 tok/s | 1-2 weeks |
| Theoretical Max | ~1200 tok/s | - |

## Recommendations

1. **Start with Triton** for rapid prototyping
2. **Focus on attention first** - biggest bottleneck
3. **Measure each optimization** - some may not help
4. **Consider memory bandwidth** - may become the limit
5. **Profile extensively** - use Nsight Compute

## Conclusion

Custom CUDA kernels are not just an alternative to CUDA graphs - they can be **superior** for your use case:

- **Better performance** through fusion and specialization
- **More flexible** than CUDA graphs
- **Solves the root problem** (CPU overhead) directly
- **No framework compatibility issues**

For batch size 1 optimization targeting 500+ tok/s, custom kernels are likely the best path forward.