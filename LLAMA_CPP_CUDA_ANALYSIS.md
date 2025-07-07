# Llama.cpp CUDA Kernel Analysis

## Key Findings

### 1. RMSNorm Implementation

Llama.cpp's RMSNorm implementation (`norm.cu`) uses **float32 precision for all intermediate calculations**, even when working with half precision inputs:

```cuda
// From rms_norm_f32 kernel
float tmp = 0.0f; // partial sum for thread in warp

for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[col];  // Input is loaded as float
    tmp += xi * xi;           // Accumulate in float32
}

// ... warp reduction ...

const float mean = tmp / ncols;
const float scale = rsqrtf(mean + eps);  // rsqrtf operates on float32

for (int col = tid; col < ncols; col += block_size) {
    dst[col] = scale * x[col];  // Final output
}
```

**Key differences from our implementation:**
- They always use float32 for accumulation and computation of the norm
- They use `rsqrtf` (single precision reciprocal square root) instead of mixed precision
- All intermediate values (mean, scale) are stored as float32

### 2. Fused RMSNorm + QKV Projection

Their fused kernel (`fusion-rmsnorm-qkv.cu`) also maintains float32 precision:

```cuda
// Step 1: Compute RMSNorm
float sum_sq = 0.0f;
for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float val = __half2float(input_ptr[i]);  // Convert to float32
    sum_sq += val * val;
}

// ... reduction ...

float rms = sqrtf(sum_sq / hidden_size + epsilon);
s_scale = 1.0f / rms;

// Apply RMSNorm and store in shared memory as float32
for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float val = __half2float(input_ptr[i]);
    float gamma_val = __half2float(gamma[i]);
    s_normalized[i] = val * scale * gamma_val;  // Stored as float32
}
```

**Critical observation:** They store normalized values in shared memory as **float32**, not half precision.

### 3. QKV Projection Handling

The projection computation is done entirely in float32:

```cuda
// Compute dot product with weight matrix row
float sum = 0.0f;
#pragma unroll 8
for (int k = 0; k < hidden_size; k++) {
    float w = __half2float(weight_ptr[weight_row * hidden_size + k]);
    sum += s_normalized[k] * w;  // s_normalized is float32
}

// Write output
*output_ptr = __float2half(sum);  // Convert back to half only at the end
```

### 4. Warp Reduction Patterns

Their warp reduction for float values is standard:

```cuda
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}
```

For half2, they use hardware intrinsics:
```cuda
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;
}
```

### 5. Precision Philosophy

Based on the code analysis, llama.cpp follows these principles:

1. **Input/Output in FP16**: Data is stored in memory as half precision
2. **Compute in FP32**: All arithmetic operations are performed in float32
3. **Explicit conversions**: They use `__half2float` and `__float2half` for conversions
4. **No mixed precision arithmetic**: They avoid operations like `__hmul` or `__hadd` in computational kernels

### 6. Memory Layout and Access Patterns

Their kernels use careful memory access patterns:
- Coalesced reads from global memory
- Shared memory for frequently accessed data
- Proper alignment and stride handling

### 7. Special Optimizations

They have architecture-specific optimizations:
- Tensor core paths for Ampere+ GPUs (placeholder in the code)
- Different block sizes based on hidden size
- Separate kernels for different model sizes

## Recommendations for Our Implementation

1. **Use float32 for all RMSNorm computations** - This appears to be critical for numerical stability
2. **Store intermediate normalized values as float32** in shared memory
3. **Perform all accumulations in float32** before converting back to half
4. **Use explicit conversion functions** rather than relying on implicit conversions
5. **Consider architecture-specific optimizations** but start with the stable float32 path

## Why This Matters

The numerical differences we're seeing likely stem from:
1. Mixed precision arithmetic introducing accumulated errors
2. Different rounding behaviors between half and float operations
3. Order of operations affecting precision (they normalize in float32, then project)

Llama.cpp's approach trades some memory bandwidth for numerical stability, which appears to be the right choice for transformer models.

## Implementation Strategy for nano-vllm

Based on this analysis, here's the recommended approach:

1. **Modify RMSNorm kernel** to use float32 for all computations:
   ```cuda
   // Convert input to float32
   float val = __half2float(input[idx]);
   
   // Accumulate in float32
   float sum_sq = 0.0f;
   sum_sq += val * val;
   
   // Compute scale in float32
   float scale = rsqrtf(sum_sq / hidden_size + eps);
   
   // Apply normalization in float32, then convert back
   output[idx] = __float2half(scale * val * __half2float(weight[idx]));
   ```

2. **Modify fused kernels** to maintain float32 precision throughout:
   - Store normalized values in shared memory as float32
   - Perform all matrix multiplications in float32
   - Only convert to half at the final output stage

3. **Epsilon handling**: Use the model's specified epsilon value (typically 1e-6 for Qwen models)

4. **Testing strategy**:
   - Compare outputs with llama.cpp using the same model
   - Verify numerical stability with long sequences
   - Check for accumulated errors over multiple layers

This approach prioritizes correctness and numerical stability over raw performance, which is essential for accurate model inference.