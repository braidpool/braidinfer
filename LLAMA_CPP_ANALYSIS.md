# llama.cpp CUDA Kernel Analysis

## Key Findings on Precision and Performance

### 1. **Mixed Precision Strategy**

llama.cpp uses a sophisticated mixed precision approach:

- **dfloat/dfloat2**: Defined as either `half/half2` or `float/float2` based on `GGML_CUDA_F16` flag
- **Accumulation**: Always done in float32 for numerical stability
- **Storage/Computation**: Can be in float16/bfloat16 for performance

### 2. **RMSNorm Implementation**

Their RMSNorm kernel (`norm.cu`):
```cuda
// Key observations:
// 1. Accumulation in float32
float tmp = 0.0f;
for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[col];  // Input read
    tmp += xi * xi;           // Float32 accumulation
}

// 2. Normalization computation
const float mean = tmp / ncols;
const float scale = rsqrtf(mean + eps);

// 3. Output scaling
for (int col = tid; col < ncols; col += block_size) {
    dst[col] = scale * x[col];  // Float32 math, output can be half
}
```

**Key difference**: They do NOT apply the weight in RMSNorm kernel itself!

### 3. **Fused RMSNorm + QKV**

Their fused kernel (`fusion-rmsnorm-qkv.cu`) shows:
- RMSNorm computation in float32
- Weight application happens AFTER normalization
- Shared memory used for normalized values (float32)
- QKV projection done with mixed precision

### 4. **Quantization Strategy**

llama.cpp achieves high performance through aggressive quantization:

- **Q4_0**: 4-bit quantization with scale factor
- **Q8_0**: 8-bit quantization  
- **Dequantization**: On-the-fly during computation
- **DP4A instructions**: Integer dot products for quantized values

Example Q4_0 computation:
```cuda
// Dequantize 4-bit values
v.x = (v.x - 8.0f) * d;  // d is scale factor
v.y = (v.y - 8.0f) * d;

// Use integer dot product for speed
sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
```

### 5. **Performance Optimizations**

1. **Kernel Fusion**: Combine RMSNorm + QKV + RoPE in single kernel
2. **Quantized Weights**: Most weights stored in INT4/INT8
3. **Tensor Cores**: Optional support for WMMA/MMA operations
4. **Warp-level Operations**: Efficient reductions and shuffles
5. **Block Sizes**: Adaptive based on hidden size (128-1024 threads)

### 6. **Key Insight: Weight Application Order**

The critical difference appears to be:
- **llama.cpp**: `output = input * (1/rms) * weight` (weight applied last)
- **Our approach**: `output = (input * weight) * (1/rms)` (weight in normalization)

This explains numerical differences when weights have extreme values!

### 7. **Tensor Core Usage**

They have tensor core implementations but:
- Disabled by default (`#define USE_TENSOR_CORES 0`)
- Only for specific operations (attention, some GEMMs)
- Not used for normalization or element-wise ops

## Performance Breakdown

Their 400+ tok/s likely comes from:
1. **Quantization** (4-bit weights = 4-8x memory bandwidth reduction)
2. **Kernel Fusion** (fewer kernel launches, better memory locality)
3. **Optimized Memory Access** (coalesced reads, shared memory usage)
4. **Integer Operations** (DP4A for quantized dot products)
5. **Adaptive Algorithms** (different kernels for different sizes)

## Recommendations for nano-vllm

1. **Fix RMSNorm**: Apply weight after normalization, not during
2. **Add Quantization**: Implement Q4_0/Q8_0 support for memory-bound ops
3. **Improve Fusion**: Combine more operations to reduce kernel launches
4. **Use Mixed Precision**: FP32 accumulation with FP16 storage
5. **Consider Integer Ops**: For quantized paths, use DP4A instructions

The key takeaway: Their performance comes primarily from quantization and memory bandwidth optimization, not from doing everything in float16. They maintain numerical stability through careful use of float32 accumulation where it matters.