# Understanding llama.cpp's 400+ tok/s Performance

## Executive Summary

The 400+ tok/s performance on Qwen3 models is achieved through **weight quantization**, not different numerical computation. Both llama.cpp and nano-vllm perform arithmetic operations in float16/float32 for numerical stability.

## Key Insights

### 1. Quantization is the Game Changer

llama.cpp's performance comes from storing weights in quantized formats:
- **Q4_0**: 4-bit weights → 4x memory bandwidth reduction
- **Q8_0**: 8-bit weights → 2x memory bandwidth reduction
- **F16/BF16**: Full precision for comparison

Performance comparison (Qwen3-4B on RTX 5090):
- Q4_K_M: 243.54 tok/s
- Q8_0: 190.60 tok/s  
- BF16: 138.59 tok/s

### 2. Computation Precision

**Both llama.cpp and nano-vllm do the same thing**:
1. Load weights (quantized in llama.cpp, bfloat16 in nano-vllm)
2. Convert to float32 for accumulation
3. Perform arithmetic in float16/float32
4. Store results

The actual computation flow in llama.cpp:
```
Quantized weight (4-bit) → Dequantize to float → Compute in float → Result
```

### 3. Why Qwen3 Works in llama.cpp

The extreme K normalization weights (96.5x) are handled correctly because:
1. RMSNorm weights are **never quantized** - kept in full F16/F32 precision
2. Arithmetic is done in float precision, same as nano-vllm
3. The numerical computation is essentially identical

### 4. Memory Bandwidth Analysis

For a 0.6B parameter model:
- **BF16**: 1.2GB of weights → ~1.2GB memory transfer per token
- **Q4_0**: 0.3GB of weights → ~0.3GB memory transfer per token
- **4x reduction** in memory bandwidth = major speedup

### 5. Additional llama.cpp Optimizations

Beyond quantization:
- **cuBLAS**: Highly optimized NVIDIA libraries
- **Tensor Cores**: Hardware acceleration for FP16 ops
- **CUDA Graphs**: Reduced kernel launch overhead (35% improvement)
- **Kernel Fusion**: RMSNorm + QKV + RoPE in single kernel

## Implications for nano-vllm

### The Numerical Stability Issue is Separate

The gibberish output from fused kernels is NOT because llama.cpp does something different numerically. Both implementations:
- Use float32 for RMSNorm accumulation
- Apply weights after normalization
- Use similar precision strategies

### The Real Issue

The numerical differences causing gibberish must come from:
1. **Operation ordering** within the tiled computation
2. **Subtle differences** in how operations are fused
3. **Compiler optimizations** that change floating-point behavior

### Path Forward

1. **Fix the numerical issue first** - The kernels need to match PyTorch exactly
2. **Then add quantization** - This is where the real performance gains are
3. **Optimize memory patterns** - Following llama.cpp's approach

## Conclusion

llama.cpp achieves 400+ tok/s through:
- **Quantization** (4x memory bandwidth reduction)
- **Not** through different numerical computation
- **Not** through lower precision arithmetic

The numerical stability issue in nano-vllm's fused kernels is a separate problem that needs to be solved independently of performance optimization.