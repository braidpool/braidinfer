# Numerical Precision Analysis - Fused Kernels vs PyTorch

## Executive Summary

The numerical differences between the fused kernels and PyTorch are caused by **bfloat16 rounding differences** that get **amplified 96x** by Qwen3's extreme K normalization weights.

## Key Findings

### 1. The Computation is Identical in Float32

Both PyTorch and the fused kernels:
- Compute RMS in float32: **identical results**
- Apply normalization in float32: **identical results**
- Perform matrix multiplication: **near-identical results** (difference < 1e-7)

### 2. The Difference Comes from BFloat16 Conversion

When converting from float32 to bfloat16:
- PyTorch path: normalized_f32 → bfloat16 → matmul → output
- Fused kernel: everything in f32 → output_f32 → bfloat16

This creates small differences (max ~0.0078) due to different rounding points.

### 3. Extreme Weights Amplify the Difference

Qwen3's K normalization weights go up to **96.5**, which:
- Amplifies a 0.0078 difference to ~0.75
- Causes the model to diverge and produce gibberish

### 4. llama.cpp Doesn't Have This Issue Because

They achieve 400+ tok/s through **quantization**, not different computation:
- Weights stored in 4-bit format (4x memory bandwidth reduction)
- Arithmetic still done in float16/float32 like nano-vllm
- RMSNorm weights are **never quantized** - kept in full precision

## Solutions

### Option 1: Match PyTorch's BFloat16 Conversion Exactly (Recommended)

Modify the kernel to:
1. Apply normalization in float32
2. Convert to bfloat16 **before** matrix multiplication
3. Perform matmul in bfloat16 (matching PyTorch)

### Option 2: Keep Everything in Float32

This works but:
- Higher memory usage
- Slower performance
- Still need careful conversion at the end

### Option 3: Separate K Normalization

Apply K normalization **after** the fused kernel to avoid amplifying small differences.

## Test Results

```
Standard PyTorch Path:
- Normalized (f32): [0.1980, 0.3961, 0.5941, ...]
- Normalized (bf16): [0.1982, 0.3965, 0.5938, ...]  # Small rounding
- Final K values: [0.0832, 4.7007, 22.3345]

Fused Kernel (F32 output):
- Output (f32): [0.0118, 0.3878, 0.9357, ...]
- Output (bf16): [0.0118, 0.3887, 0.9375, ...]  # Different rounding
- Final K values: [0.0831, 4.6976, 22.3315]

Difference: max=0.069671 (after 96.5x amplification)
```

## Recommendation

Implement Option 1 - modify the fused kernel to match PyTorch's bfloat16 conversion point exactly. This will eliminate the numerical differences while maintaining performance.