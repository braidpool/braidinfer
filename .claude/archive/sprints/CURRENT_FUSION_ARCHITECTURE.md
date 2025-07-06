# Current Fusion Architecture Analysis

## Overview
The current implementation attempts to fuse RMSNorm + QKV projection + Q/K normalization into a single kernel. This analysis documents the current fusion boundaries and identifies the refactoring needed.

## Current Fusion Boundaries

### Qwen3AttentionFused Forward Pass
```
Input: hidden_states (unnormalized)
       ↓
┌─────────────────────────────────┐
│  FusedRMSNormQKVMinimalF32      │ ← Currently fuses too much
│  - RMSNorm computation          │
│  - QKV projection               │
│  - Returns Q, K, V tensors      │
└─────────────────────────────────┘
       ↓
Add QKV bias (if present)
       ↓
Q/K Normalization (separate RMSNorm)
       ↓
Rotary Position Embeddings
       ↓
Attention (FlashInfer or custom)
       ↓
Output projection
```

### Problems with Current Approach
1. **Numerical Instability**: Fusing RMSNorm with QKV amplifies precision errors
2. **Extreme K Norm Weights**: Qwen3-0.6B has K norm weights up to 96.5x
3. **Error Amplification**: Small errors in fused kernel get amplified through layers

### llama.cpp's Approach (Working)
```
Input: hidden_states (unnormalized)
       ↓
┌─────────────────────────────────┐
│  Standalone RMSNorm             │ ← Computed separately
│  - Full float32 precision       │
│  - Returns normalized states    │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  Fused QKV + RoPE               │ ← Only fuses these
│  - QKV projection               │
│  - Rotary embeddings            │
│  - No normalization here        │
└─────────────────────────────────┘
       ↓
Q/K Normalization
       ↓
Attention
       ↓
Output projection
```

## Key Differences
1. **RMSNorm Separation**: llama.cpp computes RMSNorm separately with full precision
2. **Limited Fusion**: Only QKV projection and RoPE are fused
3. **Precision Control**: Each operation can use appropriate precision

## Refactoring Plan

### Phase 1: Separate RMSNorm
- Create standalone RMSNorm kernel with float32 precision
- Remove RMSNorm from fused kernel
- Update forward pass to call RMSNorm first

### Phase 2: Create QKV+RoPE Kernel
- Implement kernel that takes normalized input
- Fuse only QKV projection and RoPE
- Keep Q/K normalization separate

### Phase 3: Update Integration
- Modify Qwen3AttentionFused to use new kernels
- Ensure proper data flow
- Maintain compatibility with existing code

## Performance Considerations
- Separated operations may have slightly higher memory bandwidth usage
- But improved numerical stability is worth the trade-off
- Can optimize memory access patterns in individual kernels