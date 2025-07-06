# Why llama.cpp Fusion Works

## Key Insight: They Don't Fuse RMSNorm!

The llama.cpp implementation avoids the numerical stability issues by NOT fusing RMSNorm with QKV projection.

### Their Approach:
1. **RMSNorm**: Computed separately with full precision
2. **Fused QKV+RoPE**: Takes already-normalized input
3. **Q/K Normalization**: Applied after QKV projection

### Why This Works:
- RMSNorm is computed with whatever precision their RMSNorm kernel uses (likely float32)
- The normalized values are stable before entering the fused kernel
- No precision loss from trying to do too much in one kernel

### Our Problem:
We tried to fuse:
- RMSNorm computation
- QKV projection  
- All in one kernel with mixed precision

This creates a perfect storm with extreme K normalization weights:
1. Small RMSNorm precision errors
2. Amplified by matrix multiplication
3. Further amplified by K normalization (96.5x)

### Solution:
Follow llama.cpp's approach:
1. Compute RMSNorm separately (can still be optimized)
2. Fuse only QKV+RoPE
3. Apply Q/K normalization after

This way, each operation can use appropriate precision without compromising the others.