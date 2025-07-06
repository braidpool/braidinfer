# MLP Fusion Findings

## Summary

Implemented and tested MLP fusion but found that the fused Triton kernel is **17-29x slower** than separate operations.

## Performance Results

```
Standard MLP: 0.077 ms (using cuBLAS GEMMs)
Fused MLP (optimized): 21.472 ms
Fused MLP (simple): 35.321 ms
PyTorch reference: 1.196 ms
```

## Root Cause Analysis

1. **Redundant Computation**: The Triton kernels compute the full GEMM for each output element separately
2. **Memory Access Pattern**: Poor cache utilization due to accessing full weight matrices repeatedly
3. **No cuBLAS**: Missing out on highly optimized GEMM kernels that leverage Tensor Cores

## Why Fusion Failed

The MLP operation is fundamentally three GEMMs:
- gate_up_proj: [B, H] × [2I, H]ᵀ → [B, 2I]
- SiLU activation: element-wise
- down_proj: [B, I] × [H, I]ᵀ → [B, H]

Where B=1, H=896, I=4864

The GEMMs are already highly optimized by cuBLAS and use Tensor Cores. The small batch size (B=1) means:
- Limited parallelism for fusion
- Kernel launch overhead is small relative to compute
- Memory bandwidth is not the bottleneck

## Lessons Learned

1. **Not all operations benefit from fusion** - especially well-optimized GEMMs
2. **Profile first** - the bottleneck analysis showed kernel launches, but MLP GEMMs are fast
3. **Consider hardware capabilities** - Tensor Cores make unfused GEMMs very efficient

## Alternative Approaches

1. **CUDA Graphs**: Capture the MLP operations into a graph to reduce launch overhead
2. **Persistent Kernels**: Keep intermediate results in shared memory
3. **Focus on other bottlenecks**: The attention mechanism has more potential for optimization

## Recommendation

**Do not use the fused MLP kernel**. The performance regression is too severe. Instead:
1. Keep the standard implementation with separate operations
2. Consider CUDA graphs for reducing kernel launch overhead
3. Focus optimization efforts on attention mechanisms