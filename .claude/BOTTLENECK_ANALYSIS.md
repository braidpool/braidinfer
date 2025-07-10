# Bottleneck Analysis - Braidinfer @ 230 tok/s

## Executive Summary

Current performance is limited by three primary factors:
1. **Kernel Launch Overhead** (62% of time) - 360 kernels per token
2. **Memory Bandwidth** - Requiring 156 GB/s, primarily for weight loading
3. **MLP Operations** - 40% of compute time, poor memory efficiency

## Detailed Analysis

### 1. Performance Metrics

#### Current State (230 tok/s)
- **Time per token**: 4.35 ms
- **Time per layer**: 0.181 ms (24 layers total)
- **Memory bandwidth**: 156 GB/s required
- **Kernel launches**: ~360 per token (15 per layer)

#### Custom Kernel Performance
| Kernel | Time (ms) | Throughput | Memory | Arithmetic Intensity |
|--------|-----------|------------|---------|---------------------|
| FusedRMSNormQKV | 0.052 | 19,165 tok/s | 1.97 MB | 1.00 |
| ChunkAttention | 0.345 | 2,895 tok/s | 0.21 MB | 6.91 |

### 2. Time Breakdown per Layer

```
Component            Time (ms)  Percentage
-----------------------------------------
fused_rmsnorm_qkv     0.052      22.5%
attention             0.050      21.5%
mlp_gate_up          0.040      17.2%
mlp_down             0.030      12.9%
o_proj               0.020       8.6%
post_norm            0.015       6.5%
qk_rope              0.010       4.3%
mlp_activation       0.010       4.3%
residuals            0.005       2.2%
-----------------------------------------
Total                0.232     100.0%
```

### 3. Memory Bandwidth Analysis

#### Memory Access Pattern (per token)
```
Component           Memory (MB)  Percentage
------------------------------------------
gate_up_weights      399.00       57.4%
down_weights         199.50       28.7%
qkv_weights           47.25        6.8%
o_proj_weights        36.75        5.3%
kv_cache_read         11.72        1.7%
activations            0.41        0.1%
norm_params            0.08        0.0%
------------------------------------------
Total                694.71      100.0%
```

#### Bandwidth Utilization on Common GPUs
| GPU | Bandwidth | Utilization @ 230 tok/s |
|-----|-----------|------------------------|
| RTX 3090 | 936 GB/s | 16.7% |
| RTX 4090 | 1008 GB/s | 15.5% |
| A100 40GB | 1555 GB/s | 10.0% |
| H100 | 3350 GB/s | 4.7% |

### 4. Critical Bottlenecks

#### 1. Kernel Launch Overhead (PRIMARY)
- **360 kernels per token** (15 per layer × 24 layers)
- **~7.5 μs overhead per kernel**
- **Total overhead: 2.70 ms (62% of total time!)**
- This is the dominant bottleneck

#### 2. MLP Operations (40% of compute)
- Currently uses 3 separate kernels:
  - Gate + Up projection: 0.040 ms
  - SiLU activation: 0.010 ms  
  - Down projection: 0.030 ms
- Total: 0.080 ms per layer (1.92 ms total)
- Poor memory efficiency due to separate passes

#### 3. Memory Bandwidth
- MLP weights account for 86% of memory traffic
- Weight loading is not overlapped with compute
- Activation memory is fragmented across kernels

### 5. Optimization Opportunities

#### Immediate: MLP Fusion (Target: 313 tok/s)
```
Current: 3 kernels, 0.080 ms per layer
Fused:   1 kernel,  ~0.032 ms per layer (60% reduction)
Savings: 1.15 ms total
Result:  230 → 313 tok/s (36% improvement)
```

#### Next: Attention Output Fusion
```
Current: o_proj + residual = 0.025 ms per layer
Fused:   Single operation = ~0.018 ms per layer
Savings: 0.18 ms total
Result:  Additional 5% improvement
```

#### Future: Full Layer Fusion
```
Current: 15 kernels per layer
Target:  3-4 kernels per layer
Potential: 400-500 tok/s
```

### 6. Key Insights

1. **Kernel launch overhead is the dominant bottleneck** - Even with fast kernels, launching 360 of them kills performance

2. **MLP fusion is the next logical step** - It addresses both kernel count and memory efficiency

3. **Memory bandwidth is not the limiting factor** - We're only using 15-17% of available bandwidth on modern GPUs

4. **Our custom kernels are very fast** - FusedRMSNormQKV can handle 19,000 tok/s standalone

### 7. Recommended Action Plan

1. **Implement MLP Fusion** (Week 1)
   - Fuse gate_proj + up_proj + SiLU + down_proj
   - Expected: 230 → 313 tok/s

2. **Attention Output Fusion** (Week 2)
   - Fuse o_proj + residual add
   - Expected: 313 → 330 tok/s

3. **Layer-level Fusion** (Future)
   - Combine multiple operations per layer
   - Target: 400+ tok/s

The path to 400+ tok/s is clear: reduce kernel launches through aggressive fusion.