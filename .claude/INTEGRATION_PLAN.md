# Integration Plan: Fused Kernels into Nano-VLLM

## Overview

Plan to integrate the fused RMSNorm + QKV kernel into nano-vllm's model layers for production use.

## Phase 1: Model Layer Integration (2-3 days)

### 1.1 Create Kernel Module
```python
# nanovllm/kernels/__init__.py
from .fused_rmsnorm_qkv import FusedRMSNormQKV

# nanovllm/models/qwen3_optimized.py
class OptimizedQwen3DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, context):
        # Use fused kernel
        q, k, v = FusedRMSNormQKV.forward(
            hidden_states,
            self.input_layernorm.weight,
            self.self_attn.qkv_proj.weight.t(),
            self.self_attn.num_heads,
            self.self_attn.num_kv_heads
        )
        # Continue with attention...
```

### 1.2 Configuration System
- Add `use_fused_kernels` flag to Config
- Auto-detect GPU capability
- Fallback to standard implementation if needed

### 1.3 Model Loading
- Modify weight loading to prepare transposed weights
- Handle different Qwen3 variants
- Validate dimensions at load time

## Phase 2: Additional Kernel Fusion (3-4 days)

### 2.1 MLP Block Fusion
```python
@triton.jit
def fused_mlp_kernel(
    input_ptr,
    norm_weight_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    down_weight_ptr,
    output_ptr,
    # ... dimensions
):
    # Fuse: RMSNorm → Gate/Up projection → SiLU → Down projection
```

### 2.2 Attention Output Fusion
```python
@triton.jit
def fused_attn_output_kernel(
    attn_output_ptr,
    o_proj_weight_ptr,
    residual_ptr,
    output_ptr,
    # ... dimensions
):
    # Fuse: Output projection → Add residual
```

### 2.3 Full Layer Kernel (Ultimate Goal)
- Combine all operations in one kernel
- Minimize memory transfers
- Target: 1 kernel per layer

## Phase 3: Performance Optimization (2-3 days)

### 3.1 Auto-tuning
- Profile different block sizes
- Create GPU-specific configurations
- Runtime kernel selection

### 3.2 Memory Optimization
- Optimize weight layout
- Reduce memory bandwidth usage
- Explore weight compression

### 3.3 Quantization Support
- Add INT8 support to kernels
- Implement dynamic quantization
- Maintain accuracy

## Phase 4: Production Readiness (2-3 days)

### 4.1 Testing Suite
```python
# tests/test_fused_kernels.py
class TestFusedKernels(unittest.TestCase):
    def test_rmsnorm_qkv_correctness(self):
        # Test against reference
        
    def test_performance_improvement(self):
        # Verify speedup
        
    def test_different_configs(self):
        # Test all Qwen3 variants
```

### 4.2 Benchmarking
- Create comprehensive benchmark suite
- Compare against baseline
- Profile on different GPUs

### 4.3 Documentation
- User guide for enabling fused kernels
- Performance tuning guide
- Troubleshooting section

## Expected Outcomes

### Performance Targets
- **Current**: 80 tok/s
- **Phase 1**: 120 tok/s (1.5x)
- **Phase 2**: 180 tok/s (2.25x)
- **Phase 3**: 220 tok/s (2.75x)
- **Phase 4**: 250 tok/s (3.1x)

### Risk Mitigation
1. **Compatibility**: Maintain fallback paths
2. **Accuracy**: Extensive testing against reference
3. **Portability**: Test on multiple GPU types

## Implementation Priority

1. **Highest Impact First**
   - RMSNorm + QKV (done) → 1.49x
   - MLP block fusion → additional 1.3x
   - Attention output → additional 1.1x

2. **Easiest Integration First**
   - Start with opt-in flag
   - Gradual rollout
   - Monitor performance

## Code Structure

```
nanovllm/
├── kernels/
│   ├── __init__.py
│   ├── fused_rmsnorm_qkv.py
│   ├── fused_mlp.py
│   ├── fused_attention.py
│   └── utils.py
├── models/
│   ├── qwen3.py (original)
│   └── qwen3_optimized.py (with kernels)
└── tests/
    └── test_kernels.py
```

## Next Immediate Steps

1. Create `nanovllm/kernels/__init__.py`
2. Implement `qwen3_optimized.py` with single layer
3. Test end-to-end performance
4. Gradually extend to all layers
5. Add configuration options