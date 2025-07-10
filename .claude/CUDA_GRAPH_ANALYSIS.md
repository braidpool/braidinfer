# CUDA Graph Analysis for Braidinfer

## Why CUDA Graphs?

Our bottleneck analysis shows:
- **62% of time** is kernel launch overhead (2.70 ms out of 4.35 ms)
- **360 kernel launches** per token (15 per layer × 24 layers)
- **7.5 μs** overhead per kernel launch

CUDA graphs can eliminate most of this overhead by:
1. Capturing the entire forward pass as a graph
2. Replaying with a single kernel launch
3. Keeping the standard cuBLAS operations (no kernel rewrites needed!)

## Expected Performance Impact

```
Current: 230 tok/s (4.35 ms/token)
- Kernel launches: 2.70 ms (62%)
- Actual compute: 1.65 ms (38%)

With CUDA Graphs: ~600 tok/s (theoretical)
- Single graph launch: ~0.01 ms
- Actual compute: 1.65 ms
- Total: 1.66 ms/token
```

Realistic expectation: **230 → 400-500 tok/s** (accounting for overhead)

## Implementation Strategy

### Phase 1: Single Layer Graph
1. Capture one decoder layer's operations
2. Validate correctness
3. Measure speedup

### Phase 2: Full Model Graph
1. Capture all 24 layers
2. Handle dynamic shapes (if needed)
3. Optimize graph structure

### Phase 3: Advanced Features
1. Graph updates for different sequence lengths
2. Multi-stream execution
3. Memory pooling

## Key Operations to Capture

Per layer (currently 15 kernels):
1. RMSNorm + QKV projection (fused)
2. RoPE embedding
3. Attention computation
4. Output projection
5. Residual addition
6. Post-attention RMSNorm
7. Gate projection (GEMM)
8. Up projection (GEMM)
9. SiLU activation
10. Down projection (GEMM)
11. Final residual

## Challenges & Solutions

### Challenge 1: Dynamic Shapes
- **Issue**: Different sequence lengths
- **Solution**: Create graph pool for common lengths (1, 16, 32, 64, etc.)

### Challenge 2: KV Cache Updates
- **Issue**: KV cache grows dynamically
- **Solution**: Use graph capture with pre-allocated maximum size

### Challenge 3: Memory Management
- **Issue**: Graphs require fixed memory addresses
- **Solution**: Memory pooling with fixed allocations

## Implementation Plan

### Week 1: Proof of Concept
- [ ] Create simple CUDA graph wrapper
- [ ] Capture single layer forward pass
- [ ] Benchmark against standard execution
- [ ] Validate numerical correctness

### Week 2: Full Integration
- [ ] Extend to full model
- [ ] Handle edge cases
- [ ] Optimize graph structure
- [ ] Production testing

## Code Example

```python
class CudaGraphModel:
    def __init__(self, model):
        self.model = model
        self.graphs = {}  # seq_len -> graph
        
    def capture_graph(self, seq_len):
        # Warm up
        dummy_input = torch.randn(1, seq_len, 896).cuda()
        self.model(dummy_input)
        
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(dummy_input)
        
        self.graphs[seq_len] = (graph, dummy_input, output)
        
    def forward(self, input):
        seq_len = input.shape[1]
        if seq_len not in self.graphs:
            self.capture_graph(seq_len)
            
        graph, static_input, static_output = self.graphs[seq_len]
        static_input.copy_(input)
        graph.replay()
        return static_output.clone()
```

## Success Metrics

1. **Primary**: Achieve 400+ tok/s (74% improvement)
2. **Secondary**: Reduce kernel launches from 360 to <10
3. **Stretch**: Reach 500+ tok/s with advanced optimizations