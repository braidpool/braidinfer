# SPRINT.md - Current Sprint: Custom CUDA Kernels

## Previous Sprint Summary
- Identified root cause: 1,572 CPU tensor operations per forward pass
- Attempted CUDA graphs and torch.compile - both incompatible with FlashInfer
- Performance remains at ~30 tok/s (target: 500+ tok/s)
- ✓ Modified benchmarks to test batch sizes 1, 2, and 4 for single-user optimization

## Current Sprint Goal
Implement custom CUDA kernels to achieve >500 tokens/s for batch size 1 by eliminating CPU overhead.

## Sprint Tasks

### 1. Architectural Review ✓
- FlashInfer is fundamentally incompatible with CUDA graphs
- Custom kernels can fuse operations and eliminate CPU overhead
- Triton recommended for rapid prototyping

### 2. Prototype Fused Attention (Days 1-2)
- [ ] Set up Triton development environment
- [ ] Implement batch-1 specialized attention kernel
- [ ] Benchmark against current implementation
- [ ] Identify integration points

### 3. Critical Kernel Implementation (Days 3-5)
- [ ] Fused RMSNorm + QKV projection
- [ ] Optimized KV cache operations
- [ ] Fused attention + output projection
- [ ] Fused MLP block

### 4. Integration (Days 6-7)
- [ ] Create kernel wrapper for nano-vllm
- [ ] Handle memory management
- [ ] Implement fallback paths
- [ ] Test with different models

### 5. Optimization & Tuning (Days 8-9)
- [ ] Profile with Nsight Compute
- [ ] Tune for specific GPU architecture
- [ ] Optimize memory access patterns
- [ ] Minimize register usage

### 6. Testing & Documentation (Days 10-11)
- [ ] Comprehensive correctness tests
- [ ] Performance benchmarks
- [ ] Integration tests
- [ ] Documentation

### 7. Sprint Review (Day 12)
- [ ] Performance analysis
- [ ] Code review
- [ ] Next steps planning

## Success Criteria
- **Minimum**: 200 tok/s (6.5x improvement)
- **Target**: 500 tok/s (16x improvement)
- **Stretch**: 800 tok/s (26x improvement)

## Key Decisions
- Start with Triton for faster development
- Focus on batch size 1 optimization
- Fuse as many operations as possible
- Maintain compatibility with existing API