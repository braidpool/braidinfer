# Sprint: Separate RMSNorm from QKV Fusion

## Objective
Refactor the fused RMSNorm+QKV kernel to match llama.cpp's approach: compute RMSNorm separately and only fuse QKV+RoPE. This will resolve numerical stability issues with Qwen3-0.6B's extreme K normalization weights.

## Success Criteria
- [ ] RMSNorm computed separately with full float32 precision
- [ ] New fused QKV+RoPE kernel that takes normalized input
- [ ] Qwen3-0.6B produces coherent text output
- [ ] Performance remains competitive (within 10% of current fused kernel)

## Tasks

### 1. Architectural Review
- [ ] Analyze current Qwen3AttentionFused implementation
- [ ] Document the current fusion boundaries
- [ ] Plan the refactoring approach
- [ ] Identify all places that need changes

### 2. Create Standalone RMSNorm Kernel
- [ ] Implement optimized RMSNorm kernel with float32 precision
- [ ] Support in-place operation to minimize memory usage
- [ ] Add unit tests for accuracy
- [ ] Benchmark against PyTorch RMSNorm

### 3. Create QKV+RoPE Fused Kernel
- [ ] Implement fused_qkv_rope_kernel based on llama.cpp design
- [ ] Input: normalized hidden states
- [ ] Output: Q, K, V tensors with RoPE applied
- [ ] Remove RMSNorm computation from kernel
- [ ] Add support for both float16 and bfloat16

### 4. Refactor Qwen3AttentionFused
- [ ] Separate RMSNorm computation from forward pass
- [ ] Call standalone RMSNorm kernel
- [ ] Call new QKV+RoPE kernel with normalized input
- [ ] Ensure Q/K normalization happens after fusion
- [ ] Handle residual connections properly

### 5. Update Qwen3DecoderLayer
- [ ] Modify forward pass to accommodate separated RMSNorm
- [ ] Ensure layer normalization happens before attention
- [ ] Verify residual connections are correct
- [ ] Test with both custom and standard kernels

### 6. Integration Testing
- [ ] Test with simple inputs to verify correctness
- [ ] Compare outputs with standard PyTorch implementation
- [ ] Test with Qwen3-0.6B model specifically
- [ ] Verify coherent text generation

### 7. Performance Optimization
- [ ] Profile the new kernel configuration
- [ ] Optimize block sizes and memory access patterns
- [ ] Compare performance with original fused kernel
- [ ] Document performance characteristics

### 8. Cleanup and Documentation
- [ ] Remove old fused RMSNorm+QKV kernels
- [ ] Update kernel documentation
- [ ] Update QWEN3_NUMERICAL_STABILITY_GUIDE.md
- [ ] Create migration guide for other models

### 9. Extended Testing
- [ ] Test with different sequence lengths
- [ ] Test with batch sizes > 1
- [ ] Test with other Qwen3 model sizes
- [ ] Run stress tests for numerical stability

### 10. Sprint Review
- [ ] Document lessons learned
- [ ] Compare final performance metrics
- [ ] Verify all success criteria met
- [ ] Plan next optimizations

## Timeline Estimate
- Tasks 1-2: 2 hours
- Tasks 3-4: 3 hours  
- Tasks 5-6: 2 hours
- Tasks 7-8: 2 hours
- Tasks 9-10: 1 hour
- **Total: ~10 hours**

## Risks
- Performance regression from separating operations
- Integration complexity with existing codebase
- Potential issues with other model architectures

## Notes
- Priority is correctness over performance
- Follow llama.cpp's proven approach
- Keep each operation simple and verifiable