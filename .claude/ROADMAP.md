# ROADMAP.md - nano-vllm Development Roadmap

## Completed Sprints
- [x] Remove distributed infrastructure for single-GPU focus
- [x] Implement cascade attention with FlashInfer
- [x] Consolidate wrapper management (28 → 1)
- [x] Fix KV cache layout mismatch (NHD → HND)
- [x] Fix gibberish output bug (update_sequence_lengths)
- [x] Create unit tests for critical functionality
- [x] Add FlashInfer API documentation
- [x] Identify performance bottleneck (1,621 kernel launches)
- [x] Implement fused RMSNorm + QKV kernel (85.8 tok/s achieved)
- [x] Integration and Correctness Sprint (230 tok/s achieved):
    - [x] Integrate fused RMSNorm+QKV kernel into Qwen3 model
    - [x] Implement position-aware KV cache generation
    - [x] Fix chunk attention with online softmax algorithm
    - [x] Create end-to-end integration tests
    - [x] Achieve >100 tok/s stretch goal (actual: ~230 tok/s)

## Current Sprint: Additional Kernel Fusion
- [ ] MLP block fusion (Gate + Up + Down projections)
- [ ] Attention output fusion (Output projection + residual)
- [ ] Memory layout optimization
- [ ] Target: 230 → 400+ tok/s for batch size 1

## Next Sprint: Production Hardening
- [ ] Implement model warmup functionality
- [ ] Add comprehensive benchmarking suite
- [ ] Create performance regression tests
- [ ] Document deployment best practices

## Future Sprints

### Sprint: Quantization & Advanced Optimizations
- [ ] INT8/INT4 quantization kernels
- [ ] Dynamic quantization support
- [ ] TensorRT integration
- [ ] Full layer fusion (single kernel per layer)

### Sprint: Multi-Model Support
- [ ] Test with Llama, Mistral, Gemma families
- [ ] Model-specific optimizations
- [ ] Support for MoE models
- [ ] Quantization format compatibility

### Sprint: Developer Experience
- [ ] Simplified API for single-user scenarios
- [ ] Auto-tuning for hardware
- [ ] Profiling and debugging tools
- [ ] Comprehensive examples

### Sprint: Memory Optimization
- [ ] Adaptive KV cache sizing
- [ ] Memory pooling strategies
- [ ] Offloading for large models
- [ ] Dynamic batch optimization

### Sprint: Integration & Ecosystem
- [ ] LangChain integration
- [ ] REST API server
- [ ] Model repository support
- [ ] Monitoring and metrics

## Long-term Vision
Create the fastest single-GPU inference engine optimized for individual users and edge deployment, achieving near-theoretical performance limits through aggressive optimization and simplified architecture.