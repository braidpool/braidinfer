# ROADMAP.md - nano-vllm Development Roadmap

## Completed Sprints
- [x] Remove distributed infrastructure for single-GPU focus
- [x] Implement cascade attention with FlashInfer
- [x] Consolidate wrapper management (28 → 1)
- [x] Fix KV cache layout mismatch (NHD → HND)
- [x] Fix gibberish output bug (update_sequence_lengths)
- [x] Create unit tests for critical functionality
- [x] Add FlashInfer API documentation
- [x] Identify performance bottleneck (1,572 CPU tensor operations)

## Current Sprint: Batch Size 1 Performance
- [ ] Week 1: Quick wins (torch.compile, static memory, tensor ops)
- [ ] Week 2: CUDA graphs implementation
- [ ] Target: 31 → 500+ tok/s for batch size 1

## Next Sprint: Production Hardening
- [ ] Implement model warmup functionality
- [ ] Add comprehensive benchmarking suite
- [ ] Create performance regression tests
- [ ] Document deployment best practices

## Future Sprints

### Sprint: Advanced Optimizations
- [ ] Custom CUDA kernels for hot paths
- [ ] TensorRT integration
- [ ] INT8 quantization support
- [ ] Kernel fusion opportunities

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