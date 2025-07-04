# ROADMAP.md - nano-vllm Development Roadmap

## Completed Sprints
- [x] Remove distributed infrastructure for single-GPU focus
- [x] Implement cascade attention with FlashInfer
- [x] Consolidate wrapper management (28 → 1)
- [x] Fix KV cache layout mismatch (NHD → HND)
- [x] Fix gibberish output bug (update_sequence_lengths)
- [x] Create unit tests for critical functionality
- [x] Add FlashInfer API documentation

## Current Sprint
- [ ] Fix performance regression (23 tok/s → 200+ tok/s)
- [ ] Add back model warmup
- [ ] Sprint review and cleanup

## Next Sprint: Performance Recovery
- [ ] Deep performance profiling with CUDA events
- [ ] Identify specific bottlenecks (CPU vs GPU)
- [ ] Compare with vanilla FlashInfer performance
- [ ] Optimize critical path

## Future Sprints

### Sprint: Performance Optimization
- [ ] Profile and identify performance bottlenecks
- [ ] Optimize memory allocation patterns
- [ ] Implement efficient batch processing
- [ ] Add performance benchmarks

### Sprint: Testing Infrastructure
- [ ] Create comprehensive unit test suite
- [ ] Add integration tests for FlashInfer
- [ ] Add performance regression tests
- [ ] Set up CI/CD pipeline

### Sprint: Error Handling & Monitoring
- [ ] Restore error handling that was removed
- [ ] Add metrics collection back
- [ ] Implement proper logging
- [ ] Add health checks

### Sprint: API Enhancement
- [ ] Improve ChunkedLLM API
- [ ] Add streaming support
- [ ] Implement better memory management
- [ ] Add API documentation

### Sprint: Model Support
- [ ] Test with various model architectures
- [ ] Add support for quantized models
- [ ] Optimize for specific model families
- [ ] Add model-specific optimizations