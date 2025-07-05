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
- [x] Implement fused RMSNorm + QKV kernel (NOTE: kernel is 15x slower than PyTorch)
- [x] Integration and Correctness Sprint:
    - [x] Integrate fused RMSNorm+QKV kernel into Qwen3 model
    - [x] Implement position-aware KV cache generation
    - [x] Fix chunk attention with online softmax algorithm
    - [x] Create end-to-end integration tests
    - [x] Performance result: 27-29 tok/s (custom kernels actually slower)
- [x] MLP Fusion Investigation Sprint (Negative result):
    - [x] Analyzed MLP bottlenecks (40% of compute time)
    - [x] Implemented fused MLP kernels
    - [x] Discovered 17-29x performance regression
    - [x] Decision: Do not use fused MLP, cuBLAS GEMMs are superior
- [x] Foundational Kernel Fusion Sprint (o_proj + residual):
    - [x] Implemented correct tiled GEMV kernel
    - [x] Fused residual addition
    - [x] Integrated and benchmarked (2.2x slower than cuBLAS)
    - [x] Documented why fusion failed (Tensor Cores advantage)
- [x] Streaming Implementation Sprint:
    - [x] Added streaming support to LLMEngine
    - [x] Implemented yield-based token generation
    - [x] Created chat interfaces with real-time output
    - [x] Verified minimal overhead for streaming

## Current Status: Performance Reality
- **Actual Performance**: ~29 tok/s (batch size 1)
- **With Custom Kernels**: ~27 tok/s (slower!)
- **Batch Size 8**: ~237 tok/s (using FlashInfer)
- **Root Issue**: Triton kernels poorly optimized, 15x slower than PyTorch

## Next Sprint Options (Realistic)

### Option 1: Quantization (Most Promising)
- [ ] INT8/INT4 quantization with bitsandbytes or GPTQ
- [ ] Expected: 2-4x speedup (60-120 tok/s)
- [ ] Maintain model quality with proper calibration
- [ ] Easy integration with existing code

### Option 2: System Optimizations
- [ ] Memory pooling to reduce allocation overhead
- [ ] Better KV cache management
- [ ] Optimize for single-user continuous generation
- [ ] Expected: 20-30% improvement

### Option 3: Use Proven Kernels
- [ ] Integrate FlashAttention v3
- [ ] Use cuBLAS/cuDNN optimally
- [ ] Remove poorly performing custom kernels
- [ ] Expected: Return to baseline performance

## Future Sprints

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