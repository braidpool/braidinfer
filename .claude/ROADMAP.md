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
- [x] Fused Kernel Optimization Sprint (Successful kernel, failed integration):
    - [x] Re-implemented fused RMSNorm+QKV with proper tiling
    - [x] Achieved 12.72x speedup for isolated kernel
    - [x] Discovered numerical instability with Qwen3 model
    - [x] Root cause: Extreme K norm weights (up to 96.5) cause explosion
- [x] Fix Chat.py Gibberish Sprint (Incomplete - Issue Identified):
    - [x] Applied embedding scaling (1/sqrt(hidden_size))
    - [x] Verified RoPE theta configuration (1,000,000)
    - [x] Identified Layer 1 numerical explosion (values reach 1e29)
    - [x] Custom kernels disabled by default due to instability
- [x] FlashInfer Integration Debug Sprint (Root Cause Found):
    - [x] Created forensic tensor comparison
    - [x] Investigated tensor metadata (dtype, shape, stride)
    - [x] Tested multiple fixes (contiguous, normalization)
    - [x] Found: Layer 1 explodes inside FlashInfer with fused path
    - [x] Issue remains unresolved - very subtle integration bug
- [x] Separate RMSNorm from QKV Fusion Sprint (Complete):
    - [x] Architectural review - identified llama.cpp approach
    - [x] Create standalone RMSNorm kernel (2.19x faster than PyTorch)
    - [x] Create QKV+RoPE fused kernel
    - [x] Refactor Qwen3AttentionFused to use separated kernels
    - [x] Update decoder layer implementation
    - [x] Integration testing complete
    - [x] Verify numerical stability fix
    - [x] Complete kernel integration
    - [x] Documentation and cleanup
    - [x] Sprint review - Found existing implementation already stable

## Current Status: Performance & Stability ✅
- **Actual Performance**: ~29 tok/s (batch size 1)
- **Numerical Stability**: SOLVED - FusedRMSNormQKVMinimalF32 handles extreme weights correctly
- **Isolated Kernel Performance**: 12.72x faster than PyTorch
- **Batch Size 8**: ~237 tok/s (using FlashInfer)
- **Performance Gap**: 29 vs 400+ tok/s compared to llama.cpp
- **Root Cause**: System-level optimizations needed, not kernel issues

## Key Finding: Numerical Stability Already Solved

The investigation revealed that the existing `FusedRMSNormQKVMinimalF32` kernel already implements the correct float32 precision handling needed for Qwen3's extreme K normalization weights (up to 96.5x). The kernel:
- Uses float32 accumulators for variance computation
- Keeps matrix data in bfloat16 for bandwidth efficiency
- Follows llama.cpp's precision strategy
- Successfully handles extreme weights without instability

The performance gap vs llama.cpp (29 vs 400+ tok/s) is NOT due to numerical issues but rather system-level optimizations.

## Current Sprint: Demos and Examples Update

### Sprint: Update Demos to Showcase ChunkedLLM API
- [ ] Audit cli.py - fix system prompt not being seen by LLM
- [ ] Update chat.py to use ChunkedLLM with context reuse
- [ ] Add <think> tag filtering to chat.py
- [ ] Implement conversation chunk construction in chat.py
- [ ] Audit all examples in examples/ directory
- [ ] Ensure all demos properly showcase the project's capabilities

## Next Sprint Options (After Demos Update)

### Option 1: Quantization (Most Promising)
- [ ] INT8/INT4 quantization with bitsandbytes or GPTQ
- [ ] Expected: 2-4x speedup (60-120 tok/s)
- [ ] Maintain model quality with proper calibration
- [ ] Easy integration with existing code

### Option 2: Complete Kernel Fusion
- [ ] Fix numerical stability first
- [ ] Implement MLP fusion (if stability allows)
- [ ] Fuse attention output projection
- [ ] Expected: 2x speedup if Amdahl's Law permits

### Option 3: System Optimizations
- [ ] Memory pooling to reduce allocation overhead
- [ ] Better KV cache management
- [ ] Optimize for single-user continuous generation
- [ ] Expected: 20-30% improvement

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