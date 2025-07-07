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
- [x] Model Compatibility Detection for Fused Kernels Sprint:
    - [x] Design quantitative metrics for weight sensitivity
    - [x] Implement FusedKernelCompatibilityChecker class
    - [x] Create layer-wise weight analysis system
    - [x] Add compatibility scoring algorithm
    - [x] Integrate with model loading process
    - [x] Create CLI tool for compatibility checking
    - [x] Add warnings and fallback mechanisms
    - [x] Test with multiple model architectures
    - [x] Document compatibility criteria
- [x] Find Compatible Models for Fused Kernels Sprint:
    - [x] Implement LLaMA model support for TinyLlama-1.1B
    - [x] Implement ERNIE model support for ERNIE-4.5-0.3B
    - [x] Run compatibility analysis on both models
    - [x] Compare weight distributions with Qwen3
    - [x] Benchmark performance with/without fused kernels
    - [x] Create model comparison report
- [x] Numerical Stability Fix for Fused Kernels Sprint:
    - [x] Analyzed llama.cpp approach and found quantization is key to performance
    - [x] Identified bfloat16 conversion point as critical precision issue
    - [x] Fixed fused_rmsnorm_qkv_production.py to match PyTorch conversion
    - [x] Fixed fused_rmsnorm_qkv_mixed_precision.py (used by Qwen3AttentionFused)
    - [x] Achieved perfect numerical match with PyTorch (0.000000 difference)
    - [x] Tested with extreme K normalization weights (96.5x)
    - [x] Updated QWEN3_NUMERICAL_STABILITY_GUIDE.md with findings
- [x] Debug Qwen3 Attention Mechanism Sprint:
    - [x] Analyzed chat template and special token handling
    - [x] Traced token generation - found repetitive patterns
    - [x] Tested attention computation in isolation - works correctly
    - [x] Identified root cause: attention/KV cache integration issue
    - [x] Created ATTENTION_MECHANISM_ISSUE.md documentation
    - [x] Confirmed kernels are correct, integration is broken

## Current Status: Custom Kernels Integration Issue ⚠️
- **Kernel Accuracy**: Perfect match with PyTorch (0.000000 difference) ✅
- **Kernel Performance**: 12.72x speedup in isolation ✅
- **Generation Issue**: Produces gibberish due to attention/KV cache integration ❌
- **Root Cause**: Attention module expects InferenceContext, fails without proper setup

## Key Findings

### Attention Mechanism Integration Issue
The custom kernels work perfectly in isolation but fail during generation because:
- The fused path still uses the standard attention module
- The attention module requires proper InferenceContext with page_manager
- Without correct KV cache handling, attention produces garbage outputs
- Different prompts produce different repetitive patterns (context, email, nal)

### BFloat16 Conversion Point Critical
The kernels now match PyTorch exactly by converting to bfloat16 at the correct point in the computation pipeline.

### Qwen3-0.6B Incompatible with Fused Kernels
After extensive investigation, we discovered that Qwen3-0.6B cannot use fused kernels due to:
- Extreme K normalization weights (up to 96.5x)
- Small numerical differences (~0.0005) get amplified catastrophically
- Model produces gibberish with any deviation from PyTorch's exact numerics
- Now fixed by matching PyTorch's bfloat16 conversion behavior

### TinyLlama Works with Fused Kernels!
- TinyLlama-1.1B is fully compatible with fused kernels
- No extreme K normalization weights like Qwen3
- Produces coherent output with both standard and custom kernels
- Provides a working baseline for fused kernel optimization

## Next Sprint Options

### Option 1: Fix Attention Integration (High Priority)
- [ ] Debug InferenceContext passing in custom kernel path
- [ ] Fix KV cache handling for Qwen3AttentionFused
- [ ] Ensure proper sequence tracking
- [ ] Test with working models (TinyLlama)

### Option 2: Implement Custom Attention
- [ ] Create custom attention implementation that doesn't require InferenceContext
- [ ] Handle KV cache directly in custom code
- [ ] Bypass the problematic standard attention module
- [ ] Integrate with FlashInfer properly

### Option 3: Quantization (Most Promising for Performance)
- [ ] INT8/INT4 quantization with bitsandbytes or GPTQ
- [ ] Expected: 2-4x speedup (60-120 tok/s)
- [ ] Maintain model quality with proper calibration
- [ ] Easy integration with existing code

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