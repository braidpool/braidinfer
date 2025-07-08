# BIG_PICTURE.md - nano-vllm-claude-flash-attn

## Project Overview
This is nano-vllm, a single-GPU optimized implementation of vLLM focused on high-performance local LLM inference using FlashInfer for cascade attention.

## Current State
- **Architecture**: Single-GPU focused, removed distributed infrastructure
- **Key Feature**: Cascade attention using FlashInfer for efficient long-context handling
- **Performance**: 
  - Baseline: ~87 tok/s (batch size 1)
  - With custom kernels: ~230 tok/s (2.64x speedup)
  - Chunk attention capability: 2,938 tok/s
  - Next target: 400+ tok/s (with MLP fusion)
- **Recent Accomplishments**:
  - ✅ Integrated fused RMSNorm+QKV kernel into Qwen3 model
  - ✅ Implemented position-aware KV cache generation for chunks
  - ✅ Fixed chunk attention with online softmax algorithm (mathematically correct)
  - ✅ Achieved >100 tok/s stretch goal (actual: ~230 tok/s)
  - ✅ Created comprehensive test suite for kernel validation
  - ✅ Successfully integrated cascade attention with GQA and custom kernels
- **Current Sprint**: Output KV Cache Retention
  - Implementing retention of output KV cache as reusable cascade chunks
  - Handling think tag removal/masking for clean context reuse
  - Enabling efficient multi-turn conversations and chain-of-thought

## Core Components
1. **LLMEngine**: Main engine orchestrating inference
2. **ModelRunner**: Handles model execution with FlashInfer integration
3. **PageManager**: Manages paged KV cache allocation (uses HND format)
4. **FlashInferScheduler**: Schedules cascade attention operations
5. **ChunkedLLM**: High-level API with chunk-based memory management

## Technical Decisions
- Using FlashInfer as the authoritative cascade attention implementation
- KV cache uses HND layout: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
- Single shared wrapper for all layers (confirmed correct from FlashInfer docs/vLLM)
- Chunk-based API for efficient memory reuse
- Plan-run pattern for FlashInfer operations

## Model Support
- Qwen3 models (with RoPE, GQA) - ⚠️ Incompatible with fused kernels due to extreme K weights
- GPT-2 models (no RoPE, no GQA) - partial support
- LLaMA models (TinyLlama tested) - ✅ Compatible with fused kernels
- ERNIE-4.5 models - ⚠️ Implementation issues (produces gibberish)
- HuggingFace cache location support (~/.cache/huggingface/hub/)
- Trust remote code support for custom model implementations

## Optimization Progress
1. **Completed**: 
   - Fused RMSNorm + QKV kernel (2.64x speedup, integrated)
   - Chunk attention with online softmax (2,938 tok/s capability)
   - Position-aware KV cache generation
   - Custom kernel integration framework
   - Model compatibility detection system
   - TinyLlama works with fused kernels!
2. **Next**: 
   - MLP block fusion (Gate + Up + Down projections)
   - Attention output fusion (Output projection + residual)
   - Memory layout optimization
3. **Future**: 
   - INT8/INT4 quantization
   - Full layer fusion (single kernel per layer)
   - TensorRT integration

## Known Issues
1. Model warmup was removed during refactoring
2. Some error handling and metrics code was removed
3. GPT-2 weight loading has some issues with transformer. prefix
4. ~~Qwen3 models incompatible with fused kernels due to extreme K normalization weights (96.5x)~~ **FIXED!**
5. ERNIE-4.5 implementation produces gibberish (works with vanilla transformers)
6. Chat template must be model-specific (fixed in chat.py)

## Recent Breakthroughs
- **Qwen3 Custom Kernels Fixed**: Discovered and fixed critical numerical precision issue
  - Root cause: BFloat16 conversion must happen BEFORE weight multiplication in RMSNorm
  - Small 0.0078 difference was amplified 96.5x by extreme K weights
  - Custom kernels now produce coherent output matching PyTorch exactly
- **GQA Implementation**: Properly implemented Grouped Query Attention for Qwen3
- **Comprehensive Testing**: Created 10 coherence tests including factual recall ("Aistonia" test)
- **Cascade Attention Integration**: Successfully integrated cascade attention with GQA and custom kernels
  - 53.3% memory savings for shared system prompts
  - Works seamlessly with fused kernels
  - Simple API: just set enable_cascade_attention=True