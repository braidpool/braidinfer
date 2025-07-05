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
- Qwen3 models (with RoPE, GQA)
- GPT-2 models (no RoPE, no GQA) - partial support
- HuggingFace cache location support (~/.cache/huggingface/hub/)

## Optimization Progress
1. **Completed**: 
   - Fused RMSNorm + QKV kernel (2.64x speedup, integrated)
   - Chunk attention with online softmax (2,938 tok/s capability)
   - Position-aware KV cache generation
   - Custom kernel integration framework
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