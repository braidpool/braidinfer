# BIG_PICTURE.md - nano-vllm-claude-flash-attn

## Project Overview
This is nano-vllm, a single-GPU optimized implementation of vLLM focused on high-performance local LLM inference using FlashInfer for cascade attention.

## Current State
- **Architecture**: Single-GPU focused, removed distributed infrastructure
- **Key Feature**: Cascade attention using FlashInfer for efficient long-context handling
- **Performance Issue**: Currently ~23 tok/s instead of expected 200+ tok/s
- **Recent Fixes**: 
  - Fixed gibberish output (root cause: update_sequence_lengths not called)
  - Fixed KV cache layout mismatch (NHD vs HND)
  - Fixed floating point exception
  - Added missing logits computation

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

## Known Issues
1. Performance degradation (~23 tok/s vs expected 200+ tok/s)
2. Model warmup was removed
3. Some error handling and metrics code was removed during refactoring
4. GPT-2 weight loading has some issues with transformer. prefix