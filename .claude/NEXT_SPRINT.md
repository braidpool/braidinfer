# Sprint Plan: Memory-Efficient Custom Chunk Attention

## Sprint Goal
To implement a memory-efficient, high-performance chunked generation path by integrating a custom attention kernel that operates directly on the paged KV cache, avoiding any duplication or concatenation of cache data. This will fix the core correctness issue with the existing cascade attention implementation and unlock the true performance potential of the chunked API.

## Core Problem & Architecture
The current implementation fails because it tries to adapt a batch-oriented cascade attention API (FlashInfer) for a single-sequence, multi-context use case. This results in incorrect attention computation. The proposed solution is to bypass the FlashInfer wrapper and use a custom Triton kernel specifically designed for this scenario.

### Architectural Vision (The "No-Copy" Approach)
1.  **Direct Paged Cache Access**: The custom kernel will not receive a concatenated copy of the chunk KV caches. Instead, it will receive a pointer to the global `PageManager.kv_cache` and a set of indices that describe which pages belong to which chunks.
2.  **Online Softmax**: The kernel will implement an "online softmax" algorithm. This allows it to compute the correct attention distribution by iterating through the chunk pages one by one, maintaining running statistics (`max_score`, `sum_exp`) without ever materializing the full attention score matrix.
3.  **Minimal Context**: The `InferenceContext` will be enhanced to carry the list of active chunks for a given request, which the attention layer will use to orchestrate the custom kernel call.
4.  **New Generation Path**: A new path within `LLMEngine` and `Qwen3AttentionFused` will be created to specifically handle this custom kernel, leaving the standard FlashInfer path for other use cases.

---

## Detailed Implementation Plan

### Phase 1: Kernel & Wrapper Refactoring (The Core Task)
**Goal**: Modify the existing custom kernel to work directly with the paged KV cache, eliminating the current memory-inefficient concatenation step.

- **File**: `nanovllm/kernels/chunk_attention_online.py`
- **Tasks**:
    1.  **Modify `chunk_decode_attention_online_vectorized_kernel` Signature**:
        -   Remove `k_cache_ptr` and `v_cache_ptr` which expect a contiguous tensor.
        -   Add `kv_cache_ptr` (pointer to the global `PageManager.kv_cache`).
        -   Add `paged_kv_indices_ptr` (pointer to a flat list of all page numbers for the composed context).
        -   Add `paged_kv_indptr_ptr` (pointer to the boundaries of each chunk within the indices list).
        -   Add `paged_kv_last_page_len_ptr` (pointer to the length of the last page for each chunk).
        -   Add `page_size: tl.constexpr`.
    2.  **Update Kernel Logic**:
        -   Inside the kernel, when loading a K/V vector for a specific `(chunk, position)`, the logic must perform a two-step address calculation:
            1.  Determine the page index and the offset within that page for the given token position.
            2.  Use these to calculate the final memory address within the global `kv_cache_ptr`.
        -   This logic will replace the simple pointer arithmetic that worked on the concatenated cache.
    3.  **Refactor `ChunkAttentionOnline.decode_attention` Wrapper**:
        -   This function will no longer allocate or populate `k_cache_concat` or `v_cache_concat`.
        -   It will accept a list of `Chunk` objects and the global `kv_cache` tensor.
        -   It will be responsible for building the required metadata tensors for the kernel:
            -   `paged_kv_indices`: Concatenate the `page_table` from each chunk.
            -   `paged_kv_indptr`: Create pointers to the start and end of each chunk's page list within `paged_kv_indices`.
            -   `paged_kv_last_page_len`: Calculate the valid length of the last page for each chunk.
        -   It will then launch the refactored Triton kernel with the new arguments.

### Phase 2: Model and Engine Integration
**Goal**: Create the necessary pathways to call the new custom kernel from the main generation flow.

- **Files**: `nanovllm/engine/inference_context.py`, `nanovllm/engine/llm_engine.py`, `nanovllm/models/qwen3.py`
- **Tasks**:
    1.  **Enhance `InferenceContext` (`inference_context.py`)**:
        -   Add a new field: `active_chunks: Optional[List[Chunk]] = None`.
    2.  **Update `LLMEngine.generate_from_chunks` (`llm_engine.py`)**:
        -   This method will be the primary entry point for the new path.
        -   It will no longer call `build_cascade_data_for_composition`.
        -   It will create an `InferenceContext` and populate `context.active_chunks` with the list of `Chunk` objects from the `composition`.
        -   It will pass the global `kv_cache` from the `PageManager` into the context or directly to the model runner.
    3.  **Modify `Qwen3AttentionFused.forward` (`qwen3.py`)**:
        -   Add a new `if` branch triggered by a config flag (e.g., `self.use_custom_chunk_kernel`).
        -   Inside this branch:
            -   Extract the `active_chunks` list and the global `kv_cache` from the `context`.
            -   Call `ChunkAttentionOnline.decode_attention`, passing the query tensor (`q`), the list of chunks, and the global KV cache.
            -   The rest of the `forward` function (output projection, etc.) remains the same.

### Phase 3: Testing and Validation
**Goal**: Rigorously verify the correctness, performance, and memory efficiency of the new implementation.

- **File**: `tests/test_custom_chunk_attention.py` (new file)
- **Tasks**:
    1.  **Correctness Test**:
        -   Create a test that compares the output of the custom kernel path with a naive PyTorch reference.
        -   The reference implementation should manually copy the KV cache data from the paged cache into a contiguous tensor and then perform a standard `torch.nn.functional.scaled_dot_product_attention`.
        -   Assert that the outputs are numerically very close (within a reasonable tolerance for floating-point differences).
    2.  **Memory Efficiency Test**:
        -   Use `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` to measure memory usage.
        -   Create a baseline test using a KV-copy approach.
        -   Create a test for the new custom kernel path.
        -   Assert that the custom kernel path allocates significantly less memory (i.e., no large temporary tensors for the concatenated cache).
    3.  **Performance Benchmark**:
        -   Create a benchmark to measure the tokens/second throughput of the new path.
        -   Compare its performance against the (now-fixed) KV-copy approach. The custom kernel should be faster as it avoids the memory copy overhead.

---

## Success Criteria
- All tests in `tests/test_custom_chunk_attention.py` pass.
- The `test_correctness_standard_vs_chunked` test in the main suite now passes when using the new custom kernel path.
- The memory usage during a `generate_from_chunks` call is verifiably flat, with no large spikes corresponding to KV cache duplication.
- The end-to-end performance (`tok/s`) for cached chunk generation is at least **5x** faster than the prefill (non-cached) speed, demonstrating true cache reuse.
- The implementation is clean and the "no-copy" architecture is clearly reflected in the code.

## Risks and Mitigations
1.  **Risk**: Complexity of Triton kernel development and debugging.
    -   **Mitigation**: Develop the kernel iteratively. Start by making it work with a single chunk before generalizing. Use `print()` statements within the Triton kernel (for debugging) and extensive unit tests in PyTorch to validate each part of the logic.
2.  **Risk**: Performance of the custom kernel might be suboptimal initially.
    -   **Mitigation**: Focus on correctness first. Once correct, profile the kernel using the Triton profiler to identify bottlenecks (e.g., memory bank conflicts, low occupancy) and tune block sizes and data access patterns accordingly.
3.  **Risk**: Incorrectly handling all edge cases (e.g., empty chunks, single-token chunks).
    -   **Mitigation**: Add a comprehensive set of test cases covering various chunk compositions, including empty context lists, multiple small chunks, and one very large chunk.
