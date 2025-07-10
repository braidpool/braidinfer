# Architecture of Braidinfer

This document provides a detailed overview of the internal architecture of `Braidinfer`, a high-performance, single-GPU LLM inference engine. The core design focuses on maximizing throughput and minimizing latency by optimizing memory access patterns and reducing kernel launch overhead, drawing inspiration from systems like vLLM and FlashInfer.

## 1. Core Philosophy: Chunk-Based Context Management

The central architectural concept in `Braidinfer` is the **Chunk**. Instead of representing the prompt as a single, monolithic string of text, the context is broken down into logical, reusable pieces.

- **Chunk Abstraction**: A `Chunk` is a semantic unit of text (e.g., a system prompt, a document, a user query) with its own pre-computed and cached Key-Value (KV) state.
- **Dynamic Composition**: Inference requests are composed by referencing a set of chunks. The engine assembles these pieces on the fly at the attention level, avoiding costly string operations and re-computation.
- **API**: The `ChunkedLLM` class provides the primary interface for this system, allowing users to register, manage, and combine chunks for generation.

This chunk-based approach is the foundation for the system's efficiency, particularly in multi-turn conversation and RAG (Retrieval-Augmented Generation) scenarios.

## 2. Paged KV Cache and Reuse

To enable efficient chunk reuse, `Braidinfer` employs a paged KV cache, managed by the `PageManager`.

- **Paged Memory**: The entire KV cache for all layers is pre-allocated as a single contiguous tensor, divided into fixed-size "pages". The layout is HND (Heads, Num Pages, Dims) for compatibility with high-performance kernels.
- **Persistent Chunk Caching**: When a `Chunk` is registered, the `PageManager` allocates a set of pages to store its KV cache. This computation (prefill) happens only once. The pages remain allocated as long as the chunk is in use.
- **Content-Addressable Storage**: The `ChunkRegistry` manages all active chunks. It uses a SHA256 hash of a chunk's content as its unique ID, providing automatic deduplication. If two requests register a chunk with the exact same content, the underlying KV cache is shared.
- **LRU Eviction**: To manage memory, the `ChunkRegistry` uses a Least Recently Used (LRU) policy. When the cache is full, the chunk that has not been accessed for the longest time is evicted, and its pages are freed.

This architecture means that expensive computations for shared context (like a system prompt or a large document) are performed only once and reused across countless requests, dramatically improving throughput.

## 3. Fused Kernels for Performance

A key optimization strategy in `Braidinfer` is the use of **fused kernels**, which combine multiple sequential operations into a single CUDA kernel launch. This reduces memory bandwidth usage and kernel launch overhead.

The most important fused kernel is `fused_rmsnorm_qkv_mixed_precision`, which performs:
1.  **RMSNorm**: Normalizes the input hidden state.
2.  **QKV Projection**: Multiplies the normalized state by the query, key, and value weight matrices.

### Numerical Stability

Fusing these operations presented a significant numerical stability challenge, especially with models like Qwen3 that have extremely large K-normalization weights (up to 96.5x). A tiny precision error in the RMSNorm calculation would be amplified, leading to incoherent output.

The final, stable implementation (`fused_rmsnorm_qkv_mixed_precision.py`) solves this by precisely matching PyTorch's mixed-precision behavior:
1.  The variance for RMSNorm is accumulated in `float32` for high precision.
2.  The normalized hidden state is converted to `bfloat16`.
3.  This `bfloat16` tensor is then multiplied by the QKV weights.

This specific order of operations ensures that the custom kernel produces bit-for-bit identical results to the standard PyTorch implementation, guaranteeing correctness while retaining the performance benefits of fusion.

## 4. Advanced Attention: Combining Chunks with Cascade Attention and Differential RoPE

`Braidinfer`'s core innovation lies in how it combines chunked KV caches at the attention level. It uses a custom CUDA kernel, `paged_chunk_attention_kernel`, that implements two key algorithms: **Cascade Attention** (inspired by FlashInfer) for global normalization and **Differential RoPE** for correct positional awareness.

### 4.1. Cascade Attention and Online Softmax

When a query is performed against a set of chunks (e.g., a system prompt, a context document, and a user query), the attention mechanism must behave as if it were attending to a single, concatenated sequence. A naive implementation would be to physically copy the KV caches of the chunks into a contiguous block of memory, but this is inefficient.

Instead, `Braidinfer` uses a **Cascade Attention** approach. The kernel iterates through the pages of each chunk sequentially but computes a single, globally normalized attention output using an **online softmax** algorithm.

The standard softmax function is:
`softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in all_tokens)`

The online softmax algorithm computes this iteratively. It maintains three running statistics in the GPU's fast registers:
- `m_i`: The maximum attention score seen so far.
- `l_i`: The sum of the exponentials of the scores, normalized by `m_i`.
- `acc_i`: The accumulated sum of value vectors, weighted by their attention scores.

The update rule for each new key `k_j` and value `v_j` is:
1.  Compute score: `s_j = dot(query, k_j)`
2.  Find the new maximum score: `m_new = max(m_i, s_j)`
3.  Renormalize the existing accumulator: `acc_new = acc_i * exp(m_i - m_new)`
4.  Renormalize the existing sum: `l_new = l_i * exp(m_i - m_new)`
5.  Add the new value's contribution: `acc_new += exp(s_j - m_new) * v_j`
6.  Update the sum: `l_new += exp(s_j - m_new)`

This process is repeated for every token across all chunks. The final attention output is `acc / l`. This ensures that even though the chunks are processed sequentially, the final softmax is mathematically identical to one computed over the entire concatenated sequence.

### 4.2. Differential RoPE for Positional Correctness

The second challenge is ensuring correct positional information. Rotary Position Embeddings (RoPE) are sensitive to a token's absolute position in the sequence.

-   **The Problem**: A chunk's KV cache is computed with RoPE applied for its *local* positions. For example, a 100-token chunk is cached with positions 0 through 99. If this chunk is used as context starting at global position 500, its keys must be rotated as if they were at positions 500 through 599.

-   **The Solution**: `Braidinfer` uses **Differential RoPE**. RoPE rotation is a linear transformation that can be represented by a rotation matrix `R_m` for a position `m`. A key property of these matrices is that `R_{a+b} = R_a * R_b`.

Let:
-   `k_raw` be the key vector before any RoPE is applied.
-   `m_local` be the position used when the chunk was cached.
-   `m_global` be the required global position for the current inference.
-   `k_cached = R_{m_local} * k_raw` be the vector stored in the KV cache.

We want to compute `k_global = R_{m_global} * k_raw`. We can rewrite this as:
`k_global = R_{m_global - m_local} * R_{m_local} * k_raw`
`k_global = R_{m_global - m_local} * k_cached`

This means we can obtain the correctly rotated key by applying a *differential rotation* for the position difference (`delta = m_global - m_local`) to the vector we already have in the cache.

The `paged_chunk_attention_kernel` performs this for every key it loads:
1.  It calculates `delta = chunk.global_position_start + token_local_position - chunk.cached_position_start`.
2.  It fetches the `cos(delta * theta)` and `sin(delta * theta)` values from the RoPE cache.
3.  It applies the rotation to the key vector `k_cached = (k_real, k_imag)`:
    `k_global_real = k_real * cos(delta*theta) - k_imag * sin(delta*theta)`
    `k_global_imag = k_real * sin(delta*theta) + k_imag * cos(delta*theta)`

This on-the-fly rotation is the final piece that makes chunk reuse mathematically equivalent to processing a single, concatenated sequence.

### 4.3. Equivalence of Combined vs. Concatenated KV Caches

The combination of Cascade Attention and Differential RoPE guarantees that the KV cache created by combining two chunks on the fly is equivalent to the KV cache that would have been computed if the chunks' tokens were first concatenated and then processed.

**Example:**
-   `Chunk A`: Tokens `T_A`, length `L_A`. Cached with positions `0` to `L_A - 1`.
-   `Chunk B`: Tokens `T_B`, length `L_B`. Cached with positions `0` to `L_B - 1`.

When we combine them as `[Chunk A, Chunk B]`, the desired KV cache `KV_{A+B}` would have:
-   The `T_A` part computed with RoPE positions `0` to `L_A - 1`.
-   The `T_B` part computed with RoPE positions `L_A` to `L_A + L_B - 1`.

The `Braidinfer` kernel achieves this:
-   For keys from `Chunk A`, the `global_position` equals the `local_position`, so `delta = 0`. The rotation is an identity operation, and the cached keys are used as-is. This matches the first part of `KV_{A+B}`.
-   For keys from `Chunk B`, the `global_position` is `L_A + local_position`. The `delta` is `L_A`. The kernel applies the `R_{L_A}` rotation to the cached keys from `Chunk B`, effectively shifting them to their correct global positions. This matches the second part of `KV_{A+B}`.

The online softmax then ensures that attention scores are normalized correctly across this virtually constructed `KV_{A+B}` cache.

## 5. The Full Picture: A `generate_from_chunks` Call

Putting it all together, here is the lifecycle of a generation call:

1.  **API Call**: The user calls `llm.generate_from_chunks([...])`, providing IDs for a system prompt, context chunks, and a query.
2.  **Chunk Retrieval**: The `ChunkedLLM` retrieves the `Chunk` objects from the `ChunkRegistry`. If any chunk's KV cache is not populated, it is pre-filled now (a one-time cost).
3.  **Position Calculation**: The engine calculates the global position for each token in the final logical sequence. For example: System (0-50), Context (51-250), Query (251-260).
4.  **Decode Step**: For each new token to be generated, the model runner executes a decode step.
5.  **Attention Kernel Launch**: The `Qwen3AttentionFused` layer is called. It receives the single query vector for the new token and the list of active `Chunk` objects.
6.  **Paged Chunk Attention**: The `paged_chunk_attention_kernel` is launched.
    - It iterates through every token in every page of every active chunk.
    - For each key, it loads the vector from the paged KV cache.
    - It applies the differential RoPE rotation to adjust the key to its global position.
    - It applies K-normalization.
    - It computes an attention score against the query and updates the online softmax accumulator.
7.  **Output Generation**: Once the kernel finishes, it has produced a single, contextually correct output vector, which is then used to sample the next token.
