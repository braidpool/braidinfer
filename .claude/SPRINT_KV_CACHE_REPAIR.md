# Sprint: KV Cache Management System Repair

## Problem Statement

The current ChunkedLLM implementation only concatenates text from chunks and regenerates the entire KV cache each time, rather than reusing pre-computed KV caches. This defeats the entire purpose of the Output KV Cache Retention sprint.

### Core Issues Identified:

1. **`_prefill_chunk` is a stub** - The method contains only TODO comments and doesn't actually populate KV cache
2. **`generate_from_chunks` builds text prompts** - Instead of using pre-filled KV caches, it concatenates chunk texts
3. **No actual KV cache storage for chunks** - Chunks store text but not their KV cache data
4. **No integration with cascade attention** - Despite having cascade attention infrastructure, chunks don't use it

## Architecture Analysis

### What We Have:
1. **FlashInfer cascade attention** - Fully functional multi-level cascade API
2. **PageManager** - Manages KV cache pages and allocation
3. **FlashInferScheduler** - Can prepare cascade data for shared prefixes
4. **Chunk system** - Stores and manages text chunks with metadata

### What's Missing:
1. **Chunk KV cache storage** - Chunks need to store page indices for their KV cache
2. **Prefill infrastructure** - Need to populate KV cache for chunks separately
3. **Cascade data preparation** - Need to build cascade structures from chunks
4. **Position tracking** - Need to track positions for correct RoPE embeddings

## Implementation Plan

### Phase 1: Chunk KV Cache Storage
1. Extend Chunk class to store KV cache information:
   - `page_indices`: List of allocated pages
   - `kv_length`: Number of tokens in KV cache
   - `position_offset`: Starting position for RoPE
   - `cascade_level`: Level in cascade hierarchy

2. Create ChunkKVCacheManager:
   - Allocate pages for chunks
   - Track chunk-to-page mappings
   - Handle chunk eviction and page recycling

### Phase 2: Implement _prefill_chunk
1. Tokenize chunk content if not already done
2. Allocate pages from PageManager
3. Create a special sequence for the chunk
4. Run model forward pass to populate KV cache
5. Store page indices in chunk metadata
6. Mark chunk as kv_cache_allocated

### Phase 3: Cascade Attention Integration
1. Modify `generate_from_chunks` to:
   - Check if chunks have KV cache allocated
   - Build cascade data structure for prefilled chunks
   - Only generate for the query portion
   - Use MultiLevelCascadeAttentionWrapper

2. Create cascade data builder:
   - Map chunks to cascade levels (system→0, context→1, query→2)
   - Build page indices arrays for each level
   - Handle position offsets correctly

### Phase 4: Position-Aware Generation
1. Track cumulative positions across chunks
2. Apply correct RoPE embeddings for each chunk's position
3. Ensure continuous positioning for composed sequences

## Technical Details

### _prefill_chunk Implementation:
```python
def _prefill_chunk(self, chunk: Chunk, position_offset: int) -> None:
    # 1. Get/create token IDs
    if not chunk.token_ids:
        chunk.token_ids = self.tokenizer.encode(chunk.content, add_special_tokens=False)
    
    # 2. Allocate pages for this chunk
    num_tokens = len(chunk.token_ids)
    num_pages = (num_tokens + self.page_size - 1) // self.page_size
    
    # 3. Get pages from a dedicated chunk page pool
    chunk_pages = self.chunk_page_manager.allocate_pages(num_pages)
    
    # 4. Create a pseudo-sequence for prefill
    seq = ChunkSequence(
        chunk_id=chunk.chunk_id,
        token_ids=chunk.token_ids,
        pages=chunk_pages,
        position_offset=position_offset
    )
    
    # 5. Run prefill through model
    # This populates the KV cache at the allocated pages
    self.llm.model_runner.prefill_chunk(seq)
    
    # 6. Store KV cache info in chunk
    chunk.page_indices = chunk_pages
    chunk.kv_length = num_tokens
    chunk.position_offset = position_offset
    chunk.kv_cache_allocated = True
```

### generate_from_chunks with KV Reuse:
```python
def generate_from_chunks(self, ...):
    # 1. Check if chunks have KV cache
    system_chunk = self.registry.get(system_chunk_id)
    if not system_chunk.kv_cache_allocated:
        self._prefill_chunk(system_chunk, 0)
    
    # 2. Build cascade structure
    cascade_data = self._build_cascade_data([
        (system_chunk, 0),  # Level 0
        (context_chunks, 1),  # Level 1
        (query_chunk, 2)  # Level 2 (generate here)
    ])
    
    # 3. Generate only for query portion
    # The cascade attention will merge states from all levels
    output = self.llm.generate_with_cascade(
        query_tokens,
        cascade_data,
        sampling_params
    )
```

## Success Criteria

1. Chunks actually store and reuse KV cache (not just text)
2. `_prefill_chunk` allocates pages and populates KV cache
3. `generate_from_chunks` uses cascade attention with pre-filled chunks
4. Memory usage shows actual KV cache retention
5. Performance improves for repeated chunk usage
6. Tests verify KV cache is reused, not regenerated

## Risks and Mitigations

1. **Risk**: Page allocation conflicts
   - **Mitigation**: Separate page pools for chunks vs active sequences

2. **Risk**: Position embedding misalignment
   - **Mitigation**: Careful position tracking and testing

3. **Risk**: Memory fragmentation
   - **Mitigation**: Implement chunk eviction policy

## Timeline

1. **Day 1**: Chunk KV cache storage infrastructure
2. **Day 2**: Implement _prefill_chunk with page allocation
3. **Day 3**: Cascade attention integration
4. **Day 4**: Position-aware generation
5. **Day 5**: Testing and validation
6. **Day 6**: Performance benchmarking
7. **Day 7**: Documentation and cleanup