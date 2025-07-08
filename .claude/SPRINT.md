# Current Sprint: KV Cache Management System Repair

## Sprint Goal
Repair the ChunkedLLM implementation to create an actual KV cache management system that reuses pre-computed KV caches from chunks, rather than just concatenating text and regenerating everything.

## Tasks

### Phase 1: Infrastructure
- [ ] Extend Chunk class to store KV cache metadata (page_indices, kv_length, position_offset)
- [ ] Create ChunkKVCacheManager for dedicated chunk page allocation
- [ ] Add chunk-specific page pool to PageManager

### Phase 2: Core Implementation  
- [ ] Implement _prefill_chunk method to actually populate KV cache
- [ ] Create ChunkSequence class for chunk prefill operations
- [ ] Add prefill_chunk method to ModelRunner

### Phase 3: Cascade Integration
- [ ] Modify generate_from_chunks to check for and use pre-filled KV caches
- [ ] Implement _build_cascade_data to create FlashInfer cascade structures
- [ ] Add generate_with_cascade method to LLM class

### Phase 4: Position Management
- [ ] Implement position offset tracking for chunks
- [ ] Ensure correct RoPE embeddings for composed sequences
- [ ] Add position validation tests

### Phase 5: Testing & Validation
- [ ] Create tests to verify KV cache is reused, not regenerated
- [ ] Add memory usage tests to confirm retention
- [ ] Benchmark performance improvements

## Key Issues Found

1. **_prefill_chunk is completely unimplemented** - Just contains TODO comment
2. **generate_from_chunks only builds text prompts** - No KV cache reuse
3. **Chunks don't store KV cache information** - Only text and metadata
4. **No cascade attention usage** - Despite infrastructure being available

## Success Metrics

- Chunks store and reuse actual KV cache data
- Memory usage reflects retained KV caches
- Performance improvement for repeated chunk usage
- Tests pass verifying KV cache reuse

## Architecture Notes

The implementation needs to:
1. Use FlashInfer's MultiLevelCascadeAttentionWrapper for composed inference
2. Allocate dedicated pages for chunks that persist across generations
3. Track position offsets for correct attention computation
4. Map chunk types to cascade levels (system→0, context→1, query→2)