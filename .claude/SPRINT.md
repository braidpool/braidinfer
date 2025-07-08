# Current Sprint: True KV Cache Chunking Implementation

## Sprint Goal
Transition the chunk-based API from a prompt-building utility into a true KV cache management system by implementing pre-computation and direct reuse of chunk KV caches using FlashInfer's cascade attention.

## Tasks (Detailed Implementation Plan)

### Phase 1: Core Engine & Memory Management
- [ ] Modify PageManager to use chunk_page_tables: Dict[str, List[int]]
- [ ] Add allocate_for_chunk(chunk_id, num_tokens) method
- [ ] Add free_for_chunk(chunk_id) method
- [ ] Add page_table field to Chunk dataclass
- [ ] Link ChunkRegistry to PageManager for automatic cleanup

### Phase 2: Chunk Prefill Implementation
- [ ] Create ModelRunner.prefill_chunk(chunk) method
- [ ] Implement _prefill_chunk in ChunkedLLM
- [ ] Auto-prefill chunks on registration
- [ ] Ensure KV cache population without sampling

### Phase 3: Scheduler & Generation Overhaul
- [ ] Replace FlashInferScheduler.schedule with build_cascade_data_for_composition
- [ ] Create LLMEngine.generate_from_chunks(composition, params)
- [ ] Build proper multi-level cascade data structures
- [ ] Implement decode loop with cascade attention

### Phase 4: API Refactoring
- [ ] Remove ALL string building from generate_from_chunks
- [ ] Direct composition of pre-computed KV caches
- [ ] Update batch_generate_from_chunks similarly
- [ ] Ensure no re-tokenization occurs

### Phase 5: Comprehensive Testing
- [ ] Correctness: Verify identical outputs vs standard generation
- [ ] Performance: Verify >10x speedup for cached chunks
- [ ] Memory: Verify GPU memory freed on chunk deletion
- [ ] Integration: Full cascade attention validation

## Key Problems to Solve

1. **`_prefill_chunk` is a stub** - Core logic for pre-computing chunk KV cache is missing
2. **Inefficient generation flow** - Current method reconstructs prompts, negating caching benefits
3. **Inadequate scheduler** - FlashInferScheduler can't handle arbitrary chunk composition
4. **Missing memory management** - No link between chunk deletion and KV cache deallocation
5. **No cascade integration** - Despite having the infrastructure ready

## Success Criteria

- `_prefill_chunk` is fully implemented and tested
- `generate_from_chunks` directly uses pre-filled KV caches (no string building)
- Performance shows >10x speedup for subsequent cached chunk usage
- Memory tests confirm proper GPU memory management
- Project delivers on the chunked API promise

## Architecture Summary

The new architecture treats chunks as first-class citizens:
1. **Registration = Prefill**: KV cache computed immediately on chunk registration
2. **Composition Objects**: Replace string concatenation with chunk compositions
3. **Direct Cascade**: Use MultiLevelCascadeAttentionWrapper without re-tokenization
4. **Memory Linked**: Chunk deletion automatically frees GPU memory
5. **True Caching**: Subsequent calls reuse pre-computed KV caches entirely