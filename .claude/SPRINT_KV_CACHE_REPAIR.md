# Sprint: KV Cache Management System Repair

## Problem Statement

The current ChunkedLLM implementation only concatenates text from chunks and regenerates the entire KV cache each time, rather than reusing pre-computed KV caches. This defeats the entire purpose of the Output KV Cache Retention sprint.

### Core Issues Identified:

1. **`_prefill_chunk` is a stub** - The method contains only TODO comments and doesn't actually populate KV cache
2. **`generate_from_chunks` builds text prompts** - Instead of using pre-filled KV caches, it concatenates chunk texts and re-tokenizes
3. **No actual KV cache storage for chunks** - Chunks store text but not their KV cache page references
4. **No integration with cascade attention** - Despite having cascade attention infrastructure, chunks don't use it
5. **Inadequate scheduler** - FlashInferScheduler designed for simple shared prefix, not arbitrary chunk composition
6. **Missing memory management** - No link between chunk deletion and KV cache page deallocation

## Architecture Analysis

### What We Have:
1. **FlashInfer cascade attention** - Fully functional multi-level cascade API
2. **PageManager** - Manages KV cache pages and allocation (needs modification)
3. **FlashInferScheduler** - Can prepare cascade data for shared prefixes (needs overhaul)
4. **Chunk system** - Stores and manages text chunks with metadata

### What's Missing:
1. **Chunk KV cache storage** - Chunks need to store page indices for their KV cache
2. **Prefill infrastructure** - Need to populate KV cache for chunks separately
3. **Cascade data preparation** - Need to build cascade structures from chunks
4. **Position tracking** - Need to track positions for correct RoPE embeddings
5. **Chunk-aware page allocation** - PageManager needs chunk_id based allocation
6. **Direct chunk composition** - Bypass string concatenation entirely

## Implementation Plan (Updated with Gemini Suggestions)

### Phase 1: Core Engine & Memory Management Enhancements

#### PageManager Modifications (`nanovllm/engine/page_manager.py`)
- [ ] Change `seq_page_tables` to `chunk_page_tables: Dict[str, List[int]]` (chunk_id → page indices)
- [ ] Add `allocate_for_chunk(chunk_id: str, num_tokens: int) -> List[int]`
- [ ] Add `free_for_chunk(chunk_id: str)` to deallocate chunk pages
- [ ] Remove dependency on Sequence objects for chunk allocation

#### Chunk Storage (`nanovllm/chunks.py`)
- [ ] Add `page_table: Optional[List[int]] = None` field to Chunk dataclass
- [ ] Add `kv_length: int = 0` for number of tokens in KV cache
- [ ] Add `position_offset: int = 0` for RoPE positioning

#### Registry Integration (`nanovllm/chunk_registry.py`)
- [ ] Add `page_manager` reference to ChunkRegistry
- [ ] Modify `delete`, `clear`, and `_evict_lru` to call `page_manager.free_for_chunk(chunk_id)`

### Phase 2: Implement Chunk Prefill Logic

#### ModelRunner Enhancement (`nanovllm/engine/model_runner.py`)
- [ ] Create `prefill_chunk(self, chunk: Chunk)` method:
  - Take Chunk object with token_ids and page_table
  - Construct prefill inputs (input_ids, positions, kv_indices)
  - Run model forward pass with minimal InferenceContext
  - Populate KV cache without sampling

#### ChunkedLLM Implementation (`nanovllm/chunked_llm.py`)
- [ ] **Implement `_prefill_chunk(self, chunk: Chunk)`**:
  - Call `page_manager.allocate_for_chunk` to get page_table
  - Assign page_table to chunk
  - Call `model_runner.prefill_chunk(chunk)`
  - Set `chunk.kv_cache_allocated = True`
- [ ] Modify `register_chunk` to immediately prefill new chunks

### Phase 3: Scheduler and Generation Flow Refactoring

#### Scheduler Overhaul (`nanovllm/engine/flashinfer_scheduler.py`)
- [ ] Rename `schedule` to `build_cascade_data_for_composition`
- [ ] Accept composition of Chunk objects (system, context, query)
- [ ] Retrieve page_table from each Chunk
- [ ] Build multi-level cascade_data:
  - System chunk → Level 0
  - Context chunks → Level 1  
  - Query chunk → Level 2
  - Create proper FlashInfer arrays

#### Engine Updates (`nanovllm/engine/llm_engine.py`)
- [ ] Create `generate_from_chunks(composition: Dict, sampling_params)` method
- [ ] Replace add_request/step loop for chunked generation
- [ ] Call scheduler to build cascade_data
- [ ] Run decode loop with cascade attention

### Phase 4: High-Level API Changes

#### ChunkedLLM Refactoring (`nanovllm/chunked_llm.py`)
- [ ] **Refactor `generate_from_chunks`**:
  - Remove ALL string building and prompt concatenation
  - Retrieve Chunk objects from registry
  - Ensure all chunks are prefilled
  - Create composition dictionary
  - Call engine's new generate_from_chunks
- [ ] Update `batch_generate_from_chunks` similarly

### Phase 5: Testing and Validation

#### Comprehensive Tests (`tests/test_chunked_kv_caching.py`)
- [ ] **Correctness Test**: Compare standard vs chunked generation outputs
- [ ] **Performance Test**: Verify >10x speedup for cached chunks
- [ ] **Memory Test**: Verify chunk deletion frees GPU memory
- [ ] **Position Test**: Verify correct RoPE embeddings
- [ ] **Cascade Test**: Verify multi-level attention works correctly

## Technical Details (Refined Implementation)

### _prefill_chunk Implementation:
```python
def _prefill_chunk(self, chunk: Chunk) -> None:
    # 1. Get/create token IDs
    if not chunk.token_ids:
        chunk.token_ids = self.tokenizer.encode(chunk.content, add_special_tokens=False)
    
    # 2. Allocate pages from PageManager
    page_table = self.llm.model_runner.page_manager.allocate_for_chunk(
        chunk.chunk_id, 
        len(chunk.token_ids)
    )
    
    # 3. Store page table in chunk
    chunk.page_table = page_table
    chunk.kv_length = len(chunk.token_ids)
    
    # 4. Run prefill through model
    self.llm.model_runner.prefill_chunk(chunk)
    
    # 5. Mark as allocated
    chunk.kv_cache_allocated = True
```

### ModelRunner.prefill_chunk Implementation:
```python
def prefill_chunk(self, chunk: Chunk) -> None:
    """Prefill KV cache for a chunk without sampling."""
    # Convert tokens to tensors
    input_ids = torch.tensor(chunk.token_ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(len(chunk.token_ids), dtype=torch.int64, device="cuda")
    
    # Build KV indices from chunk's page table
    kv_indices = torch.tensor(chunk.page_table, dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, len(chunk.page_table)], dtype=torch.int32, device="cuda")
    last_page_len = ((len(chunk.token_ids) - 1) % self.page_manager.page_size) + 1
    last_page_lens = torch.tensor([last_page_len], dtype=torch.int32, device="cuda")
    
    # Create minimal context for prefill
    context = InferenceContext(
        is_prefill=True,
        sequences=[],  # No sequence needed
        page_manager=self.page_manager,
        wrapper=self.prefill_wrapper,
        chunk_prefill=True  # Special flag
    )
    
    # Run model forward pass (no sampling)
    with torch.no_grad():
        hidden_states = self.model(input_ids, positions, context)
        # KV cache is now populated in the allocated pages
```

### generate_from_chunks with Direct KV Reuse:
```python
def generate_from_chunks(self, system_chunk_id, query_chunk_id, 
                        context_chunk_ids=None, sampling_params=None, stream=False):
    # 1. Retrieve and prefill chunks if needed
    system_chunk = self.registry.get(system_chunk_id)
    if not system_chunk.kv_cache_allocated:
        self._prefill_chunk(system_chunk)
    
    query_chunk = self.registry.get(query_chunk_id)
    if not query_chunk.kv_cache_allocated:
        self._prefill_chunk(query_chunk)
    
    context_chunks = []
    if context_chunk_ids:
        for ctx_id in context_chunk_ids:
            ctx_chunk = self.registry.get(ctx_id)
            if not ctx_chunk.kv_cache_allocated:
                self._prefill_chunk(ctx_chunk)
            context_chunks.append(ctx_chunk)
    
    # 2. Create composition
    composition = {
        'system_chunk': system_chunk,
        'context_chunks': context_chunks,
        'query_chunk': query_chunk
    }
    
    # 3. Generate using engine's new method (NO STRING BUILDING)
    return self.llm.generate_from_chunks(composition, sampling_params, stream)
```

## Success Criteria

1. The `_prefill_chunk` function is fully implemented and tested
2. The `generate_from_chunks` method no longer builds string prompts and directly uses pre-filled chunk KV caches
3. Performance tests show >10x speedup for subsequent calls reusing cached chunks
4. Memory tests confirm that chunk deletion properly frees GPU memory
5. Correctness tests verify identical outputs between standard and chunked generation
6. The chunked API delivers on its promise of true KV cache reuse

## Risks and Mitigations

1. **Risk**: Page allocation conflicts
   - **Mitigation**: Separate page pools for chunks vs active sequences

2. **Risk**: Position embedding misalignment
   - **Mitigation**: Careful position tracking and testing

3. **Risk**: Memory fragmentation
   - **Mitigation**: Implement chunk eviction policy

## Key Architecture Changes (from Gemini Sprint)

### New Flow:
1. **Registration & Prefill**: When a chunk is registered, its KV cache is immediately pre-computed
2. **Composition**: `generate_from_chunks` assembles a Composition object with chunk IDs
3. **Scheduling**: FlashInferScheduler looks up pre-allocated page tables for each chunk
4. **Cascade Data**: Scheduler constructs precise cascade_data for MultiLevelCascadeAttentionWrapper
5. **Inference**: ModelRunner executes attention over pre-filled KV caches without concatenating prompts

### Critical Points:
- Chunks become first-class citizens in the inference engine
- Complete bypass of string concatenation and re-tokenization
- Direct composition of pre-computed KV caches using FlashInfer cascade attention
- Chunk deletion must free associated GPU memory

## Timeline

1. **Day 1**: Core engine & memory management (PageManager, Chunk modifications)
2. **Day 2**: Implement chunk prefill logic (ModelRunner.prefill_chunk, _prefill_chunk)
3. **Day 3**: Scheduler overhaul and cascade data construction
4. **Day 4**: High-level API refactoring (remove all string building)
5. **Day 5**: Comprehensive testing suite
6. **Day 6**: Performance benchmarking and optimization
7. **Day 7**: Documentation and final validation