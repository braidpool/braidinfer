# Sprint: Output KV Cache Retention and Reuse

## Sprint Goal
Implement retention of output KV cache as cascade attention chunks that can be reused in subsequent generations, enabling efficient multi-turn conversations and chain-of-thought analysis.

## Background
Currently, output KV cache is immediately deallocated after generation completes. However, cascade attention already has all the necessary position handling to allow reusing output KV cache as context chunks. This would enable:
- Reusing assistant responses in multi-turn conversations
- Building on previous reasoning traces
- Analyzing or refining previous outputs
- Significant memory and compute savings

## Key Challenge
Need to handle `<think>...</think>` blocks in outputs - either:
1. Save only the KV cache after the think block ends
2. Mask out the think block in the saved chunk

## Tasks

### 1. Architectural Review
- [ ] Analyze current KV cache deallocation flow in scheduler/page_manager
- [ ] Map out changes needed to retain output KV cache
- [ ] Determine best approach for think tag handling (save partial vs mask)
- [ ] Review ChunkedLLM API for output chunk registration

### 2. Implement KV Cache Retention
- [ ] Add `retain_output_cache` flag to SamplingParams
- [ ] Modify scheduler.postprocess() to conditionally retain KV cache
- [ ] Create mechanism to extract KV cache pages after generation
- [ ] Implement position tracking for output chunks

### 3. Think Tag Handling
- [ ] Implement token position tracking during generation
- [ ] Detect `<think>` and `</think>` token positions
- [ ] Option A: Extract KV cache only from post-think positions
- [ ] Option B: Implement masking mechanism for think blocks
- [ ] Test both approaches and choose best one

### 4. Output Chunk Registration
- [ ] Add OUTPUT chunk type to ChunkType enum
- [ ] Create method to register output KV cache as chunk
- [ ] Ensure proper position offset tracking
- [ ] Handle chunk metadata (generation params, timestamps)

### 5. Manual Deallocation API
- [ ] Add method to manually deallocate output chunks
- [ ] Implement chunk expiration/eviction policies
- [ ] Add CLI commands for output chunk management
- [ ] Document memory implications

### 6. Integration with ChunkedLLM
- [ ] Extend ChunkedLLM.generate() to return output chunk ID
- [ ] Allow output chunks to be used as context chunks
- [ ] Test cascade composition with output chunks
- [ ] Verify position handling is correct

### 7. Update Chat Interfaces
- [ ] Modify chat.py to optionally retain assistant responses
- [ ] Update cli.py to show and manage output chunks
- [ ] Add commands to reuse previous outputs
- [ ] Implement conversation memory management

### 8. Enhanced CLI Demo
- [ ] Add output chunks section to cli.py's render_chunks_table()
- [ ] Create /output command to list all output chunks
- [ ] Add /use-output <chunk-id> command to add output chunk as context
- [ ] Show output chunk preview with think tags removed
- [ ] Add /delete-output <chunk-id> command for manual deallocation
- [ ] Update help text with output chunk commands
- [ ] Add visual indicators for output vs input chunks
- [ ] Show memory usage of output chunks

### 9. Testing
- [ ] Create tests for output KV cache retention
- [ ] Test multi-turn conversations with reused outputs
- [ ] Verify think tag removal/masking works correctly
- [ ] Benchmark memory savings and performance impact
- [ ] Test edge cases (max memory, many outputs)

### 10. Documentation
- [ ] Document output chunk retention API
- [ ] Create examples of multi-turn conversation optimization
- [ ] Add memory management best practices
- [ ] Update architecture docs with new flow

### 11. Sprint Review
- [ ] Verify all tests pass
- [ ] Benchmark performance improvements
- [ ] Review memory usage patterns
- [ ] Document any limitations or issues
- [ ] Create demo showcasing the feature

## Success Criteria
1. Output KV cache can be retained and reused in subsequent generations
2. Think tags are properly handled (removed or masked)
3. Memory savings demonstrated for multi-turn conversations
4. No regression in generation quality or performance
5. Clean API for managing output chunks

## Technical Notes
- Leverage existing cascade attention position handling
- Ensure compatibility with all attention mechanisms (standard, cascade, GQA)
- Consider memory pressure and automatic eviction strategies
- Think about future extensions (e.g., persistent cache across sessions)