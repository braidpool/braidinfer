# Sprint: Fix Chunked Attention Correctness

## Sprint Goal
Ensure that chunked attention produces identical results to standard attention when given the same input tokens. Fix issues with Chinese text generation and system prompt handling in chat_chunked.py.

## Core Problems
1. **Attention Mechanism Incompatibility**: The standard attention mechanism doesn't properly handle chunked KV cache
2. **Position Encoding Mismatch**: Chunks have different position encodings than expected by attention
3. **Token Structure Mismatch**: Chunks include chat template tokens but attention expects raw content
4. **Excessive Debug Output**: Too much debug logging makes it hard to see actual issues (FIXED)

## Tasks

### Phase 1: Debug Output Cleanup
- [ ] **Reduce Debug Logging**:
  - Remove or comment out excessive debug print statements
  - Add a debug flag to control verbosity levels
  - Keep only essential debug information for tracking issues

### Phase 2: Correctness Testing Framework
- [ ] **Create Comparison Test Suite**:
  - Implement side-by-side testing of standard vs chunked generation
  - Use identical prompts and compare outputs token by token
  - Test with various prompt lengths and types
  - Save intermediate states for debugging

- [ ] **Add Attention Output Comparison**:
  - Capture attention outputs from both paths
  - Compare attention weights and hidden states
  - Identify where divergence begins

### Phase 3: Fix Attention Mechanism
- [ ] **Investigate KV Cache Handling**:
  - Verify that chunked attention accesses the correct KV cache entries
  - Check position encoding consistency between standard and chunked paths
  - Ensure proper handling of chunk boundaries

- [ ] **Fix Position Handling**:
  - Verify position indices are correctly calculated for chunks
  - Ensure RoPE is applied consistently
  - Check for off-by-one errors in position calculations

- [ ] **Verify Chunk Prefilling**:
  - Ensure chunks are properly prefilled with correct positions
  - Check that chunk KV cache entries align with expected positions
  - Verify chunk metadata (positions, lengths) are correct

### Phase 4: System Prompt and Language Issues
- [ ] **Debug System Prompt Handling**:
  - Trace how system prompts are processed in chunked mode
  - Verify tokens are correctly stored in chunks
  - Check attention mask includes system prompt tokens

- [ ] **Fix Language Generation**:
  - Investigate why Chinese is being generated
  - Check tokenizer handling in chunked mode
  - Verify model weights are loaded correctly
  - Test with explicit language constraints

### Phase 5: Integration Testing
- [ ] **Create Comprehensive Tests**:
  - Test single-turn conversations
  - Test multi-turn conversations with context
  - Test with various system prompts
  - Test edge cases (empty chunks, single token chunks)

- [ ] **Performance Verification**:
  - Ensure fixes don't degrade performance
  - Measure memory usage remains efficient
  - Verify chunk reuse works correctly

## Success Criteria
1. **Identical Outputs**: Given the same input, chunked and standard attention produce identical text
2. **Correct Language**: Model generates in the expected language (English by default)
3. **System Prompt Adherence**: Model correctly follows system prompt instructions
4. **Clean Output**: Debug output is minimal and controlled by flags
5. **All Tests Pass**: Comprehensive test suite validates correctness

## Implementation Notes
- Start with minimal test cases to isolate issues
- Use deterministic generation (temperature=0) for testing
- Compare intermediate values, not just final outputs
- Focus on correctness first, optimize later