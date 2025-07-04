# SPRINT_PERFORMANCE.md - Performance Investigation Sprint

## Sprint Goal
Investigate and fix the performance regression from ~200+ tok/s to ~23 tok/s.

## Current Status
- Gibberish output: ✓ Fixed
- All tests passing: ✓ 9/9
- Performance: ✗ 23 tok/s (expected 200+ tok/s)

## Sprint Tasks

### 1. Architectural Review
- [ ] Review all changes since last known good performance
- [ ] Identify potential bottlenecks introduced
- [ ] Compare with vanilla FlashInfer performance

### 2. Profiling
- [ ] Profile with PyTorch profiler
- [ ] Identify GPU utilization
- [ ] Check for CPU bottlenecks
- [ ] Memory access patterns

### 3. Specific Areas to Investigate
- [ ] Single wrapper vs per-layer wrappers overhead
- [ ] KV cache layout impact (HND vs NHD)
- [ ] Model loading and initialization
- [ ] Batch size and sequence length effects
- [ ] FlashInfer plan/run overhead

### 4. Benchmarking
- [ ] Create minimal FlashInfer benchmark
- [ ] Compare with vLLM performance
- [ ] Test different batch sizes
- [ ] Test different sequence lengths

### 5. Potential Optimizations
- [ ] Model warmup implementation
- [ ] Memory allocation patterns
- [ ] Workspace buffer sizing
- [ ] CUDA graph support

### 6. Testing
- [ ] Create performance regression tests
- [ ] Ensure fixes don't break functionality
- [ ] Verify improvements across different models

## Success Criteria
- Performance returns to ~200+ tok/s
- No functionality regressions
- Performance tests in place to prevent future regressions

## Notes
- Previous commit a6c6945 had better performance
- Single wrapper approach is architecturally correct per FlashInfer docs
- Need to find what else changed that impacted performance