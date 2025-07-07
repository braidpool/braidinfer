# SPRINT.md - Next Sprint Planning

## Potential Next Sprints

### Option 1: Debug ERNIE Implementation
- Fix gibberish output issue (model works with vanilla transformers)
- Investigate weight loading or architecture mismatch
- Ensure ERNIE can leverage fused kernels

### Option 2: Performance Benchmarking with TinyLlama
- Comprehensive performance comparison with/without fused kernels
- Profile kernel execution times
- Optimize TinyLlama-specific bottlenecks
- Create detailed benchmark report

### Option 3: Test More LLaMA-Family Models
- Test Mistral models
- Test Gemma models  
- Test Llama 2 models
- Expand compatibility matrix

### Option 4: Quantization Implementation
- Implement INT8/INT4 quantization
- Expected 2-4x speedup
- Start with TinyLlama as baseline

### Option 5: Fix Remaining Issues
- Re-implement model warmup functionality
- Restore error handling and metrics
- Fix GPT-2 weight loading issues

## Sprint Selection Criteria
- TinyLlama provides a working baseline for optimization
- Performance improvements should be measurable
- Expand model support to validate compatibility patterns
- Focus on practical improvements for single-GPU inference