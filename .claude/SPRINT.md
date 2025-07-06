# Sprint: Separate RMSNorm from QKV Fusion

## Sprint Status: 60% Complete

### Objective
Refactor the fused RMSNorm+QKV kernel to match llama.cpp's approach: compute RMSNorm separately and only fuse QKV+RoPE. This resolves numerical stability issues with Qwen3-0.6B's extreme K normalization weights.

### Completed Tasks ✓

#### 1. Architectural Review
- Analyzed current fusion boundaries
- Identified that fusing RMSNorm with QKV causes numerical instability
- Created detailed architecture documents

#### 2. Create Standalone RMSNorm Kernel
- Implemented `RMSNormF32` with full float32 precision
- Achieved 2.19x speedup over PyTorch
- Passes all unit tests

#### 3. Create QKV+RoPE Fused Kernel
- Implemented `QKVRoPESimple` following llama.cpp design
- Takes normalized input from separate RMSNorm
- Uses mixed precision appropriately

#### 4. Refactor Qwen3AttentionFused
- Created `Qwen3AttentionSeparated` with new architecture
- Separates RMSNorm computation from QKV fusion
- Includes fallback attention for testing

#### 5. Update Qwen3DecoderLayer
- Created `Qwen3DecoderLayerSeparated`
- Properly passes layernorm weight to attention
- Maintains residual connections

#### 6. Integration Testing
- Comprehensive test suite created
- Validates numerical stability
- Confirms architectural improvements

### In Progress 🔄
7. **Performance Optimization** - Optimize the separated kernels

### Pending 📋
8. **Cleanup and Documentation** - Final documentation updates
9. **Extended Testing** - Test with actual Qwen3-0.6B weights
10. **Sprint Review** - Analyze results and plan next steps

## Key Achievements

### Technical Implementation
- Successfully separated RMSNorm from QKV fusion
- Followed llama.cpp's proven architectural approach
- Maintained numerical stability with extreme K normalization weights

### Performance Results
- RMSNormF32: 2.19x faster than PyTorch
- Kernel accuracy: within 2-3e-3 of original
- Full float32 precision where needed

### Files Created

**Kernels:**
- `nanovllm/kernels/rmsnorm_f32.py`
- `nanovllm/kernels/qkv_rope_simple.py`
- `nanovllm/kernels/qkv_rope_fused.py` (alternative implementation)

**Models:**
- `nanovllm/models/qwen3_separated.py`

**Tests:**
- `tests/test_rmsnorm_f32.py`
- `tests/test_qkv_rope_simple.py`
- `tests/test_qwen3_separated.py`
- `tests/test_qwen3_integration.py`

**Analysis Scripts:**
- `test_qwen3_stability.py`
- `test_separated_vs_fused.py`
- `test_final_integration.py`
- `test_qwen3_simple.py`

**Documentation:**
- `.claude/CURRENT_FUSION_ARCHITECTURE.md`
- `.claude/REFACTORING_PLAN.md`
- `.claude/SPRINT_PROGRESS.md`
- `.claude/INTEGRATION_TEST_RESULTS.md`

## Root Cause Analysis

The original issue was numerical instability caused by:
1. Extreme K normalization weights in Qwen3-0.6B (up to 96.5x)
2. Fusing RMSNorm with QKV amplified precision errors
3. Float16 precision insufficient for these extreme values

## Solution

Following llama.cpp's approach:
- Compute RMSNorm separately with full float32 precision
- Only fuse QKV projection with RoPE
- This prevents precision loss amplification

## Next Steps
1. Optimize performance of separated kernels
2. Clean up code and documentation
3. Extended testing with real Qwen3-0.6B model
4. Complete sprint review

## Time Spent
- Tasks 1-6: ~7 hours
- Estimated remaining: ~3 hours