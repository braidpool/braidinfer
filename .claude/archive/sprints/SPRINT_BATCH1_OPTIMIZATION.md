# Sprint: Batch Size 1 Optimization

## Goal
Optimize Braidinfer for single-user scenarios where batch size 1 is the primary use case. Target: Get as close as possible to the theoretical FlashInfer maximum of ~1237 tokens/s.

## Current Performance Gap Analysis

### Current State
- **Theoretical FlashInfer**: 1237 tokens/s (0.808 ms/token)
- **Current Batch 1**: 31 tokens/s (32.3 ms/token)
- **Gap**: 31.5 ms of overhead per token (97.5% overhead!)

### Overhead Breakdown (Estimated)
1. **CPU Time**: ~25-30 ms
   - Python interpreter overhead
   - Scheduler logic
   - Memory management
   - Sequence tracking
   
2. **GPU Time**: ~2-7 ms
   - Kernel launch overhead
   - Memory transfers
   - Synchronization points

## Investigation Plan

### Phase 1: CPU Profiling (Days 1-2)

#### 1.1 Profile Python Overhead
```python
# Create detailed CPU profiler script
- Use cProfile/line_profiler on the generation loop
- Identify hot spots in Python code
- Measure function call overhead
- Track object allocation/deallocation
```

#### 1.2 Analyze Inference Loop
- Time each component:
  - Scheduler: `_step_async()`
  - Request handling
  - Sequence management
  - Token processing
  - Output formatting

#### 1.3 Memory Allocation Analysis
- Track torch allocations per token
- Identify unnecessary copies
- Find allocation hot spots
- Check for memory fragmentation

### Phase 2: CUDA Graph Implementation (Days 3-5)

#### 2.1 Fix FlashInfer Integration
```python
# Key fixes needed:
1. Pre-plan decode wrapper before capture
2. Use static sequence allocation
3. Implement proper state management
4. Handle workspace buffer correctly
```

#### 2.2 Static Memory Management
- Pre-allocate all buffers
- Use fixed-size pools
- Eliminate dynamic allocations
- Implement buffer recycling

#### 2.3 Graph Capture Strategy
```python
# Capture strategy:
1. Warmup with dummy sequences
2. Plan wrapper with max configuration
3. Capture graph with static buffers
4. Implement fast path for batch=1
```

### Phase 3: Optimization Implementation (Days 6-7)

#### 3.1 Fast Path for Batch Size 1
- Bypass scheduler for single sequences
- Direct model execution path
- Minimal Python overhead
- Optimized memory access

#### 3.2 Kernel Fusion Opportunities
- Fuse embedding + position encoding
- Combine layer norm + linear
- Merge attention + KV update
- Optimize output processing

#### 3.3 C++ Extension (Optional)
- Create C++ wrapper for hot path
- Minimize Python<->C++ transitions
- Direct CUDA kernel calls
- Benchmark against pure Python

## Specific Optimizations to Implement

### 1. Immediate Wins (High Impact, Low Effort)

```python
# 1. Pre-allocate reusable buffers
self.static_input_ids = torch.empty(1, dtype=torch.long, device="cuda")
self.static_positions = torch.empty(1, dtype=torch.long, device="cuda")
self.static_logits_buffer = torch.empty(1, vocab_size, dtype=dtype, device="cuda")

# 2. Eliminate unnecessary tensor creation
# Instead of: positions = torch.tensor([pos], device="cuda")
# Use: self.static_positions[0] = pos

# 3. Reduce Python function calls
# Inline hot functions where possible
```

### 2. CUDA Graph Fixes

```python
# Fix wrapper state management
class CUDAGraphRunner:
    def _prepare_wrapper_for_capture(self):
        # Pre-plan with maximum expected configuration
        self.model_runner.decode_wrapper.plan(
            kv_indptr=self.static_kv_indptr,
            kv_indices=self.static_kv_indices,
            last_page_lens=self.static_last_page_lens,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            kv_data_type=self.dtype,
            q_data_type=self.dtype
        )
```

### 3. Scheduler Bypass

```python
# Direct execution path for batch size 1
class LLMEngine:
    def generate_single_fast(self, prompt: str, max_tokens: int):
        """Fast path for single sequence generation."""
        # Skip scheduler entirely
        # Direct model execution
        # Minimal overhead
```

## Success Metrics

1. **Primary Goal**: Achieve >500 tokens/s for batch size 1
2. **Stretch Goal**: Reach >800 tokens/s (65% of theoretical max)
3. **Minimum**: Double current performance to >60 tokens/s

## Measurement Plan

1. Create comprehensive benchmark suite:
   - Pure FlashInfer baseline
   - Current implementation
   - Each optimization step
   - Final optimized version

2. Profile metrics:
   - Tokens per second
   - Latency per token (p50, p95, p99)
   - CPU utilization
   - GPU utilization
   - Memory bandwidth

## Risk Mitigation

1. **Complexity**: Keep optimizations modular and toggleable
2. **Stability**: Maintain comprehensive test coverage
3. **Compatibility**: Ensure optimizations work with all models
4. **Maintainability**: Document all optimizations thoroughly

## Next Steps

1. Start with CPU profiling to identify biggest bottlenecks
2. Implement static memory allocation
3. Fix CUDA graph implementation
4. Create fast path for single sequence
5. Benchmark and iterate