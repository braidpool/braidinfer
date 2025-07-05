# SPRINT.md - Next Sprint: MLP Block Fusion

## Previous Sprint Summary
Successfully completed Integration and Correctness sprint:
- Integrated fused RMSNorm+QKV kernel (2.64x speedup)
- Implemented chunk attention with online softmax (2,938 tok/s)
- Achieved 230 tok/s from 87 tok/s baseline
- All tests passing with <0.001 error

## Current Sprint Goal
Implement MLP block fusion to reduce kernel launches and improve memory bandwidth utilization. Target is to reach 400+ tok/s for batch size 1 inference.

## Sprint Tasks

### Week 1: MLP Fusion Implementation

#### 1. Analyze MLP Block Structure (Day 1)
- [ ] Profile current MLP execution in Qwen3MLP
- [ ] Identify kernel launch patterns for gate_up_proj and down_proj
- [ ] Measure memory bandwidth utilization
- [ ] Document fusion opportunities

#### 2. Design Fused MLP Kernel (Days 2-3)
- [ ] Design Triton kernel that fuses:
  - Gate and Up projections (currently MergedColumnParallelLinear)
  - SiLU activation
  - Down projection
  - Optional: Include post-attention layer norm
- [ ] Plan memory access patterns for optimal bandwidth
- [ ] Consider register pressure and occupancy

#### 3. Implement Fused MLP Kernel (Days 4-6)
- [ ] Create `fused_mlp_kernel.py` in `nanovllm/kernels/`
- [ ] Implement the Triton kernel with:
  - Efficient weight loading
  - Fused SiLU activation
  - Minimal memory transactions
- [ ] Add FusedMLP wrapper class
- [ ] Write unit tests

### Week 2: Integration and Optimization

#### 4. Integrate into Qwen3 Model (Days 7-8)
- [ ] Create Qwen3MLPFused class
- [ ] Modify Qwen3DecoderLayer to use fused MLP when flag is set
- [ ] Ensure weight compatibility with existing checkpoints
- [ ] Test correctness vs standard implementation

#### 5. Attention Output Fusion (Days 9-10)
- [ ] Analyze attention output projection + residual pattern
- [ ] Design kernel to fuse:
  - Output projection (o_proj)
  - Residual connection
  - Optional: Include layer norm
- [ ] Implement and integrate

#### 6. Benchmark and Optimize (Days 11-12)
- [ ] Comprehensive benchmarking of all kernels
- [ ] Profile end-to-end performance
- [ ] Identify remaining bottlenecks
- [ ] Document performance improvements

## Success Criteria
- **Minimum**: MLP fusion kernel working correctly with measurable speedup
- **Target**: Achieve 350+ tok/s with both MLP and attention output fusion
- **Stretch**: Reach 400+ tok/s for batch size 1 generation

## Technical Considerations
- MLP typically accounts for ~2/3 of compute in transformers
- Current MLP has 3 separate kernel launches per layer
- Fusion should reduce to 1 kernel launch
- Memory bandwidth is critical for decode phase
- Register pressure may limit fusion scope

## Expected Impact
- Reduce kernel launches from ~1,600 to ~800
- Improve memory bandwidth utilization by 2-3x for MLP
- Combined with existing optimizations: 3-4x total speedup