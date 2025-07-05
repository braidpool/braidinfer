# Archived Sprint: Unlocking Hardware Performance with `tl.dot`

This sprint was planned but not executed due to fundamental issues discovered with the Triton kernel implementations.

## Original Goal
Use `tl.dot` to leverage Tensor Cores and fix the performance regression in fused kernels.

## Why It Was Cancelled
- The RMSNorm+QKV kernel is 15x slower than PyTorch
- The o_proj fusion kernel is 2.2x slower than cuBLAS
- Fundamental issues with Triton implementation approach
- Even with `tl.dot`, unlikely to overcome the overhead

## Lessons Learned
- Not all operations benefit from Triton fusion
- PyTorch and cuBLAS are highly optimized
- Custom kernels require extensive optimization to compete
- Focus should be on proven optimization techniques

See SPRINT_COMPLETE_STREAMING.md for what was actually completed.