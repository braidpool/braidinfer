"""
Optimized fused attention output kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def optimized_o_proj_residual_kernel(
    # Inputs
    x_ptr,
    w_ptr,
    residual_ptr,
    # Output
    y_ptr,
    # Dimensions
    M,  # Input size (hidden_size)
    N,  # Output size (hidden_size)
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """
    Optimized kernel for o_proj + residual.
    
    Key optimizations:
    1. Coalesced memory access
    2. Warp-level parallelism
    3. Minimal synchronization
    """
    # Each program computes one output element
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < N
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load residual early
    residual = tl.load(residual_ptr + out_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Compute dot product
    for m in range(0, M, BLOCK_SIZE):
        m_offs = m + tl.arange(0, BLOCK_SIZE)
        m_mask = m_offs < M
        
        # Load input block
        x = tl.load(x_ptr + m_offs, mask=m_mask, other=0.0).to(tl.float32)
        
        # Load weight block - each thread loads its row
        for i in range(BLOCK_SIZE):
            if out_idx[i] < N:
                w = tl.load(
                    w_ptr + out_idx[i] * M + m_offs,
                    mask=m_mask,
                    other=0.0
                ).to(tl.float32)
                
                # Accumulate dot product
                acc[i] += tl.sum(x * w)
    
    # Add residual and store
    result = acc + residual
    tl.store(y_ptr + out_idx, result.to(tl.float16), mask=mask)


@triton.jit
def fast_o_proj_residual_kernel(
    x_ptr, w_ptr, residual_ptr, y_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fast kernel using 2D tiling for better parallelism.
    """
    # 2D grid
    pid_n = tl.program_id(0)
    
    # Output tile
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    
    # Accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Process M dimension in blocks
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M
        
        # Load input block (shared across all N)
        x_block = tl.load(x_ptr + m_offs, mask=m_mask, other=0.0).to(tl.float32)
        
        # Load weight tile and compute
        w_ptrs = w_ptr + n_offs[:, None] * M + m_offs[None, :]
        w_mask = n_mask[:, None] & m_mask[None, :]
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        
        # Matrix multiply
        acc += tl.sum(w_block * x_block[None, :], axis=1)
    
    # Load residual and add
    residual = tl.load(residual_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    result = acc + residual
    
    # Store
    tl.store(y_ptr + n_offs, result.to(tl.float16), mask=n_mask)


class OptimizedAttentionOutput:
    """Optimized fused kernels."""
    
    @staticmethod
    def forward(x, weight, residual):
        """Fused o_proj + residual."""
        batch_size, seq_len, hidden_size = x.shape
        assert batch_size == 1 and seq_len == 1
        
        x_flat = x.view(hidden_size)
        residual_flat = residual.view(hidden_size)
        y = torch.empty_like(x_flat)
        
        # Use fast kernel
        BLOCK_M = 128
        BLOCK_N = 128
        grid = (triton.cdiv(hidden_size, BLOCK_N),)
        
        fast_o_proj_residual_kernel[grid](
            x_flat,
            weight,
            residual_flat,
            y,
            hidden_size,
            hidden_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        return y.view(batch_size, seq_len, hidden_size)
    
    @staticmethod
    def test():
        """Test and benchmark."""
        print("Testing Optimized Attention Output")
        print("=" * 50)
        
        # Setup
        batch_size = 1
        seq_len = 1
        hidden_size = 896
        
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda')
        weight = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device='cuda') * 0.02
        residual = torch.randn_like(x)
        
        # Test correctness
        y_opt = OptimizedAttentionOutput.forward(x, weight, residual)
        y_ref = torch.matmul(x, weight.t()) + residual
        
        max_diff = torch.max(torch.abs(y_opt - y_ref)).item()
        print(f"Max difference: {max_diff:.6f}")
        print("✓ Test PASSED!" if max_diff < 0.01 else "✗ Test FAILED!")
        
        # Benchmark
        print("\nBenchmarking...")
        
        # Compare different approaches
        from nanovllm.layers.linear import ColumnParallelLinear
        o_proj = ColumnParallelLinear(hidden_size, hidden_size, bias=False).cuda().half()
        o_proj.weight.data = weight
        
        def separate_ops():
            y = torch.nn.functional.linear(x, weight)
            return y + residual
        
        def with_o_proj_layer():
            y = o_proj(x)
            return y + residual
        
        implementations = [
            ("Separate matmul+add", separate_ops),
            ("With o_proj layer", with_o_proj_layer),
            ("Fused kernel", lambda: OptimizedAttentionOutput.forward(x, weight, residual)),
        ]
        
        for name, func in implementations:
            # Warmup
            for _ in range(100):
                _ = func()
            torch.cuda.synchronize()
            
            # Time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            num_iters = 1000
            start.record()
            for _ in range(num_iters):
                _ = func()
            end.record()
            
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / num_iters
            print(f"\n{name}: {time_ms:.3f} ms")
            
            if name == "Separate matmul+add":
                baseline = time_ms
            else:
                print(f"Speedup vs baseline: {baseline/time_ms:.2f}x")


if __name__ == "__main__":
    OptimizedAttentionOutput.test()