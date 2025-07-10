"""
Fused attention output projection and residual addition.

This module implements a high-performance fused kernel that combines:
1. Output projection (GEMV): output = attention @ o_proj_weight.T
2. Residual addition: output = output + residual

Key optimization: The intermediate result is kept in registers/shared memory,
avoiding a round-trip to global memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def tiled_gemv_kernel(
    # Input vector
    x_ptr,
    # Weight matrix (transposed)
    w_ptr,
    # Output vector
    y_ptr,
    # Dimensions
    M,  # Length of input vector (hidden_size)
    N,  # Length of output vector (hidden_size)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Tiled GEMV (General Matrix-Vector multiplication) kernel.
    
    Computes: y = x @ W.T where W is [N, M]
    
    This kernel uses tiling to efficiently compute the matrix-vector product:
    - Each thread block computes BLOCK_N elements of the output
    - The weight matrix is loaded tile by tile into shared memory
    - The input vector is kept in registers and reused across tiles
    """
    # Program ID determines which output tile this block computes
    pid = tl.program_id(0)
    
    # Output indices for this block
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    
    # Initialize accumulator for output tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Loop over the M dimension (input vector) in tiles
    for m_start in range(0, M, BLOCK_M):
        # Input indices for this tile
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        
        # Load input tile - this is reused for all output elements
        x_tile = tl.load(x_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)
        
        # Load weight tile [BLOCK_N, BLOCK_M]
        # W is stored as [N, M], so W[n, m] is at w_ptr + n * M + m
        w_tile_ptr = w_ptr + n_offsets[:, None] * M + m_offsets[None, :]
        w_mask = n_mask[:, None] & m_mask[None, :]
        w_tile = tl.load(w_tile_ptr, mask=w_mask, other=0.0).to(tl.float32)
        
        # Compute dot product for this tile
        # Each output element n computes sum over m of: x[m] * W[n, m]
        acc += tl.sum(w_tile * x_tile[None, :], axis=1)
    
    # Store output tile
    tl.store(y_ptr + n_offsets, acc.to(tl.float16), mask=n_mask)


@triton.jit
def fused_o_proj_add_residual_kernel(
    # Input vector (attention output)
    x_ptr,
    # Weight matrix (o_proj weights, stored as [N, M])
    w_ptr,
    # Residual vector
    residual_ptr,
    # Output vector
    y_ptr,
    # Dimensions
    M,  # Length of input vector (hidden_size)
    N,  # Length of output vector (hidden_size)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel for output projection and residual addition.
    
    Computes: y = (x @ W.T) + residual
    
    Key optimization: The residual addition happens in registers before
    writing to global memory, saving one memory round-trip.
    """
    # Program ID determines which output tile this block computes
    pid = tl.program_id(0)
    
    # Output indices for this block
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    
    # Initialize accumulator for output tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Loop over the M dimension (input vector) in tiles
    for m_start in range(0, M, BLOCK_M):
        # Input indices for this tile
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        
        # Load input tile
        x_tile = tl.load(x_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)
        
        # Load weight tile [BLOCK_N, BLOCK_M]
        w_tile_ptr = w_ptr + n_offsets[:, None] * M + m_offsets[None, :]
        w_mask = n_mask[:, None] & m_mask[None, :]
        w_tile = tl.load(w_tile_ptr, mask=w_mask, other=0.0).to(tl.float32)
        
        # Compute dot product for this tile
        acc += tl.sum(w_tile * x_tile[None, :], axis=1)
    
    # Load residual tile and add to accumulator
    residual_tile = tl.load(residual_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    acc += residual_tile
    
    # Store final output
    tl.store(y_ptr + n_offsets, acc.to(tl.float16), mask=n_mask)


class FusedAttentionOutput:
    """Fused attention output projection and residual addition."""
    
    @staticmethod
    def gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Tiled GEMV implementation.
        
        Args:
            x: Input vector [batch_size, hidden_size]
            weight: Weight matrix [hidden_size, hidden_size]
            
        Returns:
            Output vector [batch_size, hidden_size]
        """
        batch_size, M = x.shape
        N = weight.shape[0]
        
        assert batch_size == 1, "This kernel is optimized for batch size 1"
        assert weight.shape == (N, M), f"Weight shape mismatch: expected ({N}, {M}), got {weight.shape}"
        
        # Allocate output
        y = torch.empty(batch_size, N, dtype=x.dtype, device=x.device)
        
        # Configure grid
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N),)
        
        # Launch kernel
        tiled_gemv_kernel[grid](
            x[0],  # Flatten to 1D
            weight,
            y[0],
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        return y
    
    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fused output projection and residual addition.
        
        Args:
            x: Attention output [batch_size, seq_len, hidden_size]
            weight: Output projection weights [hidden_size, hidden_size]
            residual: Residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            Output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # For now, handle batch_size=1, seq_len=1 case
        assert batch_size == 1 and seq_len == 1, "Currently optimized for decode phase"
        
        # Flatten inputs
        x_flat = x.view(-1, hidden_size)
        residual_flat = residual.view(-1, hidden_size)
        
        # Allocate output
        y = torch.empty_like(x_flat)
        
        # Configure grid
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(hidden_size, BLOCK_N),)
        
        # Launch fused kernel
        fused_o_proj_add_residual_kernel[grid](
            x_flat[0],
            weight,
            residual_flat[0],
            y[0],
            hidden_size,
            hidden_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        return y.view(batch_size, seq_len, hidden_size)
    
    @staticmethod
    def test():
        """Test the kernels for correctness."""
        print("Testing Fused Attention Output Kernels")
        print("=" * 50)
        
        # Test configuration
        batch_size = 1
        seq_len = 1
        hidden_size = 896
        
        # Create test tensors
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device='cuda')
        weight = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device='cuda') * 0.02
        
        # Test GEMV kernel
        print("\n1. Testing tiled GEMV kernel...")
        y_gemv = FusedAttentionOutput.gemv(x, weight)
        y_ref = torch.matmul(x, weight.t())
        
        gemv_diff = torch.max(torch.abs(y_gemv - y_ref)).item()
        print(f"GEMV max difference: {gemv_diff:.6f}")
        print("✓ GEMV test PASSED!" if gemv_diff < 0.001 else "✗ GEMV test FAILED!")
        
        # Test fused kernel
        print("\n2. Testing fused o_proj + residual kernel...")
        x_3d = x.unsqueeze(1)  # Add seq_len dimension
        residual = torch.randn_like(x_3d)
        
        y_fused = FusedAttentionOutput.forward(x_3d, weight, residual)
        y_ref = torch.matmul(x_3d, weight.t()) + residual
        
        fused_diff = torch.max(torch.abs(y_fused - y_ref)).item()
        print(f"Fused max difference: {fused_diff:.6f}")
        print("✓ Fused test PASSED!" if fused_diff < 0.01 else "✗ Fused test FAILED!")
        
        # Benchmark
        print("\n3. Benchmarking...")
        
        # Warmup
        for _ in range(100):
            _ = torch.matmul(x_3d, weight.t()) + residual
            _ = FusedAttentionOutput.forward(x_3d, weight, residual)
        
        torch.cuda.synchronize()
        
        # Time separate operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        num_iters = 1000
        start.record()
        for _ in range(num_iters):
            y = torch.matmul(x_3d, weight.t())
            y = y + residual
        end.record()
        
        torch.cuda.synchronize()
        separate_time = start.elapsed_time(end) / num_iters
        
        # Time fused operation
        start.record()
        for _ in range(num_iters):
            _ = FusedAttentionOutput.forward(x_3d, weight, residual)
        end.record()
        
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / num_iters
        
        print(f"\nSeparate operations: {separate_time:.3f} ms")
        print(f"Fused operation: {fused_time:.3f} ms")
        print(f"Speedup: {separate_time/fused_time:.2f}x")
        
        # Project impact
        saved_per_layer = separate_time - fused_time
        total_saved = saved_per_layer * 24
        
        print(f"\nProjected impact (24 layers):")
        print(f"Time saved per layer: {saved_per_layer:.3f} ms")
        print(f"Total time saved: {total_saved:.2f} ms")
        
        current_throughput = 230  # tok/s
        current_ms = 1000 / current_throughput
        new_ms = current_ms - total_saved
        new_throughput = 1000 / new_ms
        
        print(f"\nThroughput projection:")
        print(f"Current: {current_throughput} tok/s")
        print(f"With fusion: {new_throughput:.0f} tok/s")
        print(f"Improvement: {(new_throughput/current_throughput - 1)*100:.1f}%")


if __name__ == "__main__":
    FusedAttentionOutput.test()