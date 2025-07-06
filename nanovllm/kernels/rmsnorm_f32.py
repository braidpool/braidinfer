"""
Standalone RMSNorm kernel with full float32 precision.

This kernel is designed for models with extreme normalization weights
that require high precision normalization.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def rmsnorm_f32_kernel(
    # Input/Output pointers
    input_ptr,
    output_ptr,
    norm_weight_ptr,
    # Dimensions
    seq_len,
    hidden_dim,
    # Strides
    input_stride_seq,
    input_stride_hidden,
    output_stride_seq,
    output_stride_hidden,
    # Hyperparameters
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel with full float32 precision.
    
    Key features:
    1. All computations in float32 for maximum precision
    2. Optimized memory access patterns
    3. Support for in-place operation
    """
    # Each program handles one sequence position
    pid = tl.program_id(0)
    
    if pid >= seq_len:
        return
    
    # Step 1: Compute variance in float32
    acc_var = 0.0
    
    # Process hidden dimension in blocks
    for start_idx in range(0, hidden_dim, BLOCK_SIZE):
        # Create block indices
        block_idx = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = block_idx < hidden_dim
        
        # Load input block and convert to float32
        input_block = tl.load(
            input_ptr + pid * input_stride_seq + block_idx * input_stride_hidden,
            mask=mask,
            other=0.0
        ).to(tl.float32)
        
        # Accumulate squared values
        acc_var += tl.sum(input_block * input_block, axis=0)
    
    # Compute RMS
    rms = tl.sqrt(acc_var / hidden_dim + eps)
    
    # Step 2: Normalize and apply weight
    for start_idx in range(0, hidden_dim, BLOCK_SIZE):
        # Create block indices
        block_idx = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = block_idx < hidden_dim
        
        # Load input block
        input_block = tl.load(
            input_ptr + pid * input_stride_seq + block_idx * input_stride_hidden,
            mask=mask,
            other=0.0
        ).to(tl.float32)
        
        # Load norm weight
        weight_block = tl.load(
            norm_weight_ptr + block_idx,
            mask=mask,
            other=0.0
        ).to(tl.float32)
        
        # Normalize and apply weight
        normalized = (input_block / rms) * weight_block
        
        # Store output in float32
        tl.store(
            output_ptr + pid * output_stride_seq + block_idx * output_stride_hidden,
            normalized,
            mask=mask
        )


class RMSNormF32:
    """
    Standalone RMSNorm with full float32 precision.
    """
    
    @staticmethod
    def forward(
        input: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float = 1e-6,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply RMSNorm with full float32 precision.
        
        Args:
            input: Input tensor [seq_len, hidden_dim]
            norm_weight: Normalization weight [hidden_dim]
            eps: Epsilon for numerical stability
            output: Optional output tensor for in-place operation
            
        Returns:
            Normalized tensor in float32
        """
        assert input.dim() == 2, f"Expected 2D input, got {input.dim()}D"
        seq_len, hidden_dim = input.shape
        
        # Create output tensor if not provided
        if output is None:
            output = torch.empty_like(input, dtype=torch.float32)
        else:
            assert output.dtype == torch.float32, "Output must be float32"
            assert output.shape == input.shape, "Output shape must match input"
        
        # Determine block size based on hidden dimension
        BLOCK_SIZE = 256 if hidden_dim >= 256 else triton.next_power_of_2(hidden_dim)
        
        # Launch kernel
        grid = (seq_len,)
        rmsnorm_f32_kernel[grid](
            # Pointers
            input,
            output,
            norm_weight,
            # Dimensions
            seq_len,
            hidden_dim,
            # Strides
            input.stride(0),
            input.stride(1),
            output.stride(0),
            output.stride(1),
            # Hyperparameters
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
    
    @staticmethod
    def benchmark_vs_pytorch(seq_len: int = 512, hidden_dim: int = 1024, eps: float = 1e-6):
        """Benchmark against PyTorch implementation."""
        import time
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device='cuda')
        
        # PyTorch reference
        def pytorch_rmsnorm(x, w, eps):
            x_f32 = x.float()
            var = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 / torch.sqrt(var + eps)
            return x_normed * w.float()
        
        # Warmup
        for _ in range(10):
            _ = pytorch_rmsnorm(input, weight, eps)
            _ = RMSNormF32.forward(input, weight, eps)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(100):
            ref_output = pytorch_rmsnorm(input, weight, eps)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        # Benchmark Triton
        start = time.time()
        for _ in range(100):
            triton_output = RMSNormF32.forward(input, weight, eps)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Compare outputs
        max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
        rel_diff = torch.max(torch.abs((ref_output - triton_output) / (ref_output + 1e-8))).item()
        
        print(f"RMSNorm F32 Benchmark Results:")
        print(f"  Sequence length: {seq_len}, Hidden dim: {hidden_dim}")
        print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
        print(f"  Triton time: {triton_time*1000:.2f} ms")
        print(f"  Speedup: {pytorch_time/triton_time:.2f}x")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Max relative difference: {rel_diff:.2%}")


if __name__ == "__main__":
    # Test the kernel
    print("Testing RMSNorm F32 kernel...")
    
    # Test basic functionality
    input = torch.randn(128, 1024, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(1024, dtype=torch.bfloat16, device='cuda')
    output = RMSNormF32.forward(input, weight)
    
    print(f"Input shape: {input.shape}, dtype: {input.dtype}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    RMSNormF32.benchmark_vs_pytorch()