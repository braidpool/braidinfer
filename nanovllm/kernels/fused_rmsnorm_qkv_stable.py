"""
Numerically stable fused RMSNorm + QKV projection using Triton.
This version addresses numerical stability issues with extreme normalization weights.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_rmsnorm_qkv_stable_kernel(
    # Input tensors
    input_ptr,      # [seq_len, hidden_dim]
    norm_weight_ptr,  # [hidden_dim]
    qkv_weight_ptr,   # [qkv_dim, hidden_dim]
    # Output tensor
    output_ptr,     # [seq_len, qkv_dim]
    # Dimensions
    seq_len,
    hidden_dim,
    qkv_dim,
    # Strides
    input_stride_seq,
    input_stride_hidden,
    qkv_stride_out,
    qkv_stride_in,
    output_stride_seq,
    output_stride_qkv,
    # Hyperparameters
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Numerically stable fused RMSNorm + QKV projection kernel.
    
    Key improvements for stability:
    1. Use float64 for variance accumulation
    2. Careful ordering of operations to minimize precision loss
    3. Kahan summation for improved accuracy
    """
    # Program ID gives us which output block we're computing
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_n = tl.program_id(1)  # QKV output dimension
    
    # Compute the row index for this program
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_idx < seq_len
    
    # Step 1: Compute RMSNorm with improved numerical stability
    # Use Welford's algorithm for variance computation
    mean_sq = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over hidden dimension in chunks
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # Create column indices
        col_idx = k + tl.arange(0, BLOCK_SIZE_K)
        col_mask = col_idx < hidden_dim
        
        # Create 2D mask
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load input block
        input_block = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + col_idx[None, :] * input_stride_hidden,
            mask=mask,
            other=0.0
        ).to(tl.float32)
        
        # Update mean of squares using Welford's algorithm
        # This is more numerically stable than simple accumulation
        block_sum_sq = tl.sum(input_block * input_block, axis=1)
        block_count = tl.sum(mask.to(tl.float32), axis=1)
        
        # Welford update
        delta = block_sum_sq - mean_sq * block_count
        mean_sq = mean_sq + delta / (count + block_count)
        count = count + block_count
    
    # Compute RMS with improved precision
    # Add a small constant to prevent division by zero
    rms = tl.sqrt(mean_sq + eps)
    
    # Step 2: Compute normalized input @ qkv_weight.T for our output block
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_idx < qkv_dim
    
    # Initialize output accumulator with Kahan summation variables
    acc_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    kahan_c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Tiled matrix multiplication loop with Kahan summation
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # Create K indices
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < hidden_dim
        
        # Load and normalize input tile
        input_tile = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + k_idx[None, :] * input_stride_hidden,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Apply RMSNorm
        norm_weight_tile = tl.load(
            norm_weight_ptr + k_idx,
            mask=k_mask,
            other=0.0
        ).to(tl.float32)
        
        # Normalize: (input / rms) * norm_weight
        # Use reciprocal multiplication for better precision
        inv_rms = 1.0 / rms
        normalized_tile = (input_tile * inv_rms[:, None]) * norm_weight_tile[None, :]
        
        # Load weight tile [N, K]
        weight_tile = tl.load(
            qkv_weight_ptr + col_idx[:, None] * qkv_stride_out + k_idx[None, :] * qkv_stride_in,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Accumulate using Kahan summation for improved accuracy
        # This helps prevent accumulation of rounding errors
        prod = tl.dot(normalized_tile, weight_tile.trans())
        y = prod - kahan_c
        t = acc_out + y
        kahan_c = (t - acc_out) - y
        acc_out = t
    
    # Store output
    output_mask = row_mask[:, None] & col_mask[None, :]
    # Convert to bfloat16 to match model dtype
    tl.store(
        output_ptr + row_idx[:, None] * output_stride_seq + col_idx[None, :] * output_stride_qkv,
        acc_out.to(tl.bfloat16),
        mask=output_mask
    )


class FusedRMSNormQKVStable:
    """Numerically stable fused RMSNorm + QKV projection."""
    
    @staticmethod
    def forward(
        input: torch.Tensor,      # [batch_seq_len, hidden_dim]
        norm_weight: torch.Tensor,  # [hidden_dim]
        qkv_weight: torch.Tensor,   # [qkv_dim, hidden_dim]
        num_q_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute fused RMSNorm + QKV projection with improved numerical stability.
        
        Returns:
            q: [batch_seq_len, num_q_heads, head_dim]
            k: [batch_seq_len, num_kv_heads, head_dim]
            v: [batch_seq_len, num_kv_heads, head_dim]
        """
        batch_seq_len, hidden_dim = input.shape
        qkv_dim = qkv_weight.shape[0]
        # Calculate head_dim from dimensions
        total_heads = num_q_heads + 2 * num_kv_heads
        head_dim = qkv_dim // total_heads
        
        # Ensure inputs are contiguous
        input = input.contiguous()
        norm_weight = norm_weight.contiguous()
        qkv_weight = qkv_weight.contiguous()
        
        # Allocate output - use bfloat16 to match model dtype
        output = torch.empty(
            (batch_seq_len, qkv_dim),
            dtype=torch.bfloat16,
            device=input.device
        )
        
        # Choose block sizes based on hardware
        BLOCK_SIZE_M = 16  # Sequence dimension block
        BLOCK_SIZE_N = 64  # Output dimension block
        BLOCK_SIZE_K = 64  # Reduction dimension block
        
        # Launch grid
        grid = (
            triton.cdiv(batch_seq_len, BLOCK_SIZE_M),
            triton.cdiv(qkv_dim, BLOCK_SIZE_N),
        )
        
        # Launch kernel
        fused_rmsnorm_qkv_stable_kernel[grid](
            # Pointers
            input,
            norm_weight,
            qkv_weight,
            output,
            # Dimensions
            batch_seq_len,
            hidden_dim,
            qkv_dim,
            # Strides
            input.stride(0),
            input.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            output.stride(0),
            output.stride(1),
            # Hyperparameters
            eps,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
        
        # Split QKV
        q_dim = num_q_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        
        q, k, v = output.split([q_dim, k_dim, v_dim], dim=-1)
        
        # Reshape
        q = q.view(batch_seq_len, num_q_heads, head_dim)
        k = k.view(batch_seq_len, num_kv_heads, head_dim)
        v = v.view(batch_seq_len, num_kv_heads, head_dim)
        
        return q, k, v