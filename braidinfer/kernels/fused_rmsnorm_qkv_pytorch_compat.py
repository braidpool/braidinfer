"""
Fused RMSNorm + QKV projection with PyTorch-compatible bfloat16 conversion.

This kernel matches PyTorch's numerical behavior exactly by converting to bfloat16
at the same point PyTorch does - after normalization but before matrix multiplication.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_rmsnorm_qkv_pytorch_compat_kernel(
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
    Fused RMSNorm + QKV kernel with PyTorch-compatible precision handling.
    
    Key design choice: Convert to bfloat16 after normalization but before matmul,
    exactly matching PyTorch's behavior to avoid numerical differences with extreme weights.
    """
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_n = tl.program_id(1)  # QKV output dimension
    
    # Compute the row index for this program
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_idx < seq_len
    
    # Step 1: Compute RMSNorm with float32 accumulator for variance
    acc_var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over hidden dimension in chunks
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        col_idx = k + tl.arange(0, BLOCK_SIZE_K)
        col_mask = col_idx < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load input block - keep in original dtype
        input_block = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + col_idx[None, :] * input_stride_hidden,
            mask=mask,
            other=0.0
        )
        
        # Convert to float32 only for variance accumulation
        input_f32 = input_block.to(tl.float32)
        acc_var += tl.sum(input_f32 * input_f32, axis=1)
    
    # Compute RMS with float32 precision
    rms = tl.sqrt(acc_var / hidden_dim + eps)
    
    # Step 2: Compute normalized input @ qkv_weight.T
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_idx < qkv_dim
    
    # Initialize output accumulator in float32 (tl.dot returns float32)
    acc_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Tiled matrix multiplication loop
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < hidden_dim
        
        # Load input tile
        input_tile = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + k_idx[None, :] * input_stride_hidden,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load norm weight tile
        norm_weight_tile = tl.load(
            norm_weight_ptr + k_idx,
            mask=k_mask,
            other=0.0
        )
        
        # Apply RMSNorm: (input / rms) * norm_weight
        # Do normalization in float32 for accuracy
        normalized_f32 = input_tile.to(tl.float32) / rms[:, None] * norm_weight_tile.to(tl.float32)[None, :]
        
        # CRITICAL: Convert to bfloat16 here to match PyTorch
        normalized_tile = normalized_f32.to(tl.bfloat16)
        
        # Load weight tile
        weight_tile = tl.load(
            qkv_weight_ptr + col_idx[:, None] * qkv_stride_out + k_idx[None, :] * qkv_stride_in,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply in bfloat16 (matching PyTorch)
        # Ensure weight is also in bfloat16
        weight_bf16 = weight_tile.to(tl.bfloat16)
        acc_out += tl.dot(normalized_tile, weight_bf16.trans())
    
    # Store output - convert to bfloat16 to match PyTorch
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(
        output_ptr + row_idx[:, None] * output_stride_seq + col_idx[None, :] * output_stride_qkv,
        acc_out.to(tl.bfloat16),
        mask=output_mask
    )


class FusedRMSNormQKV:
    """Fused RMSNorm + QKV with PyTorch-compatible precision handling."""
    
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
        Compute fused RMSNorm + QKV projection with PyTorch-compatible precision.
        
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
        
        # Allocate output in bfloat16 to match PyTorch
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
        fused_rmsnorm_qkv_pytorch_compat_kernel[grid](
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