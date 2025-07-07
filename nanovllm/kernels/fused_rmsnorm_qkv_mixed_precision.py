"""
Fused RMSNorm + QKV projection with mixed precision matching standard path.

Key difference from original: Performs normalization in input dtype (bfloat16)
rather than float32 throughout, matching PyTorch's numerical behavior.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def fused_rmsnorm_qkv_mixed_precision_kernel(
    # Input tensors
    input_ptr,      # [seq_len, hidden_dim]
    norm_weight_ptr,  # [hidden_dim]
    qkv_weight_ptr,   # [qkv_dim, hidden_dim]
    qkv_bias_ptr,     # [qkv_dim] - bias parameter
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
    # Flags
    has_bias: tl.constexpr,  # Whether bias is present
    # Hyperparameters
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused RMSNorm + QKV kernel with mixed precision.
    
    Key numerical behavior:
    1. Compute variance in float32
    2. Apply normalization in input dtype (bfloat16)
    3. QKV projection in input dtype
    """
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_n = tl.program_id(1)  # QKV output dimension
    
    # Compute the row index for this program
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_idx < seq_len
    
    # Step 1: Compute RMSNorm with float32 variance ONLY
    acc_var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over hidden dimension to compute variance
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        col_idx = k + tl.arange(0, BLOCK_SIZE_K)
        col_mask = col_idx < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load input block
        input_block = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + col_idx[None, :] * input_stride_hidden,
            mask=mask,
            other=0.0
        )
        
        # Convert to float32 ONLY for variance accumulation
        input_f32 = input_block.to(tl.float32)
        acc_var += tl.sum(input_f32 * input_f32, axis=1)
    
    # Compute RMS in float32
    rms_f32 = tl.sqrt(acc_var / hidden_dim + eps)
    
    # Step 2: Apply normalization and compute QKV projection
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_idx < qkv_dim
    
    # Initialize output accumulator in float32 (required by tl.dot)
    # We'll convert to bfloat16 when storing
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
        
        # Apply RMSNorm: match PyTorch's exact behavior
        # 1. Normalize in float32
        input_f32 = input_tile.to(tl.float32)
        normalized_f32 = input_f32 / rms_f32[:, None]
        
        # 2. Convert back to original dtype (bfloat16)
        normalized_bf16 = normalized_f32.to(tl.bfloat16)
        
        # 3. Apply weight multiplication in bfloat16
        normalized = normalized_bf16 * norm_weight_tile[None, :]
        
        # Load weight tile
        weight_tile = tl.load(
            qkv_weight_ptr + col_idx[:, None] * qkv_stride_out + k_idx[None, :] * qkv_stride_in,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply - tl.dot requires float32 accumulator
        # But inputs are in bfloat16 to match standard path behavior
        acc_out += tl.dot(normalized.to(tl.float32), weight_tile.to(tl.float32).trans())
    
    # Step 3: Add bias if present
    if has_bias:
        bias_tile = tl.load(
            qkv_bias_ptr + col_idx,
            mask=col_mask,
            other=0.0
        )
        # Add bias in float32 before converting to bfloat16
        acc_out += bias_tile[None, :].to(tl.float32)
    
    # Convert to input dtype before storing (matching standard path)
    acc_out_bf16 = acc_out.to(tl.bfloat16)
    
    # Store output in input dtype
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(
        output_ptr + row_idx[:, None] * output_stride_seq + col_idx[None, :] * output_stride_qkv,
        acc_out_bf16,
        mask=output_mask
    )


class FusedRMSNormQKVMixedPrecision:
    """Fused RMSNorm + QKV with mixed precision matching standard path."""
    
    @staticmethod
    def forward(
        input: torch.Tensor,      # [batch_seq_len, hidden_dim]
        norm_weight: torch.Tensor,  # [hidden_dim]
        qkv_weight: torch.Tensor,   # [qkv_dim, hidden_dim]
        qkv_bias: Optional[torch.Tensor],  # [qkv_dim] or None
        num_q_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute fused RMSNorm + QKV projection with mixed precision.
        
        This matches the standard PyTorch path:
        - Variance computed in float32
        - Normalization applied in input dtype
        - QKV projection in input dtype
        
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
        if qkv_bias is not None:
            qkv_bias = qkv_bias.contiguous()
        
        # Allocate output in INPUT dtype (not float32!)
        output = torch.empty(
            (batch_seq_len, qkv_dim),
            dtype=input.dtype,  # Match input dtype
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
        fused_rmsnorm_qkv_mixed_precision_kernel[grid](
            # Pointers
            input,
            norm_weight,
            qkv_weight,
            qkv_bias if qkv_bias is not None else input,  # Dummy pointer if no bias
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
            # Flags
            has_bias=qkv_bias is not None,
            # Hyperparameters
            eps=eps,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Split QKV
        q_dim = num_q_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        
        q, k, v = output.split([q_dim, k_dim, v_dim], dim=-1)
        
        # Reshape to head format
        q = q.view(batch_seq_len, num_q_heads, head_dim)
        k = k.view(batch_seq_len, num_kv_heads, head_dim)
        v = v.view(batch_seq_len, num_kv_heads, head_dim)
        
        return q, k, v