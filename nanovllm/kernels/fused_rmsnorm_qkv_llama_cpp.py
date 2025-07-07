"""
Fused RMSNorm + QKV projection following llama.cpp's exact approach.

Key principles from llama.cpp:
1. ALL computations in float32
2. Load inputs as half/bfloat16, immediately convert to float32
3. Store intermediate normalized values in float32
4. Only convert back to half/bfloat16 at the very end
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_rmsnorm_qkv_llama_cpp_kernel(
    # Input tensors
    input_ptr,      # [seq_len, hidden_dim] in bfloat16
    norm_weight_ptr,  # [hidden_dim] in bfloat16
    qkv_weight_ptr,   # [qkv_dim, hidden_dim] in bfloat16
    # Output tensor
    output_ptr,     # [seq_len, qkv_dim] in bfloat16
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
    Fused RMSNorm + QKV kernel following llama.cpp approach.
    
    This kernel performs ALL operations in float32 to maintain numerical stability.
    """
    pid_m = tl.program_id(0)  # Sequence dimension
    pid_n = tl.program_id(1)  # QKV output dimension
    
    # Compute the row index for this program
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_idx < seq_len
    
    # Step 1: Compute RMSNorm entirely in float32
    acc_var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # First pass: compute variance
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        col_idx = k + tl.arange(0, BLOCK_SIZE_K)
        col_mask = col_idx < hidden_dim
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load input and immediately convert to float32
        input_block = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + col_idx[None, :] * input_stride_hidden,
            mask=mask,
            other=0.0
        )
        input_f32 = input_block.to(tl.float32)
        
        # Accumulate variance in float32
        acc_var += tl.sum(input_f32 * input_f32, axis=1)
    
    # Compute RMS in float32
    rms_f32 = tl.sqrt(acc_var / hidden_dim + eps)
    scale_f32 = 1.0 / rms_f32
    
    # Step 2: Apply normalization and compute matrix multiplication
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_idx < qkv_dim
    
    # Initialize output accumulator in float32
    acc_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Second pass: normalize and multiply
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < hidden_dim
        
        # Load input and convert to float32
        input_tile = tl.load(
            input_ptr + row_idx[:, None] * input_stride_seq + k_idx[None, :] * input_stride_hidden,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        input_f32 = input_tile.to(tl.float32)
        
        # Load norm weight and convert to float32
        norm_weight_tile = tl.load(
            norm_weight_ptr + k_idx,
            mask=k_mask,
            other=0.0
        )
        norm_weight_f32 = norm_weight_tile.to(tl.float32)
        
        # Apply normalization entirely in float32
        # normalized = (input / rms) * norm_weight
        normalized_f32 = (input_f32 * scale_f32[:, None]) * norm_weight_f32[None, :]
        
        # Load QKV weight and convert to float32
        weight_tile = tl.load(
            qkv_weight_ptr + col_idx[:, None] * qkv_stride_out + k_idx[None, :] * qkv_stride_in,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        weight_f32 = weight_tile.to(tl.float32)
        
        # Matrix multiply in float32
        # Note: weight is transposed in storage, so we use direct multiplication
        for m in range(BLOCK_SIZE_M):
            if row_mask[m]:
                for n in range(BLOCK_SIZE_N):
                    if col_mask[n]:
                        # Dot product for this output element
                        for k_inner in range(BLOCK_SIZE_K):
                            if k_mask[k_inner]:
                                acc_out[m, n] += normalized_f32[m, k_inner] * weight_f32[n, k_inner]
    
    # Convert output back to bfloat16 and store
    output_mask = row_mask[:, None] & col_mask[None, :]
    output_bf16 = acc_out.to(tl.bfloat16)
    tl.store(
        output_ptr + row_idx[:, None] * output_stride_seq + col_idx[None, :] * output_stride_qkv,
        output_bf16,
        mask=output_mask
    )


class FusedRMSNormQKVLlamaCpp:
    """Fused RMSNorm + QKV following llama.cpp's exact approach."""
    
    @staticmethod
    def forward(
        input: torch.Tensor,      # [batch_seq_len, hidden_dim] in bfloat16
        norm_weight: torch.Tensor,  # [hidden_dim] in bfloat16
        qkv_weight: torch.Tensor,   # [qkv_dim, hidden_dim] in bfloat16
        bias: torch.Tensor,         # Optional bias
        num_q_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute fused RMSNorm + QKV projection following llama.cpp approach.
        
        ALL intermediate computations are done in float32 for numerical stability.
        
        Returns:
            q: [batch_seq_len, num_q_heads, head_dim] in bfloat16
            k: [batch_seq_len, num_kv_heads, head_dim] in bfloat16
            v: [batch_seq_len, num_kv_heads, head_dim] in bfloat16
        """
        batch_seq_len, hidden_dim = input.shape
        qkv_dim = qkv_weight.shape[0]
        # Calculate head_dim from dimensions
        total_heads = num_q_heads + 2 * num_kv_heads
        head_dim = qkv_dim // total_heads
        
        # Ensure inputs are contiguous and in bfloat16
        input = input.contiguous()
        norm_weight = norm_weight.contiguous()
        qkv_weight = qkv_weight.contiguous()
        
        # Allocate output in bfloat16
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
        fused_rmsnorm_qkv_llama_cpp_kernel[grid](
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
        
        # Add bias if provided
        if bias is not None:
            output = output + bias
        
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