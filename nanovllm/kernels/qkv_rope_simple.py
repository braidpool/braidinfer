"""
Simple QKV + RoPE fused kernel following llama.cpp approach.

This implementation prioritizes correctness and clarity over maximum performance.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def qkv_matmul_kernel(
    # Input pointers
    input_ptr,            # [seq_len, hidden_dim]
    weight_ptr,           # [qkv_dim, hidden_dim]
    bias_ptr,             # [qkv_dim] or None
    # Output pointer
    output_ptr,           # [seq_len, qkv_dim]
    # Dimensions
    seq_len,
    hidden_dim,
    qkv_dim,
    # Strides
    input_stride_seq,
    input_stride_hidden,
    weight_stride_out,
    weight_stride_in,
    output_stride_seq,
    output_stride_qkv,
    # Hyperparameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Simple matrix multiplication for QKV projection."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block boundaries
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    row_idx = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_idx = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    row_mask = row_idx < seq_len
    col_mask = col_idx < qkv_dim
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Matrix multiplication with tiling
    for k_start in range(0, hidden_dim, BLOCK_SIZE_K):
        k_idx = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < hidden_dim
        
        # Load input tile
        input_ptrs = input_ptr + row_idx[:, None] * input_stride_seq + k_idx[None, :] * input_stride_hidden
        input_tile = tl.load(input_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight tile
        weight_ptrs = weight_ptr + col_idx[:, None] * weight_stride_out + k_idx[None, :] * weight_stride_in
        weight_tile = tl.load(weight_ptrs, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate
        acc += tl.dot(input_tile.to(tl.float32), weight_tile.trans().to(tl.float32))
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_idx, mask=col_mask, other=0.0)
        acc += bias[None, :].to(tl.float32)
    
    # Store result
    output_ptrs = output_ptr + row_idx[:, None] * output_stride_seq + col_idx[None, :] * output_stride_qkv
    tl.store(output_ptrs, acc, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def apply_rope_kernel(
    # Input/output pointer (in-place)
    qk_ptr,              # [seq_len, num_heads, head_dim]
    # RoPE cache
    cos_ptr,             # [max_seq_len, head_dim/2]
    sin_ptr,             # [max_seq_len, head_dim/2]
    positions_ptr,       # [seq_len]
    # Dimensions
    seq_len,
    num_heads,
    head_dim,
    # Strides
    qk_stride_seq,
    qk_stride_head,
    qk_stride_dim,
    cos_stride_pos,
    cos_stride_dim,
    sin_stride_pos,
    sin_stride_dim,
    # Hyperparameters
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embeddings to Q or K."""
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    if pid_seq >= seq_len or pid_head >= num_heads:
        return
    
    # Load position for this sequence element
    pos = tl.load(positions_ptr + pid_seq)
    
    # Process pairs of elements (real, imag)
    num_pairs = head_dim // 2
    for pair_start in range(0, num_pairs, BLOCK_SIZE):
        # Calculate how many pairs to process in this block
        num_pairs_in_block = tl.minimum(BLOCK_SIZE, num_pairs - pair_start)
        pair_idx = pair_start + tl.arange(0, BLOCK_SIZE)
        pair_mask = pair_idx < (pair_start + num_pairs_in_block)
        
        # Load cos and sin values
        cos_vals = tl.load(
            cos_ptr + pos * cos_stride_pos + pair_idx * cos_stride_dim,
            mask=pair_mask,
            other=0.0
        ).to(tl.float32)
        
        sin_vals = tl.load(
            sin_ptr + pos * sin_stride_pos + pair_idx * sin_stride_dim,
            mask=pair_mask,
            other=0.0
        ).to(tl.float32)
        
        # Calculate actual indices in the head dimension
        real_idx = pair_idx * 2
        imag_idx = pair_idx * 2 + 1
        
        # Create masks for real and imag indices
        real_mask = real_idx < head_dim
        imag_mask = imag_idx < head_dim
        
        # Calculate pointers
        real_ptrs = qk_ptr + pid_seq * qk_stride_seq + pid_head * qk_stride_head + real_idx * qk_stride_dim
        imag_ptrs = qk_ptr + pid_seq * qk_stride_seq + pid_head * qk_stride_head + imag_idx * qk_stride_dim
        
        # Load values
        real_vals = tl.load(real_ptrs, mask=pair_mask & real_mask, other=0.0).to(tl.float32)
        imag_vals = tl.load(imag_ptrs, mask=pair_mask & imag_mask, other=0.0).to(tl.float32)
        
        # Apply rotation: real' = real * cos - imag * sin
        #                 imag' = imag * cos + real * sin
        new_real = real_vals * cos_vals - imag_vals * sin_vals
        new_imag = imag_vals * cos_vals + real_vals * sin_vals
        
        # Store back
        tl.store(real_ptrs, new_real, mask=pair_mask & real_mask)
        tl.store(imag_ptrs, new_imag, mask=pair_mask & imag_mask)


class QKVRoPESimple:
    """
    Simple QKV + RoPE implementation following llama.cpp approach.
    """
    
    @staticmethod
    def forward(
        normalized_input: torch.Tensor,  # [seq_len, hidden_dim] in float32
        qkv_weight: torch.Tensor,        # [qkv_dim, hidden_dim]
        positions: torch.Tensor,         # [seq_len]
        cos_cache: torch.Tensor,         # [max_seq_len, head_dim/2]
        sin_cache: torch.Tensor,         # [max_seq_len, head_dim/2]
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply QKV projection followed by RoPE.
        
        This follows llama.cpp's approach:
        1. Project normalized input to QKV
        2. Apply RoPE to Q and K (not V)
        3. Return separate Q, K, V tensors
        """
        seq_len, hidden_dim = normalized_input.shape
        head_dim = hidden_dim // num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        
        # Ensure input is float32
        if normalized_input.dtype != torch.float32:
            normalized_input = normalized_input.float()
        
        # Step 1: QKV projection
        qkv = torch.empty(seq_len, qkv_dim, dtype=torch.float32, device=normalized_input.device)
        
        # Determine grid and block sizes
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (
            triton.cdiv(seq_len, BLOCK_SIZE_M),
            triton.cdiv(qkv_dim, BLOCK_SIZE_N)
        )
        
        qkv_matmul_kernel[grid](
            # Pointers
            normalized_input,
            qkv_weight,
            qkv_bias if qkv_bias is not None else None,
            qkv,
            # Dimensions
            seq_len,
            hidden_dim,
            qkv_dim,
            # Strides
            normalized_input.stride(0),
            normalized_input.stride(1),
            qkv_weight.stride(0),
            qkv_weight.stride(1),
            qkv.stride(0),
            qkv.stride(1),
            # Block sizes
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Step 2: Split QKV and reshape
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        
        q = qkv[:, :q_size].view(seq_len, num_heads, head_dim)
        k = qkv[:, q_size:q_size + k_size].view(seq_len, num_kv_heads, head_dim)
        v = qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
        
        # Step 3: Apply RoPE to Q and K
        BLOCK_SIZE = 32
        
        # Apply to Q
        grid_q = (seq_len, num_heads)
        apply_rope_kernel[grid_q](
            q,
            cos_cache,
            sin_cache,
            positions,
            seq_len,
            num_heads,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cos_cache.stride(0),
            cos_cache.stride(1),
            sin_cache.stride(0),
            sin_cache.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Apply to K
        grid_k = (seq_len, num_kv_heads)
        apply_rope_kernel[grid_k](
            k,
            cos_cache,
            sin_cache,
            positions,
            seq_len,
            num_kv_heads,
            head_dim,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos_cache.stride(0),
            cos_cache.stride(1),
            sin_cache.stride(0),
            sin_cache.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Convert to target dtype if needed
        target_dtype = qkv_weight.dtype
        if target_dtype != torch.float32:
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        
        return q, k, v
    
    @staticmethod
    def test():
        """Test the kernel implementation."""
        # Test parameters
        seq_len = 128
        hidden_dim = 1024
        num_heads = 16
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Create test data
        device = 'cuda'
        normalized_input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=device)
        qkv_weight = torch.randn((num_heads + 2 * num_kv_heads) * head_dim, hidden_dim, 
                                dtype=torch.bfloat16, device=device)
        qkv_bias = torch.randn((num_heads + 2 * num_kv_heads) * head_dim, 
                              dtype=torch.bfloat16, device=device)
        positions = torch.arange(seq_len, device=device)
        
        # Create RoPE cache
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(8192, device=device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            normalized_input, qkv_weight, positions,
            cos_cache, sin_cache, num_heads, num_kv_heads, qkv_bias
        )
        
        print(f"Q shape: {q.shape}, dtype: {q.dtype}")
        print(f"K shape: {k.shape}, dtype: {k.dtype}")
        print(f"V shape: {v.shape}, dtype: {v.dtype}")
        
        # Verify shapes
        assert q.shape == (seq_len, num_heads, head_dim)
        assert k.shape == (seq_len, num_kv_heads, head_dim)
        assert v.shape == (seq_len, num_kv_heads, head_dim)
        
        print("QKV+RoPE kernel test passed!")


if __name__ == "__main__":
    QKVRoPESimple.test()