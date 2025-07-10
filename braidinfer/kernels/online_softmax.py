"""
High-performance Triton kernel for online softmax with causal masking.

This kernel replaces the Python for-loop in cascade attention's _update_online_softmax
method with a GPU-optimized implementation that correctly handles causal masking.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def online_softmax_kernel(
    # Input tensors
    query_ptr,        # [num_heads, batch_size, head_dim]
    key_ptr,          # [num_heads, num_tokens, head_dim]
    value_ptr,        # [num_heads, num_tokens, head_dim]
    # Running state tensors (modified in-place)
    m_i_ptr,          # [num_heads, batch_size] - running maximum
    l_i_ptr,          # [num_heads, batch_size] - running sum
    acc_i_ptr,        # [num_heads, batch_size, head_dim] - running accumulator
    # Position information for causal masking
    query_positions_ptr,  # [batch_size] - global positions of queries
    key_positions_ptr,    # [num_tokens] - global positions of keys
    # Dimensions
    num_heads,
    batch_size,
    num_tokens,
    head_dim,
    # Strides
    query_stride_head,
    query_stride_batch,
    query_stride_dim,
    key_stride_head,
    key_stride_token,
    key_stride_dim,
    value_stride_head,
    value_stride_token,
    value_stride_dim,
    m_i_stride_head,
    m_i_stride_batch,
    l_i_stride_head,
    l_i_stride_batch,
    acc_i_stride_head,
    acc_i_stride_batch,
    acc_i_stride_dim,
    # Hyperparameters
    scale: tl.constexpr,
    apply_causal_mask: tl.constexpr,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
):
    """
    Correct online softmax kernel with proper parallelization.
    
    Parallelizes over queries (head, batch), processes tokens sequentially.
    This ensures no race conditions and correct online softmax updates.
    """
    # Program IDs - parallel over queries only
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Check bounds
    if pid_head >= num_heads:
        return
    if pid_batch >= batch_size:
        return
    
    # Load query position for causal masking
    query_pos_ptr = query_positions_ptr + pid_batch
    query_pos = tl.load(query_pos_ptr)
    
    # Load current running state into registers
    m_i_ptr_curr = m_i_ptr + pid_head * m_i_stride_head + pid_batch * m_i_stride_batch
    l_i_ptr_curr = l_i_ptr + pid_head * l_i_stride_head + pid_batch * l_i_stride_batch
    
    m_i = tl.load(m_i_ptr_curr)
    l_i = tl.load(l_i_ptr_curr)
    
    # Optimized path: Load entire accumulator into registers once (common case)
    if head_dim <= BLOCK_DIM:
        # Load entire accumulator into registers once
        acc_mask = tl.arange(0, BLOCK_DIM) < head_dim
        acc_ptrs = (acc_i_ptr +
                   pid_head * acc_i_stride_head +
                   pid_batch * acc_i_stride_batch +
                   tl.arange(0, BLOCK_DIM) * acc_i_stride_dim)
        acc_register = tl.load(acc_ptrs, mask=acc_mask, other=0.0)
        
        # Process all tokens with accumulator in registers
        for token_idx in range(num_tokens):
            key_pos_ptr = key_positions_ptr + token_idx
            key_pos = tl.load(key_pos_ptr)
            
            should_process = True
            if apply_causal_mask:
                should_process = key_pos <= query_pos
            
            if should_process:
                # Load query and key vectors
                query_ptrs = (query_ptr +
                             pid_head * query_stride_head +
                             pid_batch * query_stride_batch +
                             tl.arange(0, BLOCK_DIM) * query_stride_dim)
                query_vec = tl.load(query_ptrs, mask=acc_mask, other=0.0)
                
                key_ptrs = (key_ptr +
                           pid_head * key_stride_head +
                           token_idx * key_stride_token +
                           tl.arange(0, BLOCK_DIM) * key_stride_dim)
                key_vec = tl.load(key_ptrs, mask=acc_mask, other=0.0)
                
                # Compute complete attention score
                score = tl.sum(query_vec * key_vec) * scale
                
                # Online softmax update
                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(score - m_new)
                
                l_i = l_i * alpha + beta
                m_i = m_new
                
                # Update accumulator in registers
                value_ptrs = (value_ptr +
                             pid_head * value_stride_head +
                             token_idx * value_stride_token +
                             tl.arange(0, BLOCK_DIM) * value_stride_dim)
                value_vec = tl.load(value_ptrs, mask=acc_mask, other=0.0)
                
                acc_register = alpha * acc_register + beta * value_vec
        
        # Store final accumulator back to global memory once
        tl.store(acc_ptrs, acc_register, mask=acc_mask)
        
    else:
        # Multi-block approach for larger head dimensions  
        # Sequential loop over all tokens for this query
        for token_idx in range(num_tokens):
            # Load key position for causal masking
            key_pos_ptr = key_positions_ptr + token_idx
            key_pos = tl.load(key_pos_ptr)
            
            # Apply causal masking: only process if key_pos <= query_pos
            should_process = True
            if apply_causal_mask:
                should_process = key_pos <= query_pos
            
            if should_process:
                # Compute complete dot product across all head dimensions
                score = 0.0
                for d_start in range(0, head_dim, BLOCK_DIM):
                    d_end = min(d_start + BLOCK_DIM, head_dim)
                    d_mask = tl.arange(0, BLOCK_DIM) < (d_end - d_start)
                    
                    # Load query block
                    query_ptrs = (query_ptr +
                                 pid_head * query_stride_head +
                                 pid_batch * query_stride_batch +
                                 (d_start + tl.arange(0, BLOCK_DIM)) * query_stride_dim)
                    query_block = tl.load(query_ptrs, mask=d_mask, other=0.0)
                    
                    # Load key block
                    key_ptrs = (key_ptr +
                               pid_head * key_stride_head +
                               token_idx * key_stride_token +
                               (d_start + tl.arange(0, BLOCK_DIM)) * key_stride_dim)
                    key_block = tl.load(key_ptrs, mask=d_mask, other=0.0)
                    
                    # Accumulate dot product
                    score += tl.sum(query_block * key_block)
                
                # Apply scaling
                score = score * scale
                
                # Online softmax update
                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(score - m_new)
                
                # Update running state
                l_i = l_i * alpha + beta
                m_i = m_new
                
                # Update accumulator across all head dimensions
                for d_start in range(0, head_dim, BLOCK_DIM):
                    d_end = min(d_start + BLOCK_DIM, head_dim)
                    d_mask = tl.arange(0, BLOCK_DIM) < (d_end - d_start)
                    
                    # Load current accumulator block
                    acc_ptrs = (acc_i_ptr +
                               pid_head * acc_i_stride_head +
                               pid_batch * acc_i_stride_batch +
                               (d_start + tl.arange(0, BLOCK_DIM)) * acc_i_stride_dim)
                    acc_block = tl.load(acc_ptrs, mask=d_mask, other=0.0)
                    
                    # Load value block
                    value_ptrs = (value_ptr +
                                 pid_head * value_stride_head +
                                 token_idx * value_stride_token +
                                 (d_start + tl.arange(0, BLOCK_DIM)) * value_stride_dim)
                    value_block = tl.load(value_ptrs, mask=d_mask, other=0.0)
                    
                    # Update accumulator block: acc_new = alpha * acc_old + beta * value
                    acc_new_block = alpha * acc_block + beta * value_block
                    
                    # Store updated accumulator block
                    tl.store(acc_ptrs, acc_new_block, mask=d_mask)
    
    # Store final updated state back to global memory
    tl.store(m_i_ptr_curr, m_i)
    tl.store(l_i_ptr_curr, l_i)


def online_softmax_update(
    query: torch.Tensor,           # [num_heads, batch_size, head_dim]
    key: torch.Tensor,             # [num_heads, num_tokens, head_dim]  
    value: torch.Tensor,           # [num_heads, num_tokens, head_dim]
    m_i: torch.Tensor,             # [num_heads, batch_size] - running maximum
    l_i: torch.Tensor,             # [num_heads, batch_size] - running sum
    acc_i: torch.Tensor,           # [num_heads, batch_size, head_dim] - accumulator
    query_positions: torch.Tensor, # [batch_size] - global positions of queries
    key_positions: torch.Tensor,   # [num_tokens] - global positions of keys
    scale: float,
    apply_causal_mask: bool = True,
) -> None:
    """
    High-performance online softmax update using Triton kernel.
    
    Updates m_i, l_i, and acc_i in-place with new key-value pairs.
    
    Args:
        query: Query tensor [num_heads, batch_size, head_dim]
        key: Key tensor for current page [num_heads, num_tokens, head_dim]
        value: Value tensor for current page [num_heads, num_tokens, head_dim]
        m_i: Running maximum [num_heads, batch_size] (modified in-place)
        l_i: Running sum [num_heads, batch_size] (modified in-place)
        acc_i: Running accumulator [num_heads, batch_size, head_dim] (modified in-place)
        query_positions: Global positions of queries [batch_size]
        key_positions: Global positions of keys [num_tokens]
        scale: Attention scale factor (1/sqrt(head_dim))
        apply_causal_mask: Whether to apply causal masking
    """
    # Input validation
    assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3
    assert query.shape[0] == key.shape[0] == value.shape[0]  # num_heads
    assert query.shape[2] == key.shape[2] == value.shape[2]  # head_dim
    assert key.shape[1] == value.shape[1]  # num_tokens
    assert query_positions.shape[0] == query.shape[1]  # batch_size
    assert key_positions.shape[0] == key.shape[1]  # num_tokens
    
    num_heads, batch_size, head_dim = query.shape
    num_tokens = key.shape[1]
    
    # Ensure all tensors are contiguous for optimal memory access
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    m_i = m_i.contiguous()
    l_i = l_i.contiguous()
    acc_i = acc_i.contiguous()
    query_positions = query_positions.contiguous()
    key_positions = key_positions.contiguous()
    
    # Block size for head dimension processing - must be power of 2 and >= head_dim
    BLOCK_DIM = triton.next_power_of_2(head_dim)
    
    # Launch kernel with 2D grid: parallel over queries only
    grid = (num_heads, batch_size)
    
    online_softmax_kernel[grid](
        # Input tensors
        query, key, value,
        # State tensors (modified in-place)
        m_i, l_i, acc_i,
        # Position tensors
        query_positions, key_positions,
        # Dimensions
        num_heads, batch_size, num_tokens, head_dim,
        # Strides - query
        query.stride(0), query.stride(1), query.stride(2),
        # Strides - key
        key.stride(0), key.stride(1), key.stride(2),
        # Strides - value
        value.stride(0), value.stride(1), value.stride(2),
        # Strides - state
        m_i.stride(0), m_i.stride(1),
        l_i.stride(0), l_i.stride(1),
        acc_i.stride(0), acc_i.stride(1), acc_i.stride(2),
        # Hyperparameters
        scale=scale,
        apply_causal_mask=apply_causal_mask,
        # Block sizes
        BLOCK_DIM=BLOCK_DIM,
    )