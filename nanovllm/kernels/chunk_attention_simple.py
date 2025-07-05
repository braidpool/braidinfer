"""
Simplified chunk-aware attention kernel for initial testing.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional
import math


@triton.jit
def chunk_decode_attention_kernel_simple(
    # Query
    q_ptr, q_stride_h, q_stride_d,
    # KV cache (concatenated)
    k_ptr, v_ptr,
    kv_stride_pos, kv_stride_h, kv_stride_d,
    # Output
    out_ptr, out_stride_h, out_stride_d,
    # Softmax workspace
    m_ptr, l_ptr,
    # Dimensions
    seq_len,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr, 
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified chunk attention kernel - processes concatenated KV cache.
    """
    # Program ID = attention head
    head_idx = tl.program_id(0)
    
    # Map query head to KV head for GQA
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector
    q_offs = head_idx * q_stride_h + tl.arange(0, head_dim) * q_stride_d
    q = tl.load(q_ptr + q_offs)
    
    # Initialize softmax statistics
    m_i = -float('inf')  # max
    l_i = 0.0  # sum of exp
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    # Process KV cache in blocks
    for block_start in range(0, seq_len, BLOCK_SIZE):
        # Load K block
        block_size = tl.minimum(BLOCK_SIZE, seq_len - block_start)
        k_ptrs = k_ptr + (block_start + tl.arange(0, BLOCK_SIZE)) * kv_stride_pos + kv_head_idx * kv_stride_h
        
        # Compute scores for this block
        scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for i in range(block_size):
            if block_start + i < seq_len:
                # Load K vector
                k_offs = k_ptrs[i] + tl.arange(0, head_dim) * kv_stride_d
                k = tl.load(k_offs)
                
                # Compute dot product
                score = tl.sum(q * k, axis=0) * scale
                scores[i] = score
        
        # Update softmax statistics
        m_ij = tl.max(scores, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exponentials
        exp_scores = tl.exp(scores - m_i_new)
        l_ij = tl.sum(exp_scores, axis=0)
        
        # Update running statistics
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + l_ij
        
        # Load V block and accumulate
        v_ptrs = v_ptr + (block_start + tl.arange(0, BLOCK_SIZE)) * kv_stride_pos + kv_head_idx * kv_stride_h
        
        for i in range(block_size):
            if block_start + i < seq_len:
                v_offs = v_ptrs[i] + tl.arange(0, head_dim) * kv_stride_d
                v = tl.load(v_offs)
                
                # Scale old accumulator
                acc = acc * alpha
                
                # Add weighted V
                acc += exp_scores[i] * v
        
        m_i = m_i_new
    
    # Normalize
    acc = acc / l_i
    
    # Store output
    out_offs = head_idx * out_stride_h + tl.arange(0, head_dim) * out_stride_d
    tl.store(out_ptr + out_offs, acc.to(tl.float16))


def chunk_attention_simple(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Simple chunk attention for testing.
    
    Args:
        query: [1, num_heads, head_dim]
        k_cache: [seq_len, num_kv_heads, head_dim]
        v_cache: [seq_len, num_kv_heads, head_dim]
        scale: Attention scale
        
    Returns:
        output: [1, num_heads, head_dim]
    """
    batch_size, num_heads, head_dim = query.shape
    seq_len, num_kv_heads, _ = k_cache.shape
    
    assert batch_size == 1
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Allocate output and workspace
    output = torch.empty_like(query)
    m = torch.empty(num_heads, dtype=torch.float32, device=query.device)
    l = torch.empty(num_heads, dtype=torch.float32, device=query.device)
    
    # Launch kernel
    BLOCK_SIZE = 64
    grid = (num_heads,)
    
    chunk_decode_attention_kernel_simple[grid](
        # Query
        query, query.stride(1), query.stride(2),
        # KV cache
        k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        # Output
        output, output.stride(1), output.stride(2),
        # Workspace
        m, l,
        # Dimensions
        seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def benchmark_simple():
    """Benchmark the simple kernel."""
    # Qwen3-0.6B config
    num_heads = 14
    num_kv_heads = 2  
    head_dim = 64
    seq_len = 512
    
    # Create test data
    query = torch.randn(1, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        output = chunk_attention_simple(query, k_cache, v_cache)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    start.record()
    for _ in range(num_iters):
        output = chunk_attention_simple(query, k_cache, v_cache)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters
    
    print(f"Simple attention kernel time: {elapsed_ms:.3f} ms")
    print(f"Theoretical throughput: {1000/elapsed_ms:.1f} tok/s")
    
    # Compare with PyTorch
    scale = 1.0 / math.sqrt(head_dim)
    
    # Expand K/V for GQA
    k_expanded = k_cache.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_expanded = v_cache.repeat_interleave(num_heads // num_kv_heads, dim=1)
    
    # PyTorch attention
    scores = torch.matmul(query, k_expanded.transpose(1, 2)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(attn_weights, v_expanded.transpose(0, 1))
    
    # Check correctness
    max_diff = torch.max(torch.abs(output - pytorch_output)).item()
    print(f"Max difference from PyTorch: {max_diff}")
    
    return output


if __name__ == "__main__":
    benchmark_simple()