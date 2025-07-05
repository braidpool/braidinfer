"""
Chunk-aware attention kernels for batch size 1 optimization.

This module implements custom Triton kernels that process chunks
efficiently while maintaining the cascade attention structure.

Now uses online softmax algorithm for correct attention computation.
"""

import torch
import triton
import triton.language as tl
from typing import List, Tuple, Optional
import math

# Import the online softmax implementation
try:
    from .chunk_attention_online import ChunkAttentionOnline
except ImportError:
    # For standalone testing
    from chunk_attention_online import ChunkAttentionOnline


@triton.jit
def chunk_decode_attention_kernel(
    # Query input
    q_ptr, q_stride_h,
    # Chunk KV cache pointers (flattened)
    k_cache_ptr, v_cache_ptr,
    kv_stride_chunk, kv_stride_pos, kv_stride_h, kv_stride_d,
    # Chunk metadata
    chunk_starts_ptr, chunk_lens_ptr, chunk_levels_ptr,
    num_chunks,
    # Output
    out_ptr, out_stride_h,
    # Dimensions
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    # Config
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused chunk-based decode attention for batch size 1.
    
    Processes multiple chunks with different cascade levels in a single kernel.
    Optimized for the decode phase where we have a single query token.
    
    Args:
        q_ptr: Query tensor [1, num_heads, head_dim]
        k_cache_ptr: Chunk K caches [total_positions, num_kv_heads, head_dim]
        v_cache_ptr: Chunk V caches [total_positions, num_kv_heads, head_dim]
        chunk_starts_ptr: Start position of each chunk in KV cache
        chunk_lens_ptr: Length of each chunk
        chunk_levels_ptr: Cascade level of each chunk (0=shared, 1+=unique)
        num_chunks: Total number of chunks
        out_ptr: Output tensor [1, num_heads, head_dim]
        scale: Attention scale factor (1/sqrt(head_dim))
    """
    # Get head index
    head_idx = tl.program_id(0)
    
    # Map query head to KV head (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector for this head
    q_offset = head_idx * head_dim + tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset)
    
    # Initialize accumulator for attention output
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        chunk_start = tl.load(chunk_starts_ptr + chunk_idx)
        chunk_len = tl.load(chunk_lens_ptr + chunk_idx)
        chunk_level = tl.load(chunk_levels_ptr + chunk_idx)
            
        # Compute attention scores for this chunk
        max_score = -float('inf')
        
        # Process chunk in blocks
        for block_start in range(0, chunk_len, BLOCK_SIZE):
            block_end = tl.minimum(block_start + BLOCK_SIZE, chunk_len)
            block_size = block_end - block_start
            
            # Compute scores for this block
            scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for pos in range(block_size):
                kv_pos = chunk_start + block_start + pos
                
                # Load K vector
                k_offset = kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                k = tl.zeros([head_dim], dtype=tl.float16)
                for d in range(head_dim):
                    k[d] = tl.load(k_cache_ptr + k_offset + d)
                
                # Compute dot product
                score = tl.sum(q * k, axis=0) * scale
                scores[pos] = score
                max_score = tl.maximum(max_score, score)
        
        # Numerically stable softmax
        exp_scores = tl.exp(scores - max_score)
        sum_exp = tl.sum(exp_scores, axis=0)
        
        # Apply attention to values
        for block_start in range(0, chunk_len, BLOCK_SIZE):
            block_end = tl.minimum(block_start + BLOCK_SIZE, chunk_len)
            block_size = block_end - block_start
            
            for pos in range(block_size):
                kv_pos = chunk_start + block_start + pos
                score_idx = block_start + pos
                
                # Load V vector
                v_offset = kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                v = tl.zeros([head_dim], dtype=tl.float16)
                for d in range(head_dim):
                    v[d] = tl.load(v_cache_ptr + v_offset + d)
                
                # Apply attention weight
                weight = exp_scores[score_idx] / sum_exp
                acc += weight * v
    
    # Store output
    out_offset = head_idx * head_dim + tl.arange(0, head_dim)
    tl.store(out_ptr + out_offset, acc.to(tl.float16))


class ChunkAttention:
    """
    Wrapper for chunk-based attention kernels.
    Now uses online softmax for correct attention computation.
    """
    
    @staticmethod
    def decode_attention(
        query: torch.Tensor,
        chunk_k_caches: List[torch.Tensor],
        chunk_v_caches: List[torch.Tensor],
        chunk_lengths: List[int],
        chunk_levels: List[int],
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run chunk-based decode attention with online softmax.
        
        Args:
            query: Query tensor [1, num_heads, head_dim]
            chunk_k_caches: List of K cache tensors per chunk
            chunk_v_caches: List of V cache tensors per chunk
            chunk_lengths: Length of each chunk
            chunk_levels: Cascade level of each chunk
            scale: Attention scale (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output [1, num_heads, head_dim]
        """
        # Delegate to the online softmax implementation
        return ChunkAttentionOnline.decode_attention(
            query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels, scale
        )


def benchmark_chunk_attention():
    """Benchmark the chunk attention kernel."""
    # Test configuration matching Qwen3-0.6B
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    
    # Create test data
    query = torch.randn(1, num_heads, head_dim, dtype=torch.float16, device='cuda')
    
    # Create chunks of different sizes
    chunk_configs = [
        (50, 0),   # System prompt (level 0)
        (200, 1),  # Context 1 (level 1)
        (150, 1),  # Context 2 (level 1) 
        (20, 2),   # Query (level 2)
    ]
    
    chunk_k_caches = []
    chunk_v_caches = []
    chunk_lengths = []
    chunk_levels = []
    
    for length, level in chunk_configs:
        k_cache = torch.randn(length, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        v_cache = torch.randn(length, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        chunk_k_caches.append(k_cache)
        chunk_v_caches.append(v_cache)
        chunk_lengths.append(length)
        chunk_levels.append(level)
    
    # Warmup
    for _ in range(10):
        output = ChunkAttention.decode_attention(
            query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels
        )
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    start.record()
    for _ in range(num_iters):
        output = ChunkAttention.decode_attention(
            query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels
        )
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters
    
    print(f"Chunk attention kernel time: {elapsed_ms:.3f} ms")
    print(f"Theoretical throughput: {1000/elapsed_ms:.1f} tok/s")
    
    return output


if __name__ == "__main__":
    benchmark_chunk_attention()