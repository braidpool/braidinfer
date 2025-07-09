"""
Chunk-aware attention kernel with online softmax algorithm.

This implements the correct algorithm that accumulates statistics
across all chunks for proper attention normalization.
"""

import torch
import triton
import triton.language as tl
from typing import List, Tuple, Optional
import math


@triton.jit
def chunk_decode_attention_online_kernel(
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
    Chunk-based decode attention with online softmax algorithm.
    
    This kernel correctly accumulates statistics across all chunks
    to compute the proper attention distribution.
    
    The online softmax algorithm maintains:
    - m_i: running maximum score
    - l_i: running sum of exp(score - m_i)
    - acc_i: running weighted sum of values
    
    This allows numerically stable computation without materializing
    all attention scores.
    """
    # Get head index
    head_idx = tl.program_id(0)
    
    # Map query head to KV head (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector for this head
    q_offset = head_idx * head_dim + tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset).to(tl.float32)
    
    # Initialize online softmax statistics
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum of exp
    acc_i = tl.zeros([head_dim], dtype=tl.float32)  # Running weighted sum
    
    # Process all chunks
    for chunk_idx in range(num_chunks):
        chunk_start = tl.load(chunk_starts_ptr + chunk_idx)
        chunk_len = tl.load(chunk_lens_ptr + chunk_idx)
        
        # Skip empty chunks by processing in conditional
        if chunk_len > 0:
            # Process chunk in blocks for better memory access
            for block_start in range(0, chunk_len, BLOCK_SIZE):
                block_end = tl.minimum(block_start + BLOCK_SIZE, chunk_len)
                block_size = block_end - block_start
                
                # Process each position in the block
                for pos_idx in range(block_size):
                    kv_pos = chunk_start + block_start + pos_idx
                    
                    # Load K vector
                    k_offset = kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                    k = tl.zeros([head_dim], dtype=tl.float32)
                    for d in range(head_dim):
                        k[d] = tl.load(k_cache_ptr + k_offset + d * kv_stride_d).to(tl.float32)
                    
                    # Compute attention score
                    score = tl.sum(q * k) * scale
                    
                    # Update online softmax statistics
                    m_j = tl.maximum(m_i, score)
                    
                    # Update accumulated values with renormalization
                    if m_j > m_i:
                        # Renormalize previous accumulation
                        correction = tl.exp(m_i - m_j)
                        l_i = l_i * correction
                        acc_i = acc_i * correction
                        m_i = m_j
                    
                    # Add contribution from current position
                    exp_score = tl.exp(score - m_i)
                    l_i += exp_score
                    
                    # Load V vector and accumulate
                    v_offset = kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                    v = tl.zeros([head_dim], dtype=tl.float32)
                    for d in range(head_dim):
                        v[d] = tl.load(v_cache_ptr + v_offset + d * kv_stride_d).to(tl.float32)
                    
                    acc_i += exp_score * v
    
    # Final normalization
    output = acc_i / l_i
    
    # Store output
    out_offset = head_idx * head_dim + tl.arange(0, head_dim)
    tl.store(out_ptr + out_offset, output.to(tl.float16))


@triton.jit
def chunk_decode_attention_online_vectorized_kernel(
    # Query input
    q_ptr, q_stride_h,
    # Chunk KV cache pointers (flattened)
    k_cache_ptr, v_cache_ptr,
    kv_stride_pos, kv_stride_h, kv_stride_d,
    # Chunk metadata
    chunk_starts_ptr, chunk_lens_ptr, 
    num_chunks,
    # Output
    out_ptr, out_stride_h,
    # Dimensions
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    # Config
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Vectorized version of chunk decode attention with online softmax.
    
    This version processes the head dimension in blocks for better performance.
    """
    # Get head index
    head_idx = tl.program_id(0)
    
    # Map query head to KV head (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector for this head
    q_offset = head_idx * head_dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_D)
    q_mask = tl.arange(0, BLOCK_D) < head_dim
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Initialize online softmax statistics
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # Running weighted sum
    
    # Process all chunks
    for chunk_idx in range(num_chunks):
        chunk_start = tl.load(chunk_starts_ptr + chunk_idx)
        chunk_len = tl.load(chunk_lens_ptr + chunk_idx)
        
        # Skip empty chunks by processing in conditional
        if chunk_len > 0:
            # Process each position in the chunk
            for pos_idx in range(chunk_len):
                kv_pos = chunk_start + pos_idx
                
                # Load K vector
                k_base = k_cache_ptr + kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                k_ptrs = k_base + tl.arange(0, BLOCK_D) * kv_stride_d
                k = tl.load(k_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                
                # Compute attention score
                score = tl.sum(q * k, axis=0) * scale
                
                # Update online softmax statistics
                m_j = tl.maximum(m_i, score)
                
                # Update accumulated values with renormalization
                if m_j > m_i:
                    # Renormalize previous accumulation
                    correction = tl.exp(m_i - m_j)
                    l_i = l_i * correction
                    acc = acc * correction
                    m_i = m_j
                
                # Add contribution from current position
                exp_score = tl.exp(score - m_i)
                l_i += exp_score
                
                # Load V vector and accumulate
                v_base = v_cache_ptr + kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                v_ptrs = v_base + tl.arange(0, BLOCK_D) * kv_stride_d
                v = tl.load(v_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                
                acc += exp_score * v
    
    # Final normalization
    output = acc / l_i
    
    # Store output
    out_offset = head_idx * head_dim
    out_ptrs = out_ptr + out_offset + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs, output.to(tl.float16), mask=q_mask)


@triton.jit
def paged_chunk_attention_kernel(
    # Query input
    q_ptr, q_stride_h,
    # Global paged KV cache
    kv_cache_ptr,
    # Paged KV metadata
    paged_kv_indices_ptr,  # Flat list of page numbers
    paged_kv_indptr_ptr,   # Chunk boundaries in indices
    paged_kv_last_page_len_ptr,  # Last page lengths
    # KV cache strides
    kv_stride_layer, kv_stride_page, kv_stride_kv, kv_stride_h, kv_stride_pos, kv_stride_d,
    # Output
    out_ptr, out_stride_h,
    # Parameters
    layer_idx: tl.constexpr,
    page_size: tl.constexpr,
    num_chunks,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Paged version of chunk decode attention with online softmax.
    
    This kernel operates directly on the paged KV cache without any memory copies.
    It implements a two-step address calculation to access tokens in pages.
    """
    # Get head index
    head_idx = tl.program_id(0)
    
    # Map query head to KV head (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector for this head
    q_offset = head_idx * head_dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_D)
    q_mask = tl.arange(0, BLOCK_D) < head_dim
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Initialize online softmax statistics
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # Running weighted sum
    
    # Process all chunks
    for chunk_idx in range(num_chunks):
        # Get chunk boundaries
        chunk_start = tl.load(paged_kv_indptr_ptr + chunk_idx)
        chunk_end = tl.load(paged_kv_indptr_ptr + chunk_idx + 1)
        num_pages = chunk_end - chunk_start
        
        if num_pages > 0:
            # Get last page length for this chunk
            last_page_len = tl.load(paged_kv_last_page_len_ptr + chunk_idx)
            
            # Process all pages in this chunk
            for page_idx in range(num_pages):
                # Get global page number
                global_page_num = tl.load(paged_kv_indices_ptr + chunk_start + page_idx)
                
                # Determine how many tokens in this page
                if page_idx < num_pages - 1:
                    tokens_in_page = page_size
                else:
                    tokens_in_page = last_page_len
                
                # Process each position in the page
                for pos in range(tokens_in_page):
                    # Calculate K address
                    k_base = kv_cache_ptr + (
                        layer_idx * kv_stride_layer +
                        global_page_num * kv_stride_page +
                        0 * kv_stride_kv +  # 0 for K
                        kv_head_idx * kv_stride_h +
                        pos * kv_stride_pos
                    )
                    
                    # Load K vector
                    k_ptrs = k_base + tl.arange(0, BLOCK_D) * kv_stride_d
                    k = tl.load(k_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                    
                    # Compute attention score
                    score = tl.sum(q * k, axis=0) * scale
                    
                    # Update online softmax statistics
                    m_j = tl.maximum(m_i, score)
                    
                    # Update accumulated values with renormalization
                    if m_j > m_i:
                        # Renormalize previous accumulation
                        correction = tl.exp(m_i - m_j)
                        l_i = l_i * correction
                        acc = acc * correction
                        m_i = m_j
                    
                    # Add contribution from current position
                    exp_score = tl.exp(score - m_i)
                    l_i += exp_score
                    
                    # Calculate V address
                    v_base = kv_cache_ptr + (
                        layer_idx * kv_stride_layer +
                        global_page_num * kv_stride_page +
                        1 * kv_stride_kv +  # 1 for V
                        kv_head_idx * kv_stride_h +
                        pos * kv_stride_pos
                    )
                    
                    # Load V vector
                    v_ptrs = v_base + tl.arange(0, BLOCK_D) * kv_stride_d
                    v = tl.load(v_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                    
                    # Accumulate weighted V
                    acc += exp_score * v
    
    # Final normalization
    if l_i > 0:
        output = acc / l_i
    else:
        output = acc
    
    # Store output
    out_offset = head_idx * head_dim
    out_ptrs = out_ptr + out_offset + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs, output.to(out_ptr.dtype.element_ty), mask=q_mask)
    
    # Map query head to KV head (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load query vector for this head
    q_offset = head_idx * head_dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_D)
    q_mask = tl.arange(0, BLOCK_D) < head_dim
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Initialize online softmax statistics
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # Running weighted sum
    
    # Process all chunks
    for chunk_idx in range(num_chunks):
        chunk_start = tl.load(chunk_starts_ptr + chunk_idx)
        chunk_len = tl.load(chunk_lens_ptr + chunk_idx)
        
        # Skip empty chunks by processing in conditional
        if chunk_len > 0:
            # Process each position in the chunk
            for pos_idx in range(chunk_len):
                kv_pos = chunk_start + pos_idx
                
                # Load K vector
                k_base = k_cache_ptr + kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                k_ptrs = k_base + tl.arange(0, BLOCK_D) * kv_stride_d
                k = tl.load(k_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                
                # Compute attention score
                score = tl.sum(q * k, axis=0) * scale
                
                # Update online softmax statistics
                m_j = tl.maximum(m_i, score)
                
                # Update accumulated values with renormalization
                if m_j > m_i:
                    # Renormalize previous accumulation
                    correction = tl.exp(m_i - m_j)
                    l_i = l_i * correction
                    acc = acc * correction
                    m_i = m_j
                
                # Add contribution from current position
                exp_score = tl.exp(score - m_i)
                l_i += exp_score
                
                # Load V vector and accumulate
                v_base = v_cache_ptr + kv_pos * kv_stride_pos + kv_head_idx * kv_stride_h
                v_ptrs = v_base + tl.arange(0, BLOCK_D) * kv_stride_d
                v = tl.load(v_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                
                acc += exp_score * v
    
    # Final normalization
    output = acc / l_i
    
    # Store output
    out_offset = head_idx * head_dim
    out_ptrs = out_ptr + out_offset + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs, output.to(tl.float16), mask=q_mask)


class ChunkAttentionOnline:
    """
    Wrapper for chunk-based attention with online softmax.
    """
    
    def __init__(self, head_dim: int, scale: Optional[float] = None):
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))
    
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
            chunk_levels: Cascade level of each chunk (unused but kept for compatibility)
            scale: Attention scale (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output [1, num_heads, head_dim]
        """
        batch_size, num_heads, head_dim = query.shape
        assert batch_size == 1, "This kernel is optimized for batch size 1"
        
        # Determine number of KV heads from first chunk
        num_kv_heads = chunk_k_caches[0].shape[1] if chunk_k_caches else num_heads
        
        # Default scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # Concatenate all chunks into contiguous memory
        total_positions = sum(chunk_lengths)
        k_cache_concat = torch.empty(
            (total_positions, num_kv_heads, head_dim),
            dtype=torch.float16,
            device=query.device
        )
        v_cache_concat = torch.empty_like(k_cache_concat)
        
        # Prepare chunk metadata
        num_chunks = len(chunk_k_caches)
        chunk_starts = torch.zeros(num_chunks, dtype=torch.int32, device=query.device)
        chunk_lens = torch.tensor(chunk_lengths, dtype=torch.int32, device=query.device)
        
        # Copy chunks to contiguous memory
        pos = 0
        for i, (k_chunk, v_chunk, length) in enumerate(
            zip(chunk_k_caches, chunk_v_caches, chunk_lengths)
        ):
            if length > 0:
                k_cache_concat[pos:pos+length] = k_chunk[:length]
                v_cache_concat[pos:pos+length] = v_chunk[:length]
                chunk_starts[i] = pos
                pos += length
        
        # Allocate output
        output = torch.empty_like(query)
        
        # Launch kernel
        grid = (num_heads,)
        
        # Choose block size for head dimension
        BLOCK_D = triton.next_power_of_2(head_dim)
        
        # Use vectorized kernel
        chunk_decode_attention_online_vectorized_kernel[grid](
            # Query
            query, query.stride(1),
            # KV caches
            k_cache_concat, v_cache_concat,
            k_cache_concat.stride(0), k_cache_concat.stride(1), k_cache_concat.stride(2),
            # Chunk metadata
            chunk_starts, chunk_lens,
            num_chunks,
            # Output
            output, output.stride(1),
            # Dimensions
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            # Config
            scale=scale,
            BLOCK_D=BLOCK_D,
        )
        
        return output
    
    def decode_attention_paged(
        self,
        query: torch.Tensor,  # [num_heads, head_dim]
        chunks: List['Chunk'],  # List of chunks with page tables
        kv_cache: torch.Tensor,  # Global paged KV cache
        layer_idx: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Run chunk-based decode attention directly on paged KV cache.
        
        Args:
            query: Query tensor [num_heads, head_dim]
            chunks: List of chunks with page tables
            kv_cache: Global paged KV cache tensor
            layer_idx: Current layer index
            page_size: Size of each page in tokens
            
        Returns:
            Attention output [num_heads, head_dim]
        """
        num_heads = query.shape[0]
        head_dim = query.shape[1]
        device = query.device
        
        # Build metadata tensors
        paged_kv_indices = []
        paged_kv_indptr = [0]
        paged_kv_last_page_len = []
        
        for chunk in chunks:
            if chunk.page_table is None or len(chunk.page_table) == 0:
                # Empty chunk
                paged_kv_indptr.append(paged_kv_indptr[-1])
                paged_kv_last_page_len.append(0)
            else:
                # Add page indices
                paged_kv_indices.extend(chunk.page_table)
                paged_kv_indptr.append(len(paged_kv_indices))
                
                # Calculate last page length
                total_tokens = chunk.kv_length
                full_pages = total_tokens // page_size
                last_page_len = total_tokens % page_size
                if last_page_len == 0 and total_tokens > 0:
                    last_page_len = page_size
                paged_kv_last_page_len.append(last_page_len)
        
        # Convert to tensors
        paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=device)
        
        # Allocate output tensor
        output = torch.empty_like(query)
        
        # Get strides
        q_stride_h = query.stride(0)
        out_stride_h = output.stride(0)
        
        # KV cache strides: [num_layers, num_pages, 2, num_heads, page_size, head_dim]
        kv_strides = kv_cache.stride()
        kv_stride_layer = kv_strides[0]
        kv_stride_page = kv_strides[1]
        kv_stride_kv = kv_strides[2]
        kv_stride_h = kv_strides[3]
        kv_stride_pos = kv_strides[4]
        kv_stride_d = kv_strides[5]
        
        # Determine number of KV heads from cache shape
        num_kv_heads = kv_cache.shape[3]
        
        # Launch kernel
        grid = (num_heads,)
        BLOCK_D = triton.next_power_of_2(head_dim)
        
        paged_chunk_attention_kernel[grid](
            # Query
            query, q_stride_h,
            # Global paged KV cache
            kv_cache,
            # Paged KV metadata
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_len,
            # KV cache strides
            kv_stride_layer, kv_stride_page, kv_stride_kv, kv_stride_h, kv_stride_pos, kv_stride_d,
            # Output
            output, out_stride_h,
            # Parameters
            layer_idx=layer_idx,
            page_size=page_size,
            num_chunks=len(chunks),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=self.scale,
            BLOCK_D=BLOCK_D,
        )
        
        return output


def test_online_vs_naive():
    """Test that online softmax produces same results as naive implementation."""
    import numpy as np
    
    # Test configuration
    num_heads = 4
    num_kv_heads = 1
    head_dim = 32
    
    # Create test data
    query = torch.randn(1, num_heads, head_dim, dtype=torch.float16, device='cuda')
    
    # Create chunks
    chunk_configs = [(10, 0), (20, 1), (15, 1), (5, 2)]
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
    
    # Run online kernel
    output_online = ChunkAttentionOnline.decode_attention(
        query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels
    )
    
    # Compute naive reference
    # Concatenate all K and V
    k_all = torch.cat([k[:l] for k, l in zip(chunk_k_caches, chunk_lengths)], dim=0)
    v_all = torch.cat([v[:l] for v, l in zip(chunk_v_caches, chunk_lengths)], dim=0)
    
    # Expand for GQA
    num_kv_groups = num_heads // num_kv_heads
    k_expanded = k_all.repeat_interleave(num_kv_groups, dim=1)
    v_expanded = v_all.repeat_interleave(num_kv_groups, dim=1)
    
    # Compute attention
    scale = 1.0 / math.sqrt(head_dim)
    # query is [1, num_heads, head_dim], k_expanded is [seq_len, num_heads, head_dim]
    # Need to transpose k for matmul: [1, num_heads, head_dim] @ [1, num_heads, head_dim, seq_len]
    scores = torch.matmul(query, k_expanded.transpose(0, 1).transpose(1, 2)) * scale  # [1, num_heads, seq_len]
    attn_weights = torch.softmax(scores, dim=-1)
    # v_expanded is [seq_len, num_heads, head_dim], need [1, num_heads, seq_len, head_dim]
    v_expanded_t = v_expanded.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
    output_naive = torch.matmul(attn_weights.unsqueeze(-2), v_expanded_t).squeeze(-2)  # [1, num_heads, head_dim]
    
    # Debug shapes
    print(f"Query shape: {query.shape}")
    print(f"K all shape: {k_all.shape}")
    print(f"V all shape: {v_all.shape}")
    print(f"K expanded shape: {k_expanded.shape}")
    print(f"V expanded shape: {v_expanded.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Output online shape: {output_online.shape}")
    print(f"Output naive shape: {output_naive.shape}")
    
    # Check a specific head
    print(f"\nHead 0 comparison:")
    print(f"Online: {output_online[0, 0, :5]}")
    print(f"Naive: {output_naive[0, 0, :5]}")
    
    # Compare
    max_diff = torch.max(torch.abs(output_online - output_naive)).item()
    mean_diff = torch.mean(torch.abs(output_online - output_naive)).item()
    
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # More lenient for half precision
    assert max_diff < 0.1, f"Max difference too large: {max_diff}"
    print("✓ Online softmax test passed!")


def benchmark_online_kernel():
    """Benchmark the online chunk attention kernel."""
    # Test configuration matching Qwen3-0.5B
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
        output = ChunkAttentionOnline.decode_attention(
            query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels
        )
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    start.record()
    for _ in range(num_iters):
        output = ChunkAttentionOnline.decode_attention(
            query, chunk_k_caches, chunk_v_caches, chunk_lengths, chunk_levels
        )
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters
    
    print(f"Online chunk attention kernel time: {elapsed_ms:.3f} ms")
    print(f"Theoretical throughput: {1000/elapsed_ms:.1f} tok/s")
    
    return output


if __name__ == "__main__":
    print("Testing online softmax correctness...")
    test_online_vs_naive()
    print("\nBenchmarking online kernel...")
    benchmark_online_kernel()