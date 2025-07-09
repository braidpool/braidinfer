"""
Paged chunk attention kernel - separated to avoid compilation issues.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional
import math

from nanovllm.chunks import Chunk


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
    max_pages: tl.constexpr,  # Add max pages as a parameter
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
                    # Calculate K address with bounds checking
                    # Only process if indices are valid
                    valid_page = global_page_num < max_pages
                    if valid_page:
                        k_base = kv_cache_ptr + (
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


class PagedChunkAttention:
    """Wrapper for paged chunk attention kernel."""
    
    def __init__(self, head_dim: int, scale: Optional[float] = None):
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        
    def forward(
        self,
        query: torch.Tensor,  # [num_heads, head_dim]
        chunks: List[Chunk],
        kv_cache: torch.Tensor,  # Global paged KV cache
        layer_idx: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Perform attention computation using chunks' paged KV cache.
        
        Args:
            query: Query tensor [num_heads, head_dim]
            chunks: List of chunks with page tables
            kv_cache: Global paged KV cache tensor
            layer_idx: Current layer index
            page_size: Size of each page in tokens
            
        Returns:
            Output tensor [num_heads, head_dim]
        """
        num_heads = query.shape[0]
        head_dim = query.shape[1]
        device = query.device
        
        # Build metadata tensors
        paged_kv_indices = []
        paged_kv_indptr = [0]
        paged_kv_last_page_len = []
        
        for i, chunk in enumerate(chunks):
            if chunk.page_table is None or len(chunk.page_table) == 0:
                # Empty chunk
                print(f"[DEBUG PagedChunkAttention] Chunk {i} is empty")
                paged_kv_indptr.append(paged_kv_indptr[-1])
                paged_kv_last_page_len.append(0)
            else:
                # Add page indices
                print(f"[DEBUG PagedChunkAttention] Chunk {i}: page_table={chunk.page_table}, kv_length={chunk.kv_length}")
                paged_kv_indices.extend(chunk.page_table)
                paged_kv_indptr.append(len(paged_kv_indices))
                
                # Calculate last page length
                total_tokens = chunk.kv_length
                last_page_len = total_tokens % page_size
                if last_page_len == 0 and total_tokens > 0:
                    last_page_len = page_size
                paged_kv_last_page_len.append(last_page_len)
        
        # Convert to tensors
        if layer_idx == 0:  # Only print for first layer to reduce output
            print(f"[DEBUG PagedChunkAttention] paged_kv_indices list: {paged_kv_indices}")
            print(f"[DEBUG PagedChunkAttention] kv_cache shape: {kv_cache.shape}")
            print(f"[DEBUG PagedChunkAttention] layer_idx: {layer_idx}")
        
        # Check if indices are valid
        if paged_kv_indices:
            max_page_idx = max(paged_kv_indices)
            num_pages = kv_cache.shape[1]
            if layer_idx == 0:
                print(f"[DEBUG PagedChunkAttention] Max page index: {max_page_idx}, Num pages in cache: {num_pages}")
            if max_page_idx >= num_pages:
                print(f"[ERROR] Page index {max_page_idx} exceeds cache size {num_pages}")
                
        # Handle empty case
        if not paged_kv_indices:
            print(f"[DEBUG PagedChunkAttention] No pages to attend to, returning zeros")
            return torch.zeros_like(query)
            
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
        max_pages = kv_cache.shape[1]  # Get max pages from cache shape
        
        if layer_idx == 0:
            # Debug GQA ratio
            print(f"[DEBUG PagedChunkAttention] GQA: num_heads={num_heads}, num_kv_heads={num_kv_heads}, ratio={num_heads // num_kv_heads}")
            
            # Add debug output to check memory layout
            print(f"[DEBUG PagedChunkAttention] Strides:")
            print(f"  kv_stride_layer={kv_stride_layer}")
            print(f"  kv_stride_page={kv_stride_page}")
            print(f"  kv_stride_kv={kv_stride_kv}")
            print(f"  kv_stride_h={kv_stride_h}")
            print(f"  kv_stride_pos={kv_stride_pos}")
            print(f"  kv_stride_d={kv_stride_d}")
            print(f"  max_pages={max_pages}")
            
            # Validate parameters before kernel launch
            print(f"[DEBUG PagedChunkAttention] Launching kernel with:")
            print(f"  num_heads={num_heads}, head_dim={head_dim}")
            print(f"  paged_kv_indices shape: {paged_kv_indices.shape}, values: {paged_kv_indices.tolist()}")
            print(f"  paged_kv_indptr shape: {paged_kv_indptr.shape}, values: {paged_kv_indptr.tolist()}")
            print(f"  paged_kv_last_page_len shape: {paged_kv_last_page_len.shape}, values: {paged_kv_last_page_len.tolist()}")
            print(f"  kv_cache layer slice shape: {kv_cache[layer_idx].shape}")
            print(f"  page_size: {page_size}")
            print(f"  num_chunks: {len(chunks)}")
        
        # Synchronize before kernel launch to ensure KV cache writes are complete
        torch.cuda.synchronize()
        
        # Launch kernel
        grid = (num_heads,)
        BLOCK_D = triton.next_power_of_2(head_dim)
        
        # Get the specific layer's KV cache slice
        layer_kv_cache = kv_cache[layer_idx]
        
        paged_chunk_attention_kernel[grid](
            # Query
            query, q_stride_h,
            # Layer-specific KV cache (not the full cache)
            layer_kv_cache,
            # Paged KV metadata
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_len,
            # KV cache strides (adjusted for layer slice)
            0, kv_stride_page, kv_stride_kv, kv_stride_h, kv_stride_pos, kv_stride_d,
            # Output
            output, out_stride_h,
            # Parameters
            layer_idx=0,  # Now always 0 since we pass layer slice
            page_size=page_size,
            num_chunks=len(chunks),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=self.scale,
            BLOCK_D=BLOCK_D,
            max_pages=max_pages,
        )
        
        # Synchronize after kernel to ensure completion before returning
        torch.cuda.synchronize()
        
        return output