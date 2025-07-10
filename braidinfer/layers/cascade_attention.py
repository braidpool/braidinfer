"""
Cascade Attention implementation with Online Softmax for handling chunked KV cache.

This implementation replaces the current concatenation-based approach with 
true online softmax algorithm as described in ARCHITECTURE.md.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from braidinfer.chunks import Chunk
from braidinfer.layers.rotary_embedding import apply_rotary_emb
from braidinfer.kernels import online_softmax_update


@dataclass
class CascadeLevel:
    """Represents one level in the cascade hierarchy."""
    chunks: List[Chunk]
    kv_page_indices: torch.Tensor  # Page indices for this level
    kv_page_indptr: torch.Tensor   # Boundaries in page indices
    kv_last_page_len: torch.Tensor # Valid tokens in last page of each chunk
    position_offset: int            # Global position offset for this level
    

class CascadeAttention:
    """
    Cascade attention mechanism with online softmax algorithm.
    
    This implementation processes chunks sequentially without concatenation,
    maintaining running softmax statistics (m_i, l_i, acc_i) for numerical stability.
    """
    
    def __init__(self, num_heads: int, head_dim: int, scale: float, 
                 num_kv_heads: int, page_size: int = 16, rotary_emb = None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        
        # For GQA support
        self.heads_per_kv = num_heads // num_kv_heads
        
        # Store RoPE instance for differential RoPE
        self.rotary_emb = rotary_emb
        
    def forward(self, 
                query: torch.Tensor,           # [num_tokens * num_heads * head_dim] or [num_tokens, num_heads, head_dim]
                kv_cache: torch.Tensor,        # [num_pages, 2, page_size, num_kv_heads, head_dim]
                cascade_levels: List[CascadeLevel],
                layer_idx: int,
                causal_mask: bool = True,
                k_norm=None) -> torch.Tensor:
        """
        Perform cascade attention using online softmax algorithm.
        
        This implementation processes chunks sequentially without concatenation,
        maintaining running softmax statistics (m_i, l_i, acc_i) for numerical stability.
        
        Args:
            query: Query tensor
            kv_cache: Global KV cache containing all pages
            cascade_levels: List of cascade levels with chunk information
            layer_idx: Current layer index
            causal_mask: Whether to apply causal masking
            
        Returns:
            Attention output [num_tokens * num_heads * head_dim]
        """
        
        # Handle different query shapes
        if query.dim() == 1:
            # Flat shape: [num_tokens * num_heads * head_dim]
            total_size = query.shape[0]
            expected_size = self.num_heads * self.head_dim
            
            if total_size % expected_size == 0:
                # This is num_tokens * (num_heads * head_dim)
                batch_size = total_size // expected_size
                query = query.view(batch_size, self.num_heads, self.head_dim)
            else:
                raise ValueError(f"Query size {total_size} not divisible by num_heads*head_dim={expected_size}")
        elif query.dim() == 2:
            # Shape: [num_tokens, num_heads * head_dim]
            batch_size = query.shape[0]
            query = query.view(batch_size, self.num_heads, self.head_dim)
        elif query.dim() == 3:
            # Already in correct shape: [num_tokens, num_heads, head_dim]
            batch_size = query.shape[0]
        else:
            raise ValueError(f"Unexpected query shape: {query.shape}")
        
        # Transpose query for efficient computation: [num_heads, batch_size, head_dim]
        query = query.transpose(0, 1)
        
        # Initialize online softmax state for each query head
        # m_i: running maximum, l_i: running sum, acc_i: weighted value accumulator
        device = query.device
        dtype = query.dtype
        
        m_i = torch.full((self.num_heads, batch_size), float('-inf'), dtype=torch.float32, device=device)
        l_i = torch.zeros((self.num_heads, batch_size), dtype=torch.float32, device=device)
        acc_i = torch.zeros((self.num_heads, batch_size, self.head_dim), dtype=torch.float32, device=device)
        
        # Track cumulative position across all processed tokens for causal masking
        cumulative_tokens_processed = 0
        
        # Calculate total context length (for query position in causal masking)
        total_context_length = sum(c.kv_length for level in cascade_levels for c in level.chunks)
        
        # Process each cascade level sequentially
        for level_idx, level in enumerate(cascade_levels):
            if not level.chunks:
                continue
                
            # Process each chunk in this level
            for chunk_idx, chunk in enumerate(level.chunks):
                if not chunk.page_table or chunk.kv_length == 0:
                    continue
                
                # Get page information for this chunk
                chunk_start_idx = level.kv_page_indptr[chunk_idx].item()
                chunk_end_idx = level.kv_page_indptr[chunk_idx + 1].item()
                chunk_pages = level.kv_page_indices[chunk_start_idx:chunk_end_idx]
                
                # Process each page in this chunk
                for local_page_idx, page_idx in enumerate(chunk_pages):
                    page_idx = page_idx.item()
                    
                    # Calculate valid tokens on this page
                    if local_page_idx == len(chunk_pages) - 1:
                        tokens_on_page = level.kv_last_page_len[chunk_idx].item()
                    else:
                        tokens_on_page = self.page_size
                    
                    if tokens_on_page <= 0:
                        continue
                    
                    # Extract K and V for this page
                    page_k = kv_cache[page_idx, 0, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    page_v = kv_cache[page_idx, 1, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    
                    # Transpose to [tokens, num_kv_heads, head_dim]
                    page_k = page_k.transpose(0, 1)
                    page_v = page_v.transpose(0, 1)
                    
                    # Apply differential RoPE if needed
                    if self.rotary_emb is not None and hasattr(chunk, 'global_position_start'):
                        tokens_before_page = local_page_idx * self.page_size
                        cached_pos_start = getattr(chunk, 'cached_position_start', 0)
                        global_pos_start = getattr(chunk, 'global_position_start', cached_pos_start)
                        
                        cached_positions = torch.arange(
                            cached_pos_start + tokens_before_page,
                            cached_pos_start + tokens_before_page + tokens_on_page,
                            dtype=torch.int64, device=page_k.device
                        )
                        global_positions = torch.arange(
                            global_pos_start + tokens_before_page,
                            global_pos_start + tokens_before_page + tokens_on_page,
                            dtype=torch.int64, device=page_k.device
                        )
                        
                        if not torch.equal(cached_positions, global_positions):
                            pos_diff = global_positions - cached_positions
                            original_shape = page_k.shape
                            page_k_reshaped = page_k.reshape(tokens_on_page, self.num_kv_heads * self.head_dim)
                            dummy_q = torch.zeros_like(page_k_reshaped)
                            _, page_k_rotated = self.rotary_emb(pos_diff, dummy_q, page_k_reshaped)
                            page_k = page_k_rotated.reshape(original_shape)
                    
                    # Handle GQA: expand KV heads to match query heads
                    if self.num_kv_heads != self.num_heads:
                        heads_per_kv = self.num_heads // self.num_kv_heads
                        page_k = page_k.repeat_interleave(heads_per_kv, dim=1)  # [tokens, num_heads, head_dim]
                        page_v = page_v.repeat_interleave(heads_per_kv, dim=1)  # [tokens, num_heads, head_dim]
                    
                    # Transpose for batched matrix multiplication
                    page_k = page_k.transpose(0, 1)  # [num_heads, tokens, head_dim]
                    page_v = page_v.transpose(0, 1)  # [num_heads, tokens, head_dim]
                    
                    # Calculate global positions for queries and keys
                    # Query positions: assume queries are at the end of context (for generation)
                    query_positions = torch.full((batch_size,), total_context_length, 
                                                dtype=torch.int64, device=device)
                    
                    # Key positions for this page: start from cumulative position + page offset
                    tokens_before_page = local_page_idx * self.page_size
                    key_pos_start = cumulative_tokens_processed + tokens_before_page
                    key_positions = torch.arange(
                        key_pos_start,
                        key_pos_start + tokens_on_page,
                        dtype=torch.int64, device=device
                    )
                    
                    # Use high-performance Triton kernel for online softmax update
                    online_softmax_update(
                        query=query,                    # [num_heads, batch_size, head_dim]
                        key=page_k,                     # [num_heads, tokens_on_page, head_dim]
                        value=page_v,                   # [num_heads, tokens_on_page, head_dim]
                        m_i=m_i,                        # [num_heads, batch_size] (modified in-place)
                        l_i=l_i,                        # [num_heads, batch_size] (modified in-place)
                        acc_i=acc_i,                    # [num_heads, batch_size, head_dim] (modified in-place)
                        query_positions=query_positions, # [batch_size]
                        key_positions=key_positions,     # [tokens_on_page]
                        scale=self.scale,
                        apply_causal_mask=causal_mask
                    )
                
                # Update cumulative position counter after processing this chunk
                cumulative_tokens_processed += chunk.kv_length
        
        # Finalize: divide accumulated values by sum for final softmax normalization
        # Avoid division by zero
        l_i_safe = torch.clamp(l_i, min=1e-12)
        output = acc_i / l_i_safe.unsqueeze(-1)  # [num_heads, batch_size, head_dim]
        
        # Convert back to original dtype to match input query
        output = output.to(dtype)
        
        # Transpose back and reshape to expected output format
        output = output.transpose(0, 1).contiguous()  # [batch_size, num_heads, head_dim]
        output = output.view(-1, self.num_heads * self.head_dim)  # [batch_size, num_heads * head_dim]
        
        return output
    

    def create_cascade_levels(self, chunks: List[Chunk], 
                            page_manager) -> List[CascadeLevel]:
        """
        Create cascade levels from a list of chunks.
        
        Simple 2-level hierarchy:
        - Level 0: System prompt chunk
        - Level 1: Context and query chunks
        """
        levels = []
        
        # Separate chunks by type
        system_chunks = [c for c in chunks if c.chunk_type.value == "system_prompt"]
        other_chunks = [c for c in chunks if c.chunk_type.value != "system_prompt"]
        
        # Level 0: System chunks (shared)
        if system_chunks:
            level0 = self._create_level(system_chunks, position_offset=0)
            levels.append(level0)
        
        # Level 1: Other chunks (unique per sequence)
        if other_chunks:
            # Calculate position offset as total tokens in previous levels
            prev_tokens = sum(c.kv_length for c in system_chunks)
            level1 = self._create_level(other_chunks, position_offset=prev_tokens)
            levels.append(level1)
        
        return levels
    
    def _create_level(self, chunks: List[Chunk], position_offset: int) -> CascadeLevel:
        """Create a cascade level from chunks."""
        # Collect page information
        all_page_indices = []
        page_indptr = [0]
        last_page_lens = []
        
        for chunk in chunks:
            if chunk.page_table:
                all_page_indices.extend(chunk.page_table)
                page_indptr.append(len(all_page_indices))
                
                # Calculate last page length
                # If there's only one page, all tokens are on that page
                num_pages = len(chunk.page_table)
                if num_pages == 1:
                    last_page_len = chunk.kv_length
                else:
                    last_page_len = chunk.kv_length % self.page_size
                    if last_page_len == 0 and chunk.kv_length > 0:
                        last_page_len = self.page_size
                last_page_lens.append(last_page_len)
            else:
                # Empty chunk
                page_indptr.append(len(all_page_indices))
                last_page_lens.append(0)
        
        return CascadeLevel(
            chunks=chunks,
            kv_page_indices=torch.tensor(all_page_indices, dtype=torch.int32, device="cuda"),
            kv_page_indptr=torch.tensor(page_indptr, dtype=torch.int32, device="cuda"),
            kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32, device="cuda"),
            position_offset=position_offset
        )