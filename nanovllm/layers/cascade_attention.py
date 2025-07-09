"""
Cascade Attention implementation for handling chunked KV cache with proper position encodings.

This implementation is inspired by FlashInfer's cascade attention but adapted for
our architecture without external dependencies.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from nanovllm.chunks import Chunk


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
    Cascade attention mechanism for multi-level KV cache with proper position handling.
    
    Key concepts:
    - Level 0: Shared chunks (e.g., system prompt) used by all sequences
    - Level 1: Unique chunks per sequence
    - Proper position encoding handling across levels
    - Softmax normalization across all levels
    """
    
    def __init__(self, num_heads: int, head_dim: int, scale: float, 
                 num_kv_heads: int, page_size: int = 16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        
        # For GQA support
        self.heads_per_kv = num_heads // num_kv_heads
        
    def forward(self, 
                query: torch.Tensor,           # [num_tokens * num_heads * head_dim] or [num_tokens, num_heads, head_dim]
                kv_cache: torch.Tensor,        # [num_pages, 2, page_size, num_kv_heads, head_dim]
                cascade_levels: List[CascadeLevel],
                layer_idx: int,
                causal_mask: bool = True) -> torch.Tensor:
        """
        Perform cascade attention across multiple levels of KV cache.
        
        Args:
            query: Query tensor
            kv_cache: Global KV cache containing all pages
            cascade_levels: List of cascade levels with chunk information
            layer_idx: Current layer index
            causal_mask: Whether to apply causal masking
            
        Returns:
            Attention output [num_tokens * num_heads * head_dim]
        """
        # Debug disabled
        pass
        
        # Handle different query shapes
        if query.dim() == 1:
            # Flat shape: [num_tokens * num_heads * head_dim]
            # Qwen3 passes queries like [512] where 512 = 1 * 16 * 32
            # We need to figure out the actual dimensions
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
        
        # Debug KV cache shape disabled
        pass
        
        # Collect all K and V tensors from cascade levels
        all_keys = []
        all_values = []
        all_positions = []
        
        for level_idx, level in enumerate(cascade_levels):
            # Gather KV for this level
            level_k, level_v, level_positions = self._gather_level_kv(
                kv_cache, level, layer_idx
            )
            
            if level_k is not None:
                all_keys.append(level_k)
                all_values.append(level_v)
                all_positions.extend(level_positions)
        
        if not all_keys:
            # No KV cache available
            return torch.zeros_like(query).reshape(batch_size, -1)
        
        # Concatenate all levels
        keys = torch.cat(all_keys, dim=0)      # [total_kv_len, num_kv_heads, head_dim]
        values = torch.cat(all_values, dim=0)  # [total_kv_len, num_kv_heads, head_dim]
        
        # Debug disabled
        pass
        
        # Handle GQA by repeating KV heads
        if self.heads_per_kv > 1:
            keys = keys.repeat_interleave(self.heads_per_kv, dim=1)
            values = values.repeat_interleave(self.heads_per_kv, dim=1)
        
        # Reshape for batch matrix multiply
        # Q: [batch, num_heads, head_dim]
        # K: [total_kv_len, num_heads, head_dim]  
        # V: [total_kv_len, num_heads, head_dim]
        
        # Compute attention scores using einsum for clarity
        # Q: [batch, num_heads, head_dim]
        # K: [total_kv_len, num_heads, head_dim]
        # Result: [batch, num_heads, total_kv_len]
        scores = torch.einsum('bhd,khd->bhk', query, keys) * self.scale
        
        # Apply causal mask if needed
        if causal_mask and batch_size > 0:
            # For chunked generation, we need to handle positions differently
            # The query positions are the NEW tokens being generated
            # They should be able to attend to all previous KV positions
            
            # Get the maximum KV position (last position in the KV cache)
            if all_positions:
                max_kv_pos = max(all_positions)
                # Query positions start after the last KV position
                query_positions = torch.arange(
                    max_kv_pos + 1, max_kv_pos + 1 + batch_size, 
                    device=query.device
                )
            else:
                # No KV cache, use positions starting from 0
                query_positions = torch.arange(batch_size, device=query.device)
            
            # Debug disabled
            pass
            
            kv_positions = torch.tensor(all_positions, device=query.device) if all_positions else torch.tensor([], device=query.device)
            
            # Create causal mask
            if kv_positions.numel() > 0:
                # Mask where query_pos < kv_pos (can't attend to future)
                # Shape: [batch, total_kv_len]
                mask = query_positions.unsqueeze(1) < kv_positions.unsqueeze(0)
                # Expand mask for all heads: [batch, 1, total_kv_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
                scores.masked_fill_(mask, float('-inf'))
        
        # Softmax over all KV positions
        # scores: [batch, num_heads, total_kv_len]
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply attention to values
        # attn_weights: [batch, num_heads, total_kv_len]
        # values: [total_kv_len, num_heads, head_dim]
        # Result: [batch, num_heads, head_dim]
        output = torch.einsum('bhk,khd->bhd', attn_weights, values)
        
        # Output is already in correct shape: [batch, num_heads, head_dim]
        output = output.contiguous()
        
        # Flatten to match input shape  
        # output shape is [batch_size, num_heads * head_dim]
        output = output.view(batch_size, -1)
        
        # Debug disabled
        pass
        
        # Return 2D tensor [batch_size, hidden_dim] to match standard attention
        return output
    
    def _gather_level_kv(self, kv_cache: torch.Tensor, level: CascadeLevel, 
                        layer_idx: int) -> Tuple[Optional[torch.Tensor], 
                                                Optional[torch.Tensor], 
                                                List[int]]:
        """
        Gather K and V tensors from a cascade level.
        
        Returns:
            - Keys tensor [num_tokens, num_kv_heads, head_dim]
            - Values tensor [num_tokens, num_kv_heads, head_dim]
            - List of global positions for each token
        """
        if not level.chunks:
            return None, None, []
        
        # Debug disabled
        pass
        
        k_blocks = []
        v_blocks = []
        positions = []
        
        for chunk_idx, chunk in enumerate(level.chunks):
            if not chunk.page_table or chunk.kv_length == 0:
                continue
            
            # Get pages for this chunk
            chunk_start_idx = level.kv_page_indptr[chunk_idx].item()
            chunk_end_idx = level.kv_page_indptr[chunk_idx + 1].item()
            chunk_pages = level.kv_page_indices[chunk_start_idx:chunk_end_idx]
            
            # Gather KV from pages
            for local_page_idx, page_idx in enumerate(chunk_pages):
                page_idx = page_idx.item()
                
                # Calculate valid tokens on this page
                if local_page_idx == len(chunk_pages) - 1:
                    # Last page - use last_page_len
                    tokens_on_page = level.kv_last_page_len[chunk_idx].item()
                else:
                    tokens_on_page = self.page_size
                
                # Debug disabled
                pass
                
                if tokens_on_page > 0:
                    # Extract K and V for this page
                    # The actual KV cache shape is: [num_pages, 2, page_size, num_kv_heads, head_dim]
                    # But it seems like the shape is: [num_pages, 2, num_kv_heads, page_size, head_dim]
                    # Let me check both possibilities
                    
                    # Debug disabled
                    pass
                    
                    # The shape appears to be [num_pages, 2, num_kv_heads, page_size, head_dim]
                    # So we need to extract [num_kv_heads, tokens, head_dim] and transpose
                    page_k = kv_cache[page_idx, 0, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    page_v = kv_cache[page_idx, 1, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    
                    # Transpose to [tokens, num_kv_heads, head_dim]
                    page_k = page_k.transpose(0, 1)
                    page_v = page_v.transpose(0, 1)
                    
                    # Debug disabled
                    pass
                    
                    k_blocks.append(page_k)
                    v_blocks.append(page_v)
                    
                    # Calculate global positions for these tokens
                    page_start_pos = local_page_idx * self.page_size
                    for i in range(tokens_on_page):
                        # Use chunk's global position information if available
                        if hasattr(chunk, 'global_position_start'):
                            # Use explicit global positions
                            # Don't add page_start_pos again - it's already accounted for
                            # in how we iterate through tokens
                            tokens_before_page = local_page_idx * self.page_size
                            global_pos = chunk.global_position_start + tokens_before_page + i
                        else:
                            # Fallback to offset-based calculation
                            global_pos = level.position_offset + chunk.position_offset + page_start_pos + i
                        positions.append(global_pos)
            
            # Debug disabled
            pass
        
        if not k_blocks:
            return None, None, []
        
        # Concatenate all blocks
        keys = torch.cat(k_blocks, dim=0)
        values = torch.cat(v_blocks, dim=0)
        
        return keys, values, positions
    
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