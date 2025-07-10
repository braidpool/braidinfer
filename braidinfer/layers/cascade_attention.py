"""
Cascade Attention implementation for handling chunked KV cache with proper position encodings.

This implementation is inspired by FlashInfer's cascade attention but adapted for
our architecture without external dependencies.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from braidinfer.chunks import Chunk
from braidinfer.layers.rotary_embedding import apply_rotary_emb


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
        k_full = torch.cat(all_keys, dim=0)    # [total_kv_len, num_kv_heads, head_dim]
        v_full = torch.cat(all_values, dim=0)  # [total_kv_len, num_kv_heads, head_dim]
        
        # K normalization is already applied when storing to cache
        # No need to apply it again here - this ensures consistency
        
        # Handle GQA exactly like standard attention
        if self.num_kv_heads != self.num_heads:
            heads_per_kv = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(heads_per_kv, dim=1)
            v_full = v_full.repeat_interleave(heads_per_kv, dim=1)
        
        # Transpose exactly like standard attention for bmm
        # Standard: q.transpose(0, 1), k_full.transpose(0, 1), v_full.transpose(0, 1)
        q = query.transpose(0, 1)              # [num_heads, batch, head_dim]  
        k_full = k_full.transpose(0, 1)        # [num_heads, total_kv_len, head_dim]
        v_full = v_full.transpose(0, 1)        # [num_heads, total_kv_len, head_dim]
        
        # Compute scores exactly like standard attention
        scores = torch.bmm(q, k_full.transpose(1, 2)) * self.scale  # [num_heads, batch, total_kv_len]
        
        # Apply causal mask exactly like standard attention
        if causal_mask:
            q_len = q.shape[1]      # batch_size
            kv_len = k_full.shape[1]  # total_kv_len
            causal_mask_matrix = torch.triu(torch.ones(q_len, kv_len, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask_matrix.bool(), float('-inf'))
        
        # Softmax exactly like standard attention
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply attention exactly like standard attention
        output = torch.bmm(attn_weights, v_full)  # [num_heads, batch, head_dim]
        
        # Transpose back and reshape exactly like standard attention
        output = output.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        
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
                    
                    
                    # The shape appears to be [num_pages, 2, num_kv_heads, page_size, head_dim]
                    # So we need to extract [num_kv_heads, tokens, head_dim] and transpose
                    page_k = kv_cache[page_idx, 0, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    page_v = kv_cache[page_idx, 1, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                    
                    # Transpose to [tokens, num_kv_heads, head_dim]
                    page_k = page_k.transpose(0, 1)
                    page_v = page_v.transpose(0, 1)
                    
                    # ENABLED: Differential RoPE for position correction
                    if self.rotary_emb is not None and hasattr(chunk, 'global_position_start'):
                        # Calculate positions for this page
                        tokens_before_page = local_page_idx * self.page_size
                        cached_pos_start = getattr(chunk, 'cached_position_start', 0)
                        global_pos_start = getattr(chunk, 'global_position_start', cached_pos_start)
                        
                        # Create position tensors
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
                        
                        # Only apply differential RoPE if positions differ
                        if not torch.equal(cached_positions, global_positions):
                            # Use the model's rotary embedding to compute differential rotation
                            # We need to "unapply" the cached rotation and "reapply" for global positions
                            
                            # First, get the differential rotation: global rotation minus cached rotation
                            # This is equivalent to rotating by (global_pos - cached_pos) amount
                            pos_diff = global_positions - cached_positions
                            
                            # Apply the differential rotation using the model's RoPE
                            # This is more robust than manual cos/sin manipulation
                            original_shape = page_k.shape
                            page_k_reshaped = page_k.reshape(tokens_on_page, self.num_kv_heads * self.head_dim)
                            
                            # Create a dummy query of the same shape to use rotary_emb.forward()
                            dummy_q = torch.zeros_like(page_k_reshaped)
                            
                            # Apply differential rotation using the existing RoPE implementation
                            _, page_k_rotated = self.rotary_emb(pos_diff, dummy_q, page_k_reshaped)
                            
                            # Reshape back to original shape
                            page_k = page_k_rotated.reshape(original_shape)
                    
                    k_blocks.append(page_k)
                    v_blocks.append(page_v)
                    
                    # Calculate positions for tracking (use global positions after differential RoPE)
                    page_start_pos = local_page_idx * self.page_size
                    for i in range(tokens_on_page):
                        global_pos_start = getattr(chunk, 'global_position_start', 
                                                 getattr(chunk, 'cached_position_start', 0))
                        tokens_before_page = local_page_idx * self.page_size
                        pos_to_use = global_pos_start + tokens_before_page + i
                        positions.append(pos_to_use)
        
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