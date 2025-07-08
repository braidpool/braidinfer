"""
Grouped Query Attention (GQA) implementation for Qwen3.
Handles attention computation with proper KV cache integration.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GQAAttention:
    """Grouped Query Attention implementation that works with cascade attention."""
    
    @staticmethod
    def compute_gqa(
        q: torch.Tensor,  # [seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
        num_heads: int,
        num_kv_heads: int,
        scale: float,
        kv_cache_k: Optional[torch.Tensor] = None,  # [past_len, num_kv_heads, head_dim]
        kv_cache_v: Optional[torch.Tensor] = None,  # [past_len, num_kv_heads, head_dim]
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute grouped query attention with optional KV cache.
        
        Returns:
            - attention output [seq_len, num_heads * head_dim]
            - updated k cache (if kv_cache provided)
            - updated v cache (if kv_cache provided)
        """
        seq_len = q.shape[0]
        head_dim = q.shape[2]
        
        # Concatenate with KV cache if provided
        if kv_cache_k is not None and kv_cache_v is not None:
            # Append new K/V to cache
            k_full = torch.cat([kv_cache_k, k], dim=0)  # [past_len + seq_len, num_kv_heads, head_dim]
            v_full = torch.cat([kv_cache_v, v], dim=0)
        else:
            k_full = k
            v_full = v
        
        # Compute attention scores with GQA
        # Each KV head serves multiple Q heads
        heads_per_kv = num_heads // num_kv_heads
        
        # Reshape Q for batch processing
        q = q.transpose(0, 1)  # [num_heads, seq_len, head_dim]
        
        # Expand K/V to match Q heads
        k_expanded = k_full.repeat_interleave(heads_per_kv, dim=1)  # [full_len, num_heads, head_dim]
        v_expanded = v_full.repeat_interleave(heads_per_kv, dim=1)
        
        k_expanded = k_expanded.transpose(0, 1)  # [num_heads, full_len, head_dim]
        v_expanded = v_expanded.transpose(0, 1)
        
        # Compute attention scores
        scores = torch.bmm(q, k_expanded.transpose(1, 2)) * scale  # [num_heads, seq_len, full_len]
        
        # Apply causal mask if needed
        if is_causal:
            full_len = k_expanded.shape[1]
            if kv_cache_k is not None:
                # For cached attention, mask based on position
                past_len = kv_cache_k.shape[0]
                mask = torch.ones(seq_len, full_len, device=scores.device, dtype=torch.bool)
                # Each query position can attend to all past + its position
                for i in range(seq_len):
                    mask[i, past_len + i + 1:] = False
                mask = mask.unsqueeze(0)  # [1, seq_len, full_len]
            else:
                # Standard causal mask
                mask = torch.triu(torch.ones(seq_len, full_len, device=scores.device, dtype=torch.bool), diagonal=1)
                mask = mask.unsqueeze(0)
            
            scores.masked_fill_(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attn_weights, v_expanded)  # [num_heads, seq_len, head_dim]
        
        # Transpose and reshape - note this is Q heads, not KV heads!
        output = output.transpose(0, 1).contiguous()  # [seq_len, num_heads, head_dim]
        output = output.view(seq_len, -1)  # Let PyTorch infer the size
        
        # Return updated caches if provided
        if kv_cache_k is not None:
            return output, k_full, v_full
        else:
            return output, None, None
    
    @staticmethod
    def compute_cascade_gqa(
        q: torch.Tensor,  # [seq_len, num_heads, head_dim]
        k_chunks: list[torch.Tensor],  # List of K chunks
        v_chunks: list[torch.Tensor],  # List of V chunks
        num_heads: int,
        num_kv_heads: int,
        scale: float,
        chunk_types: list[str],  # 'shared' or 'unique' for each chunk
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute GQA with cascade attention over multiple context chunks.
        
        Args:
            q: Query tensor
            k_chunks: List of key tensors for each context chunk
            v_chunks: List of value tensors for each context chunk
            chunk_types: Type of each chunk ('shared' or 'unique')
        """
        seq_len = q.shape[0]
        head_dim = q.shape[2]
        heads_per_kv = num_heads // num_kv_heads
        
        # Concatenate all K/V chunks
        k_full = torch.cat(k_chunks, dim=0)
        v_full = torch.cat(v_chunks, dim=0)
        
        # Track chunk boundaries for masking
        chunk_boundaries = [0]
        for k_chunk in k_chunks:
            chunk_boundaries.append(chunk_boundaries[-1] + k_chunk.shape[0])
        
        # Reshape for attention computation
        q = q.transpose(0, 1)  # [num_heads, seq_len, head_dim]
        
        # Expand K/V for GQA
        k_expanded = k_full.repeat_interleave(heads_per_kv, dim=1).transpose(0, 1)
        v_expanded = v_full.repeat_interleave(heads_per_kv, dim=1).transpose(0, 1)
        
        # Compute attention scores
        scores = torch.bmm(q, k_expanded.transpose(1, 2)) * scale
        
        # Apply cascade-aware masking
        if is_causal:
            mask = torch.zeros(seq_len, k_full.shape[0], device=scores.device, dtype=torch.bool)
            
            for i in range(seq_len):
                for chunk_idx, chunk_type in enumerate(chunk_types):
                    start = chunk_boundaries[chunk_idx]
                    end = chunk_boundaries[chunk_idx + 1]
                    
                    if chunk_type == 'shared':
                        # Can attend to all shared context
                        mask[i, start:end] = False
                    else:
                        # Causal mask for unique chunks
                        # Can only attend up to current position
                        attend_up_to = min(start + i + 1, end)
                        mask[i, start:attend_up_to] = False
                        mask[i, attend_up_to:end] = True
            
            scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
        
        # Softmax and apply attention
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, v_expanded)
        
        # Reshape output
        output = output.transpose(0, 1).contiguous()
        output = output.view(seq_len, num_heads * head_dim)
        
        return output