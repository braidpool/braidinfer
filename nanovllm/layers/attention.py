"""
Attention layer for nano-vllm using paged attention.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.page_manager import PageManager


class Attention(nn.Module):
    """
    Attention layer that handles paged KV cache.

    This implementation includes a temporary fallback for standard prefill/decode
    that converts the paged KV cache to a continuous tensor before performing
    attention. This is inefficient but ensures correctness.
    """
    
    def __init__(self,
                 num_heads: int,
                 head_dim: int, 
                 scale: float,
                 num_kv_heads: int,
                 layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx
        self.kv_cache = None # Set by ModelRunner

    def _get_past_kv(self, seq: 'Sequence', page_manager: 'PageManager', context: Optional['InferenceContext'] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Gathers the paged KV cache for a single sequence into continuous tensors."""
        k_cache_blocks = []
        v_cache_blocks = []
        
        layer_kv_cache = self.kv_cache
        
        # Check if we have active chunks - if so, gather KV from chunks instead
        if context and hasattr(seq, 'active_chunks') and seq.active_chunks:
            # Gather KV cache from chunks
            for chunk in seq.active_chunks:
                if chunk.page_table and chunk.kv_length > 0:
                    # Process each page in the chunk
                    for i, page_idx in enumerate(chunk.page_table):
                        # Calculate how many tokens are on this page
                        start_pos = i * page_manager.page_size
                        end_pos = min((i + 1) * page_manager.page_size, chunk.kv_length)
                        tokens_on_page = end_pos - start_pos
                        
                        if tokens_on_page > 0:
                            page = layer_kv_cache[page_idx]
                            k_page = page[0, :, :tokens_on_page, :]
                            v_page = page[1, :, :tokens_on_page, :]
                            
                            k_page = k_page.transpose(0, 1)
                            v_page = v_page.transpose(0, 1)
                            
                            k_cache_blocks.append(k_page)
                            v_cache_blocks.append(v_page)
            
            if not k_cache_blocks:
                return None, None
                
            k_cache_cont = torch.cat(k_cache_blocks, dim=0)
            v_cache_cont = torch.cat(v_cache_blocks, dim=0)
            
            return k_cache_cont, v_cache_cont
        
        # Standard path - get KV from sequence pages
        num_cached_tokens = page_manager.seq_lengths.get(seq.seq_id, 0)
        
        if num_cached_tokens == 0:
            return None, None

        num_pages_for_seq = (num_cached_tokens + page_manager.page_size - 1) // page_manager.page_size
        
        for i in range(num_pages_for_seq):
            page_idx = seq.block_table[i]
            
            tokens_on_page = num_cached_tokens - i * page_manager.page_size
            if tokens_on_page > page_manager.page_size:
                tokens_on_page = page_manager.page_size

            page = layer_kv_cache[page_idx]
            k_page = page[0, :, :tokens_on_page, :]
            v_page = page[1, :, :tokens_on_page, :]
            
            k_page = k_page.transpose(0, 1)
            v_page = v_page.transpose(0, 1)

            k_cache_blocks.append(k_page)
            v_cache_blocks.append(v_page)

        if not k_cache_blocks:
            return None, None
            
        k_cache_cont = torch.cat(k_cache_blocks, dim=0)
        v_cache_cont = torch.cat(v_cache_blocks, dim=0)
        
        return k_cache_cont, v_cache_cont

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                context: 'InferenceContext'):
        """Forward pass for attention."""
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if context.page_manager is not None:
            if context.chunk_id is not None:
                context.page_manager.append_kv_to_cache_for_chunk(
                    self.layer_idx, k, v, context.chunk_id, context.chunk_positions
                )
            else:
                context.page_manager.append_kv_to_cache(
                    self.layer_idx, k, v, context.sequences, context.is_prefill
                )

        if context.chunk_id is not None:
            return torch.zeros(q.shape[0], self.num_heads * self.head_dim, dtype=q.dtype, device=q.device)

        if len(context.sequences) > 1:
             raise NotImplementedError("Batched attention with paged-to-continuous conversion not implemented.")
        
        seq = context.sequences[0]
        
        past_k, past_v = self._get_past_kv(seq, context.page_manager, context)
        
        if past_k is not None:
            k_full = torch.cat([past_k, k], dim=0)
            v_full = torch.cat([past_v, v], dim=0)
        else:
            k_full = k
            v_full = v

        if self.num_kv_heads != self.num_heads:
            heads_per_kv = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(heads_per_kv, dim=1)
            v_full = v_full.repeat_interleave(heads_per_kv, dim=1)

        q = q.transpose(0, 1)
        k_full = k_full.transpose(0, 1)
        v_full = v_full.transpose(0, 1)

        scores = torch.bmm(q, k_full.transpose(1, 2)) * self.scale

        if context.is_prefill:
            q_len = q.shape[1]
            kv_len = k_full.shape[1]
            causal_mask = torch.triu(torch.ones(q_len, kv_len, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask.bool(), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.bmm(attn_weights, v_full)

        output = output.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        
        return output
