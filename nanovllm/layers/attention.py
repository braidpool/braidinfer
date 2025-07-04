"""
Attention layer for nano-vllm using paged attention.
"""

import torch
from torch import nn
import flashinfer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext


class Attention(nn.Module):
    """Attention layer using paged attention."""
    
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
        
        # Reference to paged KV cache will be set by model runner
        self.kv_cache = None
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                context: 'InferenceContext'):
        """Forward pass using FlashInfer paged attention."""
        # Reshape to [seq_len, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Append K/V to cache
        if context.page_manager is not None:
            context.page_manager.append_kv_to_cache(
                self.layer_idx, k, v, 
                context.sequences,
                context.is_prefill
            )
        
        # Get the appropriate wrapper for this layer
        wrapper = context.get_wrapper(self.layer_idx)
        if wrapper is None:
            raise RuntimeError(f"Wrapper not initialized for layer {self.layer_idx}")
        
        # Get layer-specific KV cache
        if context.page_manager is not None:
            layer_kv_cache = context.page_manager.get_layer_kv_cache(self.layer_idx)
        else:
            layer_kv_cache = self.kv_cache
        
        # Run attention
        output = wrapper.run(q, layer_kv_cache)
        
        # Reshape output back to [seq_len, hidden_dim]
        output = output.view(-1, self.num_heads * self.head_dim)
        return output