"""
FlashInfer-based attention layer for nano-vllm.
Uses FlashInfer's native paged attention APIs.
"""

import torch
from torch import nn
import flashinfer
from nanovllm.utils.context import get_context


class FlashInferAttention(nn.Module):
    """Attention layer using FlashInfer's paged attention."""
    
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
        
        # FlashInfer wrappers will be set by the model runner
        self.prefill_wrapper = None
        self.decode_wrapper = None
        
        # Reference to paged KV cache will be set by model runner
        self.kv_cache = None
        
        # References to model runner components
        self.model_runner = None
        self.page_manager = None
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass using FlashInfer paged attention."""
        # Reshape to [seq_len, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        context = get_context()
        
        # First, append K/V to cache
        if self.model_runner is not None and hasattr(self.model_runner, 'current_seqs'):
            self.page_manager.append_kv_to_cache(
                self.layer_idx, k, v, 
                self.model_runner.current_seqs,
                self.model_runner.current_is_prefill
            )
        
        if context.is_prefill:
            # Prefill uses the wrapper's run method directly
            if self.prefill_wrapper is None:
                raise RuntimeError("Prefill wrapper not initialized")
            
            # Run prefill attention
            output = self.prefill_wrapper.run(q, self.kv_cache)
            
        else:
            # Decode stage
            if self.decode_wrapper is None:
                raise RuntimeError("Decode wrapper not initialized")
            
            # Run decode attention
            output = self.decode_wrapper.run(q, self.kv_cache)
        
        # Reshape output back to [seq_len, hidden_dim]
        output = output.view(-1, self.num_heads * self.head_dim)
        return output