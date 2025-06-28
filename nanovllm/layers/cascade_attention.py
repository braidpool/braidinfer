"""
Cascade attention layer for nano-vllm using FlashInfer's multi-level cascade attention.
Supports compositional context caching with multi-head attention.
"""

import torch
from torch import nn
import flashinfer
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext
    from nanovllm.engine.context_chunks import ContextChunk


class CascadeAttention(nn.Module):
    """
    Multi-level cascade attention layer supporting compositional context.
    
    Uses FlashInfer's MultiLevelCascadeAttentionWrapper to efficiently
    combine attention across multiple context chunks (system prompts, 
    documents, queries) with per-head attention patterns.
    """
    
    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 scale: float,
                 num_kv_heads: int,
                 layer_idx: int,
                 max_cascade_levels: int = 3):
        """
        Initialize cascade attention layer.
        
        Args:
            num_heads: Number of query/output heads
            head_dim: Dimension of each attention head
            scale: Attention scale factor
            num_kv_heads: Number of key/value heads (for GQA)
            layer_idx: Layer index in the model
            max_cascade_levels: Maximum number of cascade levels
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx
        self.max_cascade_levels = max_cascade_levels
        
        # Reference to paged KV cache (set by model runner)
        self.kv_cache = None
        
        # Cascade wrapper will be created per batch
        self.cascade_wrapper = None
        
        # Workspace buffer for cascade attention
        self.workspace_buffer_size = 128 * 1024 * 1024  # 128MB
    
    def create_cascade_wrapper(self, num_levels: int, workspace_buffer: torch.Tensor) -> flashinfer.MultiLevelCascadeAttentionWrapper:
        """Create cascade attention wrapper for the given number of levels."""
        return flashinfer.MultiLevelCascadeAttentionWrapper(
            num_levels=num_levels,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD"
        )
    
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                context: 'InferenceContext') -> torch.Tensor:
        """
        Forward pass using multi-level cascade attention.
        
        Args:
            q: Query tensor [seq_len, hidden_dim]
            k: Key tensor [seq_len, hidden_dim]
            v: Value tensor [seq_len, hidden_dim]
            context: Inference context with cascade information
            
        Returns:
            Attention output [seq_len, hidden_dim]
        """
        # Reshape to [seq_len, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Check if we're using cascade attention
        if hasattr(context, 'cascade_config') and context.cascade_config is not None:
            output = self._cascade_forward(q, k, v, context)
        else:
            # Fall back to regular attention
            output = self._regular_forward(q, k, v, context)
        
        # Reshape output back to [seq_len, hidden_dim]
        output = output.view(-1, self.num_heads * self.head_dim)
        return output
    
    def _cascade_forward(self,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        context: 'InferenceContext') -> torch.Tensor:
        """
        Perform cascade attention across multiple context levels.
        """
        cascade_config = context.cascade_config
        num_levels = cascade_config.num_levels
        
        # Get or create cascade wrapper
        if self.cascade_wrapper is None or self.cascade_wrapper._num_levels != num_levels:
            workspace_buffer = context.get_workspace_buffer(self.workspace_buffer_size)
            self.cascade_wrapper = self.create_cascade_wrapper(num_levels, workspace_buffer)
        
        # For decode phase, use cascade wrapper directly
        if not context.is_prefill:
            # Plan cascade attention
            self.cascade_wrapper.plan(
                qo_indptr_arr=cascade_config.qo_indptr_arr,
                paged_kv_indptr_arr=cascade_config.paged_kv_indptr_arr,
                paged_kv_indices_arr=cascade_config.paged_kv_indices_arr,
                paged_kv_last_page_len=cascade_config.paged_kv_last_page_len_arr,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=cascade_config.page_size,
                causal=False,  # Causal only on last level
                pos_encoding_mode="NONE",  # Can be configured
                sm_scale=self.scale
            )
            
            # Run cascade attention
            output = self.cascade_wrapper.run(q, self.kv_cache)
            return output
        
        # For prefill, we need to handle chunk KV appending
        # This is more complex and may need special handling
        return self._cascade_prefill(q, k, v, context)
    
    def _cascade_prefill(self,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        context: 'InferenceContext') -> torch.Tensor:
        """
        Handle prefill with cascade attention.
        
        For chunks that are being prefilled for the first time,
        we compute their attention states and cache them.
        """
        cascade_config = context.cascade_config
        
        # First, append new K/V to cache if needed
        if context.page_manager is not None:
            context.page_manager.append_kv_to_cache(
                self.layer_idx, k, v,
                context.sequences,
                context.is_prefill
            )
        
        # If we have pre-computed chunk states, we can merge them
        if hasattr(cascade_config, 'chunk_states') and cascade_config.chunk_states:
            # Use merge operations to combine pre-computed states
            return self._merge_chunk_states(q, context)
        
        # Otherwise, fall back to regular cascade attention
        # Plan cascade attention for prefill
        self.cascade_wrapper.plan(
            qo_indptr_arr=cascade_config.qo_indptr_arr,
            paged_kv_indptr_arr=cascade_config.paged_kv_indptr_arr,
            paged_kv_indices_arr=cascade_config.paged_kv_indices_arr,
            paged_kv_last_page_len=cascade_config.paged_kv_last_page_len_arr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=cascade_config.page_size,
            causal=True,  # Causal for prefill
            pos_encoding_mode="NONE",
            sm_scale=self.scale
        )
        
        # Run cascade attention
        output = self.cascade_wrapper.run(q, self.kv_cache)
        return output
    
    def _merge_chunk_states(self,
                           q: torch.Tensor,
                           context: 'InferenceContext') -> torch.Tensor:
        """
        Merge pre-computed chunk attention states.
        
        This is an optimization where chunks with pre-computed
        attention states can be merged efficiently without
        recomputing attention.
        """
        cascade_config = context.cascade_config
        chunk_states = cascade_config.chunk_states
        
        # Start with the first chunk's state
        v_merged, s_merged = chunk_states[0]
        
        # Merge remaining chunk states
        for v_chunk, s_chunk in chunk_states[1:]:
            v_merged, s_merged = flashinfer.merge_state(
                v_merged, s_merged, v_chunk, s_chunk
            )
        
        # Now compute attention with the query against merged states
        # This requires special handling as we need to compute
        # attention between q and the merged KV states
        
        # For now, return to regular cascade path
        # TODO: Implement optimized merge path
        return self._cascade_forward(q, None, None, context)
    
    def _regular_forward(self,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        context: 'InferenceContext') -> torch.Tensor:
        """
        Regular non-cascade attention forward pass.
        Falls back to the standard attention implementation.
        """
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
        
        # Run attention
        output = wrapper.run(q, self.kv_cache)
        return output


class CascadeConfig:
    """Configuration for cascade attention execution."""
    
    def __init__(self,
                 num_levels: int,
                 qo_indptr_arr: List[torch.Tensor],
                 paged_kv_indptr_arr: List[torch.Tensor],
                 paged_kv_indices_arr: List[torch.Tensor],
                 paged_kv_last_page_len_arr: List[torch.Tensor],
                 page_size: int,
                 chunk_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Initialize cascade configuration.
        
        Args:
            num_levels: Number of cascade levels
            qo_indptr_arr: Query/output indptr for each level
            paged_kv_indptr_arr: KV cache indptr for each level
            paged_kv_indices_arr: KV cache page indices for each level
            paged_kv_last_page_len_arr: Last page lengths for each level
            page_size: Page size for paged attention
            chunk_states: Optional pre-computed (V, S) states for chunks
        """
        self.num_levels = num_levels
        self.qo_indptr_arr = qo_indptr_arr
        self.paged_kv_indptr_arr = paged_kv_indptr_arr
        self.paged_kv_indices_arr = paged_kv_indices_arr
        self.paged_kv_last_page_len_arr = paged_kv_last_page_len_arr
        self.page_size = page_size
        self.chunk_states = chunk_states or []