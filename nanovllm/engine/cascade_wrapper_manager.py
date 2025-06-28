"""
Enhanced wrapper manager supporting cascade attention for nano-vllm.
"""

import torch
import flashinfer
from typing import List, Dict, Optional, Tuple

from nanovllm.engine.wrapper_manager import WrapperManager
from nanovllm.engine.context_chunks import ContextChunk, ChunkComposition
from nanovllm.layers.cascade_attention import CascadeConfig


class CascadeWrapperManager(WrapperManager):
    """
    Extended wrapper manager that supports multi-level cascade attention.
    
    Manages both regular wrappers and cascade wrappers for compositional
    context caching with multi-head attention support.
    """
    
    def __init__(self,
                 num_layers: int,
                 num_qo_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 page_size: int,
                 dtype: torch.dtype = torch.float16,
                 workspace_size: int = 128 * 1024 * 1024,
                 max_cascade_levels: int = 3):
        """
        Initialize cascade wrapper manager.
        
        Args:
            max_cascade_levels: Maximum number of cascade levels to support
        """
        super().__init__(num_layers, num_qo_heads, num_kv_heads, head_dim, 
                        page_size, dtype, workspace_size)
        
        self.max_cascade_levels = max_cascade_levels
        
        # Create cascade wrappers for each layer
        self.cascade_wrappers: List[Optional[flashinfer.MultiLevelCascadeAttentionWrapper]] = []
        
        # Shared workspace buffer for cascade attention
        self.cascade_workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")
        
        # Cache for cascade configurations
        self.cascade_config_cache: Dict[str, CascadeConfig] = {}
        
        # Initialize cascade wrappers as None (created on demand)
        for _ in range(num_layers):
            self.cascade_wrappers.append(None)
    
    def get_or_create_cascade_wrapper(self, 
                                     layer_idx: int, 
                                     num_levels: int) -> flashinfer.MultiLevelCascadeAttentionWrapper:
        """
        Get or create a cascade wrapper for the specified layer and level count.
        
        Args:
            layer_idx: Layer index
            num_levels: Number of cascade levels needed
            
        Returns:
            MultiLevelCascadeAttentionWrapper configured for the layer
        """
        # Check if we need to create or recreate the wrapper
        if (self.cascade_wrappers[layer_idx] is None or 
            self.cascade_wrappers[layer_idx]._num_levels != num_levels):
            
            # Create new cascade wrapper
            wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
                num_levels=num_levels,
                float_workspace_buffer=self.cascade_workspace,
                kv_layout="NHD",
                use_cuda_graph=False  # Can be enabled for performance
            )
            
            self.cascade_wrappers[layer_idx] = wrapper
        
        return self.cascade_wrappers[layer_idx]
    
    def plan_cascade(self,
                    cascade_config: CascadeConfig,
                    layer_idx: Optional[int] = None) -> None:
        """
        Plan cascade attention for specified layer(s).
        
        Args:
            cascade_config: Cascade configuration with level information
            layer_idx: Optional specific layer index (None for all layers)
        """
        layers = [layer_idx] if layer_idx is not None else range(self.num_layers)
        
        for idx in layers:
            wrapper = self.get_or_create_cascade_wrapper(idx, cascade_config.num_levels)
            
            # Plan cascade attention
            wrapper.plan(
                qo_indptr_arr=cascade_config.qo_indptr_arr,
                paged_kv_indptr_arr=cascade_config.paged_kv_indptr_arr,
                paged_kv_indices_arr=cascade_config.paged_kv_indices_arr,
                paged_kv_last_page_len=cascade_config.paged_kv_last_page_len_arr,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=False,  # Causal mask handled per level
                pos_encoding_mode="NONE",  # Can be configured
                use_fp16_qk_reduction=False,
                sm_scale=None  # Will use default 1/sqrt(head_dim)
            )
    
    def create_cascade_config(self,
                             chunk_composition: ChunkComposition,
                             qo_indptr_arr: List[torch.Tensor],
                             paged_kv_indptr_arr: List[torch.Tensor],
                             paged_kv_indices_arr: List[torch.Tensor],
                             paged_kv_last_page_len_arr: List[torch.Tensor]) -> CascadeConfig:
        """
        Create a cascade configuration from chunk composition.
        
        Args:
            chunk_composition: Composition of chunks
            qo_indptr_arr: Query/output indptr arrays
            paged_kv_indptr_arr: KV cache indptr arrays
            paged_kv_indices_arr: KV cache indices arrays
            paged_kv_last_page_len_arr: Last page length arrays
            
        Returns:
            CascadeConfig ready for planning
        """
        # Get cascade levels from composition
        cascade_levels = chunk_composition.get_cascade_levels()
        num_levels = len(cascade_levels)
        
        # Create configuration
        config = CascadeConfig(
            num_levels=num_levels,
            qo_indptr_arr=qo_indptr_arr,
            paged_kv_indptr_arr=paged_kv_indptr_arr,
            paged_kv_indices_arr=paged_kv_indices_arr,
            paged_kv_last_page_len_arr=paged_kv_last_page_len_arr,
            page_size=self.page_size
        )
        
        # Cache configuration
        cache_key = self._get_cascade_cache_key(chunk_composition)
        self.cascade_config_cache[cache_key] = config
        
        return config
    
    def get_cascade_wrapper(self, layer_idx: int) -> Optional[flashinfer.MultiLevelCascadeAttentionWrapper]:
        """Get cascade wrapper for a specific layer."""
        return self.cascade_wrappers[layer_idx]
    
    def run_cascade_attention(self,
                             layer_idx: int,
                             q: torch.Tensor,
                             kv_cache: torch.Tensor,
                             cascade_config: Optional[CascadeConfig] = None) -> torch.Tensor:
        """
        Run cascade attention for a specific layer.
        
        Args:
            layer_idx: Layer index
            q: Query tensor [batch_size, num_qo_heads, head_dim]
            kv_cache: KV cache tensor for the layer
            cascade_config: Optional cascade configuration (uses cached if not provided)
            
        Returns:
            Attention output [batch_size, num_qo_heads, head_dim]
        """
        wrapper = self.cascade_wrappers[layer_idx]
        if wrapper is None:
            raise RuntimeError(f"Cascade wrapper not initialized for layer {layer_idx}")
        
        # Run cascade attention
        output = wrapper.run(q, kv_cache)
        return output
    
    def merge_attention_states(self,
                              v_states: List[torch.Tensor],
                              s_states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge multiple attention states using FlashInfer's merge operations.
        
        Args:
            v_states: List of V tensors [seq_len, num_heads, head_dim]
            s_states: List of S tensors [seq_len, num_heads]
            
        Returns:
            Merged (V, S) tuple
        """
        if len(v_states) == 0:
            raise ValueError("No states to merge")
        
        if len(v_states) == 1:
            return v_states[0], s_states[0]
        
        # Start with first state
        v_merged, s_merged = v_states[0], s_states[0]
        
        # Merge remaining states
        for v, s in zip(v_states[1:], s_states[1:]):
            v_merged, s_merged = flashinfer.merge_state(v_merged, s_merged, v, s)
        
        return v_merged, s_merged
    
    def _get_cascade_cache_key(self, chunk_composition: ChunkComposition) -> str:
        """Generate cache key for cascade configuration."""
        chunk_ids = sorted([chunk.chunk_id for chunk in chunk_composition.chunks])
        return ":".join(chunk_ids)
    
    def clear_cascade_cache(self) -> None:
        """Clear cascade configuration cache."""
        self.cascade_config_cache.clear()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage information."""
        base_usage = {
            "workspace_size": self.cascade_workspace.numel() * self.cascade_workspace.element_size(),
            "num_cascade_wrappers": sum(1 for w in self.cascade_wrappers if w is not None),
            "cascade_config_cache_size": len(self.cascade_config_cache)
        }
        
        # Add base class memory usage
        base_usage["prefill_wrappers_memory"] = len(self.prefill_wrappers) * self.cascade_workspace.numel() * self.cascade_workspace.element_size()
        base_usage["decode_wrappers_memory"] = len(self.decode_wrappers) * self.cascade_workspace.numel() * self.cascade_workspace.element_size()
        
        return base_usage