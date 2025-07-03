"""
Simple cascade attention layer that properly uses FlashInfer's API.
"""

import torch
from torch import nn
import flashinfer
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext


class FlashInferCascadeAttention(nn.Module):
    """
    Cascade attention using FlashInfer's MultiLevelCascadeAttentionWrapper.
    
    This is a proper implementation that uses FlashInfer exactly as intended,
    without any custom abstractions or divergences.
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
        
        # Reference to paged KV cache (set by model runner)
        self.kv_cache = None
        
        # Workspace buffer - FlashInfer recommends 128MB
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        
        # Cascade wrapper - created on demand
        self._cascade_wrapper = None
        self._num_levels = None
    
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                context: 'InferenceContext') -> torch.Tensor:
        """
        Forward pass using cascade attention if available, otherwise regular attention.
        
        Args:
            q: Query tensor [seq_len, hidden_dim]
            k: Key tensor [seq_len, hidden_dim]  
            v: Value tensor [seq_len, hidden_dim]
            context: Inference context
            
        Returns:
            Attention output [seq_len, hidden_dim]
        """
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
        
        # Check if cascade data is provided
        if hasattr(context, 'cascade_data') and context.cascade_data is not None:
            output = self._run_cascade_attention(q, context)
        else:
            # Fall back to regular paged attention
            output = self._run_regular_attention(q, context)
        
        # Reshape output back to [seq_len, hidden_dim]
        output = output.view(-1, self.num_heads * self.head_dim)
        return output
    
    def _run_cascade_attention(self, 
                              q: torch.Tensor,
                              context: 'InferenceContext') -> torch.Tensor:
        """Run cascade attention using FlashInfer's API."""
        cascade_data = context.cascade_data
        num_levels = cascade_data['num_levels']
        
        # Create or reuse cascade wrapper
        if self._cascade_wrapper is None or self._num_levels != num_levels:
            self._cascade_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
                num_levels=num_levels,
                float_workspace_buffer=self.workspace_buffer,
                kv_layout="NHD"
            )
            self._num_levels = num_levels
        
        # Plan cascade attention
        self._cascade_wrapper.plan(
            qo_indptr_arr=cascade_data['qo_indptr_arr'],
            paged_kv_indptr_arr=cascade_data['paged_kv_indptr_arr'],
            paged_kv_indices_arr=cascade_data['paged_kv_indices_arr'],
            paged_kv_last_page_len=cascade_data['paged_kv_last_page_len_arr'],
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=cascade_data.get('page_size', context.page_manager.page_size if context.page_manager else 16),
            causal=cascade_data.get('causal', False),
            pos_encoding_mode="NONE",
            sm_scale=self.scale,
            q_data_type=str(q.dtype).replace("torch.", "")
        )
        
        # Run cascade attention
        output = self._cascade_wrapper.run(q, self.kv_cache)
        return output
    
    def _run_regular_attention(self,
                              q: torch.Tensor,
                              context: 'InferenceContext') -> torch.Tensor:
        """Run regular paged attention."""
        wrapper = context.get_wrapper(self.layer_idx)
        if wrapper is None:
            raise RuntimeError(f"Wrapper not initialized for layer {self.layer_idx}")
        return wrapper.run(q, self.kv_cache)


def prepare_cascade_data(
    sequences: List,
    shared_prefix_pages: int,
    page_size: int,
    unique_pages_per_seq: List[int],
    shared_page_indices: torch.Tensor,
    unique_page_indices: torch.Tensor,
    shared_last_page_len: int,
    unique_last_page_lens: List[int],
    apply_causal_to_last_level: bool = True
) -> dict:
    """
    Prepare cascade data in the format FlashInfer expects.
    
    This is a helper function to build the exact data structures
    FlashInfer's MultiLevelCascadeAttentionWrapper requires.
    
    Args:
        sequences: List of sequences being processed
        shared_prefix_pages: Number of pages in shared prefix
        page_size: Page size for paged KV cache
        unique_pages_per_seq: Number of unique pages per sequence
        shared_page_indices: Page indices for shared prefix
        unique_page_indices: Page indices for unique parts
        shared_last_page_len: Last page length for shared prefix
        unique_last_page_lens: Last page lengths for unique parts
        apply_causal_to_last_level: Whether to apply causal mask to last level
        
    Returns:
        Dictionary with cascade data ready for FlashInfer
    """
    batch_size = len(sequences)
    device = shared_page_indices.device
    
    # Build arrays for 2-level cascade
    # Level 0: Shared prefix (all sequences share same KV)
    shared_kv_page_indptr = torch.tensor(
        [0, shared_prefix_pages], dtype=torch.int32, device=device
    )
    shared_kv_last_page_len = torch.tensor(
        [shared_last_page_len], dtype=torch.int32, device=device
    )
    
    # Level 1: Unique parts (each sequence has its own KV)
    unique_kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, num_pages in enumerate(unique_pages_per_seq):
        unique_kv_page_indptr[i + 1] = unique_kv_page_indptr[i] + num_pages
    
    unique_kv_last_page_len = torch.tensor(
        unique_last_page_lens, dtype=torch.int32, device=device
    )
    
    # QO indptr arrays
    qo_indptr_arr = [
        torch.tensor([0, batch_size], dtype=torch.int32, device=device),  # Level 0
        torch.arange(batch_size + 1, dtype=torch.int32, device=device)    # Level 1
    ]
    
    return {
        'num_levels': 2,
        'qo_indptr_arr': qo_indptr_arr,
        'paged_kv_indptr_arr': [shared_kv_page_indptr, unique_kv_page_indptr],
        'paged_kv_indices_arr': [shared_page_indices, unique_page_indices],
        'paged_kv_last_page_len_arr': [shared_kv_last_page_len, unique_kv_last_page_len],
        'causal': apply_causal_to_last_level,
        'page_size': page_size
    }