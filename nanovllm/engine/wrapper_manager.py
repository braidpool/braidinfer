"""
FlashInfer wrapper management for nano-vllm.
"""

import torch
import flashinfer
from typing import List, Dict


class WrapperManager:
    """Manages FlashInfer prefill and decode wrappers for all layers."""
    
    def __init__(self, 
                 num_layers: int,
                 num_qo_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 page_size: int,
                 dtype: torch.dtype = torch.float16,
                 workspace_size: int = 128 * 1024 * 1024):
        """Initialize wrapper manager.
        
        Args:
            num_layers: Number of transformer layers
            num_qo_heads: Number of query/output heads per device
            num_kv_heads: Number of key/value heads per device
            head_dim: Dimension of each attention head
            page_size: Page size for paged attention
            dtype: Data type for computation
            workspace_size: Size of workspace buffer in bytes
        """
        self.num_layers = num_layers
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        
        # Create wrappers for each layer
        self.prefill_wrappers: List[flashinfer.BatchPrefillWithPagedKVCacheWrapper] = []
        self.decode_wrappers: List[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = []
        
        for _ in range(num_layers):
            # Create prefill wrapper
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=torch.empty(workspace_size, dtype=torch.uint8, device="cuda"),
                kv_layout="NHD"
            )
            
            # Create decode wrapper
            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                float_workspace_buffer=torch.empty(workspace_size, dtype=torch.uint8, device="cuda"),
                kv_layout="NHD",
                use_tensor_cores=True
            )
            
            self.prefill_wrappers.append(prefill_wrapper)
            self.decode_wrappers.append(decode_wrapper)
    
    def plan_prefill(self, 
                    q_indptr: torch.Tensor,
                    kv_indptr: torch.Tensor,
                    kv_indices: torch.Tensor,
                    last_page_lens: torch.Tensor):
        """Plan prefill for all layers.
        
        Args:
            q_indptr: Query indptr tensor
            kv_indptr: KV indptr tensor
            kv_indices: KV indices tensor
            last_page_lens: Last page lengths tensor
        """
        for wrapper in self.prefill_wrappers:
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                last_page_lens,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype
            )
    
    def plan_decode(self,
                   kv_indptr: torch.Tensor,
                   kv_indices: torch.Tensor,
                   last_page_lens: torch.Tensor):
        """Plan decode for all layers.
        
        Args:
            kv_indptr: KV indptr tensor
            kv_indices: KV indices tensor
            last_page_lens: Last page lengths tensor
        """
        for wrapper in self.decode_wrappers:
            wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_lens,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                kv_data_type=self.dtype,
                q_data_type=self.dtype
            )
    
    def get_prefill_wrapper(self, layer_idx: int) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
        """Get prefill wrapper for a specific layer."""
        return self.prefill_wrappers[layer_idx]
    
    def get_decode_wrapper(self, layer_idx: int) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
        """Get decode wrapper for a specific layer."""
        return self.decode_wrappers[layer_idx]