"""
Utilities for custom KV cache operations.
"""

import torch
from typing import List, Tuple


def append_to_paged_kv_cache_custom(
    k: torch.Tensor,
    v: torch.Tensor, 
    layer_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    kv_layout: str = "HND"
) -> None:
    """
    Custom implementation of append to paged KV cache.
    
    This replaces FlashInfer's append_paged_kv_cache for our custom kernels.
    
    Args:
        k: Key tensor, shape [num_tokens, num_heads, head_dim]
        v: Value tensor, shape [num_tokens, num_heads, head_dim]
        layer_cache: KV cache for this layer, shape depends on layout:
            - HND: [num_pages, num_heads, page_size, head_dim]
            - NHD: [num_pages, page_size, num_heads, head_dim]
        kv_indices: Page indices for the sequence
        positions: Position of each token within its sequence
        page_size: Size of each page
        kv_layout: Layout format ("HND" or "NHD")
    """
    num_tokens = k.shape[0]
    
    if kv_layout == "HND":
        # HND layout: [num_pages, num_heads, page_size, head_dim]
        # layer_cache shape: [num_pages, 2, num_heads, page_size, head_dim]
        
        for token_idx in range(num_tokens):
            # Get the position within the sequence
            seq_pos = positions[token_idx].item()
            
            # Calculate which page this position falls into
            page_offset = seq_pos // page_size
            pos_in_page = seq_pos % page_size
            
            # Get the global page index from kv_indices
            if page_offset < len(kv_indices):
                page_idx = kv_indices[page_offset].item()
                
                # Write K (kv_type=0) and V (kv_type=1)
                # k[token_idx] shape: [num_heads, head_dim]
                layer_cache[page_idx, 0, :, pos_in_page, :] = k[token_idx]
                layer_cache[page_idx, 1, :, pos_in_page, :] = v[token_idx]
            else:
                print(f"[ERROR] Token {token_idx}: page_offset {page_offset} >= len(kv_indices) {len(kv_indices)}")
                
    elif kv_layout == "NHD":
        # NHD layout: [num_pages, page_size, num_heads, head_dim]
        # layer_cache shape: [num_pages, 2, page_size, num_heads, head_dim]
        
        for token_idx in range(num_tokens):
            # Calculate which page and position within page
            seq_pos = positions[token_idx].item()
            page_offset = seq_pos // page_size
            pos_in_page = seq_pos % page_size
            
            # Get the global page index
            if page_offset < len(kv_indices):
                page_idx = kv_indices[page_offset].item()
                
                # Write K (kv_type=0) and V (kv_type=1)
                # k[token_idx] shape: [num_heads, head_dim]
                layer_cache[page_idx, 0, pos_in_page, :, :] = k[token_idx]
                layer_cache[page_idx, 1, pos_in_page, :, :] = v[token_idx]
    else:
        raise ValueError(f"Unknown kv_layout: {kv_layout}")


def build_paged_kv_indices(
    page_tables: List[List[int]],
    seq_lens: List[int],
    page_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build paged KV cache indices from page tables.
    
    Args:
        page_tables: List of page tables for each sequence
        seq_lens: Length of each sequence
        page_size: Size of each page
        
    Returns:
        kv_indices: Flat tensor of page indices
        kv_indptr: CSR-format indptr for sequences
        kv_last_page_len: Number of valid tokens in last page per sequence
    """
    kv_indices = []
    kv_indptr = [0]
    kv_last_page_len = []
    
    for page_table, seq_len in zip(page_tables, seq_lens):
        # Add page indices
        kv_indices.extend(page_table)
        kv_indptr.append(len(kv_indices))
        
        # Calculate last page length
        last_page_len = seq_len % page_size
        if last_page_len == 0 and seq_len > 0:
            last_page_len = page_size
        kv_last_page_len.append(last_page_len)
    
    return (
        torch.tensor(kv_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(kv_last_page_len, dtype=torch.int32, device="cuda")
    )