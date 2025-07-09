"""
Page manager for nano-vllm.
Manages page allocation and tracks page tables for sequences.
"""

from typing import List, Dict, Optional, Tuple, Any
import torch
from collections import deque

from nanovllm.engine.sequence import Sequence


class PageManager:
    """Page manager for paged attention KV cache."""
    
    def __init__(self, 
                 num_pages: int,
                 page_size: int,
                 num_layers: int,
                 num_kv_heads: int,
                 head_dim: int,
                 dtype: torch.dtype = torch.float16):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Free page pool
        self.free_pages: deque[int] = deque(range(num_pages))
        
        # Sequence to page mapping
        self.seq_page_tables: Dict[int, List[int]] = {}
        
        # Allocate KV cache for all layers
        # Shape: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
        # This is HND format for FlashInfer
        self.kv_cache = torch.zeros(
            num_layers, num_pages, 2, num_kv_heads, page_size, head_dim,
            dtype=dtype, device="cuda"
        )
        
        # Track current lengths for proper appending
        self.seq_lengths: Dict[int, int] = {}
        
        # Page tables for chunks (persistent allocations)
        self.chunk_page_tables: Dict[str, List[int]] = {}
        self.chunk_lengths: Dict[str, int] = {}
    
    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate pages for a sequence."""
        # If already allocated, return True
        if seq.seq_id in self.seq_page_tables:
            return True
            
        # For prefill, allocate enough pages for the prompt
        # Plus some extra for generation
        estimated_final_len = len(seq) + seq.max_tokens
        pages_needed = (estimated_final_len + self.page_size - 1) // self.page_size
        return len(self.free_pages) >= pages_needed
    
    def allocate(self, seq: Sequence):
        """Allocate pages for a sequence."""
        if seq.seq_id in self.seq_page_tables:
            # Already allocated (e.g., for chunk-based generation)
            return
        
        # Allocate pages for prompt + generation headroom
        estimated_final_len = len(seq) + seq.max_tokens
        pages_needed = (estimated_final_len + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < pages_needed:
            raise RuntimeError(f"Not enough free pages. Need {pages_needed}, have {len(self.free_pages)}")
        
        pages = []
        for _ in range(pages_needed):
            pages.append(self.free_pages.popleft())
        
        self.seq_page_tables[seq.seq_id] = pages
        self.seq_lengths[seq.seq_id] = 0
        
        # Update sequence's block_table for compatibility
        seq.block_table = pages.copy()
    
    def deallocate(self, seq: Sequence):
        """Deallocate pages for a sequence."""
        if seq.seq_id not in self.seq_page_tables:
            return
        
        # Return pages to free pool
        self.free_pages.extend(self.seq_page_tables[seq.seq_id])
        del self.seq_page_tables[seq.seq_id]
        del self.seq_lengths[seq.seq_id]
        
        # Clear sequence's block table
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append more tokens (always true with pre-allocation)."""
        if seq.seq_id not in self.seq_page_tables:
            return False
        
        # Check if we have enough pages for the new token
        current_len = self.seq_lengths.get(seq.seq_id, 0)
        pages_allocated = len(self.seq_page_tables[seq.seq_id])
        max_capacity = pages_allocated * self.page_size
        
        return current_len < max_capacity
    
    def may_append(self, seq: Sequence):
        """Handle appending tokens (page allocation already done)."""
        # With pre-allocation, we don't need to do anything here
        # The actual appending happens in append_kv_to_cache
        pass
    
    def get_layer_kv_cache(self, layer_idx: int) -> torch.Tensor:
        """Get KV cache for a specific layer."""
        return self.kv_cache[layer_idx]
    
    def build_indices_for_sequences(self, sequences: List[Sequence], 
                                   for_prefill: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build kv_indices, kv_indptr, and last_page_lens for a batch of sequences."""
        kv_indices = []
        kv_indptr = [0]
        last_page_lens = []
        
        for seq in sequences:
            if seq.seq_id not in self.seq_page_tables:
                raise ValueError(f"Sequence {seq.seq_id} has no allocated pages")
            
            pages = self.seq_page_tables[seq.seq_id]
            
            # Get the current KV cache length (before appending new tokens)
            current_kv_len = self.seq_lengths.get(seq.seq_id, 0)
            
            if for_prefill:
                # For prefill, we're appending the entire prompt
                seq_len_after_append = len(seq)
            else:
                # For decode, we're appending one new token
                seq_len_after_append = current_kv_len + 1
            
            num_pages_used = (seq_len_after_append + self.page_size - 1) // self.page_size
            kv_indices.extend(pages[:num_pages_used])
            kv_indptr.append(len(kv_indices))
            
            # Calculate last page length after appending
            if seq_len_after_append == 0:
                last_page_lens.append(0)
            else:
                last_page_len = (seq_len_after_append - 1) % self.page_size + 1
                last_page_lens.append(last_page_len)
        
        return (torch.tensor(kv_indices, dtype=torch.int32, device="cuda"),
                torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"),
                torch.tensor(last_page_lens, dtype=torch.int32, device="cuda"))
    
    def append_kv_to_cache(self, 
                          layer_idx: int,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          sequences: List[Sequence],
                          is_prefill: bool = False):
        """Append K/V values to the cache for given sequences."""
        if is_prefill:
            # For prefill, build batch indices and positions
            seq_lens = [len(seq) for seq in sequences]
            total_tokens = sum(seq_lens)
            
            # Validate K/V shapes
            # Note: K/V shapes should match the number of tokens we're appending
            # which might be less than the total sequence length if using chunks
            assert k.shape[0] == total_tokens, f"K shape mismatch: {k.shape[0]} != {total_tokens} (seq_lens={seq_lens})"
            assert v.shape[0] == total_tokens, f"V shape mismatch: {v.shape[0]} != {total_tokens}"
            
            # Build q_indptr for batch positions
            q_indptr = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(sequences))],
                                   dtype=torch.int32, device="cuda")
            
            # Build batch indices and positions manually
            # This replaces flashinfer.page.get_batch_indices_positions
            batch_indices = []
            positions = []
            for seq_idx, seq_len in enumerate(seq_lens):
                batch_indices.extend([seq_idx] * seq_len)
                positions.extend(list(range(seq_len)))
            
            batch_indices = torch.tensor(batch_indices, dtype=torch.int32, device="cuda")
            positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
        else:
            # For decode, one token per sequence
            batch_size = len(sequences)
            batch_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            positions = torch.tensor([self.seq_lengths[seq.seq_id] for seq in sequences],
                                   dtype=torch.int32, device="cuda")
        
        # Get indices for this batch
        kv_indices, kv_indptr, last_page_lens = self.build_indices_for_sequences(
            sequences, for_prefill=is_prefill
        )
        
        # Append to cache
        layer_cache = self.kv_cache[layer_idx]
        
        # Always use custom KV cache implementation (FlashInfer removed)
        from nanovllm.kernels.kv_cache_utils import append_to_paged_kv_cache_custom
        
        # Process each sequence separately for correctness
        token_offset = 0
        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq) if is_prefill else 1
            
            # Get the KV indices for this sequence
            start_idx = kv_indptr[seq_idx].item()
            end_idx = kv_indptr[seq_idx + 1].item()
            seq_kv_indices = kv_indices[start_idx:end_idx]
            
            # Get the tokens for this sequence
            seq_k = k[token_offset:token_offset + seq_len]
            seq_v = v[token_offset:token_offset + seq_len]
            
            # Get positions for this sequence - need to account for existing KV cache
            current_kv_len = self.seq_lengths.get(seq.seq_id, 0)
            if is_prefill:
                # For prefill, positions start from current KV cache length
                seq_positions = torch.arange(current_kv_len, current_kv_len + seq_len, dtype=torch.int32, device="cuda")
            else:
                # For decode, position is the current KV cache length
                seq_positions = torch.tensor([current_kv_len], dtype=torch.int32, device="cuda")
            
            # Append for this sequence
            append_to_paged_kv_cache_custom(
                k=seq_k,
                v=seq_v,
                layer_cache=layer_cache,
                kv_indices=seq_kv_indices,
                positions=seq_positions,
                page_size=self.page_size,
                kv_layout="HND"
            )
            
            token_offset += seq_len
        
        # Update sequence lengths only after all layers are processed
        # This is handled externally now
    
    def update_sequence_lengths(self, sequences: List[Sequence], is_prefill: bool):
        """Update sequence lengths after all layers have been processed."""
        if is_prefill:
            for seq in sequences:
                self.seq_lengths[seq.seq_id] = len(seq)
        else:
            for seq in sequences:
                self.seq_lengths[seq.seq_id] += 1
    
    def get_sequence_cache_info(self, seq: Sequence) -> Optional[Dict[str, Any]]:
        """Get KV cache information for a retained sequence.
        
        Returns:
            Dictionary with cache info or None if sequence not found.
        """
        if seq.seq_id not in self.seq_page_tables:
            return None
        
        return {
            "seq_id": seq.seq_id,
            "page_indices": self.seq_page_tables[seq.seq_id].copy(),
            "length": self.seq_lengths.get(seq.seq_id, 0),
            "num_pages": len(self.seq_page_tables[seq.seq_id]),
            "page_size": self.page_size,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim
        }
    
    def deallocate_by_seq_id(self, seq_id: int):
        """Deallocate pages for a sequence by ID (for manual cleanup)."""
        if seq_id not in self.seq_page_tables:
            return
        
        # Return pages to free pool
        self.free_pages.extend(self.seq_page_tables[seq_id])
        del self.seq_page_tables[seq_id]
        del self.seq_lengths[seq_id]
    
    def allocate_for_chunk(self, chunk_id: str, num_tokens: int) -> List[int]:
        """Allocate pages for a chunk's KV cache."""
        if chunk_id in self.chunk_page_tables:
            raise ValueError(f"Chunk {chunk_id} already has allocated pages")
        
        pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < pages_needed:
            raise RuntimeError(f"Out of pages for chunk. Need {pages_needed}, have {len(self.free_pages)}")
        
        # Allocate pages
        pages = []
        for _ in range(pages_needed):
            pages.append(self.free_pages.popleft())
        
        # Store mapping
        self.chunk_page_tables[chunk_id] = pages
        self.chunk_lengths[chunk_id] = num_tokens
        
        return pages
    
    def free_for_chunk(self, chunk_id: str) -> None:
        """Free pages allocated to a chunk."""
        if chunk_id not in self.chunk_page_tables:
            return
        
        # Return pages to free pool
        self.free_pages.extend(self.chunk_page_tables[chunk_id])
        del self.chunk_page_tables[chunk_id]
        
        if chunk_id in self.chunk_lengths:
            del self.chunk_lengths[chunk_id]
    
    def build_indices_for_chunk(self, chunk_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build kv_indices, kv_indptr, and last_page_lens for a single chunk."""
        if chunk_id not in self.chunk_page_tables:
            raise ValueError(f"Chunk {chunk_id} has no allocated pages")
        
        pages = self.chunk_page_tables[chunk_id]
        num_tokens = self.chunk_lengths[chunk_id]
        
        # All pages for this chunk
        kv_indices = torch.tensor(pages, dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, len(pages)], dtype=torch.int32, device="cuda")
        
        # Last page length
        if num_tokens == 0:
            last_page_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        else:
            last_page_len = (num_tokens - 1) % self.page_size + 1
            last_page_lens = torch.tensor([last_page_len], dtype=torch.int32, device="cuda")
        
        return kv_indices, kv_indptr, last_page_lens
    
    def append_kv_to_cache_for_chunk(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
                                   chunk_id: str, token_positions: torch.Tensor):
        """Append K and V tensors to KV cache for a chunk."""
        # Get chunk's page allocation
        if chunk_id not in self.chunk_page_tables:
            raise ValueError(f"Chunk {chunk_id} has no allocated pages")
        
        # Get indices for this chunk
        kv_indices, kv_indptr, last_page_lens = self.build_indices_for_chunk(chunk_id)
        
        # For chunk prefilling, we treat the entire chunk as a single "sequence"
        # Create batch indices - all zeros since we have one sequence
        num_tokens = len(token_positions)
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
        
        # Positions should be 0-based for a new chunk
        # This is important: positions are within-sequence positions, starting from 0
        positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
        
        # Append to cache
        layer_cache = self.kv_cache[layer_idx]
        
        # Always use custom KV cache implementation (FlashInfer removed)
        from nanovllm.kernels.kv_cache_utils import append_to_paged_kv_cache_custom
        
        # Debug output disabled for cleaner output
        pass
        
        # Use custom append function
        append_to_paged_kv_cache_custom(
            k=k,
            v=v,
            layer_cache=layer_cache,
            kv_indices=kv_indices,
            positions=positions,
            page_size=self.page_size,
            kv_layout="HND"  # We use HND layout
        )
        
        # Debug output disabled for cleaner output
        pass