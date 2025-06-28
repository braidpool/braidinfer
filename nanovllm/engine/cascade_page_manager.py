"""
Cascade-aware page manager for nano-vllm.
Extends PageManager to support persistent chunk storage and multi-level cascade attention.
"""

from typing import List, Dict, Optional, Tuple, Set
import torch
import flashinfer
from collections import deque, defaultdict

from nanovllm.engine.page_manager import PageManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.context_chunks import ContextChunk, ChunkComposition


class CascadePageManager(PageManager):
    """
    Extended page manager supporting persistent chunk storage for cascade attention.
    
    Manages separate page pools for:
    - Persistent chunks (system prompts, context)
    - Dynamic generation (user queries, outputs)
    """
    
    def __init__(self,
                 num_pages: int,
                 page_size: int,
                 num_layers: int,
                 num_kv_heads: int,
                 head_dim: int,
                 dtype: torch.dtype = torch.float16,
                 chunk_page_ratio: float = 0.5):
        """
        Initialize cascade page manager.
        
        Args:
            chunk_page_ratio: Fraction of pages reserved for persistent chunks
        """
        super().__init__(num_pages, page_size, num_layers, num_kv_heads, head_dim, dtype)
        
        # Split page pool
        self.num_chunk_pages = int(num_pages * chunk_page_ratio)
        self.num_dynamic_pages = num_pages - self.num_chunk_pages
        
        # Separate free pools
        self.chunk_free_pages: deque[int] = deque(range(self.num_chunk_pages))
        self.dynamic_free_pages: deque[int] = deque(range(self.num_chunk_pages, num_pages))
        
        # Override parent's free_pages to use dynamic pool by default
        self.free_pages = self.dynamic_free_pages
        
        # Chunk page tracking
        self.chunk_page_tables: Dict[str, List[int]] = {}  # chunk_id -> pages
        self.chunk_ref_counts: Dict[str, int] = defaultdict(int)  # chunk_id -> ref count
        self.page_to_chunk: Dict[int, str] = {}  # page -> chunk_id
        
        # Sequence to chunk mapping
        self.seq_chunks: Dict[int, List[str]] = {}  # seq_id -> chunk_ids
        
        # Pre-computed attention states cache
        self.chunk_attention_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def allocate_chunk_pages(self, chunk: ContextChunk) -> bool:
        """
        Allocate pages for a persistent chunk.
        
        Returns:
            True if allocation successful, False otherwise
        """
        if chunk.chunk_id in self.chunk_page_tables:
            # Already allocated, just increment ref count
            self.chunk_ref_counts[chunk.chunk_id] += 1
            return True
        
        # Calculate pages needed
        pages_needed = (chunk.seq_len + self.page_size - 1) // self.page_size
        
        if len(self.chunk_free_pages) < pages_needed:
            return False
        
        # Allocate pages
        pages = []
        for _ in range(pages_needed):
            page = self.chunk_free_pages.popleft()
            pages.append(page)
            self.page_to_chunk[page] = chunk.chunk_id
        
        self.chunk_page_tables[chunk.chunk_id] = pages
        self.chunk_ref_counts[chunk.chunk_id] = 1
        
        # Update chunk with allocation info
        chunk.kv_page_indices = pages
        chunk.kv_page_count = pages_needed
        chunk.last_page_len = (chunk.seq_len - 1) % self.page_size + 1 if chunk.seq_len > 0 else 0
        
        return True
    
    def deallocate_chunk_pages(self, chunk_id: str) -> None:
        """Deallocate pages for a chunk when ref count reaches zero."""
        if chunk_id not in self.chunk_page_tables:
            return
        
        self.chunk_ref_counts[chunk_id] -= 1
        
        if self.chunk_ref_counts[chunk_id] <= 0:
            # Return pages to chunk pool
            pages = self.chunk_page_tables[chunk_id]
            for page in pages:
                self.chunk_free_pages.append(page)
                del self.page_to_chunk[page]
            
            del self.chunk_page_tables[chunk_id]
            del self.chunk_ref_counts[chunk_id]
            
            # Clean up attention states if cached
            if chunk_id in self.chunk_attention_states:
                del self.chunk_attention_states[chunk_id]
    
    def register_sequence_chunks(self, seq_id: int, chunks: List[ContextChunk]) -> None:
        """Register chunks used by a sequence."""
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.seq_chunks[seq_id] = chunk_ids
        
        # Increment ref counts
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_ref_counts:
                self.chunk_ref_counts[chunk_id] += 1
    
    def unregister_sequence_chunks(self, seq_id: int) -> None:
        """Unregister chunks when sequence is deallocated."""
        if seq_id not in self.seq_chunks:
            return
        
        chunk_ids = self.seq_chunks[seq_id]
        for chunk_id in chunk_ids:
            self.deallocate_chunk_pages(chunk_id)
        
        del self.seq_chunks[seq_id]
    
    def deallocate(self, seq: Sequence) -> None:
        """Override to handle chunk cleanup."""
        # Unregister any chunks
        self.unregister_sequence_chunks(seq.seq_id)
        
        # Call parent deallocate for dynamic pages
        super().deallocate(seq)
    
    def build_cascade_indices(self, 
                             sequences: List[Sequence],
                             cascade_levels: List[List[ContextChunk]],
                             for_prefill: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Build indices for multi-level cascade attention.
        
        Returns:
            List of (kv_indices, kv_indptr, last_page_lens) tuples, one per cascade level
        """
        level_indices = []
        
        for level_idx, level_chunks in enumerate(cascade_levels):
            kv_indices = []
            kv_indptr = [0]
            last_page_lens = []
            
            # Handle shared chunks at this level
            if level_idx == 0:
                # Level 0: Shared across all sequences
                for chunk in level_chunks:
                    kv_indices.extend(chunk.kv_page_indices)
                kv_indptr.append(len(kv_indices))
                last_page_lens.append(chunk.last_page_len)
            else:
                # Other levels: Per-sequence chunks
                for seq in sequences:
                    seq_chunk_found = False
                    
                    # Find chunks for this sequence at this level
                    if seq.seq_id in self.seq_chunks:
                        for chunk_id in self.seq_chunks[seq.seq_id]:
                            for chunk in level_chunks:
                                if chunk.chunk_id == chunk_id:
                                    kv_indices.extend(chunk.kv_page_indices)
                                    last_page_lens.append(chunk.last_page_len)
                                    seq_chunk_found = True
                                    break
                    
                    if not seq_chunk_found:
                        # Use sequence's dynamic pages
                        seq_indices, seq_indptr, seq_last_lens = self.build_indices_for_sequences(
                            [seq], for_prefill
                        )
                        kv_indices.extend(seq_indices.tolist())
                        last_page_lens.extend(seq_last_lens.tolist())
                    
                    kv_indptr.append(len(kv_indices))
            
            level_indices.append((
                torch.tensor(kv_indices, dtype=torch.int32, device="cuda"),
                torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"),
                torch.tensor(last_page_lens, dtype=torch.int32, device="cuda")
            ))
        
        return level_indices
    
    def build_qo_indptr_array(self,
                             sequences: List[Sequence],
                             cascade_levels: List[List[ContextChunk]]) -> List[torch.Tensor]:
        """
        Build query/output indptr arrays for cascade levels.
        
        Returns:
            List of qo_indptr tensors, one per cascade level
        """
        batch_size = len(sequences)
        qo_indptr_arr = []
        
        for level_idx, level_chunks in enumerate(cascade_levels):
            if level_idx == 0:
                # Level 0: All sequences share the same chunks
                qo_indptr = torch.tensor([0, batch_size], dtype=torch.int32, device="cuda")
            else:
                # Other levels: One entry per sequence
                qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
            
            qo_indptr_arr.append(qo_indptr)
        
        return qo_indptr_arr
    
    def cache_chunk_attention_state(self, chunk_id: str, v: torch.Tensor, s: torch.Tensor) -> None:
        """
        Cache pre-computed attention states for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            v: Attention output [seq_len, num_heads, head_dim]
            s: LogSumExp values [seq_len, num_heads]
        """
        self.chunk_attention_states[chunk_id] = (v.clone(), s.clone())
    
    def get_chunk_attention_state(self, chunk_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached attention states for a chunk."""
        return self.chunk_attention_states.get(chunk_id)
    
    def get_free_pages_info(self) -> Dict[str, int]:
        """Get information about free pages."""
        return {
            "chunk_free": len(self.chunk_free_pages),
            "dynamic_free": len(self.dynamic_free_pages),
            "total_free": len(self.chunk_free_pages) + len(self.dynamic_free_pages),
            "chunk_allocated": self.num_chunk_pages - len(self.chunk_free_pages),
            "dynamic_allocated": self.num_dynamic_pages - len(self.dynamic_free_pages)
        }