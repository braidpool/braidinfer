"""
Simple scheduler that can prepare cascade data for FlashInfer.
"""

from typing import List, Tuple, Optional, Dict
import torch
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.config import Config


class FlashInferScheduler(Scheduler):
    """
    Scheduler that can prepare cascade data in FlashInfer's format.
    
    This is a minimal implementation that provides cascade data
    when sequences have a shared prefix (e.g., system prompt).
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_cascade = getattr(config, 'enable_cascade_attention', False)
        self.shared_prefix_len = getattr(config, 'cascade_shared_prefix_len', 0)
    
    def schedule(self) -> Tuple[List[Sequence], bool, Optional[Dict]]:
        """
        Schedule sequences with optional cascade data preparation.
        
        Returns:
            - List of scheduled sequences
            - Whether this is a prefill batch  
            - Optional cascade data for FlashInfer
        """
        # Use parent's scheduling logic
        sequences, is_prefill = super().schedule()
        
        if not sequences:
            return sequences, is_prefill, None
        
        # Prepare cascade data if enabled and we have a shared prefix
        cascade_data = None
        if self.enable_cascade and self.shared_prefix_len > 0:
            cascade_data = self._prepare_cascade_data(sequences, is_prefill)
        
        return sequences, is_prefill, cascade_data
    
    def _prepare_cascade_data(self, sequences: List[Sequence], is_prefill: bool) -> Optional[Dict]:
        """
        Prepare cascade data in FlashInfer's format.
        
        This creates a 2-level cascade:
        - Level 0: Shared prefix (e.g., system prompt)
        - Level 1: Unique parts per sequence
        """
        if not sequences or self.shared_prefix_len == 0:
            return None
        
        batch_size = len(sequences)
        page_size = self.page_manager.page_size
        device = torch.device("cuda")
        
        # Calculate pages for shared prefix
        shared_pages = (self.shared_prefix_len + page_size - 1) // page_size
        shared_last_page_len = self.shared_prefix_len % page_size or page_size
        
        # For simplicity, assume first N pages are for shared prefix
        shared_page_indices = torch.arange(shared_pages, dtype=torch.int32, device=device)
        
        # Calculate unique pages per sequence
        unique_pages_per_seq = []
        unique_page_indices_list = []
        unique_last_page_lens = []
        
        current_page_offset = shared_pages
        
        for seq in sequences:
            # Calculate unique length (total - shared)
            unique_len = max(0, len(seq) - self.shared_prefix_len)
            if unique_len > 0:
                unique_pages = (unique_len + page_size - 1) // page_size
                unique_last_page_len = unique_len % page_size or page_size
            else:
                unique_pages = 0
                unique_last_page_len = page_size  # FlashInfer expects valid length
            
            unique_pages_per_seq.append(unique_pages)
            unique_last_page_lens.append(unique_last_page_len)
            
            # Assign page indices for this sequence's unique part
            if unique_pages > 0:
                seq_indices = torch.arange(
                    current_page_offset, 
                    current_page_offset + unique_pages,
                    dtype=torch.int32, 
                    device=device
                )
                unique_page_indices_list.append(seq_indices)
                current_page_offset += unique_pages
        
        # Concatenate all unique page indices
        if unique_page_indices_list:
            unique_page_indices = torch.cat(unique_page_indices_list)
        else:
            unique_page_indices = torch.empty(0, dtype=torch.int32, device=device)
        
        # Build cascade data using FlashInfer's format
        from nanovllm.layers.flashinfer_cascade_attention import prepare_cascade_data
        
        cascade_data = prepare_cascade_data(
            sequences=sequences,
            shared_prefix_pages=shared_pages,
            page_size=page_size,
            unique_pages_per_seq=unique_pages_per_seq,
            shared_page_indices=shared_page_indices,
            unique_page_indices=unique_page_indices,
            shared_last_page_len=shared_last_page_len,
            unique_last_page_lens=unique_last_page_lens,
            apply_causal_to_last_level=is_prefill
        )
        
        return cascade_data