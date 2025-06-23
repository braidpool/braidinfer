"""
Virtual Sequence for direct KV cache inference.

This module provides a Sequence implementation that can reference
existing blocks in the KV cache without reallocating them.
"""

from typing import List, Optional
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class VirtualSequence(Sequence):
    """A sequence that can start with pre-existing blocks from KV cache."""
    
    def __init__(self, 
                 token_ids: List[int], 
                 sampling_params: SamplingParams,
                 existing_blocks: Optional[List[int]] = None,
                 existing_token_count: int = 0):
        """
        Initialize a virtual sequence.
        
        Args:
            token_ids: New tokens to process (not including existing cached tokens)
            sampling_params: Sampling parameters for generation
            existing_blocks: List of block IDs already in KV cache
            existing_token_count: Number of tokens in existing blocks
        """
        # Initialize base sequence with combined tokens
        if existing_blocks:
            # For virtual sequence, we only track new tokens in token_ids
            # but need to account for total length
            super().__init__(token_ids, sampling_params)
            
            # Override initialization to account for existing blocks
            self.block_table = existing_blocks.copy()
            self.existing_blocks = existing_blocks.copy()
            self.existing_token_count = existing_token_count
            
            # Adjust counts to include existing tokens
            self.num_tokens = existing_token_count + len(token_ids)
            self.num_cached_tokens = existing_token_count
            
            # Track which blocks are pre-existing (don't deallocate these)
            self.owned_blocks = []  # Blocks we allocated and own
        else:
            # Standard sequence with no existing blocks
            super().__init__(token_ids, sampling_params)
            self.existing_blocks = []
            self.existing_token_count = 0
            self.owned_blocks = []
    
    @property
    def new_token_ids(self) -> List[int]:
        """Get only the new tokens (not in existing blocks)."""
        return self.token_ids
    
    @property
    def num_cached_blocks(self):
        """Number of blocks that are already cached."""
        # For virtual sequences, all existing blocks are fully cached
        return len(self.existing_blocks)
    
    @property
    def last_block_num_tokens(self):
        """Number of tokens in the last block."""
        # For virtual sequences, we need to consider only the new tokens
        # in the last block since existing blocks are immutable
        if self.num_new_blocks == 0:
            # No new blocks, shouldn't be called
            return 0
        
        # Get tokens in the last new block
        last_block_tokens = len(self.new_token_ids) % self.block_size
        if last_block_tokens == 0 and len(self.new_token_ids) > 0:
            # Last block is full
            return self.block_size
        return last_block_tokens
    
    @property 
    def full_length(self) -> int:
        """Get total length including existing cached tokens."""
        return self.num_tokens
    
    def get_new_token_blocks(self) -> List[List[int]]:
        """Get new tokens organized by blocks for allocation."""
        blocks = []
        start_idx = 0
        
        # Calculate where new tokens should start in the block structure
        # If existing tokens don't fill the last block perfectly, we still
        # need to allocate fresh blocks for new tokens (blocks are immutable)
        
        while start_idx < len(self.new_token_ids):
            end_idx = min(start_idx + self.block_size, len(self.new_token_ids))
            blocks.append(self.new_token_ids[start_idx:end_idx])
            start_idx = end_idx
            
        return blocks
    
    @property
    def num_new_blocks(self) -> int:
        """Number of new blocks needed for new tokens."""
        if not self.new_token_ids:
            return 0
        return (len(self.new_token_ids) + self.block_size - 1) // self.block_size
    
    @property
    def num_blocks(self):
        """Total number of blocks (existing + new)."""
        # Existing blocks are immutable, so we need existing + new blocks
        return len(self.existing_blocks) + self.num_new_blocks
    
    def mark_block_as_owned(self, block_id: int):
        """Mark a block as owned by this sequence (for cleanup)."""
        if block_id not in self.owned_blocks:
            self.owned_blocks.append(block_id)
    
    def __getitem__(self, key):
        """Get token at index, only for new tokens."""
        if isinstance(key, slice):
            # Handle slice access
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.num_tokens
            
            # For slices starting at num_cached_tokens, return new tokens
            if start >= self.existing_token_count:
                new_start = start - self.existing_token_count
                new_stop = min(stop - self.existing_token_count, len(self.token_ids))
                return self.token_ids[new_start:new_stop]
            elif stop > self.existing_token_count:
                # Slice spans cached and new tokens, return only new part
                new_stop = stop - self.existing_token_count
                return self.token_ids[:new_stop]
            else:
                # Slice is entirely in cached tokens
                return []
        else:
            # Single index access
            if key < self.existing_token_count:
                # This would be from existing blocks, return dummy
                return 0
            else:
                return self.token_ids[key - self.existing_token_count]
    
    def block(self, i):
        """Get tokens for block i, accounting for existing blocks."""
        if i < len(self.existing_blocks):
            # This is an existing block - return empty since tokens are already in cache
            return []
        else:
            # This is a new block
            new_block_idx = i - len(self.existing_blocks)
            start_idx = new_block_idx * self.block_size
            end_idx = min((new_block_idx + 1) * self.block_size, len(self.new_token_ids))
            return self.new_token_ids[start_idx:end_idx]