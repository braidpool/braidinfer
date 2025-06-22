"""
Virtual Block Table for Context Manager

Provides a mapping layer between logical block references and physical blocks,
allowing for efficient activation/deactivation of chunks without data movement.
"""

from typing import Dict, List, Set, Optional, Tuple
import torch
from dataclasses import dataclass

@dataclass
class VirtualBlock:
    """Virtual block that can be mapped to a physical block"""
    virtual_id: int
    physical_id: int
    chunk_hash: str
    
class VirtualBlockTable:
    """
    Manages virtual to physical block mappings for selective attention.
    
    This allows chunks to be activated/deactivated without moving data,
    by maintaining a virtual addressing layer that can filter blocks.
    """
    
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.virtual_blocks: Dict[int, VirtualBlock] = {}
        self.chunk_to_virtual: Dict[str, List[int]] = {}  # chunk_hash -> virtual block IDs
        self.next_virtual_id = 0
        
        # Cache for filtered block tables
        self.cached_tables: Dict[frozenset, torch.Tensor] = {}
        
    def register_chunk(self, chunk_hash: str, physical_blocks: List[int]) -> List[int]:
        """
        Register a chunk's physical blocks and return virtual block IDs.
        """
        virtual_ids = []
        
        for physical_id in physical_blocks:
            virtual_id = self.next_virtual_id
            self.next_virtual_id += 1
            
            virtual_block = VirtualBlock(
                virtual_id=virtual_id,
                physical_id=physical_id,
                chunk_hash=chunk_hash
            )
            
            self.virtual_blocks[virtual_id] = virtual_block
            virtual_ids.append(virtual_id)
        
        self.chunk_to_virtual[chunk_hash] = virtual_ids
        return virtual_ids
        
    def activate_chunk(self, chunk_hash: str):
        """Activate all blocks belonging to a chunk."""
        if chunk_hash not in self.chunk_to_virtual:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        for virtual_id in self.chunk_to_virtual[chunk_hash]:
            vblock = self.virtual_blocks[virtual_id]
            if vblock.physical_id < len(self.block_manager.blocks):
                self.block_manager.blocks[vblock.physical_id].activate()
            
        # Invalidate cache
        self.cached_tables.clear()
        
    def deactivate_chunk(self, chunk_hash: str):
        """Deactivate all blocks belonging to a chunk."""
        if chunk_hash not in self.chunk_to_virtual:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        for virtual_id in self.chunk_to_virtual[chunk_hash]:
            vblock = self.virtual_blocks[virtual_id]
            if vblock.physical_id < len(self.block_manager.blocks):
                self.block_manager.blocks[vblock.physical_id].deactivate()
            
        # Invalidate cache
        self.cached_tables.clear()
        
    def get_active_physical_blocks(self) -> Set[int]:
        """Get set of active physical block IDs."""
        active_blocks = set()
        for vblock in self.virtual_blocks.values():
            if vblock.physical_id < len(self.block_manager.blocks):
                block = self.block_manager.blocks[vblock.physical_id]
                if block.is_active:
                    active_blocks.add(vblock.physical_id)
        return active_blocks
    
    def is_block_active(self, physical_id: int) -> bool:
        """Check if a physical block is active."""
        # First check if we have a virtual block for this physical ID
        for vblock in self.virtual_blocks.values():
            if vblock.physical_id == physical_id:
                if physical_id < len(self.block_manager.blocks):
                    return self.block_manager.blocks[physical_id].is_active
                return True
        # If not tracked, consider it active (for blocks not managed by context manager)
        return True
        
    def map_sequence_blocks(self, sequence_blocks: List[int], 
                           active_only: bool = True) -> Tuple[List[int], Dict[int, int]]:
        """
        Map a sequence's block table considering active blocks.
        
        Args:
            sequence_blocks: Physical block IDs from the sequence
            active_only: Whether to filter out inactive blocks
            
        Returns:
            - Filtered block list
            - Mapping from old to new positions
        """
        if not active_only:
            return sequence_blocks, {i: i for i in range(len(sequence_blocks))}
            
        filtered_blocks = []
        position_map = {}
        
        for old_pos, block_id in enumerate(sequence_blocks):
            if self.is_block_active(block_id):
                new_pos = len(filtered_blocks)
                filtered_blocks.append(block_id)
                position_map[old_pos] = new_pos
                
        return filtered_blocks, position_map
        
    def create_filtered_block_table(self, sequences: List[List[int]], 
                                   active_only: bool = True) -> torch.Tensor:
        """
        Create filtered block table for attention computation.
        
        Args:
            sequences: List of block tables (one per sequence in batch)
            active_only: Whether to filter inactive blocks
            
        Returns:
            Filtered block table tensor
        """
        if not active_only:
            # No filtering needed
            max_len = max(len(seq) for seq in sequences) if sequences else 0
            padded = [seq + [-1] * (max_len - len(seq)) for seq in sequences]
            return torch.tensor(padded, dtype=torch.int32)
            
        # Check cache
        cache_key = frozenset(self.get_active_physical_blocks())
        if cache_key in self.cached_tables and len(sequences) == 1:
            # Can reuse cached table for single sequence
            return self.cached_tables[cache_key]
            
        # Filter each sequence
        filtered_sequences = []
        for seq_blocks in sequences:
            filtered, _ = self.map_sequence_blocks(seq_blocks, active_only)
            filtered_sequences.append(filtered)
            
        # Pad to same length
        max_len = max(len(seq) for seq in filtered_sequences) if filtered_sequences else 0
        padded = [seq + [-1] * (max_len - len(seq)) for seq in filtered_sequences]
        
        block_table = torch.tensor(padded, dtype=torch.int32)
        
        # Cache for single sequence case
        if len(sequences) == 1:
            self.cached_tables[cache_key] = block_table
            
        return block_table
        
    def get_cache_invalidation_status(self) -> bool:
        """Check if block table cache needs invalidation."""
        return len(self.cached_tables) == 0
        
    def clear_cache(self):
        """Clear cached block tables."""
        self.cached_tables.clear()
        
    def get_statistics(self) -> Dict:
        """Get statistics about virtual block usage."""
        total_virtual = len(self.virtual_blocks)
        active_virtual = sum(1 for v in self.virtual_blocks.values() 
                           if v.physical_id < len(self.block_manager.blocks) 
                           and self.block_manager.blocks[v.physical_id].is_active)
        chunks = len(self.chunk_to_virtual)
        
        return {
            "total_virtual_blocks": total_virtual,
            "active_virtual_blocks": active_virtual,
            "inactive_virtual_blocks": total_virtual - active_virtual,
            "total_chunks": chunks,
            "cache_size": len(self.cached_tables)
        }