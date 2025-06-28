"""
Cascade-aware scheduler for nano-vllm.
Groups sequences by shared context chunks for efficient batch processing.
"""

from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import torch

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.cascade_page_manager import CascadePageManager
from nanovllm.engine.context_chunks import ContextChunk, ChunkComposition, ChunkType
from nanovllm.engine.chunk_registry import get_global_registry
from nanovllm.layers.cascade_attention import CascadeConfig


class CascadeSequence(Sequence):
    """Extended sequence with cascade chunk information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_composition: Optional[ChunkComposition] = None
        self.chunk_ids: List[str] = []


class CascadeScheduler(Scheduler):
    """
    Scheduler that supports cascade attention with chunk composition.
    
    Groups sequences by shared chunks to maximize cascade efficiency.
    Creates multi-level cascade configurations for batch processing.
    """
    
    def __init__(self, config: Config):
        # Initialize base scheduler but replace page manager
        super().__init__(config)
        
        # Replace with cascade page manager
        hf_config = config.hf_config
        self.page_manager = CascadePageManager(
            num_pages=config.num_kvcache_blocks,
            page_size=config.kvcache_block_size,
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            dtype=hf_config.torch_dtype,
            chunk_page_ratio=getattr(config, 'chunk_page_ratio', 0.5)
        )
        
        # Chunk registry
        self.chunk_registry = get_global_registry()
        self.chunk_registry.set_page_manager(self.page_manager)
        
        # Cascade configuration
        self.enable_cascade = getattr(config, 'enable_cascade_attention', True)
        self.max_cascade_levels = getattr(config, 'max_cascade_levels', 3)
        
        # Tracking for chunk-based grouping
        self.chunk_groups: Dict[frozenset, List[Sequence]] = defaultdict(list)
    
    def add(self, seq: Sequence):
        """Add sequence and register its chunks if cascade enabled."""
        if self.enable_cascade and isinstance(seq, CascadeSequence):
            if seq.chunk_composition:
                # Register chunks and allocate pages
                for chunk in seq.chunk_composition.chunks:
                    if chunk.seq_len == 0 and hasattr(seq, 'tokenizer'):
                        # Tokenize chunk if needed
                        chunk.token_ids = seq.tokenizer.encode(chunk.content)
                        chunk.seq_len = len(chunk.token_ids)
                    
                    # Register with chunk registry (handles deduplication)
                    registered_chunk = self.chunk_registry.register(
                        chunk.content,
                        chunk.chunk_type,
                        tokenizer=getattr(seq, 'tokenizer', None)
                    )
                    
                    # Update sequence's chunk IDs
                    seq.chunk_ids.append(registered_chunk.chunk_id)
                
                # Register chunks with page manager
                self.page_manager.register_sequence_chunks(
                    seq.seq_id,
                    seq.chunk_composition.chunks
                )
        
        super().add(seq)
    
    def schedule(self) -> Tuple[List[Sequence], bool, Optional[CascadeConfig]]:
        """
        Schedule sequences with cascade-aware batching.
        
        Returns:
            - List of scheduled sequences
            - Whether this is a prefill batch
            - Optional cascade configuration
        """
        if not self.enable_cascade:
            # Fall back to regular scheduling
            seqs, is_prefill = super().schedule()
            return seqs, is_prefill, None
        
        # Try cascade-aware prefill scheduling
        scheduled_seqs, cascade_config = self._schedule_cascade_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True, cascade_config
        
        # Try cascade-aware decode scheduling
        scheduled_seqs, cascade_config = self._schedule_cascade_decode()
        if scheduled_seqs:
            return scheduled_seqs, False, cascade_config
        
        return [], False, None
    
    def _schedule_cascade_prefill(self) -> Tuple[List[Sequence], Optional[CascadeConfig]]:
        """Schedule prefill with cascade grouping."""
        if not self.waiting:
            return [], None
        
        # Group waiting sequences by shared chunks
        chunk_groups = self._group_sequences_by_chunks(list(self.waiting))
        
        # Find best group to schedule
        best_group = None
        best_seqs = []
        best_score = -1
        
        for chunk_set, seqs in chunk_groups.items():
            # Score based on number of sequences and shared chunks
            score = len(seqs) * len(chunk_set)
            
            # Check if we can schedule this group
            num_tokens = sum(len(seq) for seq in seqs)
            if (len(seqs) <= self.max_num_seqs and 
                num_tokens <= self.max_num_batched_tokens and
                all(self.page_manager.can_allocate(seq) for seq in seqs)):
                
                if score > best_score:
                    best_score = score
                    best_group = chunk_set
                    best_seqs = seqs
        
        if not best_seqs:
            # Fall back to regular prefill if no good cascade group
            return self._fallback_prefill()
        
        # Schedule the best group
        scheduled_seqs = []
        for seq in best_seqs:
            self.page_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.remove(seq)
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # Build cascade configuration
        cascade_config = self._build_cascade_config(scheduled_seqs)
        
        return scheduled_seqs, cascade_config
    
    def _schedule_cascade_decode(self) -> Tuple[List[Sequence], Optional[CascadeConfig]]:
        """Schedule decode with cascade grouping."""
        if not self.running:
            return [], None
        
        # Group running sequences by shared chunks
        chunk_groups = self._group_sequences_by_chunks(list(self.running))
        
        # Schedule sequences from the same chunk group
        scheduled_seqs = []
        scheduled_chunks = set()
        
        for chunk_set, seqs in chunk_groups.items():
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            
            # Try to schedule sequences from this group
            group_scheduled = []
            for seq in seqs:
                if len(scheduled_seqs) + len(group_scheduled) >= self.max_num_seqs:
                    break
                
                if self.page_manager.can_append(seq):
                    self.page_manager.may_append(seq)
                    group_scheduled.append(seq)
                    self.running.remove(seq)
            
            if group_scheduled:
                scheduled_seqs.extend(group_scheduled)
                scheduled_chunks.update(chunk_set)
        
        if not scheduled_seqs:
            return [], None
        
        # Re-add scheduled sequences to running
        self.running.extendleft(reversed(scheduled_seqs))
        
        # Build cascade configuration
        cascade_config = self._build_cascade_config(scheduled_seqs)
        
        return scheduled_seqs, cascade_config
    
    def _group_sequences_by_chunks(self, sequences: List[Sequence]) -> Dict[frozenset, List[Sequence]]:
        """Group sequences by their shared chunks."""
        chunk_groups = defaultdict(list)
        
        for seq in sequences:
            if isinstance(seq, CascadeSequence) and seq.chunk_ids:
                # Use frozenset of chunk IDs as key
                chunk_key = frozenset(seq.chunk_ids)
            else:
                # Sequences without chunks go in their own group
                chunk_key = frozenset([f"seq_{seq.seq_id}"])
            
            chunk_groups[chunk_key].append(seq)
        
        return dict(chunk_groups)
    
    def _build_cascade_config(self, sequences: List[Sequence]) -> Optional[CascadeConfig]:
        """Build cascade configuration for scheduled sequences."""
        if not sequences:
            return None
        
        # Collect all unique chunks
        all_chunks = []
        chunk_map = {}
        
        for seq in sequences:
            if isinstance(seq, CascadeSequence) and seq.chunk_composition:
                for chunk in seq.chunk_composition.chunks:
                    if chunk.chunk_id not in chunk_map:
                        chunk_map[chunk.chunk_id] = chunk
                        all_chunks.append(chunk)
        
        if not all_chunks:
            return None
        
        # Organize chunks into cascade levels
        cascade_levels = self._organize_cascade_levels(all_chunks, sequences)
        
        if not cascade_levels:
            return None
        
        # Build indices for cascade attention
        is_prefill = sequences[0].status == SequenceStatus.RUNNING and len(sequences[0]) > 0
        
        level_indices = self.page_manager.build_cascade_indices(
            sequences, cascade_levels, for_prefill=is_prefill
        )
        
        qo_indptr_arr = self.page_manager.build_qo_indptr_array(
            sequences, cascade_levels
        )
        
        # Extract arrays from level indices
        kv_indices_arr = [indices[0] for indices in level_indices]
        kv_indptr_arr = [indices[1] for indices in level_indices]
        last_page_len_arr = [indices[2] for indices in level_indices]
        
        # Create cascade configuration
        config = CascadeConfig(
            num_levels=len(cascade_levels),
            qo_indptr_arr=qo_indptr_arr,
            paged_kv_indptr_arr=kv_indptr_arr,
            paged_kv_indices_arr=kv_indices_arr,
            paged_kv_last_page_len_arr=last_page_len_arr,
            page_size=self.page_manager.page_size
        )
        
        return config
    
    def _organize_cascade_levels(self, 
                                chunks: List[ContextChunk], 
                                sequences: List[Sequence]) -> List[List[ContextChunk]]:
        """
        Organize chunks into cascade levels based on sharing patterns.
        
        Level 0: Chunks shared by all sequences
        Level 1: Chunks shared by some sequences
        Level 2: Unique chunks per sequence
        """
        # Count chunk usage across sequences
        chunk_usage = defaultdict(int)
        for seq in sequences:
            if isinstance(seq, CascadeSequence):
                for chunk_id in seq.chunk_ids:
                    chunk_usage[chunk_id] += 1
        
        total_seqs = len(sequences)
        
        # Categorize chunks by usage
        shared_by_all = []
        shared_by_some = []
        unique_chunks = []
        
        for chunk in chunks:
            usage = chunk_usage.get(chunk.chunk_id, 0)
            
            if usage == total_seqs:
                shared_by_all.append(chunk)
            elif usage > 1:
                shared_by_some.append(chunk)
            else:
                unique_chunks.append(chunk)
        
        # Build levels (filter out empty levels)
        levels = []
        
        if shared_by_all:
            levels.append(shared_by_all)
        
        if shared_by_some:
            levels.append(shared_by_some)
        
        if unique_chunks or not levels:
            # Always need at least one level
            levels.append(unique_chunks if unique_chunks else chunks)
        
        # Limit to max cascade levels
        if len(levels) > self.max_cascade_levels:
            # Merge lower levels
            merged_level = []
            for level in levels[self.max_cascade_levels - 1:]:
                merged_level.extend(level)
            levels = levels[:self.max_cascade_levels - 1] + [merged_level]
        
        return levels
    
    def _fallback_prefill(self) -> Tuple[List[Sequence], Optional[CascadeConfig]]:
        """Fallback to regular prefill when cascade grouping fails."""
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.page_manager.can_allocate(seq):
                break
            
            num_seqs += 1
            self.page_manager.allocate(seq)
            num_batched_tokens += len(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # Try to build cascade config even for fallback
        cascade_config = self._build_cascade_config(scheduled_seqs) if scheduled_seqs else None
        
        return scheduled_seqs, cascade_config
    
    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> List[bool]:
        """Postprocess and update page manager sequence lengths."""
        # First run parent postprocess
        super().postprocess(seqs, token_ids)
        
        # Update sequence lengths in page manager for cascade tracking
        is_prefill = all(seq.status == SequenceStatus.RUNNING and len(seq) > 1 for seq in seqs)
        self.page_manager.update_sequence_lengths(seqs, is_prefill)