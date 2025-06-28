"""
Context chunk system for compositional caching with Cascade Attention.
Supports multi-head attention and content-based deduplication.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import torch


class ChunkType(Enum):
    """Types of context chunks."""
    SYSTEM_PROMPT = "system_prompt"
    CONTEXT = "context"
    QUERY = "query"
    RAG_RESULT = "rag_result"
    CODE = "code"


@dataclass
class ContextChunk:
    """
    A reusable context chunk with pre-computed KV cache.
    
    Chunks are identified by content hash for deduplication.
    Supports multi-head attention with per-head states.
    """
    chunk_id: str  # SHA256 hash of content
    chunk_type: ChunkType
    content: str
    seq_len: int
    
    # KV cache page allocation
    kv_page_indices: List[int] = field(default_factory=list)
    kv_page_count: int = 0
    last_page_len: int = 0
    
    # Pre-computed attention states (optional optimization)
    # Shape: [seq_len, num_heads, head_dim] for V
    # Shape: [seq_len, num_heads] for S (logsumexp)
    v_state: Optional[torch.Tensor] = None
    s_state: Optional[torch.Tensor] = None
    
    # Metadata
    token_ids: Optional[List[int]] = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    
    # Multi-head configuration
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    
    @classmethod
    def from_content(cls, content: str, chunk_type: ChunkType) -> 'ContextChunk':
        """Create a chunk from content with automatic ID generation."""
        chunk_id = hashlib.sha256(content.encode()).hexdigest()
        return cls(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=content,
            seq_len=0  # Will be set when tokenized
        )
    
    def compute_attention_state(self, 
                               v: torch.Tensor, 
                               s: torch.Tensor) -> None:
        """
        Store pre-computed attention states for this chunk.
        
        Args:
            v: Attention output [seq_len, num_heads, head_dim]
            s: LogSumExp values [seq_len, num_heads]
        """
        self.v_state = v.clone()
        self.s_state = s.clone()
        if self.num_heads is None:
            self.num_heads = v.shape[1]
            self.head_dim = v.shape[2]
    
    def has_attention_state(self) -> bool:
        """Check if pre-computed attention states are available."""
        return self.v_state is not None and self.s_state is not None
    
    def get_kv_indptr(self) -> Tuple[int, int]:
        """Get start and end indices for KV cache pages."""
        if not self.kv_page_indices:
            return (0, 0)
        return (self.kv_page_indices[0], 
                self.kv_page_indices[0] + self.kv_page_count)
    
    def update_access(self, timestamp: float) -> None:
        """Update access statistics."""
        self.last_accessed = timestamp
        self.access_count += 1


@dataclass
class ChunkComposition:
    """
    Represents a composition of multiple chunks for a request.
    Defines the cascade levels and ordering.
    """
    chunks: List[ContextChunk]
    query_text: str
    
    def get_cascade_levels(self) -> List[List[ContextChunk]]:
        """
        Organize chunks into cascade levels.
        
        Returns:
            List of chunk lists, one per cascade level:
            - Level 0: System prompts (shared across batch)
            - Level 1: Context chunks (docs, code, RAG)
            - Level 2: User queries (unique per request)
        """
        levels = [[], [], []]
        
        for chunk in self.chunks:
            if chunk.chunk_type == ChunkType.SYSTEM_PROMPT:
                levels[0].append(chunk)
            elif chunk.chunk_type in [ChunkType.CONTEXT, ChunkType.RAG_RESULT, ChunkType.CODE]:
                levels[1].append(chunk)
            elif chunk.chunk_type == ChunkType.QUERY:
                levels[2].append(chunk)
        
        # Remove empty levels
        levels = [level for level in levels if level]
        
        return levels
    
    def get_total_length(self) -> int:
        """Get total sequence length across all chunks."""
        return sum(chunk.seq_len for chunk in self.chunks)
    
    def validate_head_compatibility(self) -> bool:
        """Ensure all chunks have compatible head configurations."""
        if not self.chunks:
            return True
        
        first_chunk = next((c for c in self.chunks if c.num_heads is not None), None)
        if not first_chunk:
            return True
        
        num_heads = first_chunk.num_heads
        num_kv_heads = first_chunk.num_kv_heads
        head_dim = first_chunk.head_dim
        
        for chunk in self.chunks:
            if chunk.num_heads is not None:
                if (chunk.num_heads != num_heads or 
                    chunk.num_kv_heads != num_kv_heads or
                    chunk.head_dim != head_dim):
                    return False
        
        return True


class ChunkBuilder:
    """Helper class to build and validate chunk compositions."""
    
    def __init__(self, max_seq_len: int = 32768):
        self.max_seq_len = max_seq_len
    
    def build_composition(self,
                         system_prompt: Optional[str] = None,
                         context_chunks: Optional[List[str]] = None,
                         query: str = "") -> ChunkComposition:
        """
        Build a chunk composition from components.
        
        Args:
            system_prompt: Optional system prompt text
            context_chunks: Optional list of context texts
            query: User query text
            
        Returns:
            ChunkComposition ready for cascade attention
        """
        chunks = []
        
        if system_prompt:
            chunk = ContextChunk.from_content(system_prompt, ChunkType.SYSTEM_PROMPT)
            chunks.append(chunk)
        
        if context_chunks:
            for ctx in context_chunks:
                chunk = ContextChunk.from_content(ctx, ChunkType.CONTEXT)
                chunks.append(chunk)
        
        if query:
            chunk = ContextChunk.from_content(query, ChunkType.QUERY)
            chunks.append(chunk)
        
        composition = ChunkComposition(chunks=chunks, query_text=query)
        
        # Validate total length
        if composition.get_total_length() > self.max_seq_len:
            raise ValueError(f"Composition exceeds max length: {composition.get_total_length()} > {self.max_seq_len}")
        
        return composition