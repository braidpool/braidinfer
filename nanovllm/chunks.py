"""
Core chunk abstractions for the chunk-based API.
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import torch


class ChunkType(Enum):
    """Types of context chunks."""
    SYSTEM_PROMPT = "system_prompt"
    CONTEXT = "context"
    QUERY = "query"
    OUTPUT = "output"  # Generated output that can be reused as context


@dataclass
class Chunk:
    """
    A reusable context chunk with metadata.
    
    Chunks are identified by content hash for deduplication.
    """
    chunk_id: str  # SHA256 hash of content
    chunk_type: ChunkType
    content: str
    token_ids: List[int]
    token_count: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # KV cache allocation info (set by engine)
    kv_cache_allocated: bool = False
    cascade_level: Optional[int] = None
    page_table: Optional[List[int]] = None  # Page indices for KV cache
    kv_length: int = 0  # Number of tokens in KV cache
    position_offset: int = 0  # Starting position for RoPE embeddings
    
    @classmethod
    def from_content(cls, content: str, chunk_type: ChunkType, 
                    tokenizer=None, metadata: Optional[Dict[str, Any]] = None) -> 'Chunk':
        """Create a chunk from content with automatic ID generation."""
        chunk_id = hashlib.sha256(f"{chunk_type.value}:{content}".encode()).hexdigest()
        
        # Tokenize if tokenizer provided
        token_ids = []
        if tokenizer:
            token_ids = tokenizer.encode(content, add_special_tokens=False)
        
        return cls(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=content,
            token_ids=token_ids,
            token_count=len(token_ids),
            metadata=metadata or {}
        )
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def content_preview(self) -> str:
        """Get a preview of the content."""
        preview = self.content[:50].replace('\n', ' ')
        if len(self.content) > 50:
            preview += "..."
        return preview


class ChunkNotFoundError(Exception):
    """Raised when a requested chunk doesn't exist."""
    pass


class IncompatibleChunksError(Exception):
    """Raised when chunks cannot be composed together."""
    pass


class InvalidCompositionError(Exception):
    """Raised when chunk composition violates constraints."""
    pass