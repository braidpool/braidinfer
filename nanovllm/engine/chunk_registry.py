"""
Global chunk registry for managing reusable context chunks.
Implements LRU eviction and content-based deduplication.
"""

import time
import threading
from typing import Dict, Optional, List, Tuple, Set
from collections import OrderedDict
import torch
import pickle
import os

from nanovllm.engine.context_chunks import ContextChunk, ChunkType
from nanovllm.engine.cascade_page_manager import CascadePageManager


class ChunkRegistry:
    """
    Global registry for cached context chunks.
    
    Features:
    - Content-based deduplication using SHA256 hashes
    - LRU eviction policy when memory is full
    - Thread-safe operations
    - Optional persistence to disk
    - Memory-aware eviction
    """
    
    def __init__(self,
                 max_chunks: int = 1000,
                 eviction_batch_size: int = 10,
                 persistence_dir: Optional[str] = None):
        """
        Initialize chunk registry.
        
        Args:
            max_chunks: Maximum number of chunks to keep in memory
            eviction_batch_size: Number of chunks to evict at once
            persistence_dir: Optional directory for persistent storage
        """
        self.max_chunks = max_chunks
        self.eviction_batch_size = eviction_batch_size
        self.persistence_dir = persistence_dir
        
        # Thread-safe chunk storage (OrderedDict for LRU)
        self._lock = threading.RLock()
        self._chunks: OrderedDict[str, ContextChunk] = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_registered": 0
        }
        
        # Page manager reference (set externally)
        self.page_manager: Optional[CascadePageManager] = None
        
        # Create persistence directory if needed
        if self.persistence_dir:
            os.makedirs(self.persistence_dir, exist_ok=True)
    
    def set_page_manager(self, page_manager: CascadePageManager) -> None:
        """Set the page manager for chunk allocation."""
        self.page_manager = page_manager
    
    def register(self,
                content: str,
                chunk_type: ChunkType,
                tokenizer=None,
                compute_attention_state: bool = False) -> ContextChunk:
        """
        Register a new chunk or retrieve existing one.
        
        Args:
            content: Text content of the chunk
            chunk_type: Type of chunk (system_prompt, context, etc.)
            tokenizer: Optional tokenizer to compute token IDs
            compute_attention_state: Whether to pre-compute attention states
            
        Returns:
            Registered ContextChunk
        """
        # Create chunk to get ID
        chunk = ContextChunk.from_content(content, chunk_type)
        
        with self._lock:
            # Check if already registered
            if chunk.chunk_id in self._chunks:
                # Move to end (most recently used)
                existing_chunk = self._chunks.pop(chunk.chunk_id)
                self._chunks[chunk.chunk_id] = existing_chunk
                
                # Update access statistics
                existing_chunk.update_access(time.time())
                self.stats["hits"] += 1
                
                return existing_chunk
            
            # New chunk - check if we need to evict
            self.stats["misses"] += 1
            
            if len(self._chunks) >= self.max_chunks:
                self._evict_lru()
            
            # Tokenize if tokenizer provided
            if tokenizer:
                chunk.token_ids = tokenizer.encode(content)
                chunk.seq_len = len(chunk.token_ids)
            
            # Allocate pages if page manager available
            if self.page_manager and chunk.seq_len > 0:
                success = self.page_manager.allocate_chunk_pages(chunk)
                if not success:
                    # Try evicting and retry
                    self._evict_for_space(chunk.seq_len)
                    success = self.page_manager.allocate_chunk_pages(chunk)
                    if not success:
                        raise RuntimeError(f"Failed to allocate pages for chunk {chunk.chunk_id}")
            
            # Set timestamps
            chunk.created_at = time.time()
            chunk.last_accessed = chunk.created_at
            chunk.access_count = 1
            
            # Add to registry
            self._chunks[chunk.chunk_id] = chunk
            self.stats["total_registered"] += 1
            
            # Persist if enabled
            if self.persistence_dir:
                self._persist_chunk(chunk)
            
            return chunk
    
    def get(self, chunk_id: str) -> Optional[ContextChunk]:
        """Get chunk by ID."""
        with self._lock:
            if chunk_id in self._chunks:
                # Move to end (most recently used)
                chunk = self._chunks.pop(chunk_id)
                self._chunks[chunk_id] = chunk
                
                # Update access
                chunk.update_access(time.time())
                self.stats["hits"] += 1
                
                return chunk
            
            self.stats["misses"] += 1
            
            # Try loading from disk if persistence enabled
            if self.persistence_dir:
                chunk = self._load_chunk(chunk_id)
                if chunk:
                    # Add to in-memory cache
                    if len(self._chunks) >= self.max_chunks:
                        self._evict_lru()
                    self._chunks[chunk_id] = chunk
                    return chunk
            
            return None
    
    def get_by_content(self, content: str) -> Optional[ContextChunk]:
        """Get chunk by content (computes hash)."""
        chunk_id = ContextChunk.from_content(content, ChunkType.CONTEXT).chunk_id
        return self.get(chunk_id)
    
    def remove(self, chunk_id: str) -> bool:
        """Remove chunk from registry."""
        with self._lock:
            if chunk_id in self._chunks:
                chunk = self._chunks.pop(chunk_id)
                
                # Deallocate pages if page manager available
                if self.page_manager:
                    self.page_manager.deallocate_chunk_pages(chunk_id)
                
                # Remove from disk if persistence enabled
                if self.persistence_dir:
                    self._remove_persisted_chunk(chunk_id)
                
                return True
            
            return False
    
    def list_chunks(self, 
                   chunk_type: Optional[ChunkType] = None,
                   limit: Optional[int] = None) -> List[ContextChunk]:
        """List chunks in registry, optionally filtered by type."""
        with self._lock:
            chunks = list(self._chunks.values())
            
            if chunk_type:
                chunks = [c for c in chunks if c.chunk_type == chunk_type]
            
            # Sort by last accessed (most recent first)
            chunks.sort(key=lambda c: c.last_accessed, reverse=True)
            
            if limit:
                chunks = chunks[:limit]
            
            return chunks
    
    def get_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats["current_chunks"] = len(self._chunks)
            stats["hit_rate"] = (stats["hits"] / (stats["hits"] + stats["misses"]) 
                               if (stats["hits"] + stats["misses"]) > 0 else 0)
            
            # Add memory usage if page manager available
            if self.page_manager:
                page_info = self.page_manager.get_free_pages_info()
                stats.update(page_info)
            
            return stats
    
    def clear(self) -> None:
        """Clear all chunks from registry."""
        with self._lock:
            # Deallocate all pages
            if self.page_manager:
                for chunk_id in list(self._chunks.keys()):
                    self.page_manager.deallocate_chunk_pages(chunk_id)
            
            self._chunks.clear()
            
            # Clear persistence if enabled
            if self.persistence_dir:
                for file in os.listdir(self.persistence_dir):
                    if file.endswith(".chunk"):
                        os.remove(os.path.join(self.persistence_dir, file))
    
    def _evict_lru(self) -> None:
        """Evict least recently used chunks."""
        # Already under lock
        evict_count = min(self.eviction_batch_size, len(self._chunks))
        
        for _ in range(evict_count):
            # OrderedDict pops from beginning (oldest)
            chunk_id, chunk = self._chunks.popitem(last=False)
            
            # Deallocate pages
            if self.page_manager:
                self.page_manager.deallocate_chunk_pages(chunk_id)
            
            self.stats["evictions"] += 1
    
    def _evict_for_space(self, required_pages: int) -> None:
        """Evict chunks to free up required pages."""
        if not self.page_manager:
            return
        
        # Already under lock
        pages_per_chunk = self.page_manager.page_size
        chunks_to_evict = (required_pages + pages_per_chunk - 1) // pages_per_chunk
        
        evicted = 0
        for chunk_id in list(self._chunks.keys()):
            if evicted >= chunks_to_evict:
                break
            
            chunk = self._chunks.pop(chunk_id)
            self.page_manager.deallocate_chunk_pages(chunk_id)
            evicted += 1
            self.stats["evictions"] += 1
    
    def _persist_chunk(self, chunk: ContextChunk) -> None:
        """Persist chunk to disk."""
        if not self.persistence_dir:
            return
        
        filepath = os.path.join(self.persistence_dir, f"{chunk.chunk_id}.chunk")
        
        # Don't persist attention states (too large)
        chunk_data = {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "seq_len": chunk.seq_len,
            "token_ids": chunk.token_ids,
            "created_at": chunk.created_at,
            "num_heads": chunk.num_heads,
            "num_kv_heads": chunk.num_kv_heads,
            "head_dim": chunk.head_dim
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(chunk_data, f)
    
    def _load_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """Load chunk from disk."""
        if not self.persistence_dir:
            return None
        
        filepath = os.path.join(self.persistence_dir, f"{chunk_id}.chunk")
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, "rb") as f:
                chunk_data = pickle.load(f)
            
            # Reconstruct chunk
            chunk = ContextChunk(
                chunk_id=chunk_data["chunk_id"],
                chunk_type=chunk_data["chunk_type"],
                content=chunk_data["content"],
                seq_len=chunk_data["seq_len"],
                token_ids=chunk_data.get("token_ids"),
                created_at=chunk_data["created_at"],
                last_accessed=time.time(),
                access_count=0,
                num_heads=chunk_data.get("num_heads"),
                num_kv_heads=chunk_data.get("num_kv_heads"),
                head_dim=chunk_data.get("head_dim")
            )
            
            return chunk
            
        except Exception as e:
            print(f"Failed to load chunk {chunk_id}: {e}")
            return None
    
    def _remove_persisted_chunk(self, chunk_id: str) -> None:
        """Remove persisted chunk from disk."""
        if not self.persistence_dir:
            return
        
        filepath = os.path.join(self.persistence_dir, f"{chunk_id}.chunk")
        if os.path.exists(filepath):
            os.remove(filepath)


# Global registry instance
_global_registry: Optional[ChunkRegistry] = None


def get_global_registry() -> ChunkRegistry:
    """Get or create the global chunk registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ChunkRegistry()
    return _global_registry


def set_global_registry(registry: ChunkRegistry) -> None:
    """Set the global chunk registry."""
    global _global_registry
    _global_registry = registry