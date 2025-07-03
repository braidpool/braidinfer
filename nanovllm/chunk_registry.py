"""
Chunk registry for managing reusable context chunks.
Implements LRU eviction and content-based deduplication.
"""

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from nanovllm.chunks import Chunk, ChunkType, ChunkNotFoundError


class ChunkRegistry:
    """
    Registry for cached context chunks with LRU eviction.
    
    Features:
    - Content-based deduplication using SHA256 hashes
    - LRU eviction policy when memory is full
    - Thread-safe operations
    - Metadata filtering
    """
    
    def __init__(self, max_chunks: int = 1000, enable_deduplication: bool = True):
        """
        Initialize chunk registry.
        
        Args:
            max_chunks: Maximum number of chunks to keep in memory
            enable_deduplication: Whether to deduplicate by content hash
        """
        self.max_chunks = max_chunks
        self.enable_deduplication = enable_deduplication
        
        # Thread-safe chunk storage (OrderedDict for LRU)
        self._lock = threading.RLock()
        self._chunks: OrderedDict[str, Chunk] = OrderedDict()
        
        # Statistics
        self.stats = {
            "total_registered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0
        }
    
    def register(self, chunk: Chunk) -> str:
        """
        Register a new chunk or retrieve existing one.
        
        Args:
            chunk: Chunk to register
            
        Returns:
            chunk_id of registered chunk
        """
        with self._lock:
            # Check if already registered (deduplication)
            if self.enable_deduplication and chunk.chunk_id in self._chunks:
                # Move to end (most recently used)
                existing_chunk = self._chunks.pop(chunk.chunk_id)
                self._chunks[chunk.chunk_id] = existing_chunk
                
                # Update access statistics
                existing_chunk.update_access()
                self.stats["cache_hits"] += 1
                
                return existing_chunk.chunk_id
            
            # New chunk
            self.stats["cache_misses"] += 1
            
            # Check if we need to evict
            if len(self._chunks) >= self.max_chunks:
                self._evict_lru()
            
            # Add to registry
            self._chunks[chunk.chunk_id] = chunk
            self.stats["total_registered"] += 1
            
            return chunk.chunk_id
    
    def get(self, chunk_id: str) -> Chunk:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: ID of chunk to retrieve
            
        Returns:
            Chunk object
            
        Raises:
            ChunkNotFoundError if chunk doesn't exist
        """
        with self._lock:
            if chunk_id not in self._chunks:
                raise ChunkNotFoundError(f"Chunk {chunk_id} not found")
            
            # Move to end (most recently used)
            chunk = self._chunks.pop(chunk_id)
            self._chunks[chunk_id] = chunk
            
            # Update access
            chunk.update_access()
            self.stats["cache_hits"] += 1
            
            return chunk
    
    def list_chunks(self, chunk_type: Optional[ChunkType] = None,
                   metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List chunks with optional filtering.
        
        Args:
            chunk_type: Filter by chunk type
            metadata_filter: Filter by metadata key-value pairs
            
        Returns:
            List of chunk info dictionaries
        """
        with self._lock:
            chunks_info = []
            
            for chunk in self._chunks.values():
                # Apply filters
                if chunk_type and chunk.chunk_type != chunk_type:
                    continue
                
                if metadata_filter:
                    match = all(
                        chunk.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                # Create info dict
                chunks_info.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type.value,
                    "content_preview": chunk.content_preview,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata.copy(),
                    "created_at": datetime.fromtimestamp(chunk.created_at).isoformat(),
                    "last_accessed": datetime.fromtimestamp(chunk.last_accessed).isoformat(),
                    "access_count": chunk.access_count,
                    "kv_cache_allocated": chunk.kv_cache_allocated
                })
            
            return chunks_info
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk.
        
        Args:
            chunk_id: ID of chunk to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                return True
            return False
    
    def clear(self, chunk_type: Optional[ChunkType] = None) -> int:
        """
        Clear chunks.
        
        Args:
            chunk_type: If provided, only clear chunks of this type
            
        Returns:
            Number of chunks cleared
        """
        with self._lock:
            if chunk_type is None:
                count = len(self._chunks)
                self._chunks.clear()
                return count
            
            # Clear specific type
            to_delete = [
                chunk_id for chunk_id, chunk in self._chunks.items()
                if chunk.chunk_type == chunk_type
            ]
            
            for chunk_id in to_delete:
                del self._chunks[chunk_id]
            
            return len(to_delete)
    
    def evict_unused_chunks(self, max_age_seconds: float) -> int:
        """
        Evict chunks not accessed recently.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of chunks evicted
        """
        import time
        current_time = time.time()
        
        with self._lock:
            to_delete = [
                chunk_id for chunk_id, chunk in self._chunks.items()
                if (current_time - chunk.last_accessed) > max_age_seconds
            ]
            
            for chunk_id in to_delete:
                del self._chunks[chunk_id]
                self.stats["evictions"] += 1
            
            return len(to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total_chunks = len(self._chunks)
            hit_rate = 0.0
            if self.stats["cache_hits"] + self.stats["cache_misses"] > 0:
                hit_rate = self.stats["cache_hits"] / (
                    self.stats["cache_hits"] + self.stats["cache_misses"]
                )
            
            return {
                "total_chunks": total_chunks,
                "max_chunks": self.max_chunks,
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "total_registered": self.stats["total_registered"]
            }
    
    def _evict_lru(self, count: int = 1) -> None:
        """Evict least recently used chunks."""
        for _ in range(min(count, len(self._chunks))):
            if self._chunks:
                # Remove oldest (first) item
                self._chunks.popitem(last=False)
                self.stats["evictions"] += 1