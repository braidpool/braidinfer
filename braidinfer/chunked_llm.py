"""
ChunkedLLM - Main API implementation for chunk-based inference.
"""

import sys
import time
from typing import Dict, List, Optional, Any, Union
import torch

from braidinfer import LLM, SamplingParams
from braidinfer.chunks import Chunk, ChunkType, ChunkNotFoundError, InvalidCompositionError
from braidinfer.chunk_registry import ChunkRegistry
from transformers import AutoTokenizer


class ChunkedLLM:
    """
    LLM with chunk-based context management.
    
    Manages reusable context chunks that can be composed dynamically
    for inference using FlashInfer's cascade attention mechanism.
    """
    
    def __init__(self,
                 model_path: str,
                 max_chunks: int = 1000,
                 chunk_memory_ratio: float = 0.5,
                 enable_deduplication: bool = True,
                 **llm_kwargs):
        """
        Initialize ChunkedLLM.
        
        Args:
            model_path: Path to model
            max_chunks: Maximum number of chunks to cache
            chunk_memory_ratio: Fraction of KV cache memory for chunks
            enable_deduplication: Auto-deduplicate by content hash
            **llm_kwargs: Additional arguments for LLM initialization
        """
        self.model_path = model_path
        self.chunk_memory_ratio = chunk_memory_ratio
        
        # Calculate memory allocation
        if 'num_kvcache_blocks' in llm_kwargs:
            total_blocks = llm_kwargs.pop('num_kvcache_blocks')
        else:
            total_blocks = 256
        chunk_blocks = int(total_blocks * chunk_memory_ratio)
        
        # Create LLM instance (custom kernels are now always used)
        self.llm = LLM(
            model_path,
            num_kvcache_blocks=total_blocks,
            **llm_kwargs
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize chunk registry with page manager
        self.registry = ChunkRegistry(
            max_chunks=max_chunks,
            enable_deduplication=enable_deduplication,
            page_manager=self.llm.model_runner.page_manager
        )
        
        # Track allocated chunks for KV cache management
        self._allocated_chunks: Dict[str, int] = {}  # chunk_id -> cascade_level
        
    def register_chunk(self,
                      content: str,
                      chunk_type: ChunkType,
                      metadata: Optional[Dict[str, Any]] = None,
                      global_position_start: Optional[int] = None) -> str:
        """
        Register a new chunk or retrieve existing one.
        
        Args:
            content: Text content of the chunk
            chunk_type: Type of chunk (SYSTEM_PROMPT, CONTEXT, or QUERY)
            metadata: Optional metadata dictionary
            
        Returns:
            chunk_id (SHA256 hash of content)
        """
        # Create chunk
        chunk = Chunk.from_content(
            content=content,
            chunk_type=chunk_type,
            tokenizer=self.tokenizer,
            metadata=metadata
        )
        
        # Register in registry
        chunk_id = self.registry.register(chunk)
        
        # Don't prefill immediately - wait until generation when we know global positions
        # This allows us to use correct positions for RoPE
        registered_chunk = self.registry.get(chunk_id)
        
        return chunk_id
    
    def list_chunks(self,
                   chunk_type: Optional[ChunkType] = None,
                   metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List chunks with optional filtering.
        
        Args:
            chunk_type: Filter by chunk type
            metadata_filter: Filter by metadata key-value pairs
            
        Returns:
            List of chunk info dictionaries
        """
        return self.registry.list_chunks(chunk_type, metadata_filter)
    
    def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        Get full chunk details.
        
        Args:
            chunk_id: ID of chunk to retrieve
            
        Returns:
            Chunk details dictionary
            
        Raises:
            ChunkNotFoundError if chunk doesn't exist
        """
        chunk = self.registry.get(chunk_id)
        
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type.value,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata,
            "created_at": chunk.created_at,
            "last_accessed": chunk.last_accessed,
            "access_count": chunk.access_count,
            "kv_cache_allocated": chunk.kv_cache_allocated,
            "cascade_level": chunk.cascade_level
        }
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and free its KV cache.
        
        Args:
            chunk_id: ID of chunk to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from allocated chunks
        if chunk_id in self._allocated_chunks:
            del self._allocated_chunks[chunk_id]
        
        return self.registry.delete(chunk_id)
    
    def clear_chunks(self, chunk_type: Optional[ChunkType] = None) -> int:
        """
        Clear chunks.
        
        Args:
            chunk_type: If provided, only clear chunks of this type
            
        Returns:
            Number of chunks cleared
        """
        # Clear allocations
        if chunk_type is None:
            self._allocated_chunks.clear()
        else:
            # Clear specific type
            chunks_to_clear = []
            for chunk_id in self._allocated_chunks:
                try:
                    chunk = self.registry.get(chunk_id)
                    if chunk.chunk_type == chunk_type:
                        chunks_to_clear.append(chunk_id)
                except ChunkNotFoundError:
                    chunks_to_clear.append(chunk_id)
            
            for chunk_id in chunks_to_clear:
                del self._allocated_chunks[chunk_id]
        
        return self.registry.clear(chunk_type)
    
    def evict_unused_chunks(self, max_age_seconds: float = 3600) -> int:
        """
        Evict chunks not accessed recently.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of chunks evicted
        """
        # Get chunks before eviction
        old_chunks = set(self._allocated_chunks.keys())
        
        # Evict from registry
        count = self.registry.evict_unused_chunks(max_age_seconds)
        
        # Update allocations
        for chunk_id in old_chunks:
            try:
                self.registry.get(chunk_id)  # Check if still exists
            except ChunkNotFoundError:
                del self._allocated_chunks[chunk_id]
        
        return count
    
    def generate_from_chunks(self,
                           system_chunk_id: str,
                           query_chunk_id: str,
                           context_chunk_ids: Optional[List[str]] = None,
                           sampling_params: Optional[Dict[str, Any]] = None,
                           stream: bool = False) -> Union[Dict[str, Any], Any]:
        """
        Generate output from specific chunks.
        
        Args:
            system_chunk_id: ID of system prompt chunk
            query_chunk_id: ID of query chunk
            context_chunk_ids: Optional list of context chunk IDs
            sampling_params: Generation parameters
            
        Returns:
            Dictionary with 'text' and 'token_ids'
            
        Raises:
            ChunkNotFoundError if any chunk doesn't exist
            InvalidCompositionError if composition violates constraints
        """
        # Retrieve chunks first
        system_chunk = self.registry.get(system_chunk_id)
        if system_chunk.chunk_type != ChunkType.SYSTEM_PROMPT:
            raise InvalidCompositionError(
                f"Chunk {system_chunk_id} is not a system prompt"
            )
        
        query_chunk = self.registry.get(query_chunk_id)
        if query_chunk.chunk_type != ChunkType.QUERY:
            raise InvalidCompositionError(
                f"Chunk {query_chunk_id} is not a query"
            )
        
        context_chunks = []
        if context_chunk_ids:
            for ctx_id in context_chunk_ids:
                ctx_chunk = self.registry.get(ctx_id)
                # Allow OUTPUT chunks to be used as context
                if ctx_chunk.chunk_type not in (ChunkType.CONTEXT, ChunkType.OUTPUT):
                    raise InvalidCompositionError(
                        f"Chunk {ctx_id} is not a context or output chunk"
                    )
                context_chunks.append(ctx_chunk)
        
        # First assign global positions for proper caching
        all_chunks = [system_chunk] + context_chunks + [query_chunk]
        current_position = 0
        for chunk in all_chunks:
            if chunk.token_ids:  # Skip empty chunks
                chunk.global_position_start = current_position
                chunk.global_position_end = current_position + len(chunk.token_ids)
                current_position = chunk.global_position_end
            else:
                # Empty chunk - set both to current position
                chunk.global_position_start = current_position
                chunk.global_position_end = current_position
        
        # Prefill chunks with their correct global positions
        if not system_chunk.kv_cache_allocated:
            system_chunk.cached_position_start = system_chunk.global_position_start
            self._prefill_chunk(system_chunk)
        
        for ctx_chunk in context_chunks:
            if not ctx_chunk.kv_cache_allocated:
                ctx_chunk.cached_position_start = ctx_chunk.global_position_start
                self._prefill_chunk(ctx_chunk)
        
        if not query_chunk.kv_cache_allocated:
            query_chunk.cached_position_start = query_chunk.global_position_start
            self._prefill_chunk(query_chunk)
        
        # Create composition (NO STRING BUILDING)
        composition = {
            'system_chunk': system_chunk,
            'context_chunks': context_chunks,
            'query_chunk': query_chunk
        }
        
        # Convert sampling params
        if sampling_params is None:
            sampling_params = {}
        
        # Handle both dict and SamplingParams object
        if isinstance(sampling_params, SamplingParams):
            sp = sampling_params
        else:
            sp = SamplingParams(**sampling_params)
        
        # Generate using engine's new method (NO STRING CONCATENATION)
        result = self.llm.generate_from_chunks(composition, sp, stream)
        
        # If not streaming, consume the generator to get the final result
        if not stream:
            # For non-streaming, get the final result from generator
            final_result = None
            for item in result:
                # For non-streaming, we want the last item which has the complete result
                final_result = item
            return final_result if final_result else {"text": "", "token_ids": []}
        else:
            # For streaming, return generator directly
            return result
    
    def batch_generate_from_chunks(self,
                                 requests: List[Dict[str, Any]],
                                 sampling_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate outputs for multiple chunk compositions.
        
        Args:
            requests: List of request dictionaries, each containing:
                - system_chunk_id: ID of system prompt chunk
                - query_chunk_id: ID of query chunk  
                - context_chunk_ids: Optional list of context chunk IDs
            sampling_params: Generation parameters (shared for all)
            
        Returns:
            List of output dictionaries with 'text' and 'token_ids'
        """
        prompts = []
        
        for request in requests:
            # Validate and retrieve chunks for each request
            system_chunk = self.registry.get(request["system_chunk_id"])
            query_chunk = self.registry.get(request["query_chunk_id"])
            
            context_chunks = []
            if "context_chunk_ids" in request and request["context_chunk_ids"]:
                for ctx_id in request["context_chunk_ids"]:
                    ctx_chunk = self.registry.get(ctx_id)
                    # Validate chunk type
                    if ctx_chunk.chunk_type not in (ChunkType.CONTEXT, ChunkType.OUTPUT):
                        raise InvalidCompositionError(
                            f"Chunk {ctx_id} is not a context or output chunk"
                        )
                    context_chunks.append(ctx_chunk)
            
            # Build prompt
            prompt = self._build_prompt(system_chunk, context_chunks, query_chunk)
            prompts.append(prompt)
        
        # Convert sampling params
        if sampling_params is None:
            sampling_params = {}
        
        # Handle both dict and SamplingParams object
        if isinstance(sampling_params, SamplingParams):
            sp = sampling_params
        else:
            sp = SamplingParams(**sampling_params)
        
        # Batch generate
        outputs = self.llm.generate(prompts, sp)
        
        # Format outputs
        results = []
        for output in outputs:
            results.append({
                "text": output["text"],
                "token_ids": output.get("token_ids", [])
            })
        
        return results
    
    def preview_composition(self,
                          system_chunk_id: str,
                          query_chunk_id: str,
                          context_chunk_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Preview how chunks will be composed without running inference.
        
        Args:
            system_chunk_id: ID of system prompt chunk
            query_chunk_id: ID of query chunk
            context_chunk_ids: Optional list of context chunk IDs
            
        Returns:
            Composition preview dictionary
        """
        # Retrieve chunks
        system_chunk = self.registry.get(system_chunk_id)
        query_chunk = self.registry.get(query_chunk_id)
        
        context_chunks = []
        if context_chunk_ids:
            for ctx_id in context_chunk_ids:
                context_chunks.append(self.registry.get(ctx_id))
        
        # Calculate composition
        total_tokens = system_chunk.token_count + query_chunk.token_count
        for ctx in context_chunks:
            total_tokens += ctx.token_count
        
        # Estimate cascade levels
        # Level 0: System prompt (shared across batch)
        # Level 1: Contexts (may be shared)
        # Level 2: Query (unique per request)
        num_levels = 1  # System
        if context_chunks:
            num_levels += 1  # Contexts
        num_levels += 1  # Query
        
        # Estimate memory (rough calculation)
        # KV cache size = 2 * num_layers * num_heads * head_dim * tokens * dtype_size
        # Assuming typical values for estimation
        memory_per_token = 2 * 24 * 32 * 128 * 2  # 2 bytes for fp16
        memory_bytes = total_tokens * memory_per_token
        
        return {
            "total_tokens": total_tokens,
            "num_levels": num_levels,
            "memory_bytes": memory_bytes,
            "chunks": {
                "system": system_chunk.chunk_id,
                "contexts": [c.chunk_id for c in context_chunks],
                "query": query_chunk.chunk_id
            }
        }
    
    def generate(self,
               system_prompt: str,
               query: str,
               context: Optional[Union[str, List[str]]] = None,
               sampling_params: Optional[Dict[str, Any]] = None,
               persist_chunks: bool = True) -> Dict[str, Any]:
        """
        Convenience method that registers chunks and runs inference.
        
        Args:
            system_prompt: System prompt text
            query: Query text
            context: Optional context text or list of contexts
            sampling_params: Generation parameters
            persist_chunks: Whether to keep chunks for reuse
            
        Returns:
            Dictionary with 'text', 'token_ids', and 'chunk_ids'
        """
        # Register chunks
        system_id = self.register_chunk(system_prompt, ChunkType.SYSTEM_PROMPT)
        query_id = self.register_chunk(query, ChunkType.QUERY)
        
        context_ids = []
        if context:
            if isinstance(context, str):
                context = [context]
            for ctx in context:
                ctx_id = self.register_chunk(ctx, ChunkType.CONTEXT)
                context_ids.append(ctx_id)
        
        # Generate
        output = self.generate_from_chunks(
            system_chunk_id=system_id,
            query_chunk_id=query_id,
            context_chunk_ids=context_ids if context_ids else None,
            sampling_params=sampling_params
        )
        
        # Add chunk IDs to output
        output["chunk_ids"] = {
            "system": system_id,
            "context": context_ids,
            "query": query_id
        }
        
        # Clean up if not persisting
        if not persist_chunks:
            self.delete_chunk(system_id)
            self.delete_chunk(query_id)
            for ctx_id in context_ids:
                self.delete_chunk(ctx_id)
        
        return output
    
    def get_chunk_stats(self) -> Dict[str, Any]:
        """Get chunk statistics."""
        stats = self.registry.get_stats()
        
        # Add memory usage estimation
        total_tokens = 0
        for chunk_info in self.registry.list_chunks():
            total_tokens += chunk_info["token_count"]
        
        # Rough memory estimation
        memory_per_token = 2 * 24 * 32 * 128 * 2 / (1024 * 1024)  # MB
        memory_used_mb = total_tokens * memory_per_token
        
        stats["memory_used_mb"] = memory_used_mb
        stats["total_tokens"] = total_tokens
        
        return stats
    
    def _build_prompt(self, system_chunk: Chunk, 
                     context_chunks: List[Chunk], 
                     query_chunk: Chunk) -> str:
        """Build prompt from chunks using chat template."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": system_chunk.content
        })
        
        # Build user message with contexts and query
        user_content = []
        
        # Add contexts
        for ctx in context_chunks:
            user_content.append(ctx.content)
            user_content.append("\n\n")
        
        # Add query
        user_content.append(query_chunk.content)
        
        messages.append({
            "role": "user",
            "content": "".join(user_content).strip()
        })
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _prefill_chunk(self, chunk: Chunk) -> None:
        """
        Prefill KV cache for a chunk.
        
        This method allocates pages and populates the KV cache for a chunk,
        enabling it to be reused directly in subsequent generations.
        
        Args:
            chunk: The chunk to prefill
        """
        # Check if already allocated (defensive programming)
        if chunk.kv_cache_allocated:
            return
        
        # Get token IDs for the chunk
        if not chunk.token_ids:
            # Tokenize if not already done
            chunk.token_ids = self.tokenizer.encode(chunk.content, add_special_tokens=False)
        
        # Skip empty chunks (no tokens to prefill)
        if not chunk.token_ids:
            chunk.kv_cache_allocated = True  # Mark as "allocated" even though it's empty
            chunk.page_table = []
            chunk.kv_length = 0
            return
        
        # Check if pages already allocated in page manager
        if chunk.chunk_id in self.llm.model_runner.page_manager.chunk_page_tables:
            # Pages already allocated, just update chunk metadata
            chunk.page_table = self.llm.model_runner.page_manager.chunk_page_tables[chunk.chunk_id]
            chunk.kv_length = len(chunk.token_ids)
        else:
            # Allocate pages from PageManager
            page_table = self.llm.model_runner.page_manager.allocate_for_chunk(
                chunk.chunk_id, 
                len(chunk.token_ids)
            )
            
            # Store page table in chunk
            chunk.page_table = page_table
            chunk.kv_length = len(chunk.token_ids)
            
            # Run prefill through model
            # Debug output disabled for cleaner output
            self.llm.model_runner.prefill_chunk(chunk)
        
        # Mark as allocated
        chunk.kv_cache_allocated = True
        chunk.cascade_level = self._determine_cascade_level(chunk.chunk_type)
        
        # Store allocation info
        self._allocated_chunks[chunk.chunk_id] = chunk.cascade_level
    
    def _determine_cascade_level(self, chunk_type: ChunkType) -> int:
        """
        Determine cascade level based on chunk type.
        
        Level 0: System prompts (most shared)
        Level 1: Context chunks (somewhat shared)
        Level 2: Query chunks (least shared)
        """
        if chunk_type == ChunkType.SYSTEM_PROMPT:
            return 0
        elif chunk_type in (ChunkType.CONTEXT, ChunkType.OUTPUT):
            return 1
        else:  # QUERY
            return 2
    
    def register_output_chunk_from_retained(self, 
                                          seq_id: int, 
                                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Register an output chunk from a retained sequence's KV cache.
        
        Args:
            seq_id: Sequence ID of the retained output
            metadata: Optional metadata for the chunk
            
        Returns:
            chunk_id if successful, None if sequence not found
        """
        # Get retained sequence info from the LLM engine
        retained_info = self.llm.get_retained_sequences().get(seq_id)
        if not retained_info:
            return None
        
        # Extract the text - note this already has think tags filtered
        content = retained_info["text"]
        
        # Create metadata
        chunk_metadata = metadata or {}
        chunk_metadata.update({
            "source": "output",
            "seq_id": seq_id,
            "prompt_length": retained_info["prompt_length"],
            "completion_length": retained_info["completion_length"],
            "has_think_tags": retained_info["think_positions"] is not None
        })
        
        # Register as OUTPUT chunk
        chunk_id = self.register_chunk(
            content=content,
            chunk_type=ChunkType.OUTPUT,
            metadata=chunk_metadata
        )
        
        # Store the KV cache info with the chunk
        chunk = self.registry.get(chunk_id)
        chunk.kv_cache_allocated = True
        chunk.cascade_level = 1  # OUTPUT chunks go to level 1
        
        # Track in allocated chunks
        self._allocated_chunks[chunk_id] = 1
        
        return chunk_id
    
    def generate_and_retain_output(self,
                                 system_prompt: str,
                                 query: str,
                                 context: Optional[Union[str, List[str]]] = None,
                                 sampling_params: Optional[Dict[str, Any]] = None,
                                 persist_chunks: bool = True) -> Dict[str, Any]:
        """
        Generate output and optionally retain it as a reusable chunk.
        
        This is like generate() but sets retain_output_cache=True in sampling params.
        
        Returns:
            Dictionary with 'text', 'token_ids', 'chunk_ids', and 'output_chunk_id'
        """
        # Ensure sampling params exist and set retain flag
        if sampling_params is None:
            sampling_params = {}
        sampling_params = sampling_params.copy()  # Don't modify caller's dict
        sampling_params["retain_output_cache"] = True
        
        # Generate with retention
        output = self.generate(
            system_prompt=system_prompt,
            query=query,
            context=context,
            sampling_params=sampling_params,
            persist_chunks=persist_chunks
        )
        
        # Get the retained sequence ID (it should be the last one)
        retained_seqs = self.llm.get_retained_sequences()
        if retained_seqs:
            # Get the most recent sequence
            seq_id = max(retained_seqs.keys())
            
            # Register it as an output chunk
            output_chunk_id = self.register_output_chunk_from_retained(
                seq_id=seq_id,
                metadata={"query": query}
            )
            
            output["output_chunk_id"] = output_chunk_id
            output["retained_seq_id"] = seq_id
        
        return output