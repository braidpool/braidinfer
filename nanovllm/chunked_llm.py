"""
ChunkedLLM - Main API implementation for chunk-based inference.
"""

import time
from typing import Dict, List, Optional, Any, Union
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.chunks import Chunk, ChunkType, ChunkNotFoundError, InvalidCompositionError
from nanovllm.chunk_registry import ChunkRegistry
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
        
        # Initialize chunk registry
        self.registry = ChunkRegistry(
            max_chunks=max_chunks,
            enable_deduplication=enable_deduplication
        )
        
        # Calculate memory allocation
        if 'num_kvcache_blocks' in llm_kwargs:
            total_blocks = llm_kwargs.pop('num_kvcache_blocks')
        else:
            total_blocks = 256
        chunk_blocks = int(total_blocks * chunk_memory_ratio)
        
        # Initialize base LLM with cascade attention
        self.llm = LLM(
            model_path,
            enable_cascade_attention=True,
            num_kvcache_blocks=total_blocks,
            **llm_kwargs
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Track allocated chunks for KV cache management
        self._allocated_chunks: Dict[str, int] = {}  # chunk_id -> cascade_level
        
    def register_chunk(self,
                      content: str,
                      chunk_type: ChunkType,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
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
                           sampling_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        # Validate and retrieve chunks
        system_chunk = self.registry.get(system_chunk_id)
        query_chunk = self.registry.get(query_chunk_id)
        
        if system_chunk.chunk_type != ChunkType.SYSTEM_PROMPT:
            raise InvalidCompositionError(
                f"Chunk {system_chunk_id} is not a system prompt"
            )
        
        if query_chunk.chunk_type != ChunkType.QUERY:
            raise InvalidCompositionError(
                f"Chunk {query_chunk_id} is not a query"
            )
        
        context_chunks = []
        if context_chunk_ids:
            for ctx_id in context_chunk_ids:
                ctx_chunk = self.registry.get(ctx_id)
                if ctx_chunk.chunk_type != ChunkType.CONTEXT:
                    raise InvalidCompositionError(
                        f"Chunk {ctx_id} is not a context chunk"
                    )
                context_chunks.append(ctx_chunk)
        
        # Build prompt from chunks
        prompt = self._build_prompt(system_chunk, context_chunks, query_chunk)
        
        # Convert sampling params
        if sampling_params is None:
            sampling_params = {}
        
        sp = SamplingParams(**sampling_params)
        
        # Generate
        outputs = self.llm.generate([prompt], sp)
        
        if not outputs:
            return {"text": "", "token_ids": []}
        
        return {
            "text": outputs[0]["text"],
            "token_ids": outputs[0].get("token_ids", [])
        }
    
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
                    context_chunks.append(self.registry.get(ctx_id))
            
            # Build prompt
            prompt = self._build_prompt(system_chunk, context_chunks, query_chunk)
            prompts.append(prompt)
        
        # Convert sampling params
        if sampling_params is None:
            sampling_params = {}
        
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
    
    def _prefill_chunk(self, chunk: Chunk, position_offset: int) -> None:
        """
        Prefill KV cache for a chunk with position-aware generation.
        
        This method uses the lower-level model API to populate the KV cache
        for a chunk at the specified position offset, ensuring correct RoPE
        embeddings for composed generation.
        
        Args:
            chunk: The chunk to prefill
            position_offset: Starting position for this chunk in the composed sequence
        """
        # Get token IDs for the chunk
        token_ids = chunk.token_ids
        if not token_ids:
            # Tokenize if not already done
            token_ids = self.tokenizer.encode(chunk.content, add_special_tokens=False)
            chunk.token_ids = token_ids
        
        # Create position tensor with offset
        positions = torch.arange(
            position_offset, 
            position_offset + len(token_ids),
            dtype=torch.long,
            device="cuda"
        )
        
        # Convert token IDs to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        
        # Get the model from llm
        model = self.llm.model_runner.model
        
        # Create a minimal inference context for KV cache storage
        # This needs to be set up properly with page allocation
        # For now, we'll document the approach:
        
        # 1. Allocate KV cache pages for this chunk
        # 2. Create sequence object for tracking
        # 3. Run model forward pass with positions
        # 4. Store KV cache reference for chunk composition
        
        # TODO: Complete implementation with proper page allocation
        # This requires integration with the page manager and cascade attention setup
        
        # Forward pass through model to populate KV cache
        with torch.no_grad():
            # Create InferenceContext with chunk information
            from nanovllm.engine.inference_context import InferenceContext
            from nanovllm.engine.sequence import Sequence
            
            # Create a sequence for this chunk
            seq = Sequence(
                seq_id=f"chunk_{chunk.chunk_id}",
                prompt_token_ids=token_ids,
                max_tokens=0,  # No generation, just prefill
                temperature=1.0,
                top_p=1.0,
                top_k=-1
            )
            
            # Set up context for prefill
            context = InferenceContext(
                is_prefill=True,
                sequences=[seq],
                page_manager=self.llm.model_runner.page_manager,
                wrapper=self.llm.model_runner.prefill_wrapper
            )
            
            # Run model forward pass
            hidden_states = model(input_ids, positions, context)
            
            # Mark chunk as having KV cache allocated
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
        elif chunk_type == ChunkType.CONTEXT:
            return 1
        else:  # QUERY
            return 2