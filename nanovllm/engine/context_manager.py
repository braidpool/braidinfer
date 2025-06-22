"""
Context Manager for fine-grained KV cache control.

Provides functionality to:
- Track chunks by SHA256 hash
- Activate/deactivate chunks without deletion
- Move chunks between memory tiers (GPU/CPU/disk)
- Save and restore chunks across sessions
"""

import os
import json
import pickle
import torch
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from enum import Enum
from contextlib import contextmanager

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.virtual_block_table import VirtualBlockTable
from nanovllm.engine.context_manager_utils import resolve_chunk_hash


class ChunkType(Enum):
    INPUT = "input"
    OUTPUT = "output"

class ChunkStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CPU = "cpu"
    DISK = "disk"

@dataclass
class ChunkInfo:
    """Information about a context chunk"""
    sha256: str
    blocks: List[int]  # Block IDs
    token_ids: List[int]
    size: int  # Number of tokens
    position: Tuple[int, int]  # Start and end position in context
    metadata: Dict
    status: str  # active/inactive/cpu/disk
    chunk_type: ChunkType
    parent_chunks: List[str]  # For output chunks, track input context
    created_at: float
    memory_bytes: int  # Track memory usage
    cache_populated: bool = False  # Whether KV cache has been populated
    

class OutputTracker:
    """Context manager for tracking model outputs"""
    def __init__(self, context_manager, parent_chunks: List[str]):
        self.context_manager = context_manager
        self.parent_chunks = parent_chunks
        self.token_ids = []
        
    def add_tokens(self, token_ids: List[int]):
        """Add generated tokens"""
        self.token_ids.extend(token_ids)
    
    def finalize(self, tokenizer, metadata: Optional[Dict] = None) -> Optional[ChunkInfo]:
        """Create output chunk from collected tokens"""
        if not self.token_ids:
            return None
            
        # Create chunk from output tokens
        return self.context_manager._create_chunk_from_tokens(
            self.token_ids,
            tokenizer,
            ChunkType.OUTPUT,
            metadata,
            self.parent_chunks
        )

class ContextManager:
    """Manages context chunks with fine-grained control over KV cache"""
    
    def __init__(self, block_manager: BlockManager, config):
        self.block_manager = block_manager
        self.config = config
        
        # Chunk tracking
        self.chunks: Dict[str, ChunkInfo] = {}  # sha256 -> ChunkInfo
        self.active_chunks: Set[str] = set()  # Active chunk hashes
        
        # Memory hierarchy
        self.cpu_cache: Dict[str, Dict] = {}  # Chunks in CPU RAM
        self.disk_path = getattr(config, 'context_save_path', './saved_contexts')
        
        # Create disk directory if it doesn't exist
        if self.disk_path:
            Path(self.disk_path).mkdir(parents=True, exist_ok=True)
            
        # Track current context position
        self.current_position = 0
        
        # Output tracking
        self.output_trackers: List[OutputTracker] = []
        
        # Virtual block table for efficient filtering
        self.virtual_block_table = VirtualBlockTable(block_manager)
        
    def add_chunk(self, content: str, tokenizer, metadata: Optional[Dict] = None, populate_cache: bool = False, chunk_type: ChunkType = ChunkType.INPUT) -> ChunkInfo:
        """Add content as a managed chunk
        
        Args:
            content: Text content to add as a chunk
            tokenizer: Tokenizer to encode content
            metadata: Optional metadata for the chunk
            populate_cache: If True, immediately populate KV cache for this chunk
            
        Returns:
            ChunkInfo for the created chunk
        """
        # Tokenize content
        token_ids = tokenizer.encode(content)
        chunk = self._create_chunk_from_tokens(
            token_ids,
            tokenizer,
            chunk_type,
            metadata
        )
        
        # Optionally populate cache immediately
        if populate_cache:
            try:
                self._populate_chunk_kv_cache(chunk)
            except Exception as e:
                print(f"Warning: Failed to populate cache: {e}")
                # Don't fail the chunk creation, just warn
                
        return chunk
    
    def add_system_chunk(self, content: str, tokenizer, metadata: Optional[Dict] = None, populate_cache: bool = False) -> ChunkInfo:
        """Add content as a system message chunk that will be properly formatted
        
        Args:
            content: Text content to add as system context
            tokenizer: Tokenizer to encode content
            metadata: Optional metadata for the chunk
            populate_cache: If True, immediately populate KV cache for this chunk
            
        Returns:
            ChunkInfo for the created chunk
        """
        # Format as system message
        system_message = tokenizer.apply_chat_template(
            [{"role": "system", "content": content}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize the formatted message
        token_ids = tokenizer.encode(system_message)
        chunk = self._create_chunk_from_tokens(
            token_ids,
            tokenizer,
            ChunkType.INPUT,
            metadata
        )
        
        # Optionally populate cache immediately
        if populate_cache:
            try:
                self._populate_chunk_kv_cache(chunk)
            except Exception as e:
                print(f"Warning: Failed to populate cache: {e}")
                
        return chunk
    
    def _create_chunk_from_tokens(self, token_ids: List[int], tokenizer,
                                  chunk_type: ChunkType, metadata: Optional[Dict] = None,
                                  parent_chunks: Optional[List[str]] = None) -> ChunkInfo:
        """Internal method to create chunk from token IDs"""
        # Compute SHA256 hash of tokens
        sha256 = self.block_manager.compute_sha256(token_ids)
        
        # Check if chunk already exists
        if sha256 in self.chunks:
            chunk = self.chunks[sha256]
            # Reactivate if needed
            if chunk.status != "active":
                self.activate_chunk(sha256)
            return chunk
            
        # Allocate blocks for the chunk
        num_blocks = (len(token_ids) + self.block_manager.block_size - 1) // self.block_manager.block_size
        if len(self.block_manager.free_block_ids) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks}, have {len(self.block_manager.free_block_ids)}")
            
        # Allocate blocks and track them
        blocks = []
        for i in range(num_blocks):
            block_id = self.block_manager.free_block_ids[0]
            block = self.block_manager._allocate_block(block_id)
            
            # Set block data
            start_idx = i * self.block_manager.block_size
            end_idx = min((i + 1) * self.block_manager.block_size, len(token_ids))
            block_tokens = token_ids[start_idx:end_idx]
            
            # Compute hashes
            h = self.block_manager.compute_hash(block_tokens)
            block_sha256 = self.block_manager.compute_sha256(block_tokens)
            
            block.update(h, block_tokens, block_sha256)
            block.metadata = metadata or {}
            
            # Increment reference count since we're using this block
            block.ref_count = 1
            
            blocks.append(block_id)
            
        # Create chunk info
        start_pos = self.current_position
        end_pos = start_pos + len(token_ids)
        self.current_position = end_pos
        
        # Calculate memory usage
        num_blocks = len(blocks)
        # Get actual config values
        hf_config = getattr(self.config, 'hf_config', None)
        if hf_config:
            layers = hf_config.num_hidden_layers
            num_kv_heads = hf_config.num_key_value_heads
            head_dim = hf_config.head_dim if hasattr(hf_config, 'head_dim') else hf_config.hidden_size // hf_config.num_attention_heads
            dtype_size = hf_config.torch_dtype.itemsize if hasattr(hf_config, 'torch_dtype') else 2
        else:
            # Fallback values
            layers = 32
            num_kv_heads = 32
            head_dim = 128
            dtype_size = 2
            
        # Memory = blocks * block_size * 2 (K+V) * layers * num_kv_heads * head_dim * dtype_size
        memory_bytes = num_blocks * self.block_manager.block_size * 2 * layers * num_kv_heads * head_dim * dtype_size
        
        chunk = ChunkInfo(
            sha256=sha256,
            blocks=blocks,
            token_ids=token_ids,
            size=len(token_ids),
            position=(start_pos, end_pos),
            metadata=metadata or {},
            status="active",
            chunk_type=chunk_type,
            parent_chunks=parent_chunks or [],
            created_at=time.time(),
            memory_bytes=memory_bytes,
            cache_populated=(chunk_type == ChunkType.OUTPUT)  # Output chunks are populated during generation
        )
        
        self.chunks[sha256] = chunk
        self.active_chunks.add(sha256)
        
        # Register with virtual block table
        self.virtual_block_table.register_chunk(sha256, blocks)
        
        return chunk
    
    def populate_chunk_cache(self, chunk_hash: str) -> bool:
        """Populate the KV cache for a chunk by running it through the model
        
        This method runs a forward pass to compute and store K/V values in the
        chunk's allocated cache slots.
        
        Args:
            chunk_hash: Hash of the chunk to populate
            
        Returns:
            bool: True if cache was populated, False if already populated
        """
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        if chunk.cache_populated:
            return False
            
        if not hasattr(self, 'llm_engine') or self.llm_engine is None:
            raise RuntimeError("No LLM engine available for cache population")
            
        print(f"Populating KV cache for chunk {chunk_hash[:16]}...")
        
        # Use internal method to populate cache
        self._populate_chunk_kv_cache(chunk)
        
        return True
    
    def _populate_chunk_kv_cache(self, chunk: ChunkInfo):
        """Internal method to populate KV cache for a chunk
        
        This runs the chunk through the model using its allocated blocks,
        computing and storing K/V values in the correct cache positions.
        """
        import torch
        from nanovllm.utils.context import set_context, reset_context
        
        # Get model runner through proper chain
        if not hasattr(self, 'llm_engine') or self.llm_engine is None:
            raise RuntimeError("No LLM engine available for cache population")
            
        # Access model runner through the proper path
        # The LLM class has a model_runner attribute
        model_runner = self.llm_engine.model_runner
        
        # Prepare inputs for prefill
        input_ids = chunk.token_ids
        positions = list(range(len(chunk.token_ids)))
        
        # Calculate slot mapping for our allocated blocks
        slot_mapping = []
        for i, block_id in enumerate(chunk.blocks):
            start_idx = i * self.block_manager.block_size
            end_idx = min((i + 1) * self.block_manager.block_size, len(chunk.token_ids))
            tokens_in_block = end_idx - start_idx
            
            # Map to slots in the allocated block
            start_slot = block_id * self.block_manager.block_size
            slot_mapping.extend(range(start_slot, start_slot + tokens_in_block))
        
        # Prepare tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        positions_tensor = torch.tensor(positions, dtype=torch.int64, device='cuda')
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device='cuda')
        
        # For single sequence prefill
        cu_seqlens_q = torch.tensor([0, len(input_ids)], dtype=torch.int32, device='cuda')
        cu_seqlens_k = cu_seqlens_q  # Same for prefill
        max_seqlen_q = len(input_ids)
        max_seqlen_k = len(input_ids)
        
        # Set up context for prefill
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=None,
            active_blocks=None
        )
        
        try:
            # Run forward pass - this will compute K/V and store via store_kvcache()
            with torch.inference_mode():
                # Use the model directly to compute hidden states
                # This will trigger attention layers which call store_kvcache
                _ = model_runner.model(input_ids_tensor, positions_tensor)
            
            # Mark as populated
            chunk.cache_populated = True
            print(f"✓ KV cache populated for chunk {chunk.sha256[:16]}...")
            
        except Exception as e:
            print(f"Error populating KV cache: {e}")
            raise
        finally:
            reset_context()
    
    def populate_kv_cache_optimized(self, chunk: ChunkInfo) -> None:
        """Populate KV cache by computing only necessary attention projections
        
        This is a more efficient version that avoids full model forward pass
        when we only need to populate the KV cache.
        """
        import torch
        from nanovllm.layers.attention import store_kvcache
        
        if not hasattr(self, 'llm_engine') or self.llm_engine is None:
            raise RuntimeError("No LLM engine available for cache population")
            
        if chunk.cache_populated:
            return
            
        model = self.llm_engine.model_runner.model
        
        # Prepare inputs
        input_ids = torch.tensor(chunk.token_ids, dtype=torch.int64, device='cuda')
        positions = torch.arange(len(chunk.token_ids), device='cuda')
        
        # Create slot mapping for this chunk
        slot_mapping = []
        for i, block_id in enumerate(chunk.blocks):
            start_idx = i * self.block_manager.block_size
            end_idx = min((i + 1) * self.block_manager.block_size, len(chunk.token_ids))
            tokens_in_block = end_idx - start_idx
            
            start_slot = block_id * self.block_manager.block_size
            slot_mapping.extend(range(start_slot, start_slot + tokens_in_block))
            
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device='cuda')
        
        with torch.inference_mode():
            # Get embeddings
            hidden_states = model.model.embed_tokens(input_ids)
            
            # Process through each layer
            residual = None
            for layer_idx, layer in enumerate(model.model.layers):
                # Apply layer norm
                if residual is None:
                    residual = hidden_states
                    normed_hidden = layer.input_layernorm(hidden_states)
                else:
                    normed_hidden, residual = layer.input_layernorm(hidden_states, residual)
                
                # Get attention module
                attn = layer.self_attn
                
                # Compute QKV projections
                qkv = attn.qkv_proj(normed_hidden)
                q_size = attn.q_size
                kv_size = attn.kv_size
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                
                # Apply normalization
                q = attn.q_norm(q.view(-1, attn.num_heads, attn.head_dim)).view(q.shape)
                k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim)).view(k.shape)
                
                # Apply rotary embeddings
                q, k = attn.rotary_emb(positions, q, k)
                
                # Reshape for store_kvcache
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                
                # Store K/V in cache
                store_kvcache(k, v, attn.attn.k_cache, attn.attn.v_cache, slot_mapping_tensor)
                
                # For next layer, we need the full forward pass result
                # So we complete this layer's computation
                o = attn.attn(q, k, v)
                hidden_states = attn.o_proj(o)
                
                # MLP forward
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)
        
        chunk.cache_populated = True
        print(f"✓ KV cache populated (optimized) for chunk {chunk.sha256[:16]}...")
    
    @contextmanager
    def track_output(self, metadata: Optional[Dict] = None):
        """Context manager to track model outputs as chunks"""
        # Get current active chunks as parents
        parent_chunks = list(self.active_chunks)
        tracker = OutputTracker(self, parent_chunks)
        self.output_trackers.append(tracker)
        
        try:
            yield tracker
        finally:
            self.output_trackers.remove(tracker)
    
    def compose_chunks(self, chunk_hashes: List[str], tokenizer, metadata: Optional[Dict] = None) -> ChunkInfo:
        """Compose multiple chunks into a new chunk"""
        # Resolve potentially partial hashes
        resolved_hashes = []
        for chunk_hash in chunk_hashes:
            resolved_hash = resolve_chunk_hash(chunk_hash, self.chunks)
            resolved_hashes.append(resolved_hash)
        
        # Combine token IDs from all chunks
        combined_tokens = []
        for hash in resolved_hashes:
            chunk = self.chunks[hash]
            combined_tokens.extend(chunk.token_ids)
        
        # Create new composed chunk
        return self._create_chunk_from_tokens(
            combined_tokens,
            tokenizer,
            ChunkType.INPUT,  # Composed chunks are treated as input
            metadata,
            parent_chunks=resolved_hashes
        )
    
    def tag_chunk(self, chunk_hash: str, tag: str):
        """Add a tag to a chunk's metadata"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
        
        chunk = self.chunks[chunk_hash]
        if 'tags' not in chunk.metadata:
            chunk.metadata['tags'] = []
        if tag not in chunk.metadata['tags']:
            chunk.metadata['tags'].append(tag)
        
    def activate_chunk(self, chunk_hash: str):
        """Activate a chunk for inference"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        # If chunk is on CPU or disk, restore it first
        if chunk.status == "cpu":
            self._restore_from_cpu(chunk_hash)
        elif chunk.status == "disk":
            self.restore_chunk(chunk_hash)
            
        # Activate all blocks
        for block_id in chunk.blocks:
            self.block_manager.blocks[block_id].activate()
            
        chunk.status = "active"
        self.active_chunks.add(chunk_hash)
        
        # Update virtual block table
        self.virtual_block_table.activate_chunk(chunk_hash)
        
    def deactivate_chunk(self, chunk_hash: str):
        """Deactivate chunk (keep in cache but exclude from inference)"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        if chunk.status != "active":
            return  # Already inactive
            
        # Deactivate all blocks
        for block_id in chunk.blocks:
            self.block_manager.blocks[block_id].deactivate()
            
        chunk.status = "inactive"
        self.active_chunks.discard(chunk_hash)
        
        # Update virtual block table
        self.virtual_block_table.deactivate_chunk(chunk_hash)
        
    def extract_kv_cache_for_chunk(self, chunk: ChunkInfo) -> Optional[Dict[str, List]]:
        """Extract K/V cache tensors for a specific chunk"""
        if not chunk.cache_populated:
            return None
            
        # Access model through the chain
        if not hasattr(self, 'llm_engine') or self.llm_engine is None:
            return None
            
        k_cache_data = []
        v_cache_data = []
        
        try:
            # Access the model layers
            model = self.llm_engine.model_runner.model.model  # Qwen3Model instance
            
            for layer_idx, layer in enumerate(model.layers):
                # Get attention module that contains k_cache and v_cache
                attn_module = layer.self_attn.attn
                
                # Extract cache data for each block in this chunk
                for block_idx, block_id in enumerate(chunk.blocks):
                    # Calculate token range in this block
                    block_start = block_idx * self.block_manager.block_size
                    block_end = min(block_start + self.block_manager.block_size, len(chunk.token_ids))
                    tokens_in_block = block_end - block_start
                    
                    if tokens_in_block > 0:
                        # Extract K and V cache for this block
                        # k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
                        k_block_data = attn_module.k_cache[block_id, :tokens_in_block].clone().cpu()
                        v_block_data = attn_module.v_cache[block_id, :tokens_in_block].clone().cpu()
                        
                        k_cache_data.append({
                            'layer': layer_idx,
                            'block_id': block_id,
                            'tokens': tokens_in_block,
                            'data': k_block_data
                        })
                        v_cache_data.append({
                            'layer': layer_idx,
                            'block_id': block_id,
                            'tokens': tokens_in_block,
                            'data': v_block_data
                        })
            
            return {'k_cache': k_cache_data, 'v_cache': v_cache_data}
            
        except Exception as e:
            print(f"Warning: Failed to extract KV cache: {e}")
            return None

    def save_chunk(self, chunk_hash: str):
        """Save chunk to disk"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        # Extract actual KV cache data if populated
        kv_cache_data = None
        if chunk.cache_populated:
            kv_cache_data = self.extract_kv_cache_for_chunk(chunk)
            
        # Save to disk
        save_path = Path(self.disk_path) / f"{chunk_hash}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'chunk_info': chunk,
                'token_ids': chunk.token_ids,
                'metadata': chunk.metadata,
                'kv_cache_data': kv_cache_data  # Now includes actual tensors
            }, f)
            
            
        # Don't change status - chunk can be on disk AND in memory
        
    def restore_kv_cache_for_chunk(self, chunk: ChunkInfo, cache_data: Dict[str, List]) -> None:
        """Restore K/V cache tensors for a specific chunk"""
        if not cache_data:
            return
            
        # Access model through the chain
        if not hasattr(self, 'llm_engine') or self.llm_engine is None:
            return
            
        try:
            model = self.llm_engine.model_runner.model.model
            
            k_cache_data = cache_data['k_cache']
            v_cache_data = cache_data['v_cache']
            
            # Group by layer for efficient restoration
            by_layer = {}
            for k_entry, v_entry in zip(k_cache_data, v_cache_data):
                layer_idx = k_entry['layer']
                if layer_idx not in by_layer:
                    by_layer[layer_idx] = []
                by_layer[layer_idx].append((k_entry, v_entry))
            
            # Restore cache data
            for layer_idx, entries in by_layer.items():
                attn_module = model.layers[layer_idx].self_attn.attn
                
                for k_entry, v_entry in entries:
                    block_id = k_entry['block_id']
                    tokens = k_entry['tokens']
                    
                    # Copy data back to GPU cache
                    attn_module.k_cache[block_id, :tokens].copy_(k_entry['data'].cuda())
                    attn_module.v_cache[block_id, :tokens].copy_(v_entry['data'].cuda())
                    
            chunk.cache_populated = True
            print(f"✓ KV cache restored for chunk {chunk.sha256[:16]}...")
            
        except Exception as e:
            print(f"Warning: Failed to restore KV cache: {e}")

    def restore_chunk(self, chunk_hash: str) -> ChunkInfo:
        """Restore chunk from RAM or disk to VRAM"""
        # Try to resolve hash from existing chunks first
        try:
            full_hash = resolve_chunk_hash(chunk_hash, self.chunks)
            # If chunk exists and is in CPU RAM, restore from CPU
            if full_hash in self.chunks and self.chunks[full_hash].status == "cpu":
                return self._restore_from_cpu(full_hash)
        except ValueError:
            # Hash not found in existing chunks, continue to disk search
            pass
        
        # Try to find on disk
        disk_path = Path(self.disk_path)
        saved_files = list(disk_path.glob("*.pkl"))
        
        # Extract hashes from filenames and find matches
        saved_hashes = [f.stem for f in saved_files]
        matches = [h for h in saved_hashes if h.startswith(chunk_hash)]
        
        if len(matches) == 0:
            raise ValueError(f"Chunk not found in RAM or disk: {chunk_hash}")
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous chunk hash prefix: {chunk_hash}. Matches: {', '.join(m[:16] + '...' for m in matches)}")
            
        full_hash = matches[0]
        save_path = disk_path / f"{full_hash}.pkl"
        
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            
        chunk_info = data['chunk_info']
        kv_cache_data = data.get('kv_cache_data')
        
        # Re-add the chunk if not already loaded
        if full_hash not in self.chunks:
            self.chunks[full_hash] = chunk_info
            
            # Register with virtual block table
            if hasattr(chunk_info, 'blocks') and chunk_info.blocks:
                self.virtual_block_table.register_chunk(full_hash, chunk_info.blocks)
            
        # Restore KV cache if data is available
        if kv_cache_data and chunk_info.blocks:
            # Ensure blocks are allocated (might need to allocate if restoring to new session)
            self.restore_kv_cache_for_chunk(chunk_info, kv_cache_data)
            
        # Activate the chunk (move to VRAM)
        self.activate_chunk(full_hash)
        
        return chunk_info
        
    def unload_chunk(self, chunk_hash: str):
        """Move chunk from GPU to CPU RAM"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        if chunk.status in ["cpu", "disk"]:
            return  # Already unloaded
            
        # Extract actual KV cache data if populated
        kv_cache_data = None
        if chunk.cache_populated:
            kv_cache_data = self.extract_kv_cache_for_chunk(chunk)
            
        # Store in CPU cache
        self.cpu_cache[chunk_hash] = {
            'blocks': chunk.blocks,
            'token_ids': chunk.token_ids,
            'kv_cache_data': kv_cache_data  # Include actual KV tensors
        }
        
        # Mark blocks as CPU-resident
        for block_id in chunk.blocks:
            self.block_manager.blocks[block_id].memory_tier = "cpu"
            
        chunk.status = "cpu"
        self.active_chunks.discard(chunk_hash)
        
    def _restore_from_cpu(self, chunk_hash: str) -> ChunkInfo:
        """Restore chunk from CPU to GPU"""
        if chunk_hash not in self.cpu_cache:
            raise ValueError(f"Chunk not in CPU cache: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        cpu_data = self.cpu_cache[chunk_hash]
        
        # Restore KV cache data if available
        kv_cache_data = cpu_data.get('kv_cache_data')
        if kv_cache_data and chunk.blocks:
            self.restore_kv_cache_for_chunk(chunk, kv_cache_data)
        
        # Mark blocks as GPU-resident
        for block_id in chunk.blocks:
            self.block_manager.blocks[block_id].memory_tier = "gpu"
            
        del self.cpu_cache[chunk_hash]
        chunk.status = "active"
        self.active_chunks.add(chunk_hash)
        
        return chunk
        
    def erase_chunk(self, chunk_hash: str):
        """Completely remove a chunk"""
        # Resolve potentially partial hash
        chunk_hash = resolve_chunk_hash(chunk_hash, self.chunks)
        
        if chunk_hash not in self.chunks:
            raise ValueError(f"Unknown chunk: {chunk_hash}")
            
        chunk = self.chunks[chunk_hash]
        
        # Deallocate blocks
        for block_id in reversed(chunk.blocks):
            block = self.block_manager.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.block_manager._deallocate_block(block_id)
                
        # Remove from tracking
        del self.chunks[chunk_hash]
        self.active_chunks.discard(chunk_hash)
        
        # Remove from virtual block table
        if chunk_hash in self.virtual_block_table.chunk_to_virtual:
            # Remove virtual blocks for this chunk
            virtual_ids = self.virtual_block_table.chunk_to_virtual[chunk_hash]
            for vid in virtual_ids:
                if vid in self.virtual_block_table.virtual_blocks:
                    del self.virtual_block_table.virtual_blocks[vid]
            del self.virtual_block_table.chunk_to_virtual[chunk_hash]
            self.virtual_block_table.cached_tables.clear()
        
        # Remove from CPU cache if present
        if chunk_hash in self.cpu_cache:
            del self.cpu_cache[chunk_hash]
            
        # Remove from disk (erase removes from ALL locations)
        save_path = Path(self.disk_path) / f"{chunk_hash}.pkl"
        if save_path.exists():
            save_path.unlink()
            
    def clear_all(self):
        """Clear all chunks"""
        chunk_hashes = list(self.chunks.keys())
        for chunk_hash in chunk_hashes:
            self.erase_chunk(chunk_hash)
            
        self.current_position = 0
        
    def get_active_blocks(self) -> List[int]:
        """Get list of active block IDs for inference"""
        active_blocks = []
        for chunk_hash in self.active_chunks:
            chunk = self.chunks[chunk_hash]
            active_blocks.extend(chunk.blocks)
        return active_blocks
    
    def get_filtered_block_table(self, sequences: List[List[int]], 
                                filter_inactive: bool = True) -> torch.Tensor:
        """
        Get filtered block table for attention computation.
        
        Args:
            sequences: List of block tables from sequences
            filter_inactive: Whether to filter out inactive blocks
            
        Returns:
            Filtered block table tensor
        """
        return self.virtual_block_table.create_filtered_block_table(
            sequences, active_only=filter_inactive
        )
    
    def has_inactive_blocks(self) -> bool:
        """Check if there are any inactive blocks."""
        return len(self.active_chunks) < len(self.chunks)
    
    def get_memory_stats(self) -> Dict[str, Dict[str, int]]:
        """Get memory statistics by chunk type and tier"""
        stats = {
            'gpu': {'input': 0, 'output': 0, 'total': 0, 'count': 0},
            'cpu': {'input': 0, 'output': 0, 'total': 0, 'count': 0},
            'disk': {'input': 0, 'output': 0, 'total': 0, 'count': 0}
        }
        
        for chunk in self.chunks.values():
            # Determine memory tier based on status
            if chunk.status in ['active', 'inactive']:
                tier = 'gpu'
            elif chunk.status == 'cpu':
                tier = 'cpu'
            else:  # disk
                tier = 'disk'
                
            chunk_type = chunk.chunk_type.value
            stats[tier][chunk_type] += chunk.memory_bytes
            stats[tier]['total'] += chunk.memory_bytes
            stats[tier]['count'] += 1
        
        return stats
    
    def build_prompt_with_context(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """Build a prompt that includes active context chunks as system messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tokenizer: Tokenizer to use for formatting
            
        Returns:
            Formatted prompt string with context included
        """
        # Get all active chunks sorted by creation time
        active_chunk_infos = []
        for chunk_hash in self.active_chunks:
            chunk = self.chunks[chunk_hash]
            # Only include INPUT chunks formatted as system messages
            if chunk.chunk_type == ChunkType.INPUT:
                active_chunk_infos.append(chunk)
        
        # Sort by creation time to maintain order
        active_chunk_infos.sort(key=lambda x: x.created_at)
        
        # Build combined messages list
        combined_messages = []
        
        # Add system chunks first
        for chunk in active_chunk_infos:
            # Decode the chunk to get the original text
            # Skip chunks that look like they're already formatted
            text = tokenizer.decode(chunk.token_ids)
            if not text.startswith("<|im_start|>"):
                # Add as system message
                combined_messages.append({
                    "role": "system",
                    "content": text
                })
        
        # Add the actual messages
        combined_messages.extend(messages)
        
        # Apply chat template to combined messages
        return tokenizer.apply_chat_template(
            combined_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def get_preview(self, token_ids: List[int], tokenizer, max_length: int = 50) -> str:
        """Generate preview text from token IDs"""
        try:
            text = tokenizer.decode(token_ids[:20])  # First 20 tokens
            if len(text) > max_length:
                text = text[:max_length-3] + "..."
            # Clean up newlines and extra spaces for display
            text = ' '.join(text.split())
            return text
        except:
            return "<decode error>"
        
    def get_context_info(self) -> Dict:
        """Get current context status information"""
        total_blocks = len(self.block_manager.blocks)
        used_blocks = len(self.block_manager.used_block_ids)
        free_blocks = len(self.block_manager.free_block_ids)
        
        chunks_by_status = {
            "active": [],
            "inactive": [],
            "cpu": [],
            "disk": []
        }
        
        for chunk_hash, chunk in self.chunks.items():
            chunk_data = {
                "hash": chunk_hash,
                "size": chunk.size,
                "position": {
                    "start": chunk.position[0],
                    "end": chunk.position[1]
                },
                "metadata": chunk.metadata,
                "status": chunk.status,
                "type": chunk.chunk_type.value,
                "memory_bytes": chunk.memory_bytes,
                "parent_chunks": chunk.parent_chunks,
                "created_at": chunk.created_at,
                "cache_populated": chunk.cache_populated
            }
            chunks_by_status[chunk.status].append(chunk_data)
            
        # Check for chunks on disk
        if self.disk_path and Path(self.disk_path).exists():
            for path in Path(self.disk_path).glob("*.pkl"):
                chunk_hash = path.stem
                if chunk_hash not in self.chunks:
                    chunks_by_status["disk"].append({
                        "hash": chunk_hash,
                        "status": "disk"
                    })
                    
        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "total_context": total_blocks * self.block_manager.block_size,
            "used_context": used_blocks * self.block_manager.block_size,
            "free_context": free_blocks * self.block_manager.block_size,
            "chunks": [chunk for chunks in chunks_by_status.values() for chunk in chunks],
            "chunks_by_status": chunks_by_status,
            "virtual_block_stats": self.virtual_block_table.get_statistics()
        }