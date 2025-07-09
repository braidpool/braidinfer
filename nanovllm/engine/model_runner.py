"""
Model runner for single-GPU nano-vllm.
"""

import torch
from typing import List, Optional

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.inference_context import InferenceContext
from nanovllm.engine.model_loader import ModelLoader
from nanovllm.engine.errors import ErrorContext, handle_inference_error, InferenceError
from nanovllm.engine.metrics import MetricsContext, get_metrics_collector
from nanovllm.layers.sampler import Sampler


class ModelRunner:
    """Model runner for single-GPU nano-vllm with custom attention kernels."""
    
    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        
        # Page manager will be set by scheduler
        self.page_manager: Optional[object] = None
        
        # Setup CUDA device
        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        
        # Handle different config formats for dtype
        if hasattr(self.hf_config, 'torch_dtype') and self.hf_config.torch_dtype is not None:
            torch.set_default_dtype(self.hf_config.torch_dtype)
        else:
            torch.set_default_dtype(torch.float16)  # Default to fp16
            
        torch.set_default_device("cuda")
        
        # Load model and sampler
        self.model = ModelLoader.load_model(config)
        self.sampler = ModelLoader.create_sampler()
        
        # Store attention params from config
        # Handle different config formats
        if hasattr(self.hf_config, 'num_key_value_heads'):
            self.num_kv_heads = self.hf_config.num_key_value_heads
        elif hasattr(self.hf_config, 'num_attention_heads'):
            self.num_kv_heads = self.hf_config.num_attention_heads
        else:
            self.num_kv_heads = self.hf_config.n_head  # GPT-2
            
        if hasattr(self.hf_config, 'num_attention_heads'):
            self.num_qo_heads = self.hf_config.num_attention_heads
        else:
            self.num_qo_heads = self.hf_config.n_head  # GPT-2
            
        if hasattr(self.hf_config, 'head_dim'):
            self.head_dim = self.hf_config.head_dim
        elif hasattr(self.hf_config, 'hidden_size'):
            self.head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        else:
            self.head_dim = self.hf_config.n_embd // self.hf_config.n_head  # GPT-2
        
        # Handle dtype
        if hasattr(self.hf_config, 'torch_dtype') and self.hf_config.torch_dtype is not None:
            self.dtype = self.hf_config.torch_dtype
        else:
            self.dtype = torch.float16
            
        
        # Warmup model to compile CUDA kernels
        ModelLoader.warmup_model(self.model)
        
        # Restore default dtype
        torch.set_default_dtype(default_dtype)
    
    def set_page_manager(self, page_manager):
        """Set the page manager for KV cache."""
        self.page_manager = page_manager
        
        # Set KV cache reference in all attention layers
        if page_manager:
            from nanovllm.layers.attention import Attention
            from nanovllm.engine.model_loader import ModelLoader
            
            # Use ModelLoader's setup method to properly set KV cache for each layer
            ModelLoader.setup_attention_layers(self.model, page_manager)
    
    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for prefill stage."""
        input_ids = []
        positions = []
        seq_lens = []
        
        for seq in seqs:
            tokens = seq.prompt_token_ids
            
            # Check if this sequence has a position offset (for chunk-based generation)
            position_offset = 0
            if hasattr(seq, '_chunk_token_count'):
                position_offset = seq._chunk_token_count
                print(f"[DEBUG] Prefill positions: offset={position_offset}, tokens={len(tokens)}, "
                      f"positions={position_offset} to {position_offset + len(tokens) - 1}")
            
            positions_list = list(range(position_offset, position_offset + len(tokens)))
            
            input_ids.extend(tokens)
            positions.extend(positions_list)
            seq_lens.append(len(tokens))
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        
        # Get KV indices from page manager
        kv_indices, kv_indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            seqs, for_prefill=True
        )
        
        # Build q_indptr for queries (cumulative sum of sequence lengths)
        q_indptr = torch.zeros(len(seqs) + 1, dtype=torch.int32, device="cuda")
        for i, seq_len in enumerate(seq_lens):
            q_indptr[i + 1] = q_indptr[i] + seq_len
        
        # Build cu_seqlens_q which is cumulative sequence lengths for queries
        cu_seqlens_q = q_indptr
        
        return input_ids, positions, cu_seqlens_q
    
    def prepare_decode(self, seqs: list[Sequence]):
        """Prepare inputs for decode stage."""
        input_ids = []
        positions = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            
            # Check if this sequence has a position offset for decode
            if hasattr(seq, '_position_offset_for_decode'):
                # For chunk-based generation, positions continue from the full context
                position = seq._position_offset_for_decode + len(seq) - seq.num_prompt_tokens - 1
            else:
                position = len(seq) - 1
            
            positions.append(position)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        
        # Get KV indices from page manager
        kv_indices, kv_indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            seqs, for_prefill=False
        )
        
        # No wrapper planning needed - model handles attention internally
        
        return input_ids, positions
    
    def prepare_sample(self, seqs: list[Sequence]):
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device="cuda")
        return temperatures
    
    @torch.inference_mode()
    @handle_inference_error
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
                  seqs: list[Sequence], is_prefill: bool, cu_seqlens_q: torch.Tensor = None,
                  active_chunks=None):
        """Run model forward pass with K/V caching."""
        with ErrorContext("model forward pass", 
                         is_prefill=is_prefill, 
                         num_seqs=len(seqs),
                         input_shape=input_ids.shape):
            # Create inference context with the appropriate wrapper
            # Debug active chunks
            if active_chunks is not None:
                print(f"[DEBUG ModelRunner] Creating context with {len(active_chunks)} active chunks")
            
            context = InferenceContext(
                sequences=seqs,
                page_manager=self.page_manager,
                wrapper=None,  # No longer using FlashInfer wrappers
                is_prefill=is_prefill,
                cu_seqlens_q=cu_seqlens_q,
                cascade_data=None,  # No longer using cascade data
                active_chunks=active_chunks,
                kv_cache=self.page_manager.kv_cache if self.page_manager else None
            )
            
            # Run model
            hidden_states = self.model(input_ids, positions, context)
            
            # Compute logits
            logits = self.model.compute_logits(hidden_states, context)
            
            return logits
    
    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool, active_chunks=None):
        """Run one step of model inference."""
        # Handle both 2-tuple and 3-tuple returns
        if is_prefill:
            with ErrorContext("prefill preparation", num_seqs=len(seqs)):
                input_ids, positions, cu_seqlens_q = self.prepare_prefill(seqs)
                logits = self.run_model(input_ids, positions, seqs, is_prefill, cu_seqlens_q, active_chunks)
        else:
            with ErrorContext("decode preparation", num_seqs=len(seqs)):
                input_ids, positions = self.prepare_decode(seqs)
                logits = self.run_model(input_ids, positions, seqs, is_prefill, active_chunks=active_chunks)
        
        # Sample next tokens
        with ErrorContext("sampling", num_seqs=len(seqs)):
            temperatures = self.prepare_sample(seqs)
            next_tokens = self.sampler(logits, temperatures)
        
        # Update sequence lengths in page manager after successful generation
        if self.page_manager is not None:
            self.page_manager.update_sequence_lengths(seqs, is_prefill)
        
        # Convert to list
        next_tokens = next_tokens.tolist()
        return next_tokens
    
    @torch.inference_mode()
    def prefill_chunk(self, chunk) -> None:
        """Prefill KV cache for a chunk without sampling.
        
        This method populates the KV cache for a chunk's tokens at the
        allocated pages, without generating any new tokens.
        
        Args:
            chunk: Chunk object with token_ids and page_table
        """
        # Skip empty chunks
        if not chunk.token_ids:
            return
            
        # Convert tokens to tensors
        input_ids = torch.tensor(chunk.token_ids, dtype=torch.int64, device="cuda")
        positions = torch.arange(len(chunk.token_ids), dtype=torch.int64, device="cuda")
        
        # Create a mock sequence for this chunk to use the standard prefill path
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        # Create a dummy sequence with the chunk's tokens
        dummy_seq = Sequence(
            token_ids=chunk.token_ids,
            sampling_params=SamplingParams(max_tokens=0)  # We don't want to generate
        )
        
        # Important: Set the sequence's block table to the chunk's page table
        dummy_seq.block_table = chunk.page_table
        
        # Create an InferenceContext that knows about this chunk
        context = InferenceContext(
            sequences=[dummy_seq],
            page_manager=self.page_manager,
            wrapper=None,
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, len(chunk.token_ids)], dtype=torch.int32, device="cuda"),
            # Mark this as chunk prefilling
            chunk_id=chunk.chunk_id,
            chunk_positions=positions,
            kv_cache=self.page_manager.kv_cache if self.page_manager else None
        )
        
        # Run the model forward pass to populate KV cache
        # This will process the tokens properly through all layers
        hidden_states = self.model(input_ids, positions, context)
        
        # We don't need the output logits since we're just prefilling
        # The KV cache has been populated by the forward pass
        
        # Update chunk length in page manager
        if chunk.chunk_id not in self.page_manager.chunk_lengths:
            self.page_manager.chunk_lengths[chunk.chunk_id] = len(chunk.token_ids)