"""
Model runner for single-GPU nano-vllm.
"""

import torch
import flashinfer
from typing import List, Optional

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.inference_context import InferenceContext
from nanovllm.engine.model_loader import ModelLoader
from nanovllm.engine.errors import ErrorContext, handle_inference_error, InferenceError
from nanovllm.engine.metrics import MetricsContext, get_metrics_collector
from nanovllm.layers.sampler import Sampler


class ModelRunner:
    """Model runner that uses FlashInfer for attention operations on single GPU."""
    
    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        
        # FlashInfer page manager will be set by scheduler
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
        
        # Initialize FlashInfer wrappers
        # Handle different config formats
        if hasattr(self.hf_config, 'num_key_value_heads'):
            num_kv_heads = self.hf_config.num_key_value_heads
        elif hasattr(self.hf_config, 'num_attention_heads'):
            num_kv_heads = self.hf_config.num_attention_heads
        else:
            num_kv_heads = self.hf_config.n_head  # GPT-2
            
        if hasattr(self.hf_config, 'num_attention_heads'):
            num_qo_heads = self.hf_config.num_attention_heads
        else:
            num_qo_heads = self.hf_config.n_head  # GPT-2
            
        if hasattr(self.hf_config, 'head_dim'):
            head_dim = self.hf_config.head_dim
        elif hasattr(self.hf_config, 'hidden_size'):
            head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        else:
            head_dim = self.hf_config.n_embd // self.hf_config.n_head  # GPT-2
        
        # Create workspace buffer (shared by both wrappers)
        # Match vLLM's workspace size for better performance
        self.workspace_size = 256 * 1024 * 1024  # 256MB
        self.workspace_buffer = torch.empty(self.workspace_size, dtype=torch.uint8, device="cuda")
        
        # Create single prefill and decode wrappers to be shared by all layers
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=self.workspace_buffer,
            kv_layout="HND"
        )
        
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=self.workspace_buffer,
            kv_layout="HND",
            use_tensor_cores=True
        )
        
        # Store attention params
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
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
            kv_cache = page_manager.kv_cache
            for module in self.model.modules():
                if hasattr(module, 'kv_cache'):
                    module.kv_cache = kv_cache
    
    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for prefill stage."""
        input_ids = []
        positions = []
        seq_lens = []
        
        for seq in seqs:
            tokens = seq.prompt_token_ids
            positions_list = list(range(len(tokens)))
            
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
        
        # Plan prefill once for all layers
        # Calculate attention scale based on head dimension
        sm_scale = self.head_dim ** -0.5
        
        self.prefill_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            last_page_lens,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            causal=True,
            pos_encoding_mode="NONE",  # RoPE already applied in model
            sm_scale=sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )
        
        # Build cu_seqlens_q which is cumulative sequence lengths for queries
        cu_seqlens_q = q_indptr
        
        return input_ids, positions, cu_seqlens_q
    
    def prepare_decode(self, seqs: list[Sequence]):
        """Prepare inputs for decode stage."""
        input_ids = []
        positions = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        
        # Get KV indices from page manager
        kv_indices, kv_indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            seqs, for_prefill=False
        )
        
        # Plan decode once for all layers
        # Calculate attention scale based on head dimension
        sm_scale = self.head_dim ** -0.5
        
        self.decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_lens,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            pos_encoding_mode="NONE",  # RoPE already applied in model
            sm_scale=sm_scale,
            kv_data_type=self.dtype,
            q_data_type=self.dtype
        )
        
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
                  cascade_data=None):
        """Run model forward pass with K/V caching."""
        with ErrorContext("model forward pass", 
                         is_prefill=is_prefill, 
                         num_seqs=len(seqs),
                         input_shape=input_ids.shape):
            # Create inference context with the appropriate wrapper
            context = InferenceContext(
                sequences=seqs,
                page_manager=self.page_manager,
                wrapper=self.prefill_wrapper if is_prefill else self.decode_wrapper,
                is_prefill=is_prefill,
                cu_seqlens_q=cu_seqlens_q,
                cascade_data=cascade_data
            )
            
            # Run model
            hidden_states = self.model(input_ids, positions, context)
            
            # Compute logits
            logits = self.model.compute_logits(hidden_states, context)
            
            return logits
    
    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool, cascade_data=None):
        """Run one step of model inference."""
        # Handle both 2-tuple and 3-tuple returns
        if is_prefill:
            with ErrorContext("prefill preparation", num_seqs=len(seqs)):
                input_ids, positions, cu_seqlens_q = self.prepare_prefill(seqs)
                logits = self.run_model(input_ids, positions, seqs, is_prefill, cu_seqlens_q, cascade_data)
        else:
            with ErrorContext("decode preparation", num_seqs=len(seqs)):
                input_ids, positions = self.prepare_decode(seqs)
                logits = self.run_model(input_ids, positions, seqs, is_prefill, cascade_data=cascade_data)
        
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