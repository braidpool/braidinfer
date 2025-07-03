"""
Model runner for single-GPU nano-vllm.
"""

import torch
from typing import List, Optional

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.inference_context import InferenceContext
from nanovllm.engine.wrapper_manager import WrapperManager
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
        torch.set_default_dtype(self.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # Load model and sampler
        self.model = ModelLoader.load_model(config)
        self.sampler = ModelLoader.create_sampler()
        
        # Initialize wrapper manager
        num_kv_heads = self.hf_config.num_key_value_heads
        num_qo_heads = self.hf_config.num_attention_heads
        
        # Initialize wrapper manager
        self.wrapper_manager = WrapperManager(
            num_layers=self.hf_config.num_hidden_layers,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.hf_config.head_dim,
            page_size=self.block_size,
            dtype=self.hf_config.torch_dtype
        )
        
        # Warmup model
        ModelLoader.warmup_model(self.model)
        
        # Calculate KV cache blocks
        config.num_kvcache_blocks = ModelLoader.calculate_kvcache_blocks(
            config, self.hf_config, 1, self.block_size  # world_size=1 for single GPU
        )
        
        # Reset default device settings
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
    
    def set_page_manager(self, page_manager):
        """Set the page manager and connect it to attention layers."""
        self.page_manager = page_manager
        
        # Setup attention layers with KV cache
        ModelLoader.setup_attention_layers(self.model, page_manager)
    
    def get_metrics(self) -> dict:
        """Get current metrics summary."""
        return get_metrics_collector().get_summary()
    
    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare inputs for prefill stage."""
        input_ids = []
        positions = []
        
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.prompt_token_ids)
            positions.extend(list(range(seqlen)))
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        
        # Build indices for FlashInfer
        seq_lens = [len(seq) for seq in seqs]
        q_indptr = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seqs))],
                               dtype=torch.int32, device="cuda")
        
        # Get KV indices from page manager
        kv_indices, kv_indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            seqs, for_prefill=True
        )
        
        # Plan prefill for all layers
        self.wrapper_manager.plan_prefill(q_indptr, kv_indptr, kv_indices, last_page_lens)
        
        # Build cu_seqlens_q which is cumulative sequence lengths for queries
        cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device="cuda")
        for i, seq_len in enumerate(seq_lens):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seq_len
        
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
        
        # Plan decode for all layers
        self.wrapper_manager.plan_decode(kv_indptr, kv_indices, last_page_lens)
        
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
            # Create inference context
            context = InferenceContext(
                is_prefill=is_prefill,
                sequences=seqs,
                cu_seqlens_q=cu_seqlens_q,
                prefill_wrappers=self.wrapper_manager.prefill_wrappers if is_prefill else None,
                decode_wrappers=self.wrapper_manager.decode_wrappers if not is_prefill else None,
                page_manager=self.page_manager,
                cascade_data=cascade_data
            )
            
            # Run model forward - K/V will be appended inside attention layers
            hidden_states = self.model(input_ids, positions, context)
            logits = self.model.compute_logits(hidden_states, context)
            
            # Update sequence lengths after all layers have processed
            self.page_manager.update_sequence_lengths(seqs, is_prefill)
            
            return logits
    
    @handle_inference_error
    def run(self, seqs: list[Sequence], is_prefill: bool, cascade_data=None) -> list[int]:
        """Run inference for a batch of sequences."""
        # Generate request ID based on sequence IDs
        request_id = f"{'prefill' if is_prefill else 'decode'}_{seqs[0].seq_id}_{len(seqs)}"
        
        # Calculate total tokens
        if is_prefill:
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            num_tokens = len(seqs)  # One token per sequence in decode
        
        with MetricsContext(request_id, is_prefill, len(seqs), num_tokens):
            with ErrorContext("inference run", is_prefill=is_prefill, num_seqs=len(seqs)):
                if is_prefill:
                    input_ids, positions, cu_seqlens_q = self.prepare_prefill(seqs)
                else:
                    input_ids, positions = self.prepare_decode(seqs)
                    cu_seqlens_q = None
                
                temperatures = self.prepare_sample(seqs)
                
                logits = self.run_model(input_ids, positions, seqs, is_prefill, cu_seqlens_q, cascade_data=cascade_data)
                
                # Sample tokens
                if logits is None:
                    raise InferenceError("No logits returned from model")
                    
                try:
                    token_ids = self.sampler(logits, temperatures).tolist()
                except Exception as e:
                    raise InferenceError(f"Sampling failed: {str(e)}") from e
                
                return token_ids