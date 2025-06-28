"""
Model runner using FlashInfer for attention.
"""

import pickle
import torch
import torch.distributed as dist
import flashinfer
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3_flashinfer import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.flashinfer_attention import FlashInferAttention
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class FlashInferModelRunner:
    """Model runner that uses FlashInfer for attention operations."""
    
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        
        # FlashInfer page manager will be set by scheduler
        self.page_manager = None
        
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # Load model
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # Replace attention layers with FlashInfer attention
        self._replace_attention_layers(hf_config)
        
        # Initialize FlashInfer wrappers
        workspace_size = 128 * 1024 * 1024  # 128MB
        self.prefill_wrappers = []
        self.decode_wrappers = []
        
        for layer_idx in range(hf_config.num_hidden_layers):
            # Create wrappers for each layer
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=torch.empty(workspace_size, dtype=torch.uint8, device="cuda"),
                kv_layout="NHD"
            )
            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                float_workspace_buffer=torch.empty(workspace_size, dtype=torch.uint8, device="cuda"),
                kv_layout="NHD",
                use_tensor_cores=True
            )
            self.prefill_wrappers.append(prefill_wrapper)
            self.decode_wrappers.append(decode_wrapper)
        
        self.warmup_model()
        
        # Calculate number of KV cache blocks based on available GPU memory
        self._calculate_kvcache_blocks(config, hf_config)
        
        if not self.enforce_eager:
            # CUDA graphs support would need modification for FlashInfer
            pass
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
    
    def _replace_attention_layers(self, hf_config):
        """Replace model's attention layers with FlashInfer attention."""
        layer_idx = 0
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        num_qo_heads = hf_config.num_attention_heads // self.world_size
        
        for module in self.model.modules():
            if hasattr(module, 'attn'):
                # Replace the attention module
                old_attn = module.attn
                new_attn = FlashInferAttention(
                    num_heads=num_qo_heads,
                    head_dim=hf_config.head_dim,
                    scale=old_attn.scale,
                    num_kv_heads=num_kv_heads,
                    layer_idx=layer_idx
                )
                module.attn = new_attn
                layer_idx += 1
    
    def set_page_manager(self, page_manager):
        """Set the page manager and connect it to attention layers."""
        self.page_manager = page_manager
        
        # Connect wrappers and KV cache to attention layers
        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, FlashInferAttention):
                module.prefill_wrapper = self.prefill_wrappers[layer_idx]
                module.decode_wrapper = self.decode_wrappers[layer_idx]
                module.kv_cache = page_manager.get_layer_kv_cache(layer_idx)
                module.model_runner = self
                module.page_manager = page_manager
                layer_idx += 1
    
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        dist.destroy_process_group()
    
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break
    
    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args
    
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()
    
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)
    
    def warmup_model(self):
        """Warmup model with dummy data."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Simple warmup without running full attention
        dummy_seq_len = 128
        dummy_input = torch.zeros(dummy_seq_len, dtype=torch.int64, device="cuda")
        dummy_positions = torch.arange(dummy_seq_len, dtype=torch.int64, device="cuda")
        
        # Just run through embeddings to warmup
        with torch.inference_mode():
            _ = self.model.model.embed_tokens(dummy_input)
        
        torch.cuda.empty_cache()
    
    def _calculate_kvcache_blocks(self, config, hf_config):
        """Calculate number of KV cache blocks based on available GPU memory."""
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # Calculate bytes per block
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # For FlashInfer: [num_layers, num_pages, 2, page_size, num_kv_heads, head_dim]
        block_bytes = (hf_config.num_hidden_layers * 2 * self.block_size * 
                      num_kv_heads * hf_config.head_dim * 
                      hf_config.torch_dtype.itemsize)
        
        # Calculate available blocks
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        config.num_kvcache_blocks = int(available_memory // block_bytes)
        
        # Ensure we have at least some blocks
        if config.num_kvcache_blocks <= 0:
            # Fallback to a minimal number
            config.num_kvcache_blocks = 64
            print(f"Warning: Low GPU memory, using minimal KV cache blocks: {config.num_kvcache_blocks}")
    
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
        hf_config = self.config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        num_qo_heads = hf_config.num_attention_heads // self.world_size
        
        for wrapper in self.prefill_wrappers:
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                last_page_lens,
                num_qo_heads,
                num_kv_heads,
                hf_config.head_dim,
                self.page_manager.page_size,
                causal=True,
                q_data_type=hf_config.torch_dtype,
                kv_data_type=hf_config.torch_dtype
            )
        
        # Set context for prefill
        # Build cu_seqlens_q which is cumulative sequence lengths for queries
        cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device="cuda")
        for i, seq_len in enumerate(seq_lens):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seq_len
        
        set_context(True, cu_seqlens_q=cu_seqlens_q)
        
        return input_ids, positions
    
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
        hf_config = self.config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        num_qo_heads = hf_config.num_attention_heads // self.world_size
        
        for wrapper in self.decode_wrappers:
            wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_lens,
                num_qo_heads,
                num_kv_heads,
                hf_config.head_dim,
                self.page_manager.page_size,
                kv_data_type=hf_config.torch_dtype,
                q_data_type=hf_config.torch_dtype
            )
        
        # Set context for decode
        set_context(False)
        
        return input_ids, positions
    
    def prepare_sample(self, seqs: list[Sequence]):
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device="cuda")
        return temperatures
    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
                  seqs: list[Sequence], is_prefill: bool):
        """Run model forward pass with K/V caching."""
        # Store sequences in context for K/V appending
        self.current_seqs = seqs
        self.current_is_prefill = is_prefill
        
        # Run model forward - K/V will be appended inside attention layers
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        
        # Update sequence lengths after all layers have processed
        self.page_manager.update_sequence_lengths(seqs, is_prefill)
        
        return logits
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run inference for a batch of sequences."""
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        logits = self.run_model(input_ids, positions, seqs, is_prefill)
        
        # Sample tokens
        if self.rank == 0:
            if is_prefill:
                # For prefill, ParallelLMHead already returns only last token logits
                # No need to index again
                token_ids = self.sampler(logits, temperatures).tolist()
            else:
                # For decode, we have one logit per sequence
                token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
        
        reset_context()
        return token_ids