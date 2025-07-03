"""
Model loading and initialization for nano-vllm.
"""

import torch
from typing import Optional

from nanovllm.config import Config
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import Attention
from nanovllm.layers.flashinfer_cascade_attention import FlashInferCascadeAttention
from nanovllm.utils.loader import load_model
from nanovllm.engine.errors import ModelLoadError, MemoryError, ErrorContext


class ModelLoader:
    """Handles model loading and initialization."""
    
    @staticmethod
    def load_model(config: Config) -> Qwen3ForCausalLM:
        """Load and initialize the model.
        
        Args:
            config: Configuration object
            
        Returns:
            Loaded model instance
            
        Raises:
            ModelLoadError: If model loading fails
        """
        with ErrorContext("model loading", model_path=config.model):
            hf_config = config.hf_config
            
            # Create model instance
            try:
                # Create model instance
                model = Qwen3ForCausalLM(hf_config)
                
                # Set cascade attention flag on model layers if enabled
                if getattr(config, 'enable_cascade_attention', False):
                    for module in model.modules():
                        if hasattr(module, 'attn'):
                            module._use_cascade_attention = True
            except Exception as e:
                raise ModelLoadError(f"Failed to create model instance: {str(e)}") from e
            
            # Load weights
            try:
                load_model(model, config.model)
            except FileNotFoundError as e:
                raise ModelLoadError(f"Model file not found: {config.model}") from e
            except Exception as e:
                raise ModelLoadError(f"Failed to load model weights: {str(e)}") from e
            
            return model
    
    @staticmethod
    def create_sampler() -> Sampler:
        """Create sampler instance."""
        return Sampler()
    
    @staticmethod
    def setup_attention_layers(model: torch.nn.Module, page_manager) -> int:
        """Setup attention layers with KV cache references.
        
        Args:
            model: Model instance
            page_manager: Page manager instance
            
        Returns:
            Number of attention layers found
        """
        layer_count = 0
        
        for module in model.modules():
            if isinstance(module, (Attention, FlashInferCascadeAttention)):
                # Set KV cache reference
                module.kv_cache = page_manager.get_layer_kv_cache(layer_count)
                layer_count += 1
        
        return layer_count
    
    @staticmethod
    def warmup_model(model: torch.nn.Module, seq_len: int = 128):
        """Warmup model with dummy data.
        
        Args:
            model: Model instance
            seq_len: Sequence length for warmup
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Simple warmup without running full attention
        dummy_input = torch.zeros(seq_len, dtype=torch.int64, device="cuda")
        
        # Just run through embeddings to warmup
        with torch.inference_mode():
            _ = model.model.embed_tokens(dummy_input)
        
        torch.cuda.empty_cache()
    
    @staticmethod
    def calculate_kvcache_blocks(config: Config, 
                                hf_config, 
                                world_size: int,
                                block_size: int) -> int:
        """Calculate number of KV cache blocks based on available GPU memory.
        
        Args:
            config: Configuration object
            hf_config: HuggingFace model config
            world_size: Ignored, kept for compatibility
            block_size: Size of each KV cache block
            
        Returns:
            Number of KV cache blocks
        """
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # Calculate bytes per block (single GPU, no division)
        num_kv_heads = hf_config.num_key_value_heads
        # For FlashInfer: [num_layers, num_pages, 2, page_size, num_kv_heads, head_dim]
        block_bytes = (hf_config.num_hidden_layers * 2 * block_size * 
                      num_kv_heads * hf_config.head_dim * 
                      hf_config.torch_dtype.itemsize)
        
        # Calculate available blocks
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        num_blocks = int(available_memory // block_bytes)
        
        # Ensure we have at least some blocks
        if num_blocks <= 0:
            # Fallback to a minimal number
            num_blocks = 64
            print(f"Warning: Low GPU memory, using minimal KV cache blocks: {num_blocks}")
        
        return num_blocks