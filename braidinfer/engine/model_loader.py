"""
Model loading and initialization for nano-vllm.
"""

import torch
from typing import Optional

from braidinfer.config import Config
from braidinfer.models.qwen3 import Qwen3ForCausalLM
from braidinfer.models.gpt2 import GPT2ForCausalLM
from braidinfer.models.llama import LlamaForCausalLM
from braidinfer.models.ernie import ERNIE45ForCausalLM
from braidinfer.layers.sampler import Sampler
from braidinfer.layers.attention import Attention
from braidinfer.utils.loader import load_model
from braidinfer.engine.errors import ModelLoadError, MemoryError, ErrorContext


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
            
            # Create model instance based on model type
            try:
                model_type = hf_config.model_type
                
                if model_type in ["qwen3", "qwen2"]:
                    model = Qwen3ForCausalLM(hf_config)
                elif model_type == "gpt2":
                    model = GPT2ForCausalLM(hf_config)
                elif model_type == "llama":
                    model = LlamaForCausalLM(hf_config)
                elif model_type == "ernie4_5":
                    model = ERNIE45ForCausalLM(hf_config)
                else:
                    raise ModelLoadError(f"Unsupported model type: {model_type}")
            except Exception as e:
                raise ModelLoadError(f"Failed to create model instance: {str(e)}") from e
            
            # Load weights
            try:
                load_model(model, config.model)
                
                # Check for extreme K normalization weights after loading
                if hasattr(model, 'check_extreme_weights'):
                    model.check_extreme_weights()
            except FileNotFoundError as e:
                raise ModelLoadError(f"Model file not found: {config.model}") from e
            except Exception as e:
                raise ModelLoadError(f"Failed to load model weights: {str(e)}") from e
            
            # Move model to CUDA and convert to appropriate dtype
            dtype = hf_config.torch_dtype if hasattr(hf_config, 'torch_dtype') else torch.float16
            model = model.to(device="cuda", dtype=dtype)
            model.eval()
            
            # Print kernel configuration
            print("\n--- Kernel Configuration ---")
            print("Custom fused kernels are ENABLED.")

            # Inspect the model to see which attention layer was actually instantiated
            try:
                first_attn_module = None
                # Standard model structure (Llama, Qwen3, etc.)
                if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                    if hasattr(model.model.layers[0], 'self_attn'):
                        first_attn_module = model.model.layers[0].self_attn
                # GPT-2 model structure
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h') and len(model.transformer.h) > 0:
                    if hasattr(model.transformer.h[0], 'attn'):
                        first_attn_module = model.transformer.h[0].attn

                if first_attn_module:
                    attn_class_name = first_attn_module.__class__.__name__
                    print(f"Attention module in use: {attn_class_name}")
                    if "Fused" in attn_class_name:
                        print("-> Fused RMSNorm+QKV path is active.")
                    else:
                        print("-> Standard, separate operations path is active.")
                else:
                    print("Could not determine the specific attention module in use.")
            except Exception as e:
                print(f"Could not inspect attention modules: {e}")
            
            print("--------------------------\n")
            
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
        
        # Track which attention modules we've already set up to avoid double counting
        seen_attention = set()
        
        for name, module in model.named_modules():
            # Check for attention modules inside Qwen3AttentionFused first
            if hasattr(module, 'attn') and isinstance(module.attn, Attention):
                if id(module.attn) not in seen_attention:
                    module.attn.kv_cache = page_manager.get_layer_kv_cache(layer_count)
                    seen_attention.add(id(module.attn))
                    layer_count += 1
            # Only count standalone Attention modules that aren't already counted
            elif isinstance(module, Attention) and id(module) not in seen_attention:
                module.kv_cache = page_manager.get_layer_kv_cache(layer_count)
                seen_attention.add(id(module))
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
        # Handle different config formats
        if hasattr(hf_config, 'num_key_value_heads'):
            num_kv_heads = hf_config.num_key_value_heads
        elif hasattr(hf_config, 'num_attention_heads'):
            num_kv_heads = hf_config.num_attention_heads  # No GQA
        else:
            num_kv_heads = hf_config.n_head  # GPT-2 style
            
        if hasattr(hf_config, 'num_hidden_layers'):
            num_layers = hf_config.num_hidden_layers
        else:
            num_layers = hf_config.n_layer  # GPT-2 style
            
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            head_dim = hf_config.n_embd // hf_config.n_head  # GPT-2 style
            
        if hasattr(hf_config, 'torch_dtype') and hf_config.torch_dtype is not None:
            dtype = hf_config.torch_dtype
        else:
            dtype = torch.float16  # Default
            
        # For FlashInfer: [num_layers, num_pages, 2, page_size, num_kv_heads, head_dim]
        block_bytes = (num_layers * 2 * block_size * 
                      num_kv_heads * head_dim * 
                      dtype.itemsize)
        
        # Calculate available blocks
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        num_blocks = int(available_memory // block_bytes)
        
        # Cap KV cache size based on model size
        # For small models (< 1B params), limit KV cache to reasonable size
        model_params = hf_config.hidden_size * hf_config.num_hidden_layers * 4  # Rough estimate
        if model_params < 1e9:  # Less than 1B parameters
            # Cap at 2GB for small models
            max_kv_cache_gb = 2.0
            max_blocks = int((max_kv_cache_gb * 1024**3) // block_bytes)
            if num_blocks > max_blocks:
                print(f"Capping KV cache for small model: {num_blocks} -> {max_blocks} blocks")
                num_blocks = max_blocks
        
        # Ensure we have at least some blocks
        if num_blocks <= 0:
            # Fallback to a minimal number
            num_blocks = 64
            print(f"Warning: Low GPU memory, using minimal KV cache blocks: {num_blocks}")
        
        return num_blocks