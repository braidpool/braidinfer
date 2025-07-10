import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    bos: int = -1  # Beginning of sequence token
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    
    # Chunk configuration
    chunk_page_ratio: float = 0.5  # Fraction of pages reserved for chunks
    chunk_registry_size: int = 1000
    chunk_persistence_dir: str | None = None
    
    # Custom kernel configuration
    use_custom_kernels: bool = True  # Always True - this is the point of the project
    use_custom_chunk_kernel: bool = True  # Always True - this is the point of the project

    def __post_init__(self):
        # Check if it's a local directory first
        if os.path.isdir(self.model):
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        else:
            # Try HuggingFace model ID - this will use the cache automatically
            try:
                self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
                # Note: When using a model ID, transformers will handle the cache location
                # It will download to ~/.cache/huggingface/hub/ if not already cached
            except Exception as e:
                raise ValueError(f"Model '{self.model}' is neither a local directory nor a valid HuggingFace model ID: {e}")
        
        assert self.kvcache_block_size % 16 == 0  # Allow smaller block sizes
        
        # Handle different config attribute names
        if hasattr(self.hf_config, 'max_position_embeddings'):
            max_pos = self.hf_config.max_position_embeddings
        elif hasattr(self.hf_config, 'n_positions'):  # GPT-2 uses n_positions
            max_pos = self.hf_config.n_positions
        else:
            max_pos = self.max_model_len
            
        self.max_model_len = min(self.max_model_len, max_pos)
        assert self.max_num_batched_tokens >= self.max_model_len
