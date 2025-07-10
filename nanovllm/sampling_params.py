from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0  # 0 means no limit
    top_p: float = 1.0  # 1.0 means no nucleus sampling
    min_p: float = 0.0  # 0.0 means no min-p filtering
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: Optional[List[int]] = field(default=None)  # Additional stop tokens beyond EOS
    stop: Optional[Union[str, List[str]]] = field(default=None)  # Stop sequences (will be tokenized)
    retain_output_cache: bool = False  # Whether to retain KV cache after generation for reuse
