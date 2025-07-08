from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: Optional[List[int]] = field(default=None)  # Additional stop tokens beyond EOS
    stop: Optional[Union[str, List[str]]] = field(default=None)  # Stop sequences (will be tokenized)
    retain_output_cache: bool = False  # Whether to retain KV cache after generation for reuse
