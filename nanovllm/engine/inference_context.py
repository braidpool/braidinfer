"""
Inference context for passing state through model layers.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch

from nanovllm.engine.sequence import Sequence


@dataclass
class InferenceContext:
    """Context passed through model layers during inference."""
    
    # Stage information
    is_prefill: bool
    
    # Batch sequences
    sequences: List[Sequence]
    
    # For prefill: cumulative sequence lengths
    cu_seqlens_q: Optional[torch.Tensor] = None
    
    # FlashInfer wrappers (per layer)
    prefill_wrappers: Optional[List] = None
    decode_wrappers: Optional[List] = None
    
    # Page manager reference
    page_manager: Optional[object] = None
    
    def get_wrapper(self, layer_idx: int):
        """Get the appropriate wrapper for the current layer."""
        if self.is_prefill:
            return self.prefill_wrappers[layer_idx] if self.prefill_wrappers else None
        else:
            return self.decode_wrappers[layer_idx] if self.decode_wrappers else None