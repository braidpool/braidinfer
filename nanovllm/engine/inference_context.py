"""
Inference context for passing state through model layers.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import torch

from nanovllm.engine.sequence import Sequence

if TYPE_CHECKING:
    from nanovllm.layers.cascade_attention import CascadeConfig


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
    
    # Cascade attention configuration
    cascade_config: Optional['CascadeConfig'] = None
    
    # Workspace buffer for cascade attention
    _workspace_buffer: Optional[torch.Tensor] = None
    
    def get_wrapper(self, layer_idx: int):
        """Get the appropriate wrapper for the current layer."""
        if self.is_prefill:
            return self.prefill_wrappers[layer_idx] if self.prefill_wrappers else None
        else:
            return self.decode_wrappers[layer_idx] if self.decode_wrappers else None
    
    def get_workspace_buffer(self, size: int) -> torch.Tensor:
        """Get or allocate workspace buffer for cascade attention."""
        if self._workspace_buffer is None or self._workspace_buffer.numel() < size:
            self._workspace_buffer = torch.empty(size, dtype=torch.uint8, device="cuda")
        return self._workspace_buffer