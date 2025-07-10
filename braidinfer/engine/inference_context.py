"""
Inference context for passing state through model layers.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import torch

from braidinfer.engine.sequence import Sequence

if TYPE_CHECKING:
    from braidinfer.chunks import Chunk



@dataclass
class InferenceContext:
    """Context passed through model layers during inference."""
    
    # Stage information
    is_prefill: bool
    
    # Batch sequences
    sequences: List[Sequence]
    
    # For prefill: cumulative sequence lengths
    cu_seqlens_q: Optional[torch.Tensor] = None
    
    # Single FlashInfer wrapper shared by all layers
    wrapper: Optional[object] = None
    
    # Page manager reference
    page_manager: Optional[object] = None
    
    # Cascade attention data
    cascade_data: Optional[dict] = None
    
    # Workspace buffer for cascade attention
    _workspace_buffer: Optional[torch.Tensor] = None
    
    # For chunk prefilling - chunk_id and positions
    chunk_id: Optional[str] = None
    chunk_positions: Optional[torch.Tensor] = None
    
    # Active chunks for custom chunk attention
    active_chunks: Optional[List['Chunk']] = None
    
    # Global KV cache reference for custom chunk attention
    kv_cache: Optional[torch.Tensor] = None
    
    def get_wrapper(self, layer_idx: int = None):
        """Get the wrapper. Layer index is ignored since we use a single wrapper."""
        return self.wrapper
    
    def get_workspace_buffer(self, size: int) -> torch.Tensor:
        """Get or allocate workspace buffer for cascade attention."""
        if self._workspace_buffer is None or self._workspace_buffer.numel() < size:
            self._workspace_buffer = torch.empty(size, dtype=torch.uint8, device="cuda")
        return self._workspace_buffer