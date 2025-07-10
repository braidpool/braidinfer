"""
Layer factory for selecting between distributed and single-GPU implementations.
"""

from typing import Type, Tuple


def get_linear_layers(distributed: bool = False) -> Tuple[Type, Type, Type, Type]:
    """Get linear layer classes based on distributed mode.
    
    Args:
        distributed: Whether to use distributed (tensor parallel) layers
        
    Returns:
        Tuple of (ColumnParallelLinear, RowParallelLinear, 
                  QKVParallelLinear, MergedColumnParallelLinear)
    """
    if distributed:
        from .linear_distributed import (
            ColumnParallelLinear,
            RowParallelLinear,
            QKVParallelLinear,
            MergedColumnParallelLinear
        )
    else:
        from .linear import (
            ColumnParallelLinear,
            RowParallelLinear,
            QKVParallelLinear,
            MergedColumnParallelLinear
        )
    
    return (ColumnParallelLinear, RowParallelLinear, 
            QKVParallelLinear, MergedColumnParallelLinear)


def get_embedding_layers(distributed: bool = False) -> Tuple[Type, Type]:
    """Get embedding layer classes based on distributed mode.
    
    Args:
        distributed: Whether to use distributed (tensor parallel) layers
        
    Returns:
        Tuple of (VocabParallelEmbedding, ParallelLMHead)
    """
    if distributed:
        # Would need to create embed_head_distributed.py first
        raise NotImplementedError("Distributed embedding layers not yet separated")
    else:
        from .embed_head import (
            VocabParallelEmbedding,
            ParallelLMHead
        )
    
    return VocabParallelEmbedding, ParallelLMHead


def get_world_size(distributed: bool = False) -> int:
    """Get world size for tensor parallelism.
    
    Args:
        distributed: Whether distributed mode is enabled
        
    Returns:
        World size (1 for single GPU, actual world size for distributed)
    """
    if distributed:
        import torch.distributed as dist
        return dist.get_world_size()
    else:
        return 1