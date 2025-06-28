"""
Embedding and language model head layers for single-GPU nano-vllm.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext


class VocabParallelEmbedding(nn.Module):
    """Vocabulary embedding layer for single GPU."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert param.data.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    """Language model head for single GPU."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(num_embeddings))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, context: Optional['InferenceContext'] = None):
        # For prefill, only compute logits for the last token of each sequence
        if context is not None and context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        logits = F.linear(x, self.weight, self.bias)
        return logits