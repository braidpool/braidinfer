"""
GPT-2 model implementation for nano-vllm.
"""

import torch
from torch import nn
from transformers import GPT2Config

from braidinfer.layers.activation import NewGELUActivation
from braidinfer.layers.attention import Attention
from braidinfer.layers.layernorm import LayerNorm
from braidinfer.layers.linear import ColumnParallelLinear, RowParallelLinear
from braidinfer.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from braidinfer.engine.inference_context import InferenceContext


class GPT2Attention(nn.Module):
    """GPT-2 attention layer."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        max_position: int = 1024,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_kv_heads = self.num_heads  # GPT-2 doesn't use GQA
        self.scale = self.head_dim**-0.5

        # GPT-2 uses a single linear layer for QKV
        # Note: GPT-2 actually transposes the weight, so we need a custom loader
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.c_attn.weight.weight_loader = self._c_attn_weight_loader
        self.c_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
        )
        
        # Use standard attention (no cascade)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scale,
            self.num_kv_heads,
            self.layer_idx,
        )

    def _c_attn_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom weight loader for c_attn that handles transposed weights."""
        # GPT-2 weights are stored as [in_features, out_features]
        # PyTorch Linear expects [out_features, in_features]
        param.data.copy_(loaded_weight.t())

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        # QKV projection
        qkv = self.c_attn(hidden_states)
        
        # Split into Q, K, V
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        
        # Reshape for attention
        seq_len = hidden_states.shape[0]
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)
        
        # Apply attention (no RoPE in GPT-2)
        attn_output = self.attn(q, k, v, context)
        
        # Output projection
        output = self.c_proj(attn_output)
        return output


class GPT2MLP(nn.Module):
    """GPT-2 MLP layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=True)
        self.act = NewGELUActivation()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block."""

    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
        )
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(hidden_size, inner_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(positions, hidden_states, context)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class GPT2Model(nn.Module):
    """GPT-2 base model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        position_embeds = self.wpe(positions)
        hidden_states = hidden_states + position_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            hidden_states = block(positions, hidden_states, context)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2ForCausalLM(nn.Module):
    """GPT-2 model for causal language modeling."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)
        
        # Tie weights between input embeddings and output embeddings
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, context)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, context: 'InferenceContext' = None) -> torch.Tensor:
        logits = self.lm_head(hidden_states, context)
        return logits