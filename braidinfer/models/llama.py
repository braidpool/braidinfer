"""
LLaMA model implementation for nano-vllm.
Supports TinyLlama and other LLaMA-based models.
"""

import torch
from torch import nn
from transformers import LlamaConfig

from braidinfer.layers.activation import SiluAndMul
from braidinfer.layers.attention import Attention
from braidinfer.layers.layernorm import RMSNorm
from braidinfer.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from braidinfer.layers.rotary_embedding import get_rope
from braidinfer.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from braidinfer.kernels.fused_rmsnorm_qkv_minimal_f32 import FusedRMSNormQKVMinimalF32

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from braidinfer.engine.inference_context import InferenceContext


class LlamaAttention(nn.Module):
    """LLaMA attention layer for FlashInfer."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 2048,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # Single GPU: no tensor parallelism
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        # No layer norm for Q/K in LLaMA (unlike Qwen3)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # LLaMA doesn't use bias
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.layer_idx
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, context)
        output = self.o_proj(attn_output)
        return output


class LlamaAttentionFused(nn.Module):
    """LLaMA attention layer with fused RMSNorm+QKV kernel."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict = None,
        max_position: int = 2048,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # Determine head dimensions
        if head_dim is None:
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
            
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        # Store norm eps for fused kernel
        self.rms_norm_eps = rms_norm_eps
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Attention implementation
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.layer_idx
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
        layernorm_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with fused RMSNorm+QKV."""
        input_dtype = hidden_states.dtype
        batch_seq_len = hidden_states.shape[0]
        
        # Fused RMSNorm + QKV projection
        q, k, v = FusedRMSNormQKVMinimalF32.forward(
            hidden_states,
            layernorm_weight,
            self.qkv_proj.weight,
            self.num_heads,
            self.num_kv_heads,
            self.rms_norm_eps
        )
        
        # Reshape for rotary embeddings
        q_flat = q.view(batch_seq_len, -1)
        k_flat = k.view(batch_seq_len, -1)
        
        # Handle positions shape
        if positions.dim() == 2:
            positions_flat = positions.reshape(-1)
        else:
            positions_flat = positions
        
        # Apply rotary embeddings
        q_flat, k_flat = self.rotary_emb(positions_flat, q_flat, k_flat)
        
        # Flatten v for attention
        v_flat = v.view(batch_seq_len, -1)
        
        # Ensure contiguous
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        
        # Ensure correct dtype
        if q_flat.dtype != input_dtype:
            q_flat = q_flat.to(input_dtype)
        if k_flat.dtype != input_dtype:
            k_flat = k_flat.to(input_dtype)
        if v_flat.dtype != input_dtype:
            v_flat = v_flat.to(input_dtype)
        
        # Apply attention
        attn_output = self.attn(q_flat, k_flat, v_flat, context)
            
        # Output projection
        output = self.o_proj(attn_output)
        
        # Ensure output dtype matches input
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
            
        return output


class LlamaMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):
    """LLaMA decoder layer."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)

        # Use fused attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttentionFused(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position=max_position_embeddings,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        residual = hidden_states
        
        # Fused kernel expects unnormalized input and layernorm weight
        hidden_states = self.self_attn(positions, hidden_states, context, layernorm_weight=self.input_layernorm.weight)
        hidden_states = residual + hidden_states
            
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    """LLaMA model."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states, context)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA model for causal language modeling."""
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, context)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor:
        """Compute logits for given hidden states."""
        return self.lm_head(hidden_states, sampling_metadata)