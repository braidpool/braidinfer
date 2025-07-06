"""
Qwen3 model with separated RMSNorm and QKV+RoPE kernels.

This implementation follows llama.cpp's approach of computing RMSNorm
separately from QKV projection to avoid numerical instability.
"""

import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.flashinfer_cascade_attention import FlashInferCascadeAttention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.kernels.rmsnorm_f32 import RMSNormF32
from nanovllm.kernels.qkv_rope_simple import QKVRoPESimple
from nanovllm.kernels.chunk_attention import ChunkAttention

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext


class Qwen3AttentionSeparated(nn.Module):
    """Qwen3 attention layer with separated RMSNorm and QKV+RoPE kernels."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        rope_theta: float = 10000,
        rope_scaling: dict = None,
        max_position: int = 8192,
        use_custom_chunk_kernel: bool = False,
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
        self.use_custom_chunk_kernel = use_custom_chunk_kernel
        
        # Store norm eps for RMSNorm kernel
        self.rms_norm_eps = rms_norm_eps
        
        # QKV projection weights (without bias initially)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # RoPE embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Q/K normalization
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        
        # Attention implementation
        if use_custom_chunk_kernel:
            self.attn = None  # Will handle manually
        else:
            use_cascade = getattr(self, '_use_cascade_attention', False)
            if use_cascade:
                self.attn = FlashInferCascadeAttention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    self.num_kv_heads,
                    self.layer_idx
                )
            else:
                self.attn = Attention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    self.num_kv_heads,
                    self.layer_idx,
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
        layernorm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with separated RMSNorm and QKV+RoPE.
        
        Args:
            positions: Position indices
            hidden_states: Input hidden states (unnormalized)
            context: Inference context
            layernorm_weight: RMSNorm weight tensor
        """
        # Ensure we have layernorm weight
        if layernorm_weight is None:
            raise ValueError("layernorm_weight must be provided")
        
        # Reshape if needed
        batch_size = hidden_states.shape[0] if hidden_states.dim() == 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # Step 1: Apply RMSNorm separately with float32 precision
        normalized = RMSNormF32.forward(
            hidden_states,
            layernorm_weight,
            eps=self.rms_norm_eps
        )
        
        # Step 2: Apply QKV projection + RoPE
        # Extract cos/sin cache from rotary embedding
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cos_cache = cos_sin_cache[:, :self.head_dim // 2]
        sin_cache = cos_sin_cache[:, self.head_dim // 2:]
        
        # Apply fused QKV + RoPE
        q, k, v = QKVRoPESimple.forward(
            normalized,  # Already in float32
            self.qkv_proj.weight,
            positions,
            cos_cache,
            sin_cache,
            self.num_heads,
            self.num_kv_heads,
            self.qkv_proj.bias
        )
        
        # Step 3: Apply Q/K normalization
        # q and k are [seq_len, num_heads, head_dim]
        # RMSNorm expects last dimension to match weight dimension (head_dim)
        # We need to ensure the tensors are in the right shape
        seq_len_actual = q.shape[0]
        q = q.contiguous()  # Ensure contiguous for view
        k = k.contiguous()
        v = v.contiguous()
        
        # Apply normalization - the norm is applied to the head_dim dimension
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Reshape for attention
        q_flat = q.view(seq_len * batch_size, -1)
        k_flat = k.view(seq_len * batch_size, -1)
        v_flat = v.view(seq_len * batch_size, -1)
        
        # Ensure contiguous
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        
        # Step 4: Apply attention
        if self.use_custom_chunk_kernel and context is not None:
            # Use custom chunk attention
            attn_output = ChunkAttention.decode_attention(
                q.view(seq_len * batch_size, self.num_heads, self.head_dim),
                context.chunk_k_caches,
                context.chunk_v_caches,
                context.chunk_lengths,
                context.chunk_levels,
                scale=self.scaling
            )
        elif context is None:
            # Simple attention for testing without context
            q_reshaped = q.view(seq_len * batch_size, self.num_heads, self.head_dim)
            k_reshaped = k.view(seq_len * batch_size, self.num_kv_heads, self.head_dim)
            v_reshaped = v.view(seq_len * batch_size, self.num_kv_heads, self.head_dim)
            
            # Expand K/V for GQA if needed
            if self.num_kv_heads < self.num_heads:
                k_reshaped = k_reshaped.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v_reshaped = v_reshaped.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            
            # Simple scaled dot-product attention
            scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * self.scaling
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_reshaped)
            attn_output = attn_output.view(seq_len * batch_size, -1)
        else:
            # Use standard attention
            attn_output = self.attn(q_flat, k_flat, v_flat, context)
        
        # Step 5: Output projection
        output = self.o_proj(attn_output)
        
        return output


class Qwen3DecoderLayerSeparated(nn.Module):
    """Decoder layer using separated RMSNorm approach."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # RMSNorm for input (weight only, computation done in attention)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention with separated kernels
        self.self_attn = Qwen3AttentionSeparated(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position=max_position_embeddings,
            use_custom_chunk_kernel=getattr(config, "use_custom_chunk_kernel", False),
        )
        
        # MLP
        self.mlp = Qwen3MLP(
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
        
        # Pass layernorm weight to attention (computation done there)
        hidden_states = self.self_attn(
            positions, 
            hidden_states, 
            context, 
            layernorm_weight=self.input_layernorm.weight
        )
        hidden_states = residual + hidden_states
        
        # Post-attention norm and MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3MLP(nn.Module):
    """Standard Qwen3 MLP."""

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


class Qwen3ModelSeparated(nn.Module):
    """Qwen3 model with separated kernels."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayerSeparated(config, layer_idx) 
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
        # Apply embedding scaling as per QWEN3_NUMERICAL_STABILITY_GUIDE.md
        hidden_states = hidden_states * (1.0 / (self.config.hidden_size ** 0.5))
        
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, context)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLMSeparated(nn.Module):
    """Qwen3 causal LM with separated kernels."""
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3ModelSeparated(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, context)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, context: 'InferenceContext' = None) -> torch.Tensor:
        logits = self.lm_head(hidden_states, context)
        return logits