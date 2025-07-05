"""
Qwen3 model modified to use FlashInfer attention.
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
from nanovllm.kernels.fused_rmsnorm_qkv_production import FusedRMSNormQKV
from nanovllm.kernels.chunk_attention import ChunkAttention


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.inference_context import InferenceContext


class Qwen3Attention(nn.Module):
    """Qwen3 attention layer modified for FlashInfer."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
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

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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
        # Check if cascade attention should be used
        # This is set by the model loader based on config
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
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, context)
        output = self.o_proj(attn_output)
        return output


class Qwen3AttentionFused(nn.Module):
    """Qwen3 attention layer with fused RMSNorm+QKV kernel and custom chunk attention."""

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
        
        # Store norm weight and QKV weights for fused kernel
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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
        
        # Attention implementation based on configuration
        if use_custom_chunk_kernel:
            # Will use ChunkAttention for custom kernels
            self.attn = None  # We'll handle attention manually
        else:
            # Check if cascade attention should be used
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
                
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
    ) -> torch.Tensor:
        """
        Forward pass with fused RMSNorm+QKV kernel.
        
        Note: This expects unnormalized hidden_states as input.
        """
        # Use fused kernel for RMSNorm + QKV projection
        # Ensure input is 2D for kernel
        batch_size = hidden_states.shape[0] if hidden_states.dim() == 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # Call fused kernel (no transpose needed - weight is already [qkv_dim, hidden_dim])
        q, k, v = FusedRMSNormQKV.forward(
            hidden_states.float(),
            self.input_layernorm.weight.float(),
            self.qkv_proj.weight.float(),
            self.num_heads,
            self.num_kv_heads,
            eps=self.input_layernorm.eps
        )
        
        # Add bias if present
        if self.qkv_proj.bias is not None:
            qkv_bias = self.qkv_proj.bias
            q_bias = qkv_bias[:self.q_size]
            k_bias = qkv_bias[self.q_size:self.q_size + self.kv_size]
            v_bias = qkv_bias[self.q_size + self.kv_size:]
            
            q = q + q_bias.view(1, -1).expand(q.shape[0], -1)
            k = k + k_bias.view(1, -1).expand(k.shape[0], -1)
            v = v + v_bias.view(1, -1).expand(v.shape[0], -1)
        
        # Convert back to half precision if needed
        if hidden_states.dtype == torch.float16:
            q, k, v = q.half(), k.half(), v.half()
            
        # q, k, v are already shaped as [batch_seq_len, num_heads, head_dim]
        # Apply Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # Apply attention
        if self.use_custom_chunk_kernel and context is not None:
            # Use custom chunk attention kernel
            # This assumes context has chunk information
            attn_output = ChunkAttention.decode_attention(
                q, 
                context.chunk_k_caches,  # These would need to be set up
                context.chunk_v_caches,
                context.chunk_lengths,
                context.chunk_levels,
                scale=self.scaling
            )
        else:
            # Use standard attention
            attn_output = self.attn(q, k, v, context)
            
        # Output projection
        output = self.o_proj(attn_output)
        return output


class Qwen3MLP(nn.Module):

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


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, layer_idx: int, use_custom_kernels: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_custom_kernels = use_custom_kernels
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        if use_custom_kernels:
            # Use fused attention (no separate input_layernorm needed)
            self.input_layernorm = None
            self.self_attn = Qwen3AttentionFused(
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
        else:
            # Use standard attention
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = Qwen3Attention(
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
        
        if self.use_custom_kernels:
            # Fused kernel expects unnormalized input
            hidden_states = self.self_attn(positions, hidden_states, context)
        else:
            # Standard path with separate normalization
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states, context)
            
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config, use_custom_kernels: bool = False):
        super().__init__()
        self.config = config
        self.use_custom_kernels = use_custom_kernels
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx, use_custom_kernels) 
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


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config, use_custom_kernels: bool = False):
        super().__init__()
        self.config = config
        self.use_custom_kernels = use_custom_kernels
        self.model = Qwen3Model(config, use_custom_kernels)
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