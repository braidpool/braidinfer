"""
Qwen3 model modified to use FlashInfer attention.
"""

import torch
from torch import nn
from transformers import Qwen3Config

from braidinfer.layers.activation import SiluAndMul
from braidinfer.layers.attention import Attention
from braidinfer.layers.layernorm import RMSNorm
from braidinfer.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from braidinfer.layers.rotary_embedding import get_rope
from braidinfer.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from braidinfer.kernels.fused_rmsnorm_qkv_production import FusedRMSNormQKV
from braidinfer.kernels.fused_rmsnorm_qkv_minimal_f32 import FusedRMSNormQKVMinimalF32


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from braidinfer.engine.inference_context import InferenceContext


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
        rope_theta: float = 10000.0,  # Standard default, should be overridden by config
        rope_scaling: tuple | None = None,
        use_fused_qkv: bool = False,  # Use fused RMSNorm+QKV kernel
    ) -> None:
        super().__init__()
        self.use_fused_qkv = use_fused_qkv
        self.rms_norm_eps = rms_norm_eps
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
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.layer_idx,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
        layernorm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_fused_qkv and layernorm_weight is not None:
            # Use fused kernel
            from braidinfer.kernels.fused_rmsnorm_qkv_mixed_precision import FusedRMSNormQKVMixedPrecision

            # Ensure 2D input
            orig_shape = hidden_states.shape
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

            q, k, v = FusedRMSNormQKVMixedPrecision.forward(
                hidden_states,
                layernorm_weight,
                self.qkv_proj.weight,
                self.qkv_proj.bias if hasattr(self.qkv_proj, 'bias') and self.qkv_proj.bias is not None else None,
                self.num_heads,
                self.num_kv_heads,
                eps=self.rms_norm_eps
            )

            # Apply Q/K normalization
            q = self.q_norm(q.float()).to(hidden_states.dtype)
            k = self.k_norm(k.float()).to(hidden_states.dtype)

            # Reshape q, k, v back to [batch_seq_len, hidden_dim] for rotary embeddings
            batch_seq_len = q.shape[0]
            q_flat = q.view(batch_seq_len, -1)
            k_flat = k.view(batch_seq_len, -1)
            v_flat = v.view(batch_seq_len, -1)
        else:
            # Standard path
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
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,  # Standard default, should be overridden by config
        rope_scaling: dict = None,
        max_position: int = 8192,
        use_fused_output: bool = False,  # Disabled due to shape issues
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_fused_output = use_fused_output
        self.qkv_bias = qkv_bias

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

        # Store norm eps for fused kernel (weight will be passed from decoder layer)
        self.rms_norm_eps = rms_norm_eps
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

        # Always create standard attention module for prefill
        # Custom chunk kernel is used automatically when active_chunks present
        from braidinfer.layers.attention import Attention

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.layer_idx,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext' = None,
        residual: Optional[torch.Tensor] = None,
        layernorm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with fused RMSNorm+QKV kernel.

        Note: This expects unnormalized hidden_states as input.
        """
        # CRITICAL: Save the input dtype - this is what we must output
        input_dtype = hidden_states.dtype

        # Use fused kernel for RMSNorm + QKV projection
        # Ensure input is 2D for kernel
        batch_size = hidden_states.shape[0] if hidden_states.dim() == 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Call fused kernel (no transpose needed - weight is already [qkv_dim, hidden_dim])
        if layernorm_weight is None:
            raise ValueError("layernorm_weight must be provided for fused kernel")

        # Use the mixed precision fused kernel
        from braidinfer.kernels.fused_rmsnorm_qkv_mixed_precision import FusedRMSNormQKVMixedPrecision

        q, k, v = FusedRMSNormQKVMixedPrecision.forward(
            hidden_states,
            layernorm_weight,
            self.qkv_proj.weight,
            None if not self.qkv_bias else self.qkv_proj.bias,  # Use bias only if config says so
            self.num_heads,
            self.num_kv_heads,
            eps=self.rms_norm_eps
        )

        # Bias is now handled inside the fused kernel

        # Reshape q, k, v back to [batch_seq_len, hidden_dim] for rotary embeddings
        batch_seq_len = q.shape[0]
        q_flat = q.view(batch_seq_len, -1)
        k_flat = k.view(batch_seq_len, -1)
        v_flat = v.view(batch_seq_len, -1)

        # Flatten positions to match batch_seq_len dimension
        if positions.dim() == 2:
            positions_flat = positions.reshape(-1)
        else:
            positions_flat = positions

        # CRITICAL FIX: Apply RoPE to the normalized query and the UN-NORMALIZED key.
        # The K-normalization will be applied inside the attention mechanism.
        q_rotated, _ = self.rotary_emb(positions_flat, q_flat, k_flat)
        
        # Pass the rotated query and the ORIGINAL k and v to the attention layer.
        # The attention layer is responsible for caching k/v and applying k-norm.
        attn_output = self.attn(q_rotated, k_flat, v_flat, context)

        # Output projection
        if self.use_fused_output and residual is not None:
            # Use fused output projection + residual
            from braidinfer.kernels.fused_attention_output import FusedAttentionOutput

            # Reshape attn_output if needed
            attn_shape = attn_output.shape
            if attn_output.dim() == 2:
                attn_output = attn_output.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
            elif attn_output.dim() == 3:
                attn_output = attn_output.unsqueeze(0)  # Add batch dim

            # Ensure residual has same shape
            if residual.dim() == 2:
                residual = residual.unsqueeze(0).unsqueeze(0)
            elif residual.dim() == 3:
                residual = residual.unsqueeze(0)

            output = FusedAttentionOutput.forward(
                attn_output,
                self.o_proj.weight,
                residual
            )

            # Reshape back to original dimensions
            if len(attn_shape) == 2:
                output = output.squeeze(0).squeeze(0)
            elif len(attn_shape) == 3:
                output = output.squeeze(0)
        else:
            # Standard output projection
            output = self.o_proj(attn_output)

        # CRITICAL: Ensure output is in input dtype for residual connection
        # For extended precision layers, convert back only after projection
        if output.dtype != input_dtype:
            output = output.to(input_dtype)

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
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False)

        if use_custom_kernels:
            # Use fused attention
            # Note: We still need input_layernorm for weight loading compatibility
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = Qwen3AttentionFused(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=getattr(config, "head_dim", None),
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=attention_bias,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position=max_position_embeddings,
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
                qkv_bias=attention_bias,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position=max_position_embeddings,
                use_fused_qkv=use_custom_kernels,  # Use fused kernel when custom kernels enabled
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
            # Pass layernorm weight for fused kernel
            hidden_states = self.self_attn(positions, hidden_states, context, layernorm_weight=self.input_layernorm.weight)
            hidden_states = residual + hidden_states
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
        # Apply embedding scaling as per QWEN3_NUMERICAL_STABILITY_GUIDE.md
        hidden_states = hidden_states * (1.0 / (self.config.hidden_size ** 0.5))
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
