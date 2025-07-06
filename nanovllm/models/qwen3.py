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
from nanovllm.kernels.fused_rmsnorm_qkv_float32 import FusedRMSNormQKVFloat32
from nanovllm.kernels.fused_rmsnorm_qkv_pytorch import FusedRMSNormQKVPyTorch
from nanovllm.kernels.fused_rmsnorm_qkv_minimal_f32 import FusedRMSNormQKVMinimalF32
from nanovllm.kernels.chunk_attention import ChunkAttention


from typing import TYPE_CHECKING, Optional

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
        use_fused_output: bool = False,  # Disabled due to shape issues
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_fused_output = use_fused_output
        
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
        
        # Pre-check for extreme K normalization weights at initialization
        # This avoids the check during forward pass
        self._use_standard_computation = None  # Will be set after weights are loaded

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
        # Use fused kernel for RMSNorm + QKV projection
        # Ensure input is 2D for kernel
        batch_size = hidden_states.shape[0] if hidden_states.dim() == 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # Call fused kernel (no transpose needed - weight is already [qkv_dim, hidden_dim])
        if layernorm_weight is None:
            raise ValueError("layernorm_weight must be provided for fused kernel")
        
        # Check if this model has extreme K normalization weights
        # If so, use standard computation for numerical stability
        if hasattr(self, '_use_standard_computation'):
            use_standard = self._use_standard_computation
        else:
            # Check once and cache the result
            self._use_standard_computation = self.k_norm.weight.max().item() > 20
            use_standard = self._use_standard_computation
        
        if use_standard:
            # Use minimal float32 kernel for models with extreme weights
            # This follows llama.cpp's approach: float32 only for accumulators
            q, k, v = FusedRMSNormQKVMinimalF32.forward(
                hidden_states,  # Keep in original dtype
                layernorm_weight,  # Keep in original dtype
                self.qkv_proj.weight,  # Keep in original dtype
                self.num_heads,
                self.num_kv_heads,
                eps=self.rms_norm_eps
            )
        else:
            # Use the fused kernel for normal models
            # Also use minimal float32 approach for consistency
            q, k, v = FusedRMSNormQKVMinimalF32.forward(
                hidden_states,  # Keep in original dtype
                layernorm_weight,  # Keep in original dtype
                self.qkv_proj.weight,  # Keep in original dtype
                self.num_heads,
                self.num_kv_heads,
                eps=self.rms_norm_eps
            )
        
        # Add bias if present (both fused kernels don't include bias)
        if self.qkv_proj.bias is not None:
            qkv_bias = self.qkv_proj.bias
            q_bias = qkv_bias[:self.q_size]
            k_bias = qkv_bias[self.q_size:self.q_size + self.kv_size]
            v_bias = qkv_bias[self.q_size + self.kv_size:]
            
            # Reshape bias to match [batch, num_heads, head_dim]
            q_bias = q_bias.view(self.num_heads, self.head_dim)
            k_bias = k_bias.view(self.num_kv_heads, self.head_dim)
            v_bias = v_bias.view(self.num_kv_heads, self.head_dim)
            
            # Add bias
            q = q + q_bias.unsqueeze(0)
            k = k + k_bias.unsqueeze(0)
            v = v + v_bias.unsqueeze(0)
        
        # Apply Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Convert to original dtype for FlashInfer compatibility
        # Note: q, k, v may be in float32 if using minimal float32 kernel
        target_dtype = hidden_states.dtype if hidden_states.dtype != torch.float32 else torch.bfloat16
        if q.dtype != target_dtype:
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        
        # Reshape q and k back to [batch_seq_len, hidden_dim] for rotary embeddings
        batch_seq_len = q.shape[0]
        q_flat = q.view(batch_seq_len, -1)
        k_flat = k.view(batch_seq_len, -1)
        
        # Apply rotary embeddings
        q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
        
        # Reshape back to [batch_seq_len, num_heads, head_dim]
        q = q_flat.view(batch_seq_len, self.num_heads, self.head_dim)
        k = k_flat.view(batch_seq_len, self.num_kv_heads, self.head_dim)
        
        # Also need to flatten v for attention
        v_flat = v.view(batch_seq_len, -1)
        
        # Ensure all tensors are contiguous before passing to attention
        # This fixes the stride mismatch issue with FlashInfer
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        
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
            # Use standard attention (doesn't use custom chunk kernel)
            # The attention module expects q, k, v in flat shape, not head shape
            attn_output = self.attn(q_flat, k_flat, v_flat, context)
            
        # Output projection
        if self.use_fused_output and residual is not None:
            # Use fused output projection + residual
            from nanovllm.kernels.fused_attention_output import FusedAttentionOutput
            
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
            # Fused kernel expects unnormalized input and layernorm weight
            # Check if we should use fused output projection
            if hasattr(self.self_attn, 'use_fused_output'):
                # Pass residual for fused output projection + add
                hidden_states = self.self_attn(positions, hidden_states, context, residual, self.input_layernorm.weight)
            else:
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
    
    def check_extreme_weights(self):
        """Check for extreme K normalization weights after model is loaded."""
        if self.use_custom_kernels:
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer.self_attn, 'k_norm') and hasattr(layer.self_attn.k_norm, 'weight'):
                    if layer.self_attn.k_norm.weight is not None:
                        max_weight = layer.self_attn.k_norm.weight.max().item()
                        layer.self_attn._use_standard_computation = max_weight > 20

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