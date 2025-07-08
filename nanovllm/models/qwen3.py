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
        layernorm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_fused_qkv and layernorm_weight is not None:
            # Use fused kernel
            from nanovllm.kernels.fused_rmsnorm_qkv_mixed_precision import FusedRMSNormQKVMixedPrecision
            
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
            
            # Flatten for rotary
            batch_seq_len = q.shape[0]
            q = q.view(batch_seq_len, -1)
            k = k.view(batch_seq_len, -1)
            v = v.view(batch_seq_len, -1)
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
        use_custom_chunk_kernel: bool = False,
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
                # Use SimpleAttention as fallback for custom kernels
                from nanovllm.layers.attention import Attention
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
        self._k_norm_scale = 1.0  # Scale factor for extreme K norm weights
        self._attention_scale_compensation = 1.0  # Compensation for attention scores

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
        from nanovllm.kernels.fused_rmsnorm_qkv_mixed_precision import FusedRMSNormQKVMixedPrecision
        
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
        
        # CRITICAL FIX: Ensure Q, K, V have consistent shapes regardless of path
        # The fused kernel returns [seq_len, num_heads, head_dim]
        # but the rest of the code expects them in head format for normalization
        if not hasattr(q, 'view'):
            # If q is not a tensor, something is wrong
            raise ValueError(f"Expected tensor, got {type(q)}")
            
        # Apply Q normalization in float32 for stability
        q_normed = self.q_norm(q.float() if q.dtype != torch.float32 else q)
        
        # CRITICAL FIX: Store K in cache BEFORE normalization to keep values in float16 range
        # K normalization will be applied during attention computation
        
        # Check if this layer has extreme K norm weights
        if self._use_standard_computation is None:
            k_norm_max = self.k_norm.weight.max().item()
            self._use_standard_computation = k_norm_max > 20.0
            self._k_norm_scale = min(k_norm_max / 10.0, 10.0) if k_norm_max > 20.0 else 1.0
        
        # Ensure scale is set even if _use_standard_computation was pre-set
        if self._use_standard_computation and self._k_norm_scale == 1.0:
            k_norm_max = self.k_norm.weight.max().item()
            self._k_norm_scale = min(k_norm_max / 10.0, 10.0)
        
        # Apply K normalization normally
        k_normed = self.k_norm(k.float() if k.dtype != torch.float32 else k)
        
        # IMPORTANT: For KV caching, we should store k (before normalization)
        # to keep values in reasonable float16 range
        k_unnormalized = k
        
        # Convert to appropriate precision
        # Always convert to input dtype (which might be bfloat16)
        q = q_normed.to(input_dtype)
        k = k_normed.to(input_dtype)
        # CRITICAL FIX: Ensure V is properly converted and not reused
        # Make a deep copy to prevent any aliasing issues
        v = v.to(input_dtype).detach().clone().contiguous()
        
        
        # Reshape q and k back to [batch_seq_len, hidden_dim] for rotary embeddings
        batch_seq_len = q.shape[0]
        q_flat = q.view(batch_seq_len, -1)
        k_flat = k.view(batch_seq_len, -1)
        
        # Flatten positions to match batch_seq_len dimension
        if positions.dim() == 2:
            positions_flat = positions.reshape(-1)
        else:
            positions_flat = positions
        
        # Apply rotary embeddings
        q_flat, k_flat = self.rotary_emb(positions_flat, q_flat, k_flat)
        
        # Also need to flatten v for attention - do this BEFORE making contiguous
        v_flat = v.view(batch_seq_len, -1)
        
        # Ensure all tensors are contiguous before passing to attention
        # This fixes the stride mismatch issue with FlashInfer
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        
        # NOTE: For KV caching, we should store k_unnormalized (before K norm)
        # This keeps values in float16 range and applies K norm during attention
        # Context/cache handling would happen in the attention module
        
        # Ensure tensors are in the expected dtype for attention module
        # The attention module expects the model's dtype, not float32
        if q_flat.dtype != input_dtype:
            q_flat = q_flat.to(input_dtype)
        if k_flat.dtype != input_dtype:
            k_flat = k_flat.to(input_dtype)
        if v_flat.dtype != input_dtype:
            v_flat = v_flat.to(input_dtype)
        
        # Apply attention without scale compensation
        # The K normalization already handles the scaling appropriately
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