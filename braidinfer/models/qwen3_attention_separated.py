"""
Qwen3 Attention with separated RMSNorm and QKV+RoPE kernels.

This implementation follows llama.cpp's approach:
1. Compute RMSNorm separately with full float32 precision
2. Apply QKV projection and RoPE together
3. Handle extreme K normalization weights properly
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

from braidinfer.layers.layernorm import RMSNorm
from braidinfer.layers.linear import QKVParallelLinear
from braidinfer.layers.rotary_embedding import get_rope
from braidinfer.kernels.rmsnorm_f32 import RMSNormF32
from braidinfer.kernels.qkv_rope_simple import QKVRoPESimple

if TYPE_CHECKING:
    from braidinfer.core import CacheContext


class Qwen3AttentionSeparated(nn.Module):
    """
    Qwen3 attention with separated RMSNorm and QKV+RoPE.
    
    This implementation is designed for numerical stability with
    extreme K normalization weights (up to 96.5x).
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,  # Default, should be overridden by config
        rope_theta: float = 10000.0,  # Standard default, should be overridden by config
        rope_scaling: dict = None,
        max_position: int = 8192,
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
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_dim = self.q_size + 2 * self.kv_size
        
        # Create layers
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=qkv_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        
        # Create RoPE embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Create attention module
        try:
            from flashinfer import CascadeAttentionWrapper
            self.attn = CascadeAttentionWrapper(
                num_heads,
                num_kv_heads,
                self.head_dim,
                layer_idx,
            )
            self.use_flashinfer = True
        except ImportError:
            # Fallback to standard attention
            self.attn = None
            self.use_flashinfer = False
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: Optional['CacheContext'] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with separated RMSNorm and QKV+RoPE.
        
        Args:
            positions: Position indices
            hidden_states: Input hidden states
            context: Cache context for KV cache
            seq_lens: Sequence lengths for batched inference
            
        Returns:
            Output hidden states
        """
        # Step 1: Apply RMSNorm separately with float32 precision
        # This is critical for numerical stability
        normalized = RMSNormF32.forward(
            hidden_states,
            self.input_layernorm.weight,
            eps=self.input_layernorm.eps
        )
        
        # Step 2: Apply QKV projection with RoPE
        # Get RoPE cache
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cos_cache = cos_sin_cache[:, :self.head_dim // 2]
        sin_cache = cos_sin_cache[:, self.head_dim // 2:]
        
        # Apply fused QKV+RoPE
        # IMPORTANT: Pass None for bias if config says no bias (Qwen3 case)
        # to avoid using potentially corrupted bias values
        bias_to_use = None if not hasattr(self.qkv_proj, 'bias') or self.qkv_proj.bias is None else self.qkv_proj.bias
        
        q, k, v = QKVRoPESimple.forward(
            normalized,  # float32 input
            self.qkv_proj.weight,
            positions,
            cos_cache,
            sin_cache,
            self.num_heads,
            self.num_kv_heads,
            bias_to_use
        )
        
        # Step 3: Apply Q/K normalization
        # Reshape for normalization: [seq_len, num_heads, head_dim] -> [seq_len * num_heads, head_dim]
        seq_len = q.shape[0]
        q_for_norm = q.reshape(-1, self.head_dim)
        k_for_norm = k.reshape(-1, self.head_dim)
        
        # Apply normalization with float32 precision
        q_normed = RMSNormF32.forward(
            q_for_norm,
            self.q_norm.weight,
            eps=self.q_norm.eps
        )
        k_normed = RMSNormF32.forward(
            k_for_norm,
            self.k_norm.weight,
            eps=self.k_norm.eps
        )
        
        # Reshape back
        q = q_normed.reshape(seq_len, self.num_heads, self.head_dim)
        k = k_normed.reshape(seq_len, self.num_kv_heads, self.head_dim)
        
        # Step 4: Apply attention
        if self.use_flashinfer and context is not None:
            # Use FlashInfer cascade attention
            # Reshape for FlashInfer: [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
            q_flat = q.view(seq_len, -1)
            k_flat = k.view(seq_len, -1)
            v_flat = v.view(seq_len, -1)
            
            # Apply attention through context
            attn_output = context.compute_attention(
                self.layer_idx,
                q_flat,
                k_flat,
                v_flat,
                wrapper=self.attn
            )
        else:
            # Fallback to standard attention with GQA support
            # Handle grouped query attention
            if self.num_kv_heads != self.num_heads:
                # Repeat KV heads to match Q heads
                num_repeat = self.num_heads // self.num_kv_heads
                k = k.unsqueeze(2).repeat(1, 1, num_repeat, 1).reshape(seq_len, self.num_heads, self.head_dim)
                v = v.unsqueeze(2).repeat(1, 1, num_repeat, 1).reshape(seq_len, self.num_heads, self.head_dim)
            
            # Convert to [batch_size=1, num_heads, seq_len, head_dim]
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply softmax and compute output
            attn_weights = torch.softmax(scores, dim=-1)
            # Handle NaN from softmax of all -inf (first token case)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape back: [1, num_heads, seq_len, head_dim] -> [seq_len, num_heads * head_dim]
            attn_output = attn_output.squeeze(0).transpose(0, 1).contiguous()
            attn_output = attn_output.view(seq_len, -1)
        
        # Step 5: Apply output projection
        output = self.o_proj(attn_output)
        
        return output