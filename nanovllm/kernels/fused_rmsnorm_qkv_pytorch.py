"""
PyTorch-based "fused" RMSNorm + QKV projection for numerical stability.

This version uses standard PyTorch operations to ensure exact compatibility
with the reference implementation, while still providing a fused interface.
"""

import torch
from typing import Tuple


class FusedRMSNormQKVPyTorch:
    """PyTorch-based fused RMSNorm + QKV projection for exact numerical compatibility."""
    
    @staticmethod
    def forward(
        input: torch.Tensor,      # [batch_seq_len, hidden_dim]
        norm_weight: torch.Tensor,  # [hidden_dim]
        qkv_weight: torch.Tensor,   # [qkv_dim, hidden_dim]
        num_q_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute fused RMSNorm + QKV projection using PyTorch operations.
        
        This ensures exact numerical compatibility with the standard path.
        
        Returns:
            q: [batch_seq_len, num_q_heads, head_dim]
            k: [batch_seq_len, num_kv_heads, head_dim]
            v: [batch_seq_len, num_kv_heads, head_dim]
        """
        batch_seq_len, hidden_dim = input.shape
        qkv_dim = qkv_weight.shape[0]
        # Calculate head_dim from dimensions
        total_heads = num_q_heads + 2 * num_kv_heads
        head_dim = qkv_dim // total_heads
        
        # Ensure float32 computation
        input = input.float()
        norm_weight = norm_weight.float()
        qkv_weight = qkv_weight.float()
        
        # RMSNorm using exact PyTorch operations
        var = input.pow(2).mean(dim=-1, keepdim=True)
        rms_norm = input * torch.rsqrt(var + eps) * norm_weight
        
        # QKV projection
        qkv = torch.matmul(rms_norm, qkv_weight.t())
        
        # Split QKV
        q_dim = num_q_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        
        q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)
        
        # Reshape
        q = q.view(batch_seq_len, num_q_heads, head_dim)
        k = k.view(batch_seq_len, num_kv_heads, head_dim)
        v = v.view(batch_seq_len, num_kv_heads, head_dim)
        
        return q, k, v