#!/usr/bin/env python3
"""
Simple test for fused RMSNorm+QKV kernel integration.
"""

import torch
import unittest
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3AttentionFused
from nanovllm.layers.linear import QKVParallelLinear
from nanovllm.layers.layernorm import RMSNorm


class TestFusedKernelSimple(unittest.TestCase):
    """Simple test for fused kernel."""
    
    def test_fused_attention_forward(self):
        """Test fused attention forward pass."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Configuration
        hidden_size = 896
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64
        seq_len = 10
        
        # Create fused attention layer
        attn = Qwen3AttentionFused(
            layer_idx=0,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_custom_chunk_kernel=False,  # Don't use chunk kernel yet
        ).cuda()
        
        # Create inputs
        hidden_states = torch.randn(seq_len, hidden_size).cuda()
        positions = torch.arange(seq_len).cuda()
        
        # Test forward pass without context (will skip attention)
        with torch.no_grad():
            # We need to handle the case where attn expects context
            # For now, just test the fused kernel part
            
            # Import the fused kernel
            from nanovllm.kernels.fused_rmsnorm_qkv_production import FusedRMSNormQKV
            
            # Call fused kernel (no transpose needed)
            q, k, v = FusedRMSNormQKV.forward(
                hidden_states.float(),
                attn.input_layernorm.weight.float(),
                attn.qkv_proj.weight.float(),
                num_heads,
                num_kv_heads,
                eps=attn.input_layernorm.eps
            )
            
            # Check shapes
            self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
            self.assertEqual(k.shape, (seq_len, num_kv_heads, head_dim))
            self.assertEqual(v.shape, (seq_len, num_kv_heads, head_dim))
            
            # Apply Q/K normalization
            q_normed = attn.q_norm(q)
            k_normed = attn.k_norm(k)
            
            # Apply rotary embeddings
            q_rot, k_rot = attn.rotary_emb(positions, q_normed, k_normed)
            
            # Check final shapes
            self.assertEqual(q_rot.shape, (seq_len, num_heads, head_dim))
            self.assertEqual(k_rot.shape, (seq_len, num_kv_heads, head_dim))
            
    def test_compare_with_standard(self):
        """Compare fused kernel output with standard operations."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Configuration
        hidden_size = 896
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64
        seq_len = 5
        
        # Create layers
        norm = RMSNorm(hidden_size, eps=1e-6).cuda()
        qkv_proj = QKVParallelLinear(
            hidden_size, head_dim, num_heads, num_kv_heads, bias=False
        ).cuda()
        
        # Print weight shape for debugging
        print(f"QKV weight shape: {qkv_proj.weight.shape}")
        print(f"Expected output size: {(num_heads + 2 * num_kv_heads) * head_dim}")
        
        # Create input
        hidden_states = torch.randn(seq_len, hidden_size).cuda()
        
        with torch.no_grad():
            # Standard path
            normed = norm(hidden_states)
            qkv = qkv_proj(normed)
            q_size = num_heads * head_dim
            kv_size = num_kv_heads * head_dim
            q_std, k_std, v_std = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q_std = q_std.view(seq_len, num_heads, head_dim)
            k_std = k_std.view(seq_len, num_kv_heads, head_dim)
            v_std = v_std.view(seq_len, num_kv_heads, head_dim)
            
            # Fused path
            from nanovllm.kernels.fused_rmsnorm_qkv_production import FusedRMSNormQKV
            
            print(f"Input shape to kernel: {hidden_states.shape}")
            print(f"Norm weight shape: {norm.weight.shape}")
            print(f"QKV weight shape (no transpose): {qkv_proj.weight.shape}")
            
            q_fused, k_fused, v_fused = FusedRMSNormQKV.forward(
                hidden_states.float(),
                norm.weight.float(),
                qkv_proj.weight.float(),  # No transpose!
                num_heads,
                num_kv_heads,
                eps=norm.eps
            )
            
            print(f"q_std shape: {q_std.shape}, q_fused shape: {q_fused.shape}")
            print(f"k_std shape: {k_std.shape}, k_fused shape: {k_fused.shape}")
            print(f"v_std shape: {v_std.shape}, v_fused shape: {v_fused.shape}")
            
            # Compare outputs
            q_diff = torch.max(torch.abs(q_std - q_fused)).item()
            k_diff = torch.max(torch.abs(k_std - k_fused)).item()
            v_diff = torch.max(torch.abs(v_std - v_fused)).item()
            
            print(f"Q max diff: {q_diff}")
            print(f"K max diff: {k_diff}")
            print(f"V max diff: {v_diff}")
            
            # Should be very close
            self.assertLess(q_diff, 0.001)
            self.assertLess(k_diff, 0.001)
            self.assertLess(v_diff, 0.001)


if __name__ == "__main__":
    unittest.main()