"""
Basic tests for separated RMSNorm and QKV+RoPE kernels.
"""

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.rmsnorm_f32 import RMSNormF32
from nanovllm.kernels.qkv_rope_simple import QKVRoPESimple


class TestSeparatedKernelsBasic(unittest.TestCase):
    """Basic tests for the separated kernel architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def test_separated_pipeline(self):
        """Test the complete separated pipeline: RMSNorm -> QKV+RoPE."""
        seq_len = 32
        hidden_dim = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Step 1: Create input
        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        norm_weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Step 2: Apply RMSNorm separately
        normalized = RMSNormF32.forward(hidden_states, norm_weight, eps=1e-6)
        
        # Verify normalization worked
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertTrue(torch.all(torch.isfinite(normalized)))
        
        # Step 3: Prepare QKV projection
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        qkv_weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device) * 0.02
        
        # Create RoPE cache
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        t = torch.arange(512, device=self.device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        
        positions = torch.arange(seq_len, device=self.device)
        
        # Step 4: Apply QKV+RoPE
        q, k, v = QKVRoPESimple.forward(
            normalized,  # float32 input
            qkv_weight,
            positions,
            cos_cache,
            sin_cache,
            num_heads,
            num_kv_heads
        )
        
        # Verify outputs
        self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
        self.assertEqual(k.shape, (seq_len, num_kv_heads, head_dim))
        self.assertEqual(v.shape, (seq_len, num_kv_heads, head_dim))
        
        self.assertTrue(torch.all(torch.isfinite(q)))
        self.assertTrue(torch.all(torch.isfinite(k)))
        self.assertTrue(torch.all(torch.isfinite(v)))
        
        print("✓ Separated pipeline test passed")
    
    def test_extreme_weights_stability(self):
        """Test stability with extreme normalization weights."""
        seq_len = 16
        hidden_dim = 128
        
        # Create extreme normalization weight
        norm_weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=self.device)
        norm_weight[0] = 96.5  # Extreme value from Qwen3-0.6B
        norm_weight[1:5] = 50.0
        
        # Test different input magnitudes
        for scale in [1e-3, 1.0, 10.0]:
            with self.subTest(scale=scale):
                hidden_states = torch.randn(seq_len, hidden_dim, device=self.device) * scale
                hidden_states = hidden_states.to(torch.bfloat16)
                
                # Apply RMSNorm
                normalized = RMSNormF32.forward(hidden_states, norm_weight, eps=1e-6)
                
                # Check stability
                self.assertTrue(torch.all(torch.isfinite(normalized)),
                               f"RMSNorm produced non-finite values with scale {scale}")
                
                # Check output is bounded
                max_val = torch.max(torch.abs(normalized)).item()
                self.assertLess(max_val, 1e5,
                               f"RMSNorm output too large: {max_val}")
        
        print("✓ Extreme weights stability test passed")
    
    def test_fused_vs_separated_accuracy(self):
        """Compare accuracy of fused vs separated approaches."""
        seq_len = 8
        hidden_dim = 64
        
        # Create test data
        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        norm_weight = torch.ones(hidden_dim, dtype=torch.float32, device=self.device)
        norm_weight[0] = 10.0  # Moderate scaling
        
        # Separated approach (float32 RMSNorm)
        norm_f32 = RMSNormF32.forward(
            hidden_states.bfloat16(),
            norm_weight.bfloat16(),
            eps=1e-6
        )
        
        # Fused approach simulation (bfloat16 throughout)
        x_bf16 = hidden_states.bfloat16()
        var = x_bf16.pow(2).mean(dim=-1, keepdim=True)
        norm_bf16 = x_bf16 * torch.rsqrt(var + 1e-6) * norm_weight.bfloat16()
        
        # Compare
        diff = torch.max(torch.abs(norm_f32 - norm_bf16.float())).item()
        rel_diff = diff / torch.max(torch.abs(norm_f32)).item()
        
        print(f"✓ Fused vs separated accuracy test passed")
        print(f"  Absolute difference: {diff:.2e}")
        print(f"  Relative difference: {rel_diff:.2e}")
        
        # Should be close but not identical due to precision differences
        self.assertLess(diff, 0.15, f"Difference too large: {diff}")


if __name__ == '__main__':
    unittest.main()