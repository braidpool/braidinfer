"""
Tests for numerical stability issues encountered during the sprint.

These tests specifically check for the issues that caused gibberish output
with extreme K normalization weights.
"""

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.rmsnorm_f32 import RMSNormF32
from nanovllm.kernels.fused_rmsnorm_qkv_minimal_f32 import FusedRMSNormQKVMinimalF32
from nanovllm.layers.layernorm import RMSNorm


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability with extreme normalization weights."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def test_extreme_k_norm_weights(self):
        """Test with extreme K normalization weights like Qwen3-0.6B."""
        seq_len = 8
        hidden_dim = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Create input
        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Create extreme K normalization weights
        k_norm = RMSNorm(head_dim, eps=1e-6).to(self.device)
        k_norm.weight.data[0] = 96.5  # Extreme value from Qwen3-0.6B
        k_norm.weight.data[1:5] = 50.0
        
        # Create test tensor
        k = torch.randn(seq_len, num_kv_heads, head_dim, device=self.device)
        
        # Apply normalization
        k_normed = k_norm(k)
        
        # Check output is finite
        self.assertTrue(torch.all(torch.isfinite(k_normed)), 
                       "K normalization produced non-finite values")
        
        # Check amplification
        k_norm_ratio = torch.norm(k_normed) / torch.norm(k)
        print(f"K norm amplification: {k_norm_ratio.item():.2f}x")
        
        # Even with extreme weights, output should be bounded
        self.assertLess(torch.max(torch.abs(k_normed)).item(), 1e6,
                       "K normalization produced unbounded values")
    
    def test_precision_loss_amplification(self):
        """Test how precision errors get amplified through layers."""
        hidden_dim = 256
        
        # Simulate precision error
        x = torch.ones(1, hidden_dim, device=self.device)
        error = torch.randn_like(x) * 1e-4  # Small error
        
        # Simulate extreme normalization
        scale = 96.5  # Extreme K norm weight
        
        # In float16
        x_f16 = x.half()
        error_f16 = error.half()
        amplified_f16 = (x_f16 + error_f16) * scale
        
        # In float32
        x_f32 = x.float()
        error_f32 = error.float()
        amplified_f32 = (x_f32 + error_f32) * scale
        
        # Compare relative errors
        rel_error_f16 = torch.max(torch.abs((amplified_f16 - x_f16 * scale) / (x_f16 * scale + 1e-8))).item()
        rel_error_f32 = torch.max(torch.abs((amplified_f32 - x_f32 * scale) / (x_f32 * scale + 1e-8))).item()
        
        print(f"Relative error with float16: {rel_error_f16:.2e}")
        print(f"Relative error with float32: {rel_error_f32:.2e}")
        
        # Float32 should have lower error than float16
        # But with small errors, the ratio might not be as dramatic
        if rel_error_f16 > 1e-6:  # Only compare if error is meaningful
            self.assertLess(rel_error_f32, rel_error_f16,
                           "Float32 should have lower error than float16")
    
    def test_layer_cascading_effect(self):
        """Test how errors cascade through multiple layers."""
        hidden_dim = 256
        num_layers = 5
        
        # Initial small error
        x = torch.randn(1, hidden_dim, device=self.device)
        initial_norm = torch.norm(x).item()
        
        # Simulate layers with extreme normalization
        scales = [96.5, 50.0, 30.0, 25.0, 20.0]  # Decreasing but still high
        
        # Track norm growth
        norms_f16 = [initial_norm]
        norms_f32 = [initial_norm]
        
        x_f16 = x.half()
        x_f32 = x.float()
        
        for i, scale in enumerate(scales):
            # Add small noise (simulation of computation error)
            noise = torch.randn_like(x) * 1e-4
            
            # Float16 path
            x_f16 = x_f16 + noise.half()
            x_f16 = x_f16 * scale
            norms_f16.append(torch.norm(x_f16).item())
            
            # Float32 path
            x_f32 = x_f32 + noise.float()
            x_f32 = x_f32 * scale
            norms_f32.append(torch.norm(x_f32).item())
        
        # Check growth rates
        growth_f16 = norms_f16[-1] / norms_f16[0]
        growth_f32 = norms_f32[-1] / norms_f32[0]
        
        print(f"Norm growth over {num_layers} layers:")
        print(f"  Float16: {growth_f16:.2e}x")
        print(f"  Float32: {growth_f32:.2e}x")
        
        # Both will grow, but float32 should be more stable
        self.assertLess(growth_f32, growth_f16,
                       "Float32 should have more stable growth")
    
    def test_rmsnorm_f32_stability(self):
        """Test that RMSNormF32 maintains stability with extreme inputs."""
        seq_len = 16
        hidden_dim = 256
        
        # Create extreme inputs
        test_cases = [
            ("normal", torch.randn(seq_len, hidden_dim, device=self.device)),
            ("large", torch.randn(seq_len, hidden_dim, device=self.device) * 1e3),
            ("small", torch.randn(seq_len, hidden_dim, device=self.device) * 1e-3),
            ("mixed", torch.cat([
                torch.ones(seq_len//2, hidden_dim, device=self.device) * 1e3,
                torch.ones(seq_len//2, hidden_dim, device=self.device) * 1e-3
            ], dim=0))
        ]
        
        # Extreme normalization weight
        norm_weight = torch.ones(hidden_dim, device=self.device)
        norm_weight[0] = 96.5
        
        for name, input_tensor in test_cases:
            with self.subTest(case=name):
                # Apply RMSNormF32
                output = RMSNormF32.forward(
                    input_tensor.bfloat16(),
                    norm_weight.bfloat16(),
                    eps=1e-6
                )
                
                # Check output
                self.assertTrue(torch.all(torch.isfinite(output)),
                              f"RMSNormF32 produced non-finite values for {name} input")
                
                # Check output magnitude is reasonable
                output_norm = torch.norm(output).item()
                self.assertLess(output_norm, 1e6,
                              f"RMSNormF32 output too large for {name} input")
    
    def test_fused_vs_separated_stability(self):
        """Compare numerical stability of fused vs separated approach."""
        seq_len = 8
        hidden_dim = 256
        num_heads = 8
        num_kv_heads = 2
        qkv_dim = (num_heads + 2 * num_kv_heads) * hidden_dim // num_heads
        
        # Create test data
        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        norm_weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=self.device)
        qkv_weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device) * 0.1
        
        # Add some extreme values to norm weight
        norm_weight[0] = 96.5
        
        # Test fused approach
        q_fused, k_fused, v_fused = FusedRMSNormQKVMinimalF32.forward(
            hidden_states,
            norm_weight,
            qkv_weight,
            num_heads,
            num_kv_heads,
            eps=1e-6
        )
        
        # Test separated approach
        normalized = RMSNormF32.forward(hidden_states, norm_weight, eps=1e-6)
        qkv = torch.matmul(normalized, qkv_weight.t().float())
        
        head_dim = hidden_dim // num_heads
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        
        q_sep = qkv[:, :q_size].view(seq_len, num_heads, head_dim)
        k_sep = qkv[:, q_size:q_size + k_size].view(seq_len, num_kv_heads, head_dim)
        v_sep = qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
        
        # Compare outputs
        q_diff = torch.max(torch.abs(q_fused - q_sep)).item()
        k_diff = torch.max(torch.abs(k_fused - k_sep)).item()
        v_diff = torch.max(torch.abs(v_fused - v_sep)).item()
        
        print(f"Fused vs Separated differences:")
        print(f"  Q: {q_diff:.2e}")
        print(f"  K: {k_diff:.2e}")
        print(f"  V: {v_diff:.2e}")
        
        # Should be reasonably close (allowing for numerical differences)
        self.assertLess(q_diff, 5e-2, "Q difference too large")
        self.assertLess(k_diff, 5e-2, "K difference too large")
        self.assertLess(v_diff, 5e-2, "V difference too large")
        
        # Both should be stable
        self.assertTrue(torch.all(torch.isfinite(q_fused)))
        self.assertTrue(torch.all(torch.isfinite(q_sep)))


if __name__ == '__main__':
    unittest.main()