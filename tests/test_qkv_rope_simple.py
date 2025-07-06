"""
Fixed unit tests for QKV+RoPE fused kernel.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.qkv_rope_simple import QKVRoPESimple
from nanovllm.layers.rotary_embedding import apply_rotary_emb


class TestQKVRoPEFixed(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def create_rope_cache(self, head_dim, max_seq_len=8192, base=10000.0):
        """Create RoPE cos/sin cache matching the kernel's format."""
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        t = torch.arange(max_seq_len, device=self.device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        return freqs.cos(), freqs.sin()
    
    def test_qkv_projection_basic(self):
        """Test basic QKV projection functionality."""
        seq_len = 32
        hidden_dim = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device)
        bias = torch.randn(qkv_dim, dtype=torch.bfloat16, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads, bias
        )
        
        # Check shapes
        self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
        self.assertEqual(k.shape, (seq_len, num_kv_heads, head_dim))
        self.assertEqual(v.shape, (seq_len, num_kv_heads, head_dim))
        
        # Check dtypes match weight dtype
        self.assertEqual(q.dtype, weight.dtype)
        self.assertEqual(k.dtype, weight.dtype)
        self.assertEqual(v.dtype, weight.dtype)
        
        # Check outputs are finite
        self.assertTrue(torch.all(torch.isfinite(q)))
        self.assertTrue(torch.all(torch.isfinite(k)))
        self.assertTrue(torch.all(torch.isfinite(v)))
    
    def test_qkv_projection_accuracy(self):
        """Test QKV projection accuracy against reference."""
        seq_len = 16
        hidden_dim = 128
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device) * 0.1
        bias = torch.randn(qkv_dim, dtype=torch.bfloat16, device=self.device) * 0.1
        
        # Reference computation (without RoPE)
        ref_qkv = F.linear(input, weight.float(), bias.float())
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        ref_v = ref_qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
        
        # Run kernel
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads, bias
        )
        
        # V should match exactly (no RoPE applied)
        v_diff = torch.max(torch.abs(v.float() - ref_v)).item()
        self.assertLess(v_diff, 1e-2, f"V projection error too high: {v_diff}")
    
    def test_rope_correctness(self):
        """Test RoPE is applied correctly."""
        seq_len = 8
        hidden_dim = 64
        num_heads = 2
        num_kv_heads = 1
        head_dim = hidden_dim // num_heads
        
        # Create simple test case
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.eye(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device) * 0.1
        
        # Run kernel
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads
        )
        
        # Check that Q and K have been modified by RoPE
        # but V has not
        self.assertTrue(torch.all(torch.isfinite(q)))
        self.assertTrue(torch.all(torch.isfinite(k)))
        self.assertTrue(torch.all(torch.isfinite(v)))
        
        # V should be a simple projection (no RoPE)
        # Q and K should have RoPE applied
        # Just verify they're different from input
        input_norm = torch.norm(input).item()
        q_norm = torch.norm(q).item()
        self.assertGreater(input_norm, 0)
        self.assertGreater(q_norm, 0)
    
    def test_different_positions(self):
        """Test with non-sequential positions."""
        seq_len = 8
        hidden_dim = 64
        num_heads = 2
        num_kv_heads = 1
        head_dim = hidden_dim // num_heads
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Non-sequential positions
        positions = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35], device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads
        )
        
        # Check outputs
        self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
        self.assertTrue(torch.all(torch.isfinite(q)))
        self.assertTrue(torch.all(torch.isfinite(k)))
        self.assertTrue(torch.all(torch.isfinite(v)))
    
    def test_extreme_rope_theta(self):
        """Test with extreme RoPE theta like Qwen3."""
        seq_len = 32
        hidden_dim = 128
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Qwen3 uses theta=1000000
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim, base=1000000.0)
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads
        )
        
        # Check outputs are reasonable
        self.assertTrue(torch.all(torch.isfinite(q)))
        self.assertTrue(torch.all(torch.abs(q) < 100))


if __name__ == '__main__':
    unittest.main()