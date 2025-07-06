"""
Unit tests for QKV+RoPE fused kernel.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.qkv_rope_simple import QKVRoPESimple


class TestQKVRoPESimple(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def create_rope_cache(self, head_dim, max_seq_len=8192, base=10000.0):
        """Create RoPE cos/sin cache."""
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        t = torch.arange(max_seq_len, device=self.device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        return freqs.cos(), freqs.sin()
    
    def apply_rope_reference(self, x, cos, sin):
        """Reference RoPE implementation."""
        # x: [seq_len, num_heads, head_dim]
        # cos, sin: [seq_len, head_dim/2]
        seq_len, num_heads, head_dim = x.shape
        
        # Reshape x to separate real and imaginary parts
        x = x.view(seq_len, num_heads, head_dim // 2, 2)
        real = x[..., 0]
        imag = x[..., 1]
        
        # Expand cos/sin for broadcasting
        cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim/2]
        sin = sin.unsqueeze(1)
        
        # Apply rotation
        new_real = real * cos - imag * sin
        new_imag = real * sin + imag * cos
        
        # Combine back
        result = torch.stack([new_real, new_imag], dim=-1)
        return result.view(seq_len, num_heads, head_dim)
    
    def test_qkv_projection_only(self):
        """Test QKV projection without RoPE."""
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
        
        # Reference computation
        ref_qkv = F.linear(input, weight.float(), bias.float())
        ref_qkv = ref_qkv.float()  # Ensure float32 for comparison
        
        # Split reference
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        ref_q = ref_qkv[:, :q_size].view(seq_len, num_heads, head_dim)
        ref_k = ref_qkv[:, q_size:q_size + k_size].view(seq_len, num_kv_heads, head_dim)
        ref_v = ref_qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
        
        # Kernel computation (without RoPE for this test)
        positions = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        cos_cache = torch.ones(8192, head_dim // 2, device=self.device)
        sin_cache = torch.zeros(8192, head_dim // 2, device=self.device)
        
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads, bias
        )
        
        # Convert to float32 for comparison
        q = q.float()
        k = k.float()
        v = v.float()
        
        # Check shapes
        self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
        self.assertEqual(k.shape, (seq_len, num_kv_heads, head_dim))
        self.assertEqual(v.shape, (seq_len, num_kv_heads, head_dim))
        
        # Check V is unchanged (no RoPE applied)
        v_diff = torch.max(torch.abs(v - ref_v)).item()
        self.assertLess(v_diff, 1e-3, f"V projection error: {v_diff}")
    
    def test_rope_application(self):
        """Test RoPE is correctly applied to Q and K."""
        seq_len = 16
        hidden_dim = 128
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Create positions and RoPE cache
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        
        # Get cos/sin for these positions
        cos_vals = cos_cache[positions]
        sin_vals = sin_cache[positions]
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads
        )
        
        # Reference: compute QKV without RoPE first
        ref_qkv = F.linear(input, weight.float()).float()
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        ref_q = ref_qkv[:, :q_size].view(seq_len, num_heads, head_dim)
        ref_k = ref_qkv[:, q_size:q_size + k_size].view(seq_len, num_kv_heads, head_dim)
        ref_v = ref_qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
        
        # Apply RoPE to reference Q and K
        ref_q_rope = self.apply_rope_reference(ref_q, cos_vals, sin_vals)
        ref_k_rope = self.apply_rope_reference(ref_k, cos_vals, sin_vals)
        
        # Compare (convert to same dtype)
        q_float = q.float()
        k_float = k.float()
        v_float = v.float()
        
        # Check Q with RoPE
        q_diff = torch.max(torch.abs(q_float - ref_q_rope)).item()
        self.assertLess(q_diff, 1e-2, f"Q RoPE error: {q_diff}")
        
        # Check K with RoPE
        k_diff = torch.max(torch.abs(k_float - ref_k_rope)).item()
        self.assertLess(k_diff, 1e-2, f"K RoPE error: {k_diff}")
        
        # Check V unchanged
        v_diff = torch.max(torch.abs(v_float - ref_v)).item()
        self.assertLess(v_diff, 1e-3, f"V should be unchanged: {v_diff}")
    
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
        
        # Non-sequential positions (e.g., for cached decoding)
        positions = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35], device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        
        # Run kernel
        q, k, v = QKVRoPESimple.forward(
            input, weight, positions, cos_cache, sin_cache,
            num_heads, num_kv_heads
        )
        
        # Check output shapes and finiteness
        self.assertEqual(q.shape, (seq_len, num_heads, head_dim))
        self.assertTrue(torch.all(torch.isfinite(q)), "Q contains non-finite values")
        self.assertTrue(torch.all(torch.isfinite(k)), "K contains non-finite values")
        self.assertTrue(torch.all(torch.isfinite(v)), "V contains non-finite values")
    
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
        self.assertTrue(torch.all(torch.isfinite(q)), "Q contains non-finite values with large theta")
        self.assertTrue(torch.all(torch.abs(q) < 100), "Q values unreasonably large")
    
    def test_performance(self):
        """Benchmark performance vs separate operations."""
        import time
        
        seq_len = 512
        hidden_dim = 1024
        num_heads = 16
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        cos_cache, sin_cache = self.create_rope_cache(head_dim)
        
        # Warmup
        for _ in range(10):
            _ = QKVRoPESimple.forward(input, weight, positions, cos_cache, sin_cache,
                                    num_heads, num_kv_heads)
        
        torch.cuda.synchronize()
        
        # Benchmark fused kernel
        start = time.time()
        for _ in range(100):
            q, k, v = QKVRoPESimple.forward(input, weight, positions, cos_cache, sin_cache,
                                          num_heads, num_kv_heads)
        torch.cuda.synchronize()
        fused_time = time.time() - start
        
        print(f"\nQKV+RoPE Performance:")
        print(f"  Sequence length: {seq_len}, Hidden dim: {hidden_dim}")
        print(f"  Fused kernel time: {fused_time*10:.2f} ms per iteration")


if __name__ == '__main__':
    unittest.main()