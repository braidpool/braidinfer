"""
Unit test for the online softmax Triton kernel.

This test verifies that the new Triton-based online softmax kernel produces
numerically correct results compared to a reference implementation and properly
handles causal masking.
"""

import unittest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from braidinfer.kernels import online_softmax_update


class TestOnlineSoftmaxKernel(unittest.TestCase):
    """Test suite for the online softmax Triton kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test configuration
        self.num_heads = 8
        self.head_dim = 64
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping Triton kernel tests")
    
    def test_kernel_vs_reference_implementation(self):
        """Test kernel output against reference Python implementation."""
        batch_size = 4
        num_tokens = 16
        
        # Create test data
        query = torch.randn(self.num_heads, batch_size, self.head_dim, 
                           dtype=torch.float32, device=self.device)
        key = torch.randn(self.num_heads, num_tokens, self.head_dim,
                         dtype=torch.float32, device=self.device)
        value = torch.randn(self.num_heads, num_tokens, self.head_dim,
                           dtype=torch.float32, device=self.device)
        
        # Initialize running state
        m_i_kernel = torch.full((self.num_heads, batch_size), float('-inf'), 
                               dtype=torch.float32, device=self.device)
        l_i_kernel = torch.zeros((self.num_heads, batch_size), 
                                dtype=torch.float32, device=self.device)
        acc_i_kernel = torch.zeros((self.num_heads, batch_size, self.head_dim), 
                                  dtype=torch.float32, device=self.device)
        
        # Copy for reference implementation
        m_i_ref = m_i_kernel.clone()
        l_i_ref = l_i_kernel.clone()
        acc_i_ref = acc_i_kernel.clone()
        
        # Positions (no causal masking for simplicity)
        query_positions = torch.arange(batch_size, dtype=torch.int64, device=self.device)
        key_positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        
        # Test kernel
        online_softmax_update(
            query=query,
            key=key,
            value=value,
            m_i=m_i_kernel,
            l_i=l_i_kernel,
            acc_i=acc_i_kernel,
            query_positions=query_positions,
            key_positions=key_positions,
            scale=self.scale,
            apply_causal_mask=False
        )
        
        # Reference implementation (simplified online softmax)
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale  # [num_heads, batch_size, num_tokens]
        
        for token_idx in range(num_tokens):
            s_j = scores[:, :, token_idx]  # [num_heads, batch_size]
            v_j = value[:, token_idx, :]   # [num_heads, head_dim]
            
            # Online softmax update
            m_new = torch.maximum(m_i_ref, s_j)
            alpha = torch.exp(m_i_ref - m_new)
            beta = torch.exp(s_j - m_new)
            
            acc_i_ref *= alpha.unsqueeze(-1)
            l_i_ref *= alpha
            
            acc_i_ref += beta.unsqueeze(-1) * v_j.unsqueeze(1)
            l_i_ref += beta
            
            m_i_ref.copy_(m_new)
        
        # Compare results
        tolerance = 1e-5  # Using float32, so tighter tolerance than float16
        
        max_diff_m = torch.max(torch.abs(m_i_kernel - m_i_ref)).item()
        max_diff_l = torch.max(torch.abs(l_i_kernel - l_i_ref)).item()
        max_diff_acc = torch.max(torch.abs(acc_i_kernel - acc_i_ref)).item()
        
        self.assertLess(max_diff_m, tolerance, f"m_i differs by {max_diff_m}")
        self.assertLess(max_diff_l, tolerance, f"l_i differs by {max_diff_l}")
        self.assertLess(max_diff_acc, tolerance, f"acc_i differs by {max_diff_acc}")
    
    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        batch_size = 2
        num_tokens = 8
        
        # Create test data
        query = torch.randn(self.num_heads, batch_size, self.head_dim,
                           dtype=torch.float32, device=self.device)
        key = torch.randn(self.num_heads, num_tokens, self.head_dim,
                         dtype=torch.float32, device=self.device)
        value = torch.randn(self.num_heads, num_tokens, self.head_dim,
                           dtype=torch.float32, device=self.device)
        
        # Test with different query positions to verify causal masking
        for query_pos in [0, num_tokens // 2, num_tokens - 1]:
            # Initialize state
            m_i = torch.full((self.num_heads, batch_size), float('-inf'), 
                           dtype=torch.float32, device=self.device)
            l_i = torch.zeros((self.num_heads, batch_size), 
                             dtype=torch.float32, device=self.device)
            acc_i = torch.zeros((self.num_heads, batch_size, self.head_dim), 
                               dtype=torch.float32, device=self.device)
            
            # Positions for causal masking
            query_positions = torch.full((batch_size,), query_pos, 
                                       dtype=torch.int64, device=self.device)
            key_positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
            
            # Apply kernel with causal masking
            online_softmax_update(
                query=query,
                key=key,
                value=value,
                m_i=m_i,
                l_i=l_i,
                acc_i=acc_i,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.scale,
                apply_causal_mask=True
            )
            
            # Verify that the results are reasonable (not all zeros/infs)
            self.assertFalse(torch.isnan(acc_i).any(), "acc_i contains NaN")
            self.assertFalse(torch.isinf(acc_i).any(), "acc_i contains Inf")
            self.assertTrue(torch.isfinite(l_i).all(), "l_i contains non-finite values")
            
            # For query_pos = 0, only the first key should contribute
            if query_pos == 0:
                # l_i should be roughly the exponential of one score
                # This is a weak test, but verifies basic functionality
                self.assertTrue((l_i > 0).all(), "l_i should be positive when causal masking allows tokens")
    
    def test_incremental_updates(self):
        """Test that multiple kernel calls produce correct incremental updates."""
        batch_size = 3
        tokens_per_call = 4
        num_calls = 3
        
        query = torch.randn(self.num_heads, batch_size, self.head_dim,
                           dtype=torch.float32, device=self.device)
        
        # Initialize state
        m_i = torch.full((self.num_heads, batch_size), float('-inf'), 
                        dtype=torch.float32, device=self.device)
        l_i = torch.zeros((self.num_heads, batch_size), 
                         dtype=torch.float32, device=self.device)
        acc_i = torch.zeros((self.num_heads, batch_size, self.head_dim), 
                           dtype=torch.float32, device=self.device)
        
        # Prepare query positions
        total_context = tokens_per_call * num_calls
        query_positions = torch.full((batch_size,), total_context, 
                                   dtype=torch.int64, device=self.device)
        
        # Simulate multiple kernel calls (like processing multiple pages)
        all_keys = []
        all_values = []
        
        for call_idx in range(num_calls):
            # Create key/value for this call
            key = torch.randn(self.num_heads, tokens_per_call, self.head_dim,
                             dtype=torch.float32, device=self.device)
            value = torch.randn(self.num_heads, tokens_per_call, self.head_dim,
                               dtype=torch.float32, device=self.device)
            
            all_keys.append(key)
            all_values.append(value)
            
            # Key positions for this call
            key_pos_start = call_idx * tokens_per_call
            key_positions = torch.arange(key_pos_start, key_pos_start + tokens_per_call,
                                       dtype=torch.int64, device=self.device)
            
            # Apply kernel
            online_softmax_update(
                query=query,
                key=key,
                value=value,
                m_i=m_i,
                l_i=l_i,
                acc_i=acc_i,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.scale,
                apply_causal_mask=True
            )
        
        # Compare with single-shot computation
        all_keys_concat = torch.cat(all_keys, dim=1)  # [num_heads, total_tokens, head_dim]
        all_values_concat = torch.cat(all_values, dim=1)
        all_key_positions = torch.arange(total_context, dtype=torch.int64, device=self.device)
        
        # Initialize reference state
        m_i_ref = torch.full((self.num_heads, batch_size), float('-inf'), 
                            dtype=torch.float32, device=self.device)
        l_i_ref = torch.zeros((self.num_heads, batch_size), 
                             dtype=torch.float32, device=self.device)
        acc_i_ref = torch.zeros((self.num_heads, batch_size, self.head_dim), 
                               dtype=torch.float32, device=self.device)
        
        # Single-shot kernel call
        online_softmax_update(
            query=query,
            key=all_keys_concat,
            value=all_values_concat,
            m_i=m_i_ref,
            l_i=l_i_ref,
            acc_i=acc_i_ref,
            query_positions=query_positions,
            key_positions=all_key_positions,
            scale=self.scale,
            apply_causal_mask=True
        )
        
        # Compare incremental vs single-shot results
        tolerance = 1e-4  # Slightly looser tolerance for accumulated numerical errors
        
        max_diff_m = torch.max(torch.abs(m_i - m_i_ref)).item()
        max_diff_l = torch.max(torch.abs(l_i - l_i_ref)).item()
        max_diff_acc = torch.max(torch.abs(acc_i - acc_i_ref)).item()
        
        self.assertLess(max_diff_m, tolerance, 
                       f"Incremental m_i differs from single-shot by {max_diff_m}")
        self.assertLess(max_diff_l, tolerance, 
                       f"Incremental l_i differs from single-shot by {max_diff_l}")
        self.assertLess(max_diff_acc, tolerance, 
                       f"Incremental acc_i differs from single-shot by {max_diff_acc}")
    
    def test_edge_cases(self):
        """Test edge cases like empty inputs, single token, etc."""
        # Test single token
        query = torch.randn(self.num_heads, 1, self.head_dim,
                           dtype=torch.float32, device=self.device)
        key = torch.randn(self.num_heads, 1, self.head_dim,
                         dtype=torch.float32, device=self.device)
        value = torch.randn(self.num_heads, 1, self.head_dim,
                           dtype=torch.float32, device=self.device)
        
        m_i = torch.full((self.num_heads, 1), float('-inf'), 
                        dtype=torch.float32, device=self.device)
        l_i = torch.zeros((self.num_heads, 1), 
                         dtype=torch.float32, device=self.device)
        acc_i = torch.zeros((self.num_heads, 1, self.head_dim), 
                           dtype=torch.float32, device=self.device)
        
        query_positions = torch.tensor([0], dtype=torch.int64, device=self.device)
        key_positions = torch.tensor([0], dtype=torch.int64, device=self.device)
        
        # Should not crash
        try:
            online_softmax_update(
                query=query,
                key=key,
                value=value,
                m_i=m_i,
                l_i=l_i,
                acc_i=acc_i,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.scale,
                apply_causal_mask=True
            )
            # Verify reasonable output
            self.assertFalse(torch.isnan(acc_i).any())
            self.assertTrue((l_i > 0).all())
        except Exception as e:
            self.fail(f"Single token test failed: {e}")


if __name__ == '__main__':
    unittest.main()