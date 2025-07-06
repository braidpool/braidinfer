"""
Unit tests for the standalone RMSNorm F32 kernel.
"""

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.rmsnorm_f32 import RMSNormF32


class TestRMSNormF32(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def pytorch_rmsnorm(self, x, weight, eps=1e-6):
        """Reference PyTorch implementation."""
        x_f32 = x.float()
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f32 / torch.sqrt(var + eps)
        return x_normed * weight.float()
    
    def test_correctness_small(self):
        """Test correctness on small tensors."""
        seq_len, hidden_dim = 4, 8
        eps = 1e-6
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Compute reference
        ref_output = self.pytorch_rmsnorm(input, weight, eps)
        
        # Compute with kernel
        kernel_output = RMSNormF32.forward(input, weight, eps)
        
        # Check output properties
        self.assertEqual(kernel_output.shape, input.shape)
        self.assertEqual(kernel_output.dtype, torch.float32)
        
        # Check correctness
        max_diff = torch.max(torch.abs(ref_output - kernel_output)).item()
        self.assertLess(max_diff, 1e-5, f"Max difference {max_diff} exceeds threshold")
    
    def test_correctness_large(self):
        """Test correctness on realistic sizes."""
        seq_len, hidden_dim = 512, 1024
        eps = 1e-6
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Compute reference
        ref_output = self.pytorch_rmsnorm(input, weight, eps)
        
        # Compute with kernel
        kernel_output = RMSNormF32.forward(input, weight, eps)
        
        # Check correctness with relative error
        rel_error = torch.max(torch.abs((ref_output - kernel_output) / (ref_output + 1e-8))).item()
        self.assertLess(rel_error, 1e-4, f"Relative error {rel_error} exceeds threshold")
    
    def test_extreme_values(self):
        """Test with extreme normalization weights like in Qwen3-0.6B."""
        seq_len, hidden_dim = 128, 1024
        eps = 1e-6
        
        # Create input with normal values
        input = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Create weight with extreme values (simulating Qwen3-0.6B)
        weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device=self.device)
        # Add some extreme values
        weight[0] = 96.5  # Extreme value like in Qwen3-0.6B
        weight[100:110] = 50.0
        
        # Compute reference
        ref_output = self.pytorch_rmsnorm(input, weight, eps)
        
        # Compute with kernel
        kernel_output = RMSNormF32.forward(input, weight, eps)
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(kernel_output)), "Output contains non-finite values")
        
        # Check correctness - may need higher tolerance due to extreme values
        rel_error = torch.max(torch.abs((ref_output - kernel_output) / (ref_output + 1e-8))).item()
        self.assertLess(rel_error, 1e-3, f"Relative error {rel_error} exceeds threshold for extreme values")
    
    def test_in_place_operation(self):
        """Test in-place operation support."""
        seq_len, hidden_dim = 64, 256
        eps = 1e-6
        
        # Create test data
        input = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Pre-allocate output buffer
        output_buffer = torch.empty(seq_len, hidden_dim, dtype=torch.float32, device=self.device)
        
        # Compute with pre-allocated buffer
        result = RMSNormF32.forward(input, weight, eps, output=output_buffer)
        
        # Check that result uses the same buffer
        self.assertTrue(result.data_ptr() == output_buffer.data_ptr(), "Output buffer not used")
        
        # Check correctness
        ref_output = self.pytorch_rmsnorm(input, weight, eps)
        rel_error = torch.max(torch.abs((ref_output - result) / (ref_output + 1e-8))).item()
        self.assertLess(rel_error, 1e-4, f"Relative error {rel_error} exceeds threshold")
    
    def test_different_dtypes(self):
        """Test with different input dtypes."""
        seq_len, hidden_dim = 32, 128
        eps = 1e-6
        
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                input = torch.randn(seq_len, hidden_dim, dtype=dtype, device=self.device)
                weight = torch.randn(hidden_dim, dtype=dtype, device=self.device)
                
                # Compute with kernel
                output = RMSNormF32.forward(input, weight, eps)
                
                # Check output is always float32
                self.assertEqual(output.dtype, torch.float32)
                
                # Check correctness
                ref_output = self.pytorch_rmsnorm(input, weight, eps)
                rel_error = torch.max(torch.abs((ref_output - output) / (ref_output + 1e-8))).item()
                self.assertLess(rel_error, 1e-4, f"Relative error {rel_error} exceeds threshold for {dtype}")
    
    def test_numerical_stability(self):
        """Test numerical stability with very small variance."""
        seq_len, hidden_dim = 16, 64
        eps = 1e-6
        
        # Create input with very small variance
        input = torch.ones(seq_len, hidden_dim, dtype=torch.bfloat16, device=self.device)
        input += torch.randn_like(input) * 1e-4  # Add tiny noise
        weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=self.device)
        
        # Compute with kernel
        output = RMSNormF32.forward(input, weight, eps)
        
        # Check output is finite and reasonable
        self.assertTrue(torch.all(torch.isfinite(output)), "Output contains non-finite values")
        self.assertTrue(torch.all(torch.abs(output) < 10), "Output values unreasonably large")


if __name__ == '__main__':
    unittest.main()