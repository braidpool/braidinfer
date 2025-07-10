"""
Verify that embedding scaling is correctly implemented in Qwen3Model.

According to QWEN3_NUMERICAL_STABILITY_GUIDE.md, embeddings must be scaled
by 1/sqrt(hidden_size) to prevent numerical instability.
"""

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingScalingImplementation(unittest.TestCase):
    """Test that embedding scaling is correctly implemented."""
    
    def test_scaling_is_implemented_in_code(self):
        """Verify that the scaling code exists in qwen3.py."""
        # Read the qwen3.py file to verify the implementation
        qwen3_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'braidinfer', 'models', 'qwen3.py'
        )
        
        with open(qwen3_path, 'r') as f:
            content = f.read()
        
        # Check for the scaling line
        scaling_line = "hidden_states = hidden_states * (1.0 / (self.config.hidden_size ** 0.5))"
        self.assertIn(scaling_line, content,
                     "Embedding scaling not found in qwen3.py")
        
        # Also check it's in the forward method of Qwen3Model
        lines = content.split('\n')
        in_forward = False
        found_scaling = False
        
        for i, line in enumerate(lines):
            if 'class Qwen3Model' in line:
                # Found the model class
                for j in range(i, min(i + 50, len(lines))):
                    if 'def forward' in lines[j]:
                        in_forward = True
                    elif in_forward and scaling_line in lines[j]:
                        found_scaling = True
                        break
                    elif in_forward and 'def ' in lines[j] and 'forward' not in lines[j]:
                        # We've left the forward method
                        break
        
        self.assertTrue(found_scaling,
                       "Embedding scaling found but not in Qwen3Model.forward method")
    
    def test_scaling_value_calculation(self):
        """Test the scaling calculation for various hidden sizes."""
        test_cases = [
            (256, 1.0 / 16.0),      # sqrt(256) = 16
            (512, 1.0 / 22.627417),  # sqrt(512) ≈ 22.627
            (1024, 1.0 / 32.0),      # sqrt(1024) = 32 (Qwen3-0.6B)
            (2048, 1.0 / 45.254834), # sqrt(2048) ≈ 45.255
        ]
        
        for hidden_size, expected_scale in test_cases:
            with self.subTest(hidden_size=hidden_size):
                calculated_scale = 1.0 / (hidden_size ** 0.5)
                self.assertAlmostEqual(calculated_scale, expected_scale, places=6,
                                     msg=f"Incorrect scale for hidden_size={hidden_size}")
    
    def test_numerical_impact_of_scaling(self):
        """Test the numerical impact of embedding scaling."""
        hidden_size = 1024  # Qwen3-0.6B
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        # Create mock embeddings with reasonable magnitude
        embeddings = torch.randn(vocab_size, hidden_size) * 0.1
        
        # Create input ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Get embeddings
        raw_embeds = embeddings[input_ids.flatten()].reshape(batch_size, seq_len, hidden_size)
        
        # Apply scaling
        scale = 1.0 / (hidden_size ** 0.5)
        scaled_embeds = raw_embeds * scale
        
        # Check magnitudes
        raw_norm = torch.norm(raw_embeds, dim=-1).mean().item()
        scaled_norm = torch.norm(scaled_embeds, dim=-1).mean().item()
        
        # The scaled norm should be approximately 1/32 of the raw norm
        expected_ratio = scale
        actual_ratio = scaled_norm / raw_norm
        
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5,
                             msg=f"Scaling doesn't reduce magnitude correctly. "
                             f"Expected ratio: {expected_ratio}, Actual: {actual_ratio}")
        
        # Verify the scaled embeddings are in a reasonable range for stability
        self.assertLess(scaled_norm, 1.0,
                       "Scaled embedding norm should be < 1.0 for stability")


if __name__ == '__main__':
    unittest.main()