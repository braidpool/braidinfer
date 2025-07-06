"""
Test the separated RMSNorm + QKV implementation for Qwen3.
"""

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3Config
from nanovllm.models.qwen3_separated import (
    Qwen3AttentionSeparated, 
    Qwen3DecoderLayerSeparated,
    Qwen3ForCausalLMSeparated
)
from nanovllm.models.qwen3 import Qwen3Attention


class TestQwen3Separated(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
        
        # Create a small test config
        self.config = Qwen3Config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=512,
            num_hidden_layers=2,
            vocab_size=1000,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )
    
    def test_attention_layer_shapes(self):
        """Test that attention layer produces correct shapes."""
        layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Create dummy layernorm weight
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        
        # Test inputs
        seq_len = 16
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        
        # Check output shape
        self.assertEqual(output.shape, (seq_len, self.config.hidden_size))
    
    def test_decoder_layer(self):
        """Test decoder layer with separated kernels."""
        layer = Qwen3DecoderLayerSeparated(self.config, layer_idx=0).to(self.device)
        
        # Test inputs
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, hidden_states.shape)
        
        # Check output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output contains non-finite values")
    
    def test_full_model_forward(self):
        """Test full model forward pass."""
        model = Qwen3ForCausalLMSeparated(self.config).to(self.device)
        
        # Test inputs
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass
        hidden_states = model(input_ids, positions)
        
        # Check shape
        self.assertEqual(hidden_states.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # Compute logits
        logits = model.compute_logits(hidden_states)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
    
    def test_numerical_stability(self):
        """Test with extreme normalization weights."""
        layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Create extreme K normalization weights (like Qwen3-0.6B)
        layer.k_norm.weight.data = torch.randn_like(layer.k_norm.weight) * 50.0
        layer.k_norm.weight.data[0] = 96.5  # Extreme value
        
        # Create layernorm weight
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        
        # Test inputs
        seq_len = 16
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        
        # Check output is finite despite extreme weights
        self.assertTrue(torch.all(torch.isfinite(output)), 
                       "Output contains non-finite values with extreme K norm weights")
        
        # Check output magnitude is reasonable
        output_norm = torch.norm(output, dim=-1).mean().item()
        self.assertLess(output_norm, 1000, f"Output norm {output_norm} is too large")
    
    def test_compare_with_standard(self):
        """Compare outputs with standard implementation."""
        # Create both implementations
        separated_layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        standard_layer = Qwen3Attention(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Copy weights
        standard_layer.qkv_proj.weight.data = separated_layer.qkv_proj.weight.data.clone()
        standard_layer.o_proj.weight.data = separated_layer.o_proj.weight.data.clone()
        standard_layer.q_norm.weight.data = separated_layer.q_norm.weight.data.clone()
        standard_layer.k_norm.weight.data = separated_layer.k_norm.weight.data.clone()
        
        # Test inputs
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        
        # Normalize input for standard layer
        from nanovllm.layers.layernorm import RMSNorm
        input_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device)
        normalized_input = input_norm(hidden_states)
        
        # Forward passes
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        separated_output = separated_layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        standard_output = standard_layer(positions, normalized_input)
        
        # Compare outputs (allow some tolerance due to float32 vs mixed precision)
        max_diff = torch.max(torch.abs(separated_output - standard_output)).item()
        rel_diff = torch.max(torch.abs((separated_output - standard_output) / (standard_output + 1e-8))).item()
        
        print(f"\nComparison with standard implementation:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Max relative difference: {rel_diff:.2%}")
        
        # We expect some difference due to float32 RMSNorm
        self.assertLess(rel_diff, 0.05, f"Relative difference {rel_diff} exceeds 5%")


if __name__ == '__main__':
    unittest.main()