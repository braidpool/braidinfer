"""
Fixed unit tests for Qwen3 separated model implementation.
"""

import unittest
import torch
from transformers import Qwen3Config

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.models.qwen3_separated import (
    Qwen3AttentionSeparated,
    Qwen3DecoderLayerSeparated, 
    Qwen3ModelSeparated,
    Qwen3ForCausalLMSeparated
)


class TestQwen3SeparatedFixed(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
        
        # Create a small test config
        self.config = self.create_test_config()
    
    def create_test_config(self):
        """Create a small test configuration."""
        return Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )
    
    def test_attention_forward(self):
        """Test basic forward pass of attention layer."""
        layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Test input
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        
        # Check output
        self.assertEqual(output.shape, (seq_len, self.config.hidden_size))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_decoder_layer(self):
        """Test decoder layer with separated kernels."""
        layer = Qwen3DecoderLayerSeparated(self.config, layer_idx=0).to(self.device)
        
        # Test input
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states)
        
        # Check output
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_full_model_forward(self):
        """Test full model forward pass."""
        model = Qwen3ForCausalLMSeparated(self.config).to(self.device)
        
        # Test input
        seq_len = 16
        batch_size = 1
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        # Forward pass
        hidden_states = model(input_ids, positions)
        
        # Check output
        self.assertEqual(hidden_states.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertTrue(torch.all(torch.isfinite(hidden_states)))
        
        # Test logits computation
        logits = model.compute_logits(hidden_states)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertTrue(torch.all(torch.isfinite(logits)))
    
    def test_extreme_k_norm_weights(self):
        """Test with extreme K normalization weights."""
        layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Set extreme K norm weight
        layer.k_norm.weight.data[0] = 96.5  # Extreme value from Qwen3-0.6B
        
        # Test input
        seq_len = 8
        hidden_states = torch.randn(seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device)
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        
        # Forward pass
        output = layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        
        # Check output is still stable
        self.assertTrue(torch.all(torch.isfinite(output)),
                       "Output contains non-finite values with extreme K norm weights")
        
        # Check output magnitude is reasonable
        output_norm = torch.norm(output, dim=-1).mean().item()
        self.assertLess(output_norm, 1000, f"Output norm {output_norm} is too large")
    
    def test_batch_processing(self):
        """Test with batch dimension."""
        layer = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
        ).to(self.device)
        
        # Test with batch dimension
        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size, device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        layernorm_weight = torch.ones(self.config.hidden_size, device=self.device)
        
        # Forward pass
        output = layer(positions.flatten(), hidden_states, layernorm_weight=layernorm_weight)
        
        # Check output
        self.assertEqual(output.shape, (batch_size * seq_len, self.config.hidden_size))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == '__main__':
    unittest.main()