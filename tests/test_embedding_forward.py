#!/usr/bin/env python3
"""Test basic embedding and forward pass functionality."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanovllm.config import Config
from nanovllm.engine.model_loader import ModelLoader


class TestEmbeddingForward(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        self.model = ModelLoader.load_model(self.config)
        self.model.eval()
        
    def test_embedding_forward(self):
        """Test embedding layer forward pass."""
        # Create input tokens on CUDA
        tokens = torch.tensor([1, 2, 3, 4, 5], device="cuda")
        
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(tokens)
        
        # Check output
        self.assertEqual(embeddings.shape[0], len(tokens))
        self.assertEqual(embeddings.shape[1], self.config.hf_config.hidden_size)
        self.assertEqual(embeddings.device.type, "cuda")
        
        # Check dtype matches config
        expected_dtype = self.config.hf_config.torch_dtype if hasattr(self.config.hf_config, 'torch_dtype') else torch.float16
        self.assertEqual(embeddings.dtype, expected_dtype)
        
        # Check values are reasonable
        self.assertFalse(torch.isnan(embeddings).any())
        self.assertFalse(torch.isinf(embeddings).any())
        
    def test_mlp_forward(self):
        """Test MLP (feed-forward) layer forward pass."""
        # Create simple input
        seq_len = 5
        hidden_size = self.config.hf_config.hidden_size
        dtype = self.config.hf_config.torch_dtype if hasattr(self.config.hf_config, 'torch_dtype') else torch.float16
        
        # Create tensors on CUDA with correct dtype
        hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
        
        # Test MLP from first layer
        mlp = self.model.model.layers[0].mlp
        
        with torch.no_grad():
            output = mlp(hidden_states)
        
        # Check output
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(output.dtype, dtype)
        
        # Should be different from input
        self.assertFalse(torch.allclose(output, hidden_states))
        
    def test_final_norm_and_lm_head(self):
        """Test final normalization and language model head."""
        seq_len = 5
        hidden_size = self.config.hf_config.hidden_size
        dtype = self.config.hf_config.torch_dtype if hasattr(self.config.hf_config, 'torch_dtype') else torch.float16
        
        # Create input on CUDA with correct dtype
        hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
        
        with torch.no_grad():
            # Apply final norm
            normed = self.model.model.norm(hidden_states)
            
            # Apply lm_head to last token
            logits = self.model.lm_head(normed[-1:])
        
        # Check shapes
        self.assertEqual(normed.shape, hidden_states.shape)
        self.assertEqual(logits.shape[0], 1)  # Single token
        self.assertEqual(logits.shape[1], self.model.lm_head.weight.shape[0])  # Vocab size
        
        # Check device and dtype
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(logits.dtype, dtype)
        
        # Check values are reasonable
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
        
    def test_logit_statistics(self):
        """Test that logit statistics are reasonable."""
        # Simple input
        tokens = torch.tensor([1, 2, 3], device="cuda")
        
        with torch.no_grad():
            # Just test embeddings -> norm -> lm_head without attention layers
            hidden = self.model.model.embed_tokens(tokens)
            
            # Apply layer norms without going through attention
            for i in range(min(2, len(self.model.model.layers))):
                # Just apply the layer norms
                layer = self.model.model.layers[i]
                hidden = layer.input_layernorm(hidden)
                # Skip attention, just do MLP
                mlp_out = layer.mlp(hidden)
                hidden = hidden + mlp_out
                hidden = layer.post_attention_layernorm(hidden)
            
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden[-1:])
        
        # Check statistics
        logit_mean = logits.mean().item()
        logit_std = logits.std().item()
        
        # Mean can be large when skipping attention
        self.assertLess(abs(logit_mean), 30.0, f"Logit mean too large: {logit_mean}")
        
        # Should have reasonable variance
        self.assertGreater(logit_std, 0.5, f"Logit std too low: {logit_std}")
        self.assertLess(logit_std, 10.0, f"Logit std too high: {logit_std}")
        
        # Max should not be extreme
        max_logit = logits.max().item()
        self.assertLess(max_logit, 50.0, f"Max logit too high: {max_logit}")


if __name__ == '__main__':
    unittest.main()