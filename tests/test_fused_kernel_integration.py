#!/usr/bin/env python3
"""
Test integration of fused RMSNorm+QKV kernel with Qwen3 model.
"""

import torch
import unittest
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM


class TestFusedKernelIntegration(unittest.TestCase):
    """Test fused kernel integration."""
    
    def setUp(self):
        """Set up test configuration."""
        # Create a small test config
        self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.config.num_hidden_layers = 2  # Small for testing
        self.config.hidden_size = 896
        self.config.num_attention_heads = 14
        self.config.num_key_value_heads = 2
        
    def test_model_creation(self):
        """Test that models can be created with and without custom kernels."""
        # Standard model
        model_standard = Qwen3ForCausalLM(self.config, use_custom_kernels=False)
        self.assertFalse(model_standard.use_custom_kernels)
        self.assertIsNotNone(model_standard.model.layers[0].input_layernorm)
        
        # Fused kernel model
        model_fused = Qwen3ForCausalLM(self.config, use_custom_kernels=True)
        self.assertTrue(model_fused.use_custom_kernels)
        self.assertIsNone(model_fused.model.layers[0].input_layernorm)
        
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct shapes."""
        batch_size = 1
        seq_len = 10
        
        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0)
        
        # Test both models
        for use_custom in [False, True]:
            with self.subTest(use_custom_kernels=use_custom):
                model = Qwen3ForCausalLM(self.config, use_custom_kernels=use_custom)
                model.eval()
                
                with torch.no_grad():
                    # Forward pass through model
                    hidden_states = model.model.embed_tokens(input_ids)
                    
                    # Flatten positions for attention
                    positions_flat = positions.flatten()
                    
                    # Pass through first layer
                    hidden_states = model.model.layers[0](positions_flat, hidden_states)
                    
                    # Check shape
                    expected_shape = (batch_size * seq_len, self.config.hidden_size)
                    self.assertEqual(hidden_states.shape, expected_shape)
                    
    def test_output_similarity(self):
        """Test that fused and standard models produce similar outputs."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        batch_size = 1
        seq_len = 5
        
        # Create models
        model_standard = Qwen3ForCausalLM(self.config, use_custom_kernels=False).cuda()
        model_fused = Qwen3ForCausalLM(self.config, use_custom_kernels=True).cuda()
        
        # Copy relevant weights from standard to fused
        # We need to handle the different structure
        fused_state = model_fused.state_dict()
        standard_state = model_standard.state_dict()
        
        # Copy all matching keys
        for key in fused_state:
            if key in standard_state:
                fused_state[key] = standard_state[key]
            # Handle the layer norm weight that moved into attention
            elif "self_attn.input_layernorm.weight" in key:
                # Get layer index
                layer_idx = key.split(".")[2]
                standard_key = f"model.layers.{layer_idx}.input_layernorm.weight"
                if standard_key in standard_state:
                    fused_state[key] = standard_state[standard_key]
                    
        model_fused.load_state_dict(fused_state)
        
        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        positions = torch.arange(seq_len).unsqueeze(0).cuda()
        
        model_standard.eval()
        model_fused.eval()
        
        with torch.no_grad():
            # Get embeddings
            hidden_standard = model_standard.model.embed_tokens(input_ids)
            hidden_fused = model_fused.model.embed_tokens(input_ids)
            
            # Flatten positions
            positions_flat = positions.flatten()
            
            # Pass through first layer
            output_standard = model_standard.model.layers[0](positions_flat, hidden_standard)
            output_fused = model_fused.model.layers[0](positions_flat, hidden_fused)
            
            # Check outputs are close
            max_diff = torch.max(torch.abs(output_standard - output_fused)).item()
            print(f"Max difference: {max_diff}")
            
            # Should be very close (allowing for numerical differences)
            self.assertLess(max_diff, 0.01)


if __name__ == "__main__":
    unittest.main()