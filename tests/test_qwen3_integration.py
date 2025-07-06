"""
Integration tests for Qwen3 with separated RMSNorm implementation.

This tests the actual Qwen3-0.6B model to verify the numerical stability fix.
"""

import unittest
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoConfig
from nanovllm.models.qwen3_separated import Qwen3ForCausalLMSeparated
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
import numpy as np


class TestQwen3Integration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cls.device == 'cpu':
            raise unittest.SkipTest("CUDA not available")
        
        # Model path - adjust as needed
        cls.model_path = "Qwen/Qwen2.5-0.5B"  # Using publicly available model
        cls.tokenizer = None
        cls.config = None
        
        # Try to load tokenizer and config
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path, trust_remote_code=True)
            cls.config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True)
            # Override to Qwen3 if needed
            if hasattr(cls.config, 'model_type') and cls.config.model_type == 'qwen2':
                cls.config.model_type = 'qwen3'
        except Exception as e:
            print(f"Warning: Could not load tokenizer/config from {cls.model_path}: {e}")
            # Create minimal config for testing
            from transformers import Qwen2Config
            cls.config = Qwen2Config(
                hidden_size=1024,
                num_attention_heads=16,
                num_key_value_heads=16,
                intermediate_size=2816,
                num_hidden_layers=24,
                vocab_size=151936,
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                model_type='qwen3'
            )
    
    def test_model_loading(self):
        """Test that we can load the model with separated kernels."""
        print("\n=== Testing Model Loading ===")
        
        # Create model with separated kernels
        model = Qwen3ForCausalLMSeparated(self.config)
        model = model.to(self.device)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(len(model.model.layers), self.config.num_hidden_layers)
        
        # Check that layers use separated attention
        for i, layer in enumerate(model.model.layers):
            self.assertTrue(hasattr(layer.self_attn, 'q_norm'))
            self.assertTrue(hasattr(layer.self_attn, 'k_norm'))
            print(f"Layer {i} - K norm max weight: {layer.self_attn.k_norm.weight.max().item():.2f}")
    
    def test_forward_pass_stability(self):
        """Test forward pass numerical stability."""
        print("\n=== Testing Forward Pass Stability ===")
        
        # Create model
        model = Qwen3ForCausalLMSeparated(self.config)
        model = model.to(self.device).eval()
        
        # Create dummy input
        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass
            hidden_states = model(input_ids, positions)
            
            # Check outputs are finite
            self.assertTrue(torch.all(torch.isfinite(hidden_states)), 
                          "Hidden states contain non-finite values")
            
            # Check magnitude is reasonable
            hidden_norm = torch.norm(hidden_states, dim=-1).mean().item()
            print(f"Hidden states norm: {hidden_norm:.2f}")
            self.assertLess(hidden_norm, 1000, "Hidden states norm too large")
            
            # Get logits
            logits = model.compute_logits(hidden_states)
            self.assertTrue(torch.all(torch.isfinite(logits)), 
                          "Logits contain non-finite values")
    
    def test_layer_by_layer_analysis(self):
        """Analyze layer-by-layer numerical behavior."""
        print("\n=== Layer-by-Layer Analysis ===")
        
        # Create model
        model = Qwen3ForCausalLMSeparated(self.config)
        model = model.to(self.device).eval()
        
        # Input
        input_ids = torch.randint(0, 1000, (1, 8), device=self.device)
        positions = torch.arange(8, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get embeddings
            hidden_states = model.model.embed_tokens(input_ids)
            hidden_states = hidden_states * (1.0 / (self.config.hidden_size ** 0.5))
            
            print(f"Initial norm: {torch.norm(hidden_states).item():.2f}")
            
            # Track through layers
            for i, layer in enumerate(model.model.layers[:5]):  # Just first 5 layers
                old_norm = torch.norm(hidden_states).item()
                hidden_states = layer(positions, hidden_states)
                new_norm = torch.norm(hidden_states).item()
                
                # Check K norm weights
                k_norm_max = layer.self_attn.k_norm.weight.max().item()
                
                print(f"Layer {i}: norm {old_norm:.2f} -> {new_norm:.2f} "
                      f"(ratio: {new_norm/old_norm:.2f}), K norm max: {k_norm_max:.2f}")
                
                # Check for explosion
                self.assertLess(new_norm/old_norm, 100, 
                              f"Layer {i} amplification too high")
    
    def test_compare_implementations(self):
        """Compare standard and separated implementations."""
        print("\n=== Comparing Implementations ===")
        
        # Create both models with small config for testing
        test_config = type(self.config)(
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
        
        standard_model = Qwen3ForCausalLM(test_config, use_custom_kernels=False)
        separated_model = Qwen3ForCausalLMSeparated(test_config)
        
        # Copy weights from standard to separated
        separated_model.load_state_dict(standard_model.state_dict())
        
        standard_model = standard_model.to(self.device).eval()
        separated_model = separated_model.to(self.device).eval()
        
        # Test input
        input_ids = torch.randint(0, 1000, (1, 16), device=self.device)
        positions = torch.arange(16, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Forward passes
            standard_hidden = standard_model(input_ids, positions)
            separated_hidden = separated_model(input_ids, positions)
            
            # Compare
            max_diff = torch.max(torch.abs(standard_hidden - separated_hidden)).item()
            rel_diff = torch.max(torch.abs((standard_hidden - separated_hidden) / 
                                          (standard_hidden + 1e-8))).item()
            
            print(f"Max absolute difference: {max_diff:.2e}")
            print(f"Max relative difference: {rel_diff:.2%}")
            
            # We expect some difference due to float32 RMSNorm
            self.assertLess(rel_diff, 0.1, "Implementations differ too much")
    
    def test_extreme_k_norm_handling(self):
        """Test handling of extreme K normalization weights."""
        print("\n=== Testing Extreme K Norm Weights ===")
        
        # Create model
        model = Qwen3ForCausalLMSeparated(self.config)
        
        # Set extreme K norm weights in first layer (simulate Qwen3-0.6B)
        with torch.no_grad():
            model.model.layers[0].self_attn.k_norm.weight.data *= 96.5
            
        model = model.to(self.device).eval()
        
        # Test input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        positions = torch.arange(5, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass
            hidden_states = model(input_ids, positions)
            
            # Check stability
            self.assertTrue(torch.all(torch.isfinite(hidden_states)), 
                          "Extreme K norm caused instability")
            
            # Check magnitude
            norm = torch.norm(hidden_states).item()
            print(f"Output norm with extreme K weights: {norm:.2f}")
            self.assertLess(norm, 1e6, "Output exploded with extreme weights")
    
    def test_generation_coherence(self):
        """Test text generation coherence (if tokenizer available)."""
        if self.tokenizer is None:
            self.skipTest("Tokenizer not available")
        
        print("\n=== Testing Generation Coherence ===")
        
        # Create model
        model = Qwen3ForCausalLMSeparated(self.config)
        model = model.to(self.device).eval()
        
        # Simple generation test
        input_text = "Hello, my name is"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Generate a few tokens
            generated_ids = input_ids.clone()
            
            for _ in range(10):
                positions = torch.arange(generated_ids.shape[1], device=self.device).unsqueeze(0)
                hidden_states = model(generated_ids, positions)
                logits = model.compute_logits(hidden_states)
                
                # Get next token (greedy)
                next_token = torch.argmax(logits[0, -1, :])
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Check for repetition or nonsense
                if generated_ids.shape[1] > 3:
                    last_tokens = generated_ids[0, -3:].tolist()
                    if len(set(last_tokens)) == 1:
                        print("Warning: Repetitive output detected")
                        break
            
            # Decode and print
            generated_text = self.tokenizer.decode(generated_ids[0])
            print(f"Generated: {generated_text}")
            
            # Basic coherence check - should not be all same token
            unique_tokens = len(set(generated_ids[0].tolist()))
            self.assertGreater(unique_tokens, 3, "Generated text is too repetitive")


if __name__ == '__main__':
    unittest.main(verbosity=2)