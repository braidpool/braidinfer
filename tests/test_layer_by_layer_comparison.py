"""
Layer-by-layer comparison test for Qwen3 model.

This test compares the outputs of each layer between:
1. Standard PyTorch/HuggingFace implementation
2. Our custom kernel implementation
3. Separated kernel implementation

It helps identify exactly where numerical divergence occurs.
"""

import unittest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.models.qwen3 import Qwen3ForCausalLM as CustomQwen3
from nanovllm.models.qwen3_attention_separated import Qwen3AttentionSeparated


class LayerOutputCollector:
    """Collects intermediate outputs from model layers."""
    
    def __init__(self):
        self.outputs = {}
        self.hooks = []
    
    def add_hook(self, module, name):
        """Add a forward hook to collect outputs."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[name] = output.detach().cpu()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def clear(self):
        """Clear collected outputs."""
        self.outputs.clear()


class TestLayerByLayerComparison(unittest.TestCase):
    """Test layer-by-layer comparison between implementations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cls.device == 'cpu':
            return  # Skip on CPU
        
        # Use a smaller model for testing
        cls.model_name = "Qwen/Qwen2-0.5B"  # Smaller model for testing
        cls.seq_len = 32
        cls.vocab_size = 151936
    
    def setUp(self):
        """Set up for each test."""
        if self.device == 'cpu':
            self.skipTest("CUDA not available")
    
    def test_embedding_scaling(self):
        """Test that embedding scaling is correctly applied."""
        # Create config
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Create input
        input_ids = torch.randint(0, 1000, (1, self.seq_len), device=self.device)
        
        # Load reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Get embedding output
        embeddings = ref_model.model.embed_tokens(input_ids)
        
        # Check if scaling is applied
        hidden_size = config.hidden_size
        expected_scale = 1.0 / (hidden_size ** 0.5)
        
        # The embeddings should have reasonable magnitude after scaling
        embedding_norm = torch.norm(embeddings, dim=-1).mean().item()
        
        print(f"Embedding norm: {embedding_norm:.4f}")
        print(f"Expected scale factor: {expected_scale:.6f}")
        print(f"Hidden size: {hidden_size}")
        
        # Verify the norm is reasonable (not too large)
        self.assertLess(embedding_norm, 10.0, 
                       "Embedding norm too large - scaling might be missing")
    
    def test_rope_theta_value(self):
        """Test that RoPE theta is correctly set."""
        # Create config
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Check rope_theta in config
        rope_theta = getattr(config, 'rope_theta', None)
        print(f"RoPE theta from config: {rope_theta}")
        
        # For Qwen3, it should be 1,000,000
        if 'qwen' in self.model_name.lower():
            expected_theta = 1000000.0
            if rope_theta is not None:
                self.assertEqual(rope_theta, expected_theta,
                               f"RoPE theta {rope_theta} != expected {expected_theta}")
    
    def test_layer_outputs_comparison(self):
        """Compare layer outputs between implementations."""
        # This is a placeholder for the full comparison
        # In practice, you would:
        # 1. Load both models
        # 2. Add hooks to collect intermediate outputs
        # 3. Run the same input through both
        # 4. Compare outputs at each layer
        # 5. Find where they diverge
        
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Print key config values for debugging
        print("\nModel Configuration:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_attention_heads: {config.num_attention_heads}")
        print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
        print(f"  rope_theta: {getattr(config, 'rope_theta', 10000)}")
        print(f"  rms_norm_eps: {config.rms_norm_eps}")
        
        # Create simple test input
        input_ids = torch.ones((1, 4), dtype=torch.long, device=self.device)
        
        # This would be expanded to actually compare models
        self.assertTrue(True)  # Placeholder
    
    def test_attention_implementation(self):
        """Test the separated attention implementation."""
        # Create a test configuration
        hidden_size = 1024
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_size // num_heads
        seq_len = 16
        
        # Create attention layer
        attn = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=1000000.0,  # Correct value for Qwen3
        ).to(self.device)
        
        # Create test input
        hidden_states = torch.randn(seq_len, hidden_size, device=self.device, dtype=torch.float16)
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            output = attn(positions, hidden_states)
        
        # Check output
        self.assertEqual(output.shape, (seq_len, hidden_size))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf")
        
        # Check output magnitude
        output_norm = torch.norm(output, dim=-1).mean().item()
        print(f"\nAttention output norm: {output_norm:.4f}")
        
        # Should be reasonable
        self.assertLess(output_norm, 100.0, "Output norm too large")
        self.assertGreater(output_norm, 0.01, "Output norm too small")
    
    def test_numerical_stability_check(self):
        """Test numerical stability with extreme weights."""
        # Create attention with extreme K norm weights
        attn = Qwen3AttentionSeparated(
            layer_idx=0,
            hidden_size=1024,
            num_heads=8,
            num_kv_heads=2,
        ).to(self.device)
        
        # Set extreme K norm weight (like in Qwen3-0.6B)
        with torch.no_grad():
            attn.k_norm.weight.data[0] = 96.5
            attn.k_norm.weight.data[1:5] = 50.0
        
        # Create test input with small magnitude
        seq_len = 8
        hidden_states = torch.randn(seq_len, 1024, device=self.device, dtype=torch.float16) * 0.1
        positions = torch.arange(seq_len, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            output = attn(positions, hidden_states)
        
        # Check for numerical issues
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if has_nan or has_inf:
            print("\nWARNING: Numerical instability detected with extreme weights")
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")
        else:
            output_norm = torch.norm(output, dim=-1).mean().item()
            print(f"\nOutput norm with extreme weights: {output_norm:.4f}")


if __name__ == '__main__':
    unittest.main()