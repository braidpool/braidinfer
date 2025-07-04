#!/usr/bin/env python3
"""Test model loading functionality."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanovllm.config import Config
from nanovllm.engine.model_loader import ModelLoader


class TestModelLoading(unittest.TestCase):
    def test_load_qwen3_model(self):
        """Test loading Qwen3 model."""
        config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        model = ModelLoader.load_model(config)
        
        # Check model is loaded and on CUDA
        self.assertIsNotNone(model)
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        
        # Check model structure
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.model.embed_tokens)
        self.assertIsNotNone(model.model.layers)
        self.assertIsNotNone(model.model.norm)
        self.assertIsNotNone(model.lm_head)
        
        # Check layer count
        self.assertEqual(len(model.model.layers), config.hf_config.num_hidden_layers)
        
    def test_model_dtype(self):
        """Test model dtype is correct."""
        config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        model = ModelLoader.load_model(config)
        
        # Check dtype - should match config
        expected_dtype = config.hf_config.torch_dtype if hasattr(config.hf_config, 'torch_dtype') else torch.float16
        for param in model.parameters():
            self.assertEqual(param.dtype, expected_dtype)
            break
    
    def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        model = ModelLoader.load_model(config)
        
        # Check embedding shape
        embed_weight = model.model.embed_tokens.weight
        vocab_size, hidden_size = embed_weight.shape
        
        self.assertGreater(vocab_size, 150000)  # Qwen3 has large vocab
        self.assertEqual(hidden_size, config.hf_config.hidden_size)
        
        # Check lm_head matches
        lm_head_weight = model.lm_head.weight
        self.assertEqual(lm_head_weight.shape[0], vocab_size)
        self.assertEqual(lm_head_weight.shape[1], hidden_size)
    
    def test_attention_heads(self):
        """Test attention head configuration."""
        config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        model = ModelLoader.load_model(config)
        
        # Check first layer attention
        first_attn = model.model.layers[0].self_attn
        
        # Check QKV projection exists
        self.assertIsNotNone(first_attn.qkv_proj)
        
        # Check dimensions
        qkv_weight = first_attn.qkv_proj.weight
        hidden_size = config.hf_config.hidden_size
        num_heads = config.hf_config.num_attention_heads
        num_kv_heads = config.hf_config.num_key_value_heads
        head_dim = config.hf_config.head_dim
        
        # QKV weight should be [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
        expected_out_features = (num_heads + 2 * num_kv_heads) * head_dim
        self.assertEqual(qkv_weight.shape[0], expected_out_features)
        self.assertEqual(qkv_weight.shape[1], hidden_size)


if __name__ == '__main__':
    unittest.main()