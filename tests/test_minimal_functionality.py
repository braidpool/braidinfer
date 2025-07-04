#!/usr/bin/env python3
"""Minimal tests that don't require loading full models."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.config import Config


class TestMinimalFunctionality(unittest.TestCase):
    def test_sequence_creation(self):
        """Test basic sequence creation."""
        tokens = [1, 2, 3, 4, 5]
        params = SamplingParams(max_tokens=10, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=params)
        
        # Check attributes
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.token_ids, tokens)
        self.assertEqual(seq.max_tokens, 10)
        self.assertIsNotNone(seq.seq_id)
        
    def test_sequence_append(self):
        """Test appending tokens to sequence."""
        tokens = [1, 2, 3]
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams())
        
        # Append token
        seq.append_token(4)
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.token_ids[-1], 4)
        
        # Append another
        seq.append_token(5)
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 5])
    
    def test_config_creation(self):
        """Test config creation."""
        config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        # Check basic attributes
        self.assertEqual(config.model, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.max_model_len, 512)
        self.assertEqual(config.kvcache_block_size, 16)
        self.assertTrue(config.enforce_eager)
    
    def test_sampling_params(self):
        """Test sampling parameters."""
        params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
            ignore_eos=True
        )
        
        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.max_tokens, 100)
        self.assertTrue(params.ignore_eos)
        
        # Test defaults
        default_params = SamplingParams()
        self.assertEqual(default_params.temperature, 1.0)
        self.assertEqual(default_params.max_tokens, 64)
        self.assertFalse(default_params.ignore_eos)
    
    def test_tensor_device_operations(self):
        """Test basic tensor operations on CUDA."""
        # Create tensor on CUDA
        x = torch.tensor([1, 2, 3], device="cuda")
        
        # Check device
        self.assertEqual(x.device.type, "cuda")
        
        # Basic operations
        y = x + 1
        self.assertEqual(y.tolist(), [2, 3, 4])
        self.assertEqual(y.device.type, "cuda")
        
        # Move to CPU and back
        x_cpu = x.cpu()
        self.assertEqual(x_cpu.device.type, "cpu")
        
        x_cuda = x_cpu.cuda()
        self.assertEqual(x_cuda.device.type, "cuda")
        self.assertTrue(torch.equal(x, x_cuda))


if __name__ == '__main__':
    unittest.main()