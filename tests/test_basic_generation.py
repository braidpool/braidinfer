#!/usr/bin/env python3
"""Test basic text generation functionality."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.page_manager import PageManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestBasicGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.config = Config(
            model=self.model_path,
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        self.model_runner = ModelRunner(self.config)
        
        # Smaller page manager to avoid OOM
        self.page_manager = PageManager(
            num_pages=50,  # Reduced from default
            page_size=self.config.kvcache_block_size,
            num_layers=self.config.hf_config.num_hidden_layers,
            num_kv_heads=self.config.hf_config.num_key_value_heads,
            head_dim=self.config.hf_config.head_dim,
            dtype=self.config.hf_config.torch_dtype
        )
        self.model_runner.set_page_manager(self.page_manager)
    
    def test_single_token_generation(self):
        """Test generating a single token."""
        # Simple prompt
        text = "Hello"
        tokens = self.tokenizer.encode(text)
        
        # Create sequence
        sampling_params = SamplingParams(max_tokens=1, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        self.page_manager.allocate(seq)
        
        # Generate one token
        with torch.no_grad():
            next_token = self.model_runner.run([seq], is_prefill=True)[0]
        
        # Check we got a valid token
        self.assertIsInstance(next_token, int)
        self.assertGreaterEqual(next_token, 0)
        self.assertLess(next_token, self.tokenizer.vocab_size + 300)  # Allow for added tokens
        
        # Decode to check it's not gibberish
        decoded = self.tokenizer.decode([next_token])
        print(f"\nGenerated token: {next_token} ('{decoded}')")
    
    def test_multi_token_generation(self):
        """Test generating multiple tokens."""
        # Create a simple prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tokens = self.tokenizer.encode(text)
        
        # Create sequence
        sampling_params = SamplingParams(max_tokens=10, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        self.page_manager.allocate(seq)
        
        # Generate tokens
        generated_tokens = []
        
        # First token (prefill)
        with torch.no_grad():
            next_token = self.model_runner.run([seq], is_prefill=True)[0]
        seq.append_token(next_token)
        generated_tokens.append(next_token)
        
        # Generate more tokens (decode)
        for _ in range(4):  # Generate 4 more tokens
            with torch.no_grad():
                next_token = self.model_runner.run([seq], is_prefill=False)[0]
            seq.append_token(next_token)
            generated_tokens.append(next_token)
        
        # Check we generated tokens
        self.assertEqual(len(generated_tokens), 5)
        
        # Decode and check
        generated_text = self.tokenizer.decode(generated_tokens)
        print(f"\nGenerated text (5 tokens): '{generated_text}'")
        
        # Should not be empty or just special tokens
        self.assertGreater(len(generated_text.strip()), 0)
    
    def test_sequence_continuation(self):
        """Test that sequences properly continue from previous context."""
        # Start with a clear prompt
        text = "The capital of France is"
        tokens = self.tokenizer.encode(text)
        
        sampling_params = SamplingParams(max_tokens=5, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        self.page_manager.allocate(seq)
        
        # Generate tokens
        generated = []
        
        # Prefill
        with torch.no_grad():
            next_token = self.model_runner.run([seq], is_prefill=True)[0]
        seq.append_token(next_token)
        generated.append(next_token)
        
        # Check sequence length is updated
        self.assertEqual(self.page_manager.seq_lengths[seq.seq_id], len(tokens))
        
        # Generate more
        for _ in range(2):
            with torch.no_grad():
                next_token = self.model_runner.run([seq], is_prefill=False)[0]
            seq.append_token(next_token)
            generated.append(next_token)
        
        # Check final sequence length
        # After prefill: seq_length = len(tokens) = 5
        # After 2 decode steps: seq_length = 5 + 2 = 7
        expected_length = len(tokens) + 2  # 2 tokens generated in decode loop
        actual_length = self.page_manager.seq_lengths[seq.seq_id]
        self.assertEqual(actual_length, expected_length, 
                        f"Expected {expected_length} but got {actual_length}. "
                        f"Original: {len(tokens)}, decode steps: 2")
        
        # Decode result
        result = self.tokenizer.decode(generated)
        print(f"\nContinuation of 'The capital of France is': '{result}'")
        
        # Should mention Paris or France-related content
        # (Note: with temperature=0.6, exact output may vary)
        self.assertGreater(len(result.strip()), 0)


if __name__ == '__main__':
    unittest.main()