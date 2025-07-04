#!/usr/bin/env python3
"""Test that sequence lengths are properly updated in PageManager."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.page_manager import PageManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestSequenceLengthUpdate(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        self.model_runner = ModelRunner(self.config)
        self.page_manager = PageManager(
            num_pages=100,
            page_size=self.config.kvcache_block_size,
            num_layers=self.config.hf_config.num_hidden_layers,
            num_kv_heads=self.config.hf_config.num_key_value_heads,
            head_dim=self.config.hf_config.head_dim,
            dtype=self.config.hf_config.torch_dtype
        )
        self.model_runner.set_page_manager(self.page_manager)
        
    def test_sequence_length_updated_after_prefill(self):
        """Test that sequence lengths are updated after prefill."""
        # Create a sequence with some tokens
        tokens = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=1, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        
        # Allocate pages
        self.page_manager.allocate(seq)
        
        # Check initial state
        self.assertEqual(self.page_manager.seq_lengths.get(seq.seq_id, 0), 0,
                        "Sequence length should be 0 before prefill")
        
        # Run prefill
        with torch.no_grad():
            self.model_runner.run([seq], is_prefill=True)
        
        # Check that sequence length was updated
        self.assertEqual(self.page_manager.seq_lengths.get(seq.seq_id), len(tokens),
                        f"Sequence length should be {len(tokens)} after prefill")
    
    def test_sequence_length_updated_after_decode(self):
        """Test that sequence lengths are incremented after decode."""
        # Create and prefill a sequence
        tokens = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=5, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        
        self.page_manager.allocate(seq)
        
        # Run prefill
        with torch.no_grad():
            next_token = self.model_runner.run([seq], is_prefill=True)[0]
        
        seq.append_token(next_token)
        initial_length = self.page_manager.seq_lengths.get(seq.seq_id)
        
        # Run decode
        with torch.no_grad():
            next_token = self.model_runner.run([seq], is_prefill=False)[0]
        
        # Check that sequence length was incremented
        self.assertEqual(self.page_manager.seq_lengths.get(seq.seq_id), initial_length + 1,
                        "Sequence length should increment by 1 after decode")
    
    def test_kv_indices_correct_after_update(self):
        """Test that KV indices are correct after sequence length updates."""
        tokens = list(range(20))  # 20 tokens = more than one page
        sampling_params = SamplingParams(max_tokens=5, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        
        self.page_manager.allocate(seq)
        
        # Run prefill
        with torch.no_grad():
            self.model_runner.run([seq], is_prefill=True)
        
        # Get indices for decode
        indices, indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            [seq], for_prefill=False
        )
        
        # With 20 tokens and page_size=16, we should have 2 pages
        # First page: 16 tokens, Second page: 4 tokens
        expected_pages = 2
        self.assertEqual(len(indices), expected_pages,
                        f"Should have {expected_pages} pages for {len(tokens)} tokens")
        
        # Last page should have 4 tokens
        self.assertEqual(last_page_lens[0], 4,
                        "Last page should have 4 tokens")


if __name__ == '__main__':
    unittest.main()