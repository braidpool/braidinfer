#!/usr/bin/env python3
"""Test basic KV cache functionality."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanovllm.engine.page_manager import PageManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestKVCacheBasics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 2
        self.num_kv_heads = 4
        self.head_dim = 64
        self.page_size = 16
        self.num_pages = 10
        
        self.page_manager = PageManager(
            num_pages=self.num_pages,
            page_size=self.page_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.float16
        )
    
    def test_kv_cache_shape(self):
        """Test that KV cache has correct shape."""
        # Should be [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
        expected_shape = (self.num_layers, self.num_pages, 2, self.num_kv_heads, 
                         self.page_size, self.head_dim)
        self.assertEqual(self.page_manager.kv_cache.shape, expected_shape)
        self.assertEqual(self.page_manager.kv_cache.device.type, "cuda")
        self.assertEqual(self.page_manager.kv_cache.dtype, torch.float16)
    
    def test_page_allocation(self):
        """Test basic page allocation."""
        tokens = [1, 2, 3, 4, 5]
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams(max_tokens=10))
        
        # Allocate pages
        self.page_manager.allocate(seq)
        
        # Check allocation
        self.assertIn(seq.seq_id, self.page_manager.seq_page_tables)
        allocated_pages = self.page_manager.seq_page_tables[seq.seq_id]
        
        # Should allocate pages for prompt + max_tokens
        # 5 tokens + 10 max_tokens = 15 total, needs 1 page with page_size=16
        self.assertEqual(len(allocated_pages), 1)
        
        # Page should be marked as used
        page_id = allocated_pages[0]
        self.assertNotIn(page_id, self.page_manager.free_pages)
    
    def test_sequence_length_tracking(self):
        """Test that sequence lengths are tracked correctly."""
        tokens = [1, 2, 3, 4, 5]
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams(max_tokens=10))
        
        self.page_manager.allocate(seq)
        
        # Initially should be 0
        self.assertEqual(self.page_manager.seq_lengths.get(seq.seq_id, 0), 0)
        
        # Update sequence length
        self.page_manager.update_sequence_lengths([seq], is_prefill=True)
        
        # Should now be length of tokens
        self.assertEqual(self.page_manager.seq_lengths[seq.seq_id], len(tokens))
    
    def test_indices_for_prefill(self):
        """Test building indices for prefill."""
        tokens = list(range(20))  # 20 tokens
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams(max_tokens=10))
        
        self.page_manager.allocate(seq)
        # Don't update sequence lengths before building indices for prefill
        # The indices builder expects to see the length BEFORE appending
        
        # Build indices
        indices, indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            [seq], for_prefill=True
        )
        
        # Check shapes and device
        self.assertEqual(indices.device.type, "cuda")
        self.assertEqual(indptr.device.type, "cuda") 
        self.assertEqual(last_page_lens.device.type, "cuda")
        
        # With 20 tokens and page_size=16, need 2 pages
        self.assertEqual(len(indices), 2)
        # For prefill with 20 tokens, last page has 4 tokens (20 % 16 = 4)
        self.assertEqual(last_page_lens[0].item(), 4)
    
    def test_indices_for_decode(self):
        """Test building indices for decode."""
        tokens = list(range(20))  # 20 tokens
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams(max_tokens=10))
        
        self.page_manager.allocate(seq)
        self.page_manager.update_sequence_lengths([seq], is_prefill=True)
        
        # Build indices for decode
        indices, indptr, last_page_lens = self.page_manager.build_indices_for_sequences(
            [seq], for_prefill=False
        )
        
        # For decode, we're adding 1 token to the existing 20
        # So we'll have 21 tokens total, which means 5 tokens in last page
        self.assertEqual(last_page_lens[0].item(), 5)
    
    def test_free_sequence(self):
        """Test freeing sequence resources."""
        tokens = [1, 2, 3, 4, 5]
        seq = Sequence(token_ids=tokens, sampling_params=SamplingParams(max_tokens=10))
        
        self.page_manager.allocate(seq)
        allocated_pages = list(self.page_manager.seq_page_tables[seq.seq_id])
        
        # Free the sequence
        self.page_manager.deallocate(seq)
        
        # Check cleanup
        self.assertNotIn(seq.seq_id, self.page_manager.seq_page_tables)
        self.assertNotIn(seq.seq_id, self.page_manager.seq_lengths)
        
        # Pages should be returned to free pool
        for page_id in allocated_pages:
            self.assertIn(page_id, self.page_manager.free_pages)


if __name__ == '__main__':
    unittest.main()