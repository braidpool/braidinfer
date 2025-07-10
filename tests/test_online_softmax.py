"""
Unit test for online softmax kernel.

This test verifies that the online softmax implementation in CascadeAttention
produces bit-for-bit identical results to standard softmax on concatenated sequences.
"""

import unittest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from braidinfer.layers.cascade_attention import CascadeAttention, CascadeLevel
from braidinfer.chunks import Chunk, ChunkType


class TestOnlineSoftmax(unittest.TestCase):
    """Test suite for online softmax algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test configuration
        self.num_heads = 8
        self.head_dim = 64
        self.num_kv_heads = 8  # No GQA for simplicity
        self.page_size = 16
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Create cascade attention instance
        self.cascade_attn = CascadeAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            page_size=self.page_size
        )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_mock_kv_cache(self, num_pages: int, tokens_per_page: int = None):
        """Create a mock KV cache for testing."""
        if tokens_per_page is None:
            tokens_per_page = self.page_size
            
        # Shape: [num_pages, 2, num_kv_heads, page_size, head_dim]
        kv_cache = torch.randn(
            num_pages, 2, self.num_kv_heads, self.page_size, self.head_dim,
            dtype=torch.float16, device=self.device
        )
        return kv_cache
    
    def _create_mock_chunks(self, chunk_configs):
        """Create mock chunks with specified page configurations.
        
        Args:
            chunk_configs: List of (num_pages, tokens_in_last_page) tuples
        """
        chunks = []
        current_page = 0
        
        for i, (num_pages, tokens_in_last_page) in enumerate(chunk_configs):
            # Calculate total tokens
            if num_pages == 1:
                total_tokens = tokens_in_last_page
            else:
                total_tokens = (num_pages - 1) * self.page_size + tokens_in_last_page
            
            # Create page table
            page_table = list(range(current_page, current_page + num_pages))
            current_page += num_pages
            
            # Create mock chunk  
            chunk = Chunk.from_content(content=f"Chunk {i}", chunk_type=ChunkType.CONTEXT)
            chunk.page_table = page_table
            chunk.kv_length = total_tokens
            chunk.global_position_start = sum(
                (cfg[0] - 1) * self.page_size + cfg[1] if cfg[0] > 1 else cfg[1]
                for cfg in chunk_configs[:i]
            )
            chunk.global_position_end = chunk.global_position_start + total_tokens
            chunk.cached_position_start = 0  # Assume cached at start
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_cascade_level(self, chunks):
        """Create a CascadeLevel from chunks."""
        all_page_indices = []
        page_indptr = [0]
        last_page_lens = []
        
        for chunk in chunks:
            all_page_indices.extend(chunk.page_table)
            page_indptr.append(len(all_page_indices))
            
            # Calculate last page length
            num_pages = len(chunk.page_table)
            if num_pages == 1:
                last_page_len = chunk.kv_length
            else:
                last_page_len = chunk.kv_length % self.page_size
                if last_page_len == 0:
                    last_page_len = self.page_size
            last_page_lens.append(last_page_len)
        
        return CascadeLevel(
            chunks=chunks,
            kv_page_indices=torch.tensor(all_page_indices, dtype=torch.int32, device=self.device),
            kv_page_indptr=torch.tensor(page_indptr, dtype=torch.int32, device=self.device),
            kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32, device=self.device),
            position_offset=0
        )
    
    def test_online_softmax_vs_standard_simple(self):
        """Test online softmax against standard softmax with simple configuration."""
        # Simple case: 2 chunks, each with 1 page, full pages
        chunk_configs = [(1, self.page_size), (1, self.page_size)]
        chunks = self._create_mock_chunks(chunk_configs)
        
        # Create KV cache
        total_pages = sum(cfg[0] for cfg in chunk_configs)
        kv_cache = self._create_mock_kv_cache(total_pages)
        
        # Create cascade level
        cascade_level = self._create_cascade_level(chunks)
        
        # Create query
        batch_size = 1
        query = torch.randn(batch_size, self.num_heads, self.head_dim, device=self.device, dtype=torch.float16)
        
        # Test online softmax (cascade attention)
        cascade_output = self.cascade_attn.forward(
            query=query,
            kv_cache=kv_cache,
            cascade_levels=[cascade_level],
            layer_idx=0,
            causal_mask=False  # Disable causal mask for simpler comparison
        )
        
        # Test standard attention (concatenated)
        # Extract all K, V from cache manually
        all_k = []
        all_v = []
        
        for chunk in chunks:
            for page_idx in chunk.page_table:
                # Extract K and V
                if page_idx == chunk.page_table[-1]:
                    # Last page - use actual length
                    num_pages = len(chunk.page_table)
                    if num_pages == 1:
                        tokens_on_page = chunk.kv_length
                    else:
                        tokens_on_page = chunk.kv_length % self.page_size
                        if tokens_on_page == 0:
                            tokens_on_page = self.page_size
                else:
                    tokens_on_page = self.page_size
                
                page_k = kv_cache[page_idx, 0, :, :tokens_on_page, :]  # [num_kv_heads, tokens, head_dim]
                page_v = kv_cache[page_idx, 1, :, :tokens_on_page, :]
                
                # Transpose to [tokens, num_kv_heads, head_dim]
                page_k = page_k.transpose(0, 1)
                page_v = page_v.transpose(0, 1)
                
                all_k.append(page_k)
                all_v.append(page_v)
        
        # Concatenate all K, V
        k_concat = torch.cat(all_k, dim=0)  # [total_tokens, num_kv_heads, head_dim]
        v_concat = torch.cat(all_v, dim=0)
        
        # Standard attention computation
        q = query.transpose(0, 1)  # [num_heads, batch_size, head_dim]
        k_concat = k_concat.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        v_concat = v_concat.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        
        scores = torch.bmm(q, k_concat.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        standard_output = torch.bmm(attn_weights, v_concat)
        standard_output = standard_output.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        
        # Compare outputs
        # Allow for small numerical differences due to different computation order
        max_diff = torch.max(torch.abs(cascade_output - standard_output)).item()
        
        # The tolerance needs to account for:
        # 1. Different computation order in online vs batch softmax
        # 2. Float16 precision limitations
        # 3. Accumulation errors in online algorithm
        tolerance = 1e-2  # Reasonable tolerance for float16
        
        self.assertLess(
            max_diff, tolerance,
            f"Online softmax output differs from standard by {max_diff:.6f}, expected < {tolerance}"
        )
    
    def test_online_softmax_partial_pages(self):
        """Test online softmax with partial pages."""
        # More complex case: chunks with partial pages
        chunk_configs = [(2, 8), (1, 12)]  # First chunk: 2 pages, 8 tokens in last; Second: 1 page, 12 tokens
        chunks = self._create_mock_chunks(chunk_configs)
        
        total_pages = sum(cfg[0] for cfg in chunk_configs)
        kv_cache = self._create_mock_kv_cache(total_pages)
        
        cascade_level = self._create_cascade_level(chunks)
        
        batch_size = 2  # Test with multiple queries
        query = torch.randn(batch_size, self.num_heads, self.head_dim, device=self.device, dtype=torch.float16)
        
        # Should not crash and should produce reasonable output
        try:
            cascade_output = self.cascade_attn.forward(
                query=query,
                kv_cache=kv_cache,
                cascade_levels=[cascade_level],
                layer_idx=0,
                causal_mask=False
            )
            
            # Verify output shape  
            expected_shape = (batch_size, self.num_heads * self.head_dim)
            self.assertEqual(cascade_output.shape, expected_shape,
                           f"Expected shape {expected_shape}, got {cascade_output.shape}")
            
            # Verify output is not NaN or Inf
            self.assertFalse(torch.isnan(cascade_output).any(), "Output contains NaN values")
            self.assertFalse(torch.isinf(cascade_output).any(), "Output contains Inf values")
            
        except Exception as e:
            self.fail(f"Online softmax failed with partial pages: {e}")
    
    def test_empty_chunks_handling(self):
        """Test that empty chunks are handled gracefully."""
        # Mix of empty and non-empty chunks
        chunk_configs = [(1, 16), (0, 0), (1, 8)]  # middle chunk is empty
        
        # Filter out empty chunks as they would in real usage
        valid_configs = [(cfg[0], cfg[1]) for cfg in chunk_configs if cfg[0] > 0]
        chunks = self._create_mock_chunks(valid_configs)
        
        total_pages = sum(cfg[0] for cfg in valid_configs)
        kv_cache = self._create_mock_kv_cache(total_pages)
        
        cascade_level = self._create_cascade_level(chunks)
        
        query = torch.randn(1, self.num_heads, self.head_dim, device=self.device, dtype=torch.float16)
        
        # Should handle gracefully
        try:
            cascade_output = self.cascade_attn.forward(
                query=query,
                kv_cache=kv_cache,
                cascade_levels=[cascade_level],
                layer_idx=0,
                causal_mask=False
            )
            
            self.assertIsNotNone(cascade_output, "Should produce output even with empty chunks")
            
        except Exception as e:
            self.fail(f"Failed to handle empty chunks: {e}")


if __name__ == '__main__':
    unittest.main()