"""
Tests for custom chunk attention kernel with paged KV cache.
"""

import unittest
import torch
import math
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.kernels.paged_chunk_attention import PagedChunkAttention
from nanovllm.chunks import Chunk, ChunkType
from nanovllm.engine.page_manager import PageManager


class TestCustomChunkAttention(unittest.TestCase):
    """Test suite for custom chunk attention kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        self.num_heads = 8
        self.num_kv_heads = 2  # For GQA
        self.head_dim = 64
        self.num_layers = 4
        self.page_size = 16
        self.num_pages = 100
        
        # Create page manager
        self.page_manager = PageManager(
            num_pages=self.num_pages,
            page_size=self.page_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.float16
        )
        
        # Create chunk attention instance
        self.chunk_attention = PagedChunkAttention(
            head_dim=self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim)
        )
        
    def tearDown(self):
        """Clean up after tests."""
        pass
        
    def create_test_chunk(self, chunk_id: str, num_tokens: int, seed: int = 42) -> Chunk:
        """Create a test chunk with allocated pages and populated KV cache."""
        torch.manual_seed(seed)
        
        # Create chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            chunk_type=ChunkType.CONTEXT,
            content=f"Test chunk {chunk_id}",
            token_ids=list(range(num_tokens)),
            token_count=num_tokens,
            kv_length=num_tokens
        )
        
        # Allocate pages
        chunk.page_table = self.page_manager.allocate_for_chunk(chunk_id, num_tokens)
        
        # Populate KV cache with test data
        for layer_idx in range(self.num_layers):
            # Create random K and V tensors
            k = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
            v = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
            
            # Append to cache
            positions = torch.arange(num_tokens, device=self.device)
            self.page_manager.append_kv_to_cache_for_chunk(
                layer_idx, k, v, chunk_id, positions
            )
        
        return chunk
        
    def test_correctness_vs_pytorch(self):
        """Test that custom kernel produces same results as PyTorch reference."""
        # Create test chunks
        chunks = [
            self.create_test_chunk("system", 20, seed=1),
            self.create_test_chunk("context1", 30, seed=2),
            self.create_test_chunk("context2", 25, seed=3),
            self.create_test_chunk("query", 10, seed=4)
        ]
        
        # Create query tensor
        query = torch.randn(self.num_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
        
        # Test for each layer
        for layer_idx in range(self.num_layers):
            # Run custom kernel
            output_custom = self.chunk_attention.forward(
                query=query,
                chunks=chunks,
                kv_cache=self.page_manager.kv_cache,
                layer_idx=layer_idx,
                page_size=self.page_size
            )
            
            # Compute PyTorch reference
            # First, gather all K and V from chunks
            all_k = []
            all_v = []
            
            for chunk in chunks:
                if chunk.page_table is None or len(chunk.page_table) == 0:
                    continue
                    
                # Extract K and V from paged cache
                for token_idx in range(chunk.kv_length):
                    page_idx = token_idx // self.page_size
                    page_offset = token_idx % self.page_size
                    global_page_num = chunk.page_table[page_idx]
                    
                    # Extract K
                    k_token = self.page_manager.kv_cache[
                        layer_idx, global_page_num, 0, :, page_offset, :
                    ]  # [num_kv_heads, head_dim]
                    all_k.append(k_token)
                    
                    # Extract V
                    v_token = self.page_manager.kv_cache[
                        layer_idx, global_page_num, 1, :, page_offset, :
                    ]  # [num_kv_heads, head_dim]
                    all_v.append(v_token)
            
            # Stack K and V
            k_concat = torch.stack(all_k, dim=0)  # [seq_len, num_kv_heads, head_dim]
            v_concat = torch.stack(all_v, dim=0)  # [seq_len, num_kv_heads, head_dim]
            
            # Expand K and V for GQA
            num_kv_groups = self.num_heads // self.num_kv_heads
            k_expanded = k_concat.repeat_interleave(num_kv_groups, dim=1)  # [seq_len, num_heads, head_dim]
            v_expanded = v_concat.repeat_interleave(num_kv_groups, dim=1)  # [seq_len, num_heads, head_dim]
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(self.head_dim)
            # query shape: [num_heads, head_dim]
            # k_expanded shape: [seq_len, num_heads, head_dim]
            # Need to compute query @ k^T for each head
            # Reshape for batch matmul
            q_reshaped = query.unsqueeze(1)  # [num_heads, 1, head_dim]
            k_reshaped = k_expanded.transpose(0, 1)  # [num_heads, seq_len, head_dim]
            scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale  # [num_heads, 1, seq_len]
            scores = scores.squeeze(1)  # [num_heads, seq_len]
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_len]
            
            # Apply to values
            # v_expanded shape: [seq_len, num_heads, head_dim]
            v_reshaped = v_expanded.transpose(0, 1)  # [num_heads, seq_len, head_dim]
            attn_weights_expanded = attn_weights.unsqueeze(1)  # [num_heads, 1, seq_len]
            output_ref = torch.bmm(attn_weights_expanded, v_reshaped).squeeze(1)  # [num_heads, head_dim]
            
            # Compare outputs
            max_diff = torch.max(torch.abs(output_custom - output_ref)).item()
            mean_diff = torch.mean(torch.abs(output_custom - output_ref)).item()
            
            # More lenient for half precision
            self.assertLess(max_diff, 0.1, 
                           f"Layer {layer_idx}: Max difference {max_diff} exceeds threshold")
            self.assertLess(mean_diff, 0.01,
                           f"Layer {layer_idx}: Mean difference {mean_diff} exceeds threshold")
            
    def test_memory_efficiency(self):
        """Test that custom kernel doesn't allocate large temporary buffers."""
        # Create test chunks
        chunks = [
            self.create_test_chunk("system", 100, seed=1),
            self.create_test_chunk("context1", 200, seed=2),
            self.create_test_chunk("context2", 150, seed=3),
            self.create_test_chunk("query", 50, seed=4)
        ]
        
        # Create query tensor
        query = torch.randn(self.num_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
        
        # Measure baseline memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        
        # Run custom kernel
        output = self.chunk_attention.forward(
            query=query,
            chunks=chunks,
            kv_cache=self.page_manager.kv_cache,
            layer_idx=0,
            page_size=self.page_size
        )
        
        # Measure peak memory usage
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - baseline_memory
        
        # Calculate expected temporary memory
        # Should only need small buffers for metadata, not full KV concatenation
        total_tokens = sum(chunk.kv_length for chunk in chunks)
        expected_concat_size = total_tokens * self.num_kv_heads * self.head_dim * 2  # K and V
        expected_concat_memory = expected_concat_size * 2  # float16 = 2 bytes
        
        # Memory increase should be much less than full concatenation
        # Allow some overhead for metadata and kernel workspace
        max_allowed_memory = expected_concat_memory * 0.1  # 10% of concat size
        
        self.assertLess(memory_increase, max_allowed_memory,
                       f"Memory increase {memory_increase} exceeds allowed {max_allowed_memory}")
        
    def test_performance_benchmark(self):
        """Benchmark performance of custom chunk attention."""
        # Create test chunks with realistic sizes
        chunks = [
            self.create_test_chunk("system", 50, seed=1),    # System prompt
            self.create_test_chunk("context1", 200, seed=2),  # Context 1
            self.create_test_chunk("context2", 150, seed=3),  # Context 2  
            self.create_test_chunk("query", 20, seed=4)       # Query
        ]
        
        # Create query tensor
        query = torch.randn(self.num_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
        
        # Warmup
        for _ in range(10):
            output = self.chunk_attention.forward(
                query=query,
                chunks=chunks,
                kv_cache=self.page_manager.kv_cache,
                layer_idx=0,
                page_size=self.page_size
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        num_iters = 1000
        start.record()
        for _ in range(num_iters):
            output = self.chunk_attention.forward(
                query=query,
                chunks=chunks,
                kv_cache=self.page_manager.kv_cache,
                layer_idx=0,
                page_size=self.page_size
            )
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / num_iters
        
        # Calculate tokens/second
        tokens_per_second = 1000 / elapsed_ms
        
        print(f"\nPerformance Benchmark:")
        print(f"  Time per attention: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {tokens_per_second:.1f} tok/s")
        
        # Ensure reasonable performance (>100 tok/s minimum)
        self.assertGreater(tokens_per_second, 100,
                          f"Performance {tokens_per_second:.1f} tok/s is too low")
        
    def test_empty_chunks(self):
        """Test handling of empty chunks."""
        # Create mix of empty and non-empty chunks
        chunks = [
            self.create_test_chunk("system", 20, seed=1),
            Chunk(chunk_id="empty1", chunk_type=ChunkType.CONTEXT, content="", 
                  token_ids=[], token_count=0, kv_length=0),
            self.create_test_chunk("context", 30, seed=2),
            Chunk(chunk_id="empty2", chunk_type=ChunkType.CONTEXT, content="", 
                  token_ids=[], token_count=0, kv_length=0),
        ]
        
        # Create query tensor
        query = torch.randn(self.num_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
        
        # Should run without errors
        output = self.chunk_attention.forward(
            query=query,
            chunks=chunks,
            kv_cache=self.page_manager.kv_cache,
            layer_idx=0,
            page_size=self.page_size
        )
        
        # Output should be valid
        self.assertEqual(output.shape, (self.num_heads, self.head_dim))
        self.assertFalse(torch.isnan(output).any())
        
    def test_single_token_chunks(self):
        """Test handling of single-token chunks."""
        # Create single-token chunks
        chunks = [
            self.create_test_chunk("tok1", 1, seed=1),
            self.create_test_chunk("tok2", 1, seed=2),
            self.create_test_chunk("tok3", 1, seed=3),
        ]
        
        # Create query tensor
        query = torch.randn(self.num_heads, self.head_dim, 
                          dtype=torch.float16, device=self.device)
        
        # Should run without errors
        output = self.chunk_attention.forward(
            query=query,
            chunks=chunks,
            kv_cache=self.page_manager.kv_cache,
            layer_idx=0,
            page_size=self.page_size
        )
        
        # Output should be valid
        self.assertEqual(output.shape, (self.num_heads, self.head_dim))
        self.assertFalse(torch.isnan(output).any())


if __name__ == "__main__":
    unittest.main()