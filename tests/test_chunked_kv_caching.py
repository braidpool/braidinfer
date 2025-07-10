#!/usr/bin/env python3
"""
Comprehensive tests for the chunked KV cache management system.

This test suite verifies that the ChunkedLLM truly reuses pre-computed
KV caches instead of just concatenating text.
"""

import unittest
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from braidinfer import LLM, ChunkedLLM, SamplingParams, ChunkType


class TestChunkedKVCaching(unittest.TestCase):
    """Test suite for verifying KV cache reuse in ChunkedLLM."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
        
        # Check if model exists
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model not found at {cls.model_path}")
        
        # Initialize standard LLM for comparison
        cls.standard_llm = LLM(
            cls.model_path,
            max_num_seqs=1,
            enforce_eager=True
        )
        
        # Initialize ChunkedLLM with KV cache management
        cls.chunked_llm = ChunkedLLM(
            cls.model_path,
            max_chunks=100,
            chunk_memory_ratio=0.5,
            enable_deduplication=True,
            enforce_eager=True
        )
        
        # Test content
        cls.system_prompt = "You are a helpful AI assistant."
        cls.context = "The capital of France is Paris. The Eiffel Tower is located in Paris."
        cls.query = "What famous landmark is in the capital of France?"
        
        # Sampling params for deterministic output
        cls.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding
            max_tokens=50
        )
    
    def test_correctness_standard_vs_chunked(self):
        """Test that chunked generation produces same output as standard."""
        # Standard generation
        full_prompt = f"System: {self.system_prompt}\n\nContext: {self.context}\n\nQuestion: {self.query}\n\nAnswer:"
        
        standard_output = self.standard_llm.generate(
            [full_prompt],
            self.sampling_params
        )[0]
        
        # Chunked generation
        system_chunk_id = self.chunked_llm.register_chunk(
            self.system_prompt,
            ChunkType.SYSTEM_PROMPT
        )
        
        context_chunk_id = self.chunked_llm.register_chunk(
            self.context,
            ChunkType.CONTEXT
        )
        
        query_chunk_id = self.chunked_llm.register_chunk(
            self.query,
            ChunkType.QUERY
        )
        
        chunked_output = self.chunked_llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            query_chunk_id=query_chunk_id,
            context_chunk_ids=[context_chunk_id],
            sampling_params=self.sampling_params
        )
        
        # Compare outputs
        self.assertEqual(
            standard_output["text"].strip(),
            chunked_output["text"].strip(),
            "Chunked output should match standard output"
        )
    
    def test_performance_speedup_cached_chunks(self):
        """Test that subsequent calls with cached chunks are significantly faster."""
        # Register chunks (this will prefill them)
        system_chunk_id = self.chunked_llm.register_chunk(
            self.system_prompt,
            ChunkType.SYSTEM_PROMPT
        )
        
        context_chunk_id = self.chunked_llm.register_chunk(
            self.context,
            ChunkType.CONTEXT
        )
        
        # First query - includes prefill time
        query1 = "What is the capital of France?"
        query1_chunk_id = self.chunked_llm.register_chunk(
            query1,
            ChunkType.QUERY
        )
        
        start_time = time.time()
        output1 = self.chunked_llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            query_chunk_id=query1_chunk_id,
            context_chunk_ids=[context_chunk_id],
            sampling_params=self.sampling_params
        )
        first_call_time = time.time() - start_time
        
        # Second query - should reuse system and context chunks
        query2 = "Tell me about the Eiffel Tower."
        query2_chunk_id = self.chunked_llm.register_chunk(
            query2,
            ChunkType.QUERY
        )
        
        start_time = time.time()
        output2 = self.chunked_llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            query_chunk_id=query2_chunk_id,
            context_chunk_ids=[context_chunk_id],
            sampling_params=self.sampling_params
        )
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster (at least 5x)
        speedup = first_call_time / second_call_time
        self.assertGreater(
            speedup,
            5.0,
            f"Expected >5x speedup for cached chunks, got {speedup:.1f}x"
        )
        
        print(f"\nPerformance test results:")
        print(f"First call (with prefill): {first_call_time:.3f}s")
        print(f"Second call (cached): {second_call_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
    
    def test_memory_management(self):
        """Test that chunk deletion properly frees GPU memory."""
        # Get initial memory usage
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        
        # Register a large chunk
        large_content = "This is a test. " * 1000  # ~4000 tokens
        chunk_id = self.chunked_llm.register_chunk(
            large_content,
            ChunkType.CONTEXT
        )
        
        # Memory should increase after registration (KV cache allocated)
        torch.cuda.synchronize()
        after_register_memory = torch.cuda.memory_allocated()
        memory_increase = after_register_memory - initial_memory
        
        self.assertGreater(
            memory_increase,
            1_000_000,  # At least 1MB for KV cache
            "Memory should increase after chunk registration"
        )
        
        # Delete the chunk
        self.chunked_llm.delete_chunk(chunk_id)
        
        # Memory should decrease after deletion
        torch.cuda.synchronize()
        after_delete_memory = torch.cuda.memory_allocated()
        
        # Allow some tolerance for fragmentation
        self.assertLess(
            after_delete_memory - initial_memory,
            memory_increase * 0.1,  # Should free at least 90% of allocated memory
            "Memory should be freed after chunk deletion"
        )
        
        print(f"\nMemory management test results:")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"After registration: {after_register_memory / 1024 / 1024:.1f} MB")
        print(f"After deletion: {after_delete_memory / 1024 / 1024:.1f} MB")
        print(f"Memory freed: {(after_register_memory - after_delete_memory) / 1024 / 1024:.1f} MB")
    
    def test_chunk_kv_cache_allocated(self):
        """Test that chunks have KV cache allocated after registration."""
        # Register a chunk
        chunk_id = self.chunked_llm.register_chunk(
            "Test content for KV cache allocation.",
            ChunkType.CONTEXT
        )
        
        # Get chunk info
        chunk_info = self.chunked_llm.get_chunk(chunk_id)
        
        # Verify KV cache is allocated
        self.assertTrue(
            chunk_info["kv_cache_allocated"],
            "Chunk should have KV cache allocated after registration"
        )
        
        # Verify page table exists
        chunk = self.chunked_llm.registry.get(chunk_id)
        self.assertIsNotNone(
            chunk.page_table,
            "Chunk should have page table assigned"
        )
        self.assertGreater(
            len(chunk.page_table),
            0,
            "Page table should not be empty"
        )
    
    def test_no_string_concatenation(self):
        """Verify that generate_from_chunks doesn't build string prompts."""
        # This test is more of a code inspection test, but we can verify
        # that the _build_prompt method is not called during generation
        
        # Register chunks
        system_id = self.chunked_llm.register_chunk(
            "System prompt",
            ChunkType.SYSTEM_PROMPT
        )
        query_id = self.chunked_llm.register_chunk(
            "Query",
            ChunkType.QUERY
        )
        
        # Monkey patch _build_prompt to detect if it's called
        original_build_prompt = self.chunked_llm._build_prompt
        build_prompt_called = False
        
        def mock_build_prompt(*args, **kwargs):
            nonlocal build_prompt_called
            build_prompt_called = True
            return original_build_prompt(*args, **kwargs)
        
        self.chunked_llm._build_prompt = mock_build_prompt
        
        try:
            # Generate (should NOT call _build_prompt)
            output = self.chunked_llm.generate_from_chunks(
                system_chunk_id=system_id,
                query_chunk_id=query_id,
                sampling_params=self.sampling_params
            )
            
            self.assertFalse(
                build_prompt_called,
                "_build_prompt should not be called in true KV cache implementation"
            )
        finally:
            # Restore original method
            self.chunked_llm._build_prompt = original_build_prompt


if __name__ == "__main__":
    unittest.main(verbosity=2)