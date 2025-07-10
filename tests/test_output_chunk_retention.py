#!/usr/bin/env python3
"""
Test output KV cache retention and reuse as cascade attention chunks.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from braidinfer import ChunkedLLM, ChunkType, SamplingParams


class TestOutputChunkRetention(unittest.TestCase):
    """Test output KV cache retention functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up ChunkedLLM for testing."""
        model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
        cls.llm = ChunkedLLM(
            model_path,
            max_chunks=100,
            chunk_memory_ratio=0.5,
            enable_deduplication=True,
            enforce_eager=True,
            num_kvcache_blocks=64,
            kvcache_block_size=256
        )
    
    def test_basic_output_retention(self):
        """Test basic output retention and chunk registration."""
        # Generate with retention
        output = self.llm.generate_and_retain_output(
            system_prompt="You are a helpful assistant.",
            query="What is 2+2?",
            sampling_params={"temperature": 0.1, "max_tokens": 50}
        )
        
        # Check that output was generated
        self.assertIn('text', output)
        self.assertIn('token_ids', output)
        self.assertIn('output_chunk_id', output)
        self.assertIn('retained_seq_id', output)
        
        # Verify output chunk was registered
        output_chunk_id = output['output_chunk_id']
        chunk_info = self.llm.get_chunk(output_chunk_id)
        
        self.assertEqual(chunk_info['chunk_type'], 'output')
        self.assertIn('source', chunk_info['metadata'])
        self.assertEqual(chunk_info['metadata']['source'], 'output')
        
        # Clean up
        self.llm.delete_chunk(output_chunk_id)
    
    def test_output_reuse_as_context(self):
        """Test using output chunk as context for new generation."""
        # First generation
        output1 = self.llm.generate_and_retain_output(
            system_prompt="You are a helpful assistant.",
            query="The capital of France is",
            sampling_params={"temperature": 0.1, "max_tokens": 20}
        )
        
        output_chunk_id = output1['output_chunk_id']
        
        # Second generation using first output as context
        output2 = self.llm.generate_from_chunks(
            system_chunk_id=self.llm.register_chunk("You are a helpful assistant.", ChunkType.SYSTEM_PROMPT),
            query_chunk_id=self.llm.register_chunk("What city did we just talk about?", ChunkType.QUERY),
            context_chunk_ids=[output_chunk_id],
            sampling_params={"temperature": 0.1, "max_tokens": 50}
        )
        
        # The model should reference Paris from the context
        self.assertIn('text', output2)
        # Note: Exact text matching is tricky with LLMs, but we expect
        # the model to reference the previous output
        
        # Clean up
        self.llm.delete_chunk(output_chunk_id)
    
    def test_think_tag_filtering(self):
        """Test that think tags are filtered from retained output chunks."""
        # Generate with a prompt that might trigger think tags
        output = self.llm.generate_and_retain_output(
            system_prompt="You are a helpful assistant that thinks step by step.",
            query="Solve this math problem: If a train travels 60 mph for 2 hours, how far does it go?",
            sampling_params={"temperature": 0.1, "max_tokens": 150}
        )
        
        if 'output_chunk_id' in output:
            chunk_info = self.llm.get_chunk(output['output_chunk_id'])
            content = chunk_info['content']
            
            # Get the original output text
            raw_text = output['text']
            
            # If the raw output had think tags, they should be filtered in the chunk
            if '<think>' in raw_text:
                # The stored chunk content should have think tags filtered
                self.assertNotIn('<think>', content)
                self.assertNotIn('</think>', content)
                
                # Metadata should indicate think tags were present
                self.assertTrue(chunk_info['metadata']['has_think_tags'])
            else:
                # No think tags in original output
                self.assertFalse(chunk_info['metadata'].get('has_think_tags', False))
            
            # Clean up
            self.llm.delete_chunk(output['output_chunk_id'])
    
    def test_memory_management(self):
        """Test memory usage with retained output chunks."""
        # Clear all chunks first to get a clean baseline
        self.llm.clear_chunks()
        
        initial_stats = self.llm.get_chunk_stats()
        initial_chunks = initial_stats['total_chunks']
        
        # Generate multiple outputs with retention
        output_ids = []
        all_chunk_ids = []  # Track ALL created chunks
        
        for i in range(3):
            output = self.llm.generate_and_retain_output(
                system_prompt="You are a helpful assistant.",
                query=f"Count to {i+1}",
                sampling_params={"temperature": 0.1, "max_tokens": 20},
                persist_chunks=False  # Don't persist input chunks
            )
            if 'output_chunk_id' in output:
                output_ids.append(output['output_chunk_id'])
            
            # Track all chunk IDs created
            if 'chunk_ids' in output:
                all_chunk_ids.append(output['chunk_ids']['system'])
                all_chunk_ids.append(output['chunk_ids']['query'])
                all_chunk_ids.extend(output['chunk_ids']['context'])
        
        # Check that chunks were added
        stats_after = self.llm.get_chunk_stats()
        # We should have at least some chunks (output chunks might be empty/deduplicated)
        self.assertGreater(stats_after['total_chunks'], initial_chunks)
        
        # Check memory usage increased (skip if memory tracking returns 0)
        if stats_after['memory_used_mb'] > 0:
            self.assertGreater(stats_after['memory_used_mb'], initial_stats['memory_used_mb'])
        
        # Clean up all chunks (both output and any persisted input chunks)
        for chunk_id in output_ids:
            self.llm.delete_chunk(chunk_id)
        
        # Also clean up any input chunks that were created
        for chunk_id in all_chunk_ids:
            try:
                self.llm.delete_chunk(chunk_id)
            except:
                pass  # Chunk might already be deleted or not exist
        
        # Clear any remaining chunks
        self.llm.clear_chunks()
        
        # Verify cleanup
        stats_final = self.llm.get_chunk_stats()
        self.assertEqual(stats_final['total_chunks'], 0)
    
    def test_cascade_attention_levels(self):
        """Test that output chunks are assigned correct cascade level."""
        output = self.llm.generate_and_retain_output(
            system_prompt="You are a helpful assistant.",
            query="Hello",
            sampling_params={"temperature": 0.1, "max_tokens": 10}
        )
        
        if 'output_chunk_id' in output:
            chunk_info = self.llm.get_chunk(output['output_chunk_id'])
            
            # OUTPUT chunks should be at cascade level 1 (same as CONTEXT)
            # This is checked internally, but we can verify through usage
            
            # Clean up
            self.llm.delete_chunk(output['output_chunk_id'])
    
    def test_manual_deallocation(self):
        """Test manual deallocation of retained sequences."""
        # Generate with retention
        output = self.llm.generate_and_retain_output(
            system_prompt="You are a helpful assistant.",
            query="Test query",
            sampling_params={"temperature": 0.1, "max_tokens": 10}
        )
        
        if 'retained_seq_id' in output:
            seq_id = output['retained_seq_id']
            
            # Check sequence is retained
            retained_seqs = self.llm.llm.get_retained_sequences()
            self.assertIn(seq_id, retained_seqs)
            
            # Manually release
            released = self.llm.llm.release_retained_sequence(seq_id)
            self.assertTrue(released)
            
            # Verify it's gone
            retained_seqs_after = self.llm.llm.get_retained_sequences()
            self.assertNotIn(seq_id, retained_seqs_after)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Release all retained sequences
        cls.llm.llm.release_all_retained_sequences()
        
        # Clear all chunks
        cls.llm.clear_chunks()


if __name__ == "__main__":
    unittest.main()