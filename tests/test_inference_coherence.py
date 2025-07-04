#!/usr/bin/env python3
"""Test that model produces coherent output, not gibberish."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from nanovllm import ChunkedLLM, ChunkType, SamplingParams


class TestInferenceCoherence(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm = ChunkedLLM(self.model_path, max_model_len=2048, enable_cascade=True)
        
    def test_simple_math_question(self):
        """Test that model can answer simple math questions coherently."""
        system_id = self.llm.register_chunk(
            "You are a helpful assistant.",
            chunk_type=ChunkType.SYSTEM_PROMPT
        )
        
        query_id = self.llm.register_chunk(
            "What is 2+2?",
            chunk_type=ChunkType.QUERY
        )
        
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.6
        )
        
        result = self.llm.generate_from_chunks(
            system_chunk_id=system_id,
            query_chunk_id=query_id,
            sampling_params=sampling_params.__dict__
        )
        
        output = result['text']
        
        # Check that output is not gibberish
        # Gibberish would have repeated nonsense words
        words = output.split()
        unique_words = set(words)
        
        # Should have reasonable word diversity
        self.assertGreater(len(unique_words), 5, 
                          "Output should have more than 5 unique words")
        
        # Should not have excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        self.assertLess(max_repetition, len(words) * 0.3,
                       "No word should repeat more than 30% of the time")
        
        # Should mention "4" or "four" somewhere
        output_lower = output.lower()
        self.assertTrue("4" in output or "four" in output_lower,
                       "Answer should contain '4' or 'four'")
        
    def test_greeting_response(self):
        """Test that model can respond to greetings coherently."""
        system_id = self.llm.register_chunk(
            "You are a helpful assistant.",
            chunk_type=ChunkType.SYSTEM_PROMPT
        )
        
        query_id = self.llm.register_chunk(
            "Hello, how are you?",
            chunk_type=ChunkType.QUERY
        )
        
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.6
        )
        
        result = self.llm.generate_from_chunks(
            system_chunk_id=system_id,
            query_chunk_id=query_id,
            sampling_params=sampling_params.__dict__
        )
        
        output = result['text']
        
        # Should not be repetitive gibberish
        self.assertNotIn("Question Instructions", output,
                        "Should not contain gibberish pattern 'Question Instructions'")
        self.assertNotIn("licants", output,
                        "Should not contain gibberish word 'licants'")
        
        # Should be a reasonable length
        self.assertGreater(len(output.split()), 3,
                          "Response should have more than 3 words")
        
    def test_no_think_directive(self):
        """Test that /no_think directive works."""
        system_id = self.llm.register_chunk(
            "You are a helpful assistant.",
            chunk_type=ChunkType.SYSTEM_PROMPT
        )
        
        query_id = self.llm.register_chunk(
            "/no_think\nWhat is the capital of France?",
            chunk_type=ChunkType.QUERY
        )
        
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.6
        )
        
        result = self.llm.generate_from_chunks(
            system_chunk_id=system_id,
            query_chunk_id=query_id,
            sampling_params=sampling_params.__dict__
        )
        
        output = result['text']
        
        # Should mention Paris
        self.assertIn("Paris", output,
                     "Answer should contain 'Paris'")
        
        # Should not have excessive think blocks
        think_count = output.count("<think>")
        self.assertLessEqual(think_count, 1,
                           "Should have at most one <think> block with /no_think")


if __name__ == '__main__':
    unittest.main()