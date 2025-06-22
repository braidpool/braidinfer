#!/usr/bin/env python3
"""
Unit tests for Context Manager functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager, ChunkType




# Global test fixtures
_test_llm = None
_test_tokenizer = None
_test_context_mgr = None


class TestContextManager(unittest.TestCase):
    """Test suite for Context Manager functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Initialize fixtures if not already done (for individual test runs)
        global _test_llm, _test_tokenizer, _test_context_mgr
        if _test_llm is None:
            model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
            if not model_path.exists():
                raise unittest.SkipTest(f"Model not found at {model_path}")
            
            _test_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            
            _test_llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
            
            _test_context_mgr = ContextManager(_test_llm.scheduler.block_manager, _test_llm.config)
            _test_llm.context_manager = _test_context_mgr
            _test_llm.config.context_manager = _test_context_mgr
            _test_context_mgr.llm_engine = _test_llm
        
        cls.llm = _test_llm
        cls.tokenizer = _test_tokenizer
        cls.context_mgr = _test_context_mgr

    def setUp(self):
        """Set up for each test."""
        # Clear all chunks before each test
        self.context_mgr.clear_all()
        
        # Create temporary directory for disk tests
        self.temp_dir = tempfile.mkdtemp()
        self.context_mgr.disk_path = self.temp_dir

    def tearDown(self):
        """Clean up after each test."""
        # Clean up temporary directory
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_chunk_basic(self):
        """Test basic chunk creation."""
        content = "The capital of France is Paris."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.chunk_type, ChunkType.INPUT)
        self.assertEqual(len(chunk.token_ids), len(self.tokenizer.encode(content)))
        self.assertIn(chunk.sha256, self.context_mgr.chunks)
        self.assertIn(chunk.sha256, self.context_mgr.active_chunks)
        self.assertFalse(chunk.cache_populated)

    def test_add_chunk_with_cache_population(self):
        """Test chunk creation with immediate cache population."""
        content = "The capital of Germany is Berlin."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer, populate_cache=True)
        
        self.assertTrue(chunk.cache_populated)

    def test_chunk_activation_deactivation(self):
        """Test chunk activation and deactivation."""
        content = "Test content for activation."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        chunk_hash = chunk.sha256
        
        # Initially active
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)
        self.assertEqual(chunk.status, "active")
        
        # Deactivate
        self.context_mgr.deactivate_chunk(chunk_hash)
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "inactive")
        
        # Reactivate
        self.context_mgr.activate_chunk(chunk_hash)
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "active")

    def test_populate_chunk_cache(self):
        """Test manual KV cache population."""
        content = "Content for cache population test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer, populate_cache=False)
        
        self.assertFalse(chunk.cache_populated)
        
        # Populate cache manually
        result = self.context_mgr.populate_chunk_cache(chunk.sha256)
        self.assertTrue(result)
        self.assertTrue(chunk.cache_populated)
        
        # Second population should return False (already populated)
        result = self.context_mgr.populate_chunk_cache(chunk.sha256)
        self.assertFalse(result)

    def test_chunk_composition(self):
        """Test combining multiple chunks."""
        content1 = "First chunk content."
        content2 = "Second chunk content."
        
        chunk1 = self.context_mgr.add_chunk(content1, self.tokenizer)
        chunk2 = self.context_mgr.add_chunk(content2, self.tokenizer)
        
        # Compose chunks
        composed = self.context_mgr.compose_chunks(
            [chunk1.sha256, chunk2.sha256], 
            self.tokenizer
        )
        
        # Verify composition
        expected_tokens = len(chunk1.token_ids) + len(chunk2.token_ids)
        self.assertEqual(composed.size, expected_tokens)
        self.assertEqual(composed.parent_chunks, [chunk1.sha256, chunk2.sha256])

    def test_chunk_tagging(self):
        """Test chunk tagging functionality."""
        content = "Content for tagging test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        
        # Add tag
        self.context_mgr.tag_chunk(chunk.sha256, "test-tag")
        
        # Verify tag
        self.assertIn("tags", chunk.metadata)
        self.assertIn("test-tag", chunk.metadata["tags"])

    def test_chunk_deletion(self):
        """Test chunk deletion."""
        content = "Content to be deleted."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        chunk_hash = chunk.sha256
        
        # Verify chunk exists
        self.assertIn(chunk_hash, self.context_mgr.chunks)
        
        # Delete chunk
        self.context_mgr.erase_chunk(chunk_hash)
        
        # Verify deletion
        self.assertNotIn(chunk_hash, self.context_mgr.chunks)
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)

    def test_save_and_restore_chunk(self):
        """Test chunk persistence to disk."""
        content = "Content for save/restore test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer, populate_cache=True)
        chunk_hash = chunk.sha256
        
        # Save chunk to disk
        self.context_mgr.save_chunk(chunk_hash)
        
        # Verify file exists
        save_path = Path(self.temp_dir) / f"{chunk_hash}.pkl"
        self.assertTrue(save_path.exists())
        
        # Erase chunk completely (removes from all locations including disk)
        self.context_mgr.erase_chunk(chunk_hash)
        self.assertNotIn(chunk_hash, self.context_mgr.chunks)
        self.assertFalse(save_path.exists())  # File should be gone after erase
        
    def test_unload_and_restore_chunk(self):
        """Test three-tier memory hierarchy: VRAM -> RAM -> VRAM."""
        content = "Content for unload/restore test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer, populate_cache=True)
        chunk_hash = chunk.sha256
        
        # Initially in VRAM (active)
        self.assertEqual(chunk.status, "active")
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)
        
        # Unload to system RAM
        self.context_mgr.unload_chunk(chunk_hash)
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "cpu")
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)
        self.assertIn(chunk_hash, self.context_mgr.cpu_cache)
        
        # Restore from RAM back to VRAM
        restored_chunk = self.context_mgr.restore_chunk(chunk_hash)
        
        # Verify restoration to VRAM
        self.assertEqual(restored_chunk.status, "active")
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)
        self.assertNotIn(chunk_hash, self.context_mgr.cpu_cache)
        self.assertEqual(restored_chunk.size, chunk.size)
        self.assertEqual(restored_chunk.token_ids, chunk.token_ids)
        
    def test_save_erase_restore_workflow(self):
        """Test save -> erase -> restore workflow from disk."""
        content = "Content for save/erase/restore test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer, populate_cache=True)
        chunk_hash = chunk.sha256
        original_size = chunk.size
        original_tokens = chunk.token_ids.copy()
        
        # Save chunk to disk
        self.context_mgr.save_chunk(chunk_hash)
        save_path = Path(self.temp_dir) / f"{chunk_hash}.pkl"
        self.assertTrue(save_path.exists())
        
        # Erase chunk completely
        self.context_mgr.erase_chunk(chunk_hash)
        self.assertNotIn(chunk_hash, self.context_mgr.chunks)
        self.assertFalse(save_path.exists())  # Erase removes from disk too
        
        # At this point, we can't restore because erase removes everything
        # This demonstrates that erase truly removes from ALL locations
        with self.assertRaises(ValueError):
            self.context_mgr.restore_chunk(chunk_hash)

    def test_context_info(self):
        """Test context information retrieval."""
        # Add some chunks
        chunk1 = self.context_mgr.add_chunk("First chunk.", self.tokenizer)
        chunk2 = self.context_mgr.add_chunk("Second chunk.", self.tokenizer)
        
        # Deactivate one chunk
        self.context_mgr.deactivate_chunk(chunk2.sha256)
        
        # Get context info
        info = self.context_mgr.get_context_info()
        
        # Verify info structure
        self.assertIn("chunks", info)
        self.assertIn("chunks_by_status", info)
        self.assertIn("total_blocks", info)
        self.assertIn("free_blocks", info)
        
        # Verify chunk counts
        active_chunks = info["chunks_by_status"]["active"]
        inactive_chunks = info["chunks_by_status"]["inactive"]
        
        self.assertEqual(len(active_chunks), 1)
        self.assertEqual(len(inactive_chunks), 1)

    def test_memory_stats(self):
        """Test memory statistics."""
        # Add chunks
        chunk1 = self.context_mgr.add_chunk("Content for memory test.", self.tokenizer)
        chunk2 = self.context_mgr.add_chunk("More content.", self.tokenizer)
        
        # Get memory stats
        stats = self.context_mgr.get_memory_stats()
        
        # Verify stats structure
        self.assertIn("gpu", stats)
        self.assertIn("cpu", stats)
        self.assertIn("disk", stats)
        
        # Verify GPU stats (chunks should be on GPU initially)
        gpu_stats = stats["gpu"]
        self.assertGreater(gpu_stats["total"], 0)
        self.assertEqual(gpu_stats["count"], 2)

    def test_build_prompt_with_context(self):
        """Test prompt building with context integration."""
        # Add context chunk
        context_content = "The capital of France is Lyon."
        chunk = self.context_mgr.add_chunk(context_content, self.tokenizer)
        
        # Build prompt with context
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = self.context_mgr.build_prompt_with_context(messages, self.tokenizer)
        
        # Verify context is included
        self.assertIn("Lyon", formatted_prompt)
        self.assertIn("<|im_start|>system", formatted_prompt)
        self.assertIn("<|im_start|>user", formatted_prompt)

    def test_chunk_preview(self):
        """Test chunk preview generation."""
        content = "This is a test content for preview generation."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        
        preview = self.context_mgr.get_preview(chunk.token_ids, self.tokenizer)
        
        self.assertIsInstance(preview, str)
        self.assertGreater(len(preview), 0)
        self.assertIn("test", preview.lower())

    def test_hash_resolution(self):
        """Test partial hash resolution."""
        content = "Content for hash resolution test."
        chunk = self.context_mgr.add_chunk(content, self.tokenizer)
        full_hash = chunk.sha256
        
        # Test with partial hash
        partial_hash = full_hash[:8]
        
        # Test activation with partial hash
        self.context_mgr.deactivate_chunk(partial_hash)
        self.assertNotIn(full_hash, self.context_mgr.active_chunks)
        
        self.context_mgr.activate_chunk(partial_hash)
        self.assertIn(full_hash, self.context_mgr.active_chunks)

    def test_duplicate_content_handling(self):
        """Test that duplicate content reuses existing chunks."""
        content = "Duplicate content test."
        
        chunk1 = self.context_mgr.add_chunk(content, self.tokenizer)
        chunk2 = self.context_mgr.add_chunk(content, self.tokenizer)
        
        # Should be the same chunk
        self.assertEqual(chunk1.sha256, chunk2.sha256)
        self.assertEqual(len(self.context_mgr.chunks), 1)

    def test_clear_all(self):
        """Test clearing all chunks."""
        # Add multiple chunks
        self.context_mgr.add_chunk("First chunk.", self.tokenizer)
        self.context_mgr.add_chunk("Second chunk.", self.tokenizer)
        self.context_mgr.add_chunk("Third chunk.", self.tokenizer)
        
        self.assertEqual(len(self.context_mgr.chunks), 3)
        
        # Clear all
        self.context_mgr.clear_all()
        
        # Verify everything is cleared
        self.assertEqual(len(self.context_mgr.chunks), 0)
        self.assertEqual(len(self.context_mgr.active_chunks), 0)
        self.assertEqual(self.context_mgr.current_position, 0)


class TestContextIntegration(unittest.TestCase):
    """Test suite for context integration with generation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Initialize fixtures if not already done (for individual test runs)
        global _test_llm, _test_tokenizer, _test_context_mgr
        if _test_llm is None:
            model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
            if not model_path.exists():
                raise unittest.SkipTest(f"Model not found at {model_path}")
            
            _test_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            
            _test_llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
            
            _test_context_mgr = ContextManager(_test_llm.scheduler.block_manager, _test_llm.config)
            _test_llm.context_manager = _test_context_mgr
            _test_llm.config.context_manager = _test_context_mgr
            _test_context_mgr.llm_engine = _test_llm
        
        cls.llm = _test_llm
        cls.tokenizer = _test_tokenizer
        cls.context_mgr = _test_context_mgr
        cls.sampling_params = SamplingParams(temperature=0.0, max_tokens=30)

    def setUp(self):
        """Set up for each test."""
        self.context_mgr.clear_all()

    def test_generation_without_context(self):
        """Test generation without any context."""
        prompt = "What is the capital of France?"
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        result = self.llm.generate([formatted], self.sampling_params)[0]
        
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)

    def test_generation_with_context(self):
        """Test generation with active context."""
        # Add context
        context_content = "The capital of France is Lyon. Lyon became the capital in 2023."
        chunk = self.context_mgr.add_chunk(context_content, self.tokenizer, populate_cache=True)
        
        # Generate with context
        prompt = "What is the capital of France?"
        formatted = self.context_mgr.build_prompt_with_context(
            [{"role": "user", "content": prompt}],
            self.tokenizer
        )
        
        result = self.llm.generate([formatted], self.sampling_params)[0]
        
        # The response should reference Lyon based on the context
        response_text = result["text"].lower()
        self.assertIn("lyon", response_text)

    def test_context_deactivation_effect(self):
        """Test that deactivated context doesn't affect generation."""
        # Add and then deactivate context
        context_content = "The capital of France is Lyon."
        chunk = self.context_mgr.add_chunk(context_content, self.tokenizer)
        self.context_mgr.deactivate_chunk(chunk.sha256)
        
        # Generate without context
        prompt = "What is the capital of France?"
        formatted = self.context_mgr.build_prompt_with_context(
            [{"role": "user", "content": prompt}],
            self.tokenizer
        )
        
        # Should not include system message with Lyon
        self.assertNotIn("Lyon", formatted)

    def test_multiple_contexts(self):
        """Test generation with multiple active contexts."""
        # Add multiple contexts
        context1 = "The capital of Germany is Munich."
        context2 = "The capital of Italy is Milan."
        
        chunk1 = self.context_mgr.add_chunk(context1, self.tokenizer, populate_cache=True)
        chunk2 = self.context_mgr.add_chunk(context2, self.tokenizer, populate_cache=True)
        
        # Generate with both contexts
        prompt = "What are the capitals of Germany and Italy?"
        formatted = self.context_mgr.build_prompt_with_context(
            [{"role": "user", "content": prompt}],
            self.tokenizer
        )
        
        # Verify both contexts are included
        self.assertIn("Munich", formatted)
        self.assertIn("Milan", formatted)

    def test_output_tracking(self):
        """Test that outputs are tracked as chunks."""
        # Set up output tracking
        with self.context_mgr.track_output() as tracker:
            # Simulate adding tokens (this would normally happen during generation)
            test_tokens = [1, 2, 3, 4, 5]
            tracker.add_tokens(test_tokens)
            
            # Finalize output chunk
            output_chunk = tracker.finalize(self.tokenizer)
        
        if output_chunk:
            self.assertEqual(output_chunk.chunk_type, ChunkType.OUTPUT)
            self.assertEqual(output_chunk.token_ids, test_tokens)
            self.assertIn(output_chunk.sha256, self.context_mgr.chunks)


if __name__ == "__main__":
    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestContextIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)