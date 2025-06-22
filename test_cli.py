#!/usr/bin/env python3
"""
Unit tests for CLI functionality and slash commands.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock, Mock
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from cli import handle_slash_command
from rich.console import Console


class MockConsole:
    """Mock console to capture rich output."""
    def __init__(self):
        self.output = []
    
    def print(self, *args, **kwargs):
        # Convert rich markup to plain text for testing
        for arg in args:
            # Handle Rich Table objects
            if hasattr(arg, '__rich__') or hasattr(arg, '_rows'):
                # For tables, we'll just add a simple representation
                text = str(type(arg).__name__).lower()
                if hasattr(arg, 'title') and arg.title:
                    text = str(arg.title).lower()
            else:
                text = str(arg)
                # Simple removal of rich markup
                import re
                text = re.sub(r'\[/?[^\]]+\]', '', text)
            self.output.append(text)
    
    def get_output(self):
        return '\n'.join(self.output)
    
    def clear(self):
        self.output = []


# Global test fixtures
_test_llm = None
_test_tokenizer = None
_test_context_mgr = None


class TestCLICommands(unittest.TestCase):
    """Test suite for CLI slash commands."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
        
        # Set different port to avoid conflicts
        import os
        os.environ["MASTER_PORT"] = "2335"
        
        # Initialize fixtures if not already done (for individual test runs)
        global _test_llm, _test_tokenizer, _test_context_mgr
        if _test_llm is None:
            if not cls.model_path.exists():
                raise unittest.SkipTest(f"Model not found at {cls.model_path}")
                
            from transformers import AutoTokenizer
            from nanovllm import LLM
            from nanovllm.engine.context_manager import ContextManager
            
            _test_tokenizer = AutoTokenizer.from_pretrained(
                str(cls.model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            
            _test_llm = LLM(cls.model_path, enforce_eager=True, tensor_parallel_size=1)
            
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

        # Create temporary directory and files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.context_mgr.disk_path = self.temp_dir

        # Create test file
        self.test_file = Path(self.temp_dir) / "test_content.txt"
        self.test_file.write_text("This is test content for loading.\nIt has multiple lines.")

    def tearDown(self):
        """Clean up after each test."""
        # Clean up temporary directory
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_command(self):
        """Test /load command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Test successful load
        handle_slash_command("load", str(self.test_file), self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk was created
        self.assertEqual(len(self.context_mgr.chunks), 1)
        chunk = list(self.context_mgr.chunks.values())[0]
        self.assertIn("source", chunk.metadata)
        self.assertEqual(chunk.metadata["source"], str(self.test_file))

        # Verify success message was printed
        self.assertIn("✓ Added chunk:", mock_console.get_output())

    def test_load_command_nonexistent_file(self):
        """Test /load command with nonexistent file."""

        # Create mock console

        mock_console = MockConsole()

        
        handle_slash_command("load", "/nonexistent/file.txt", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify no chunks were created
        self.assertEqual(len(self.context_mgr.chunks), 0)

        # Verify error message was printed
        self.assertIn("File not found: /nonexistent/file.txt", mock_console.get_output())

    def test_load_command_no_args(self):
        """Test /load command without arguments."""

        # Create mock console

        mock_console = MockConsole()

        
        handle_slash_command("load", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify usage message was printed
        self.assertIn("Usage: /load <filename>", mock_console.get_output())

    def test_context_command(self):
        """Test /context command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add some test chunks
        chunk1 = self.context_mgr.add_chunk("First chunk", self.tokenizer)
        chunk2 = self.context_mgr.add_chunk("Second chunk", self.tokenizer)
        self.context_mgr.deactivate_chunk(chunk2.sha256)

        # Test context command
        handle_slash_command("context", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify output contains relevant information (check if any call contains expected strings)
        output = mock_console.get_output()
        # Should see the table, memory stats, and chunk information
        self.assertIn("memory usage", output.lower())
        self.assertIn("token summary", output.lower())
        # One chunk is active, one is inactive - we should see both in summary
        self.assertIn("active tokens", output.lower())

    def test_activate_command(self):
        """Test /activate command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add and deactivate a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)
        self.context_mgr.deactivate_chunk(chunk.sha256)
        
        # Test activation
        handle_slash_command("activate", chunk.sha256[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk is active
        self.assertIn(chunk.sha256, self.context_mgr.active_chunks)

        # Verify success message
        output = mock_console.get_output()
        self.assertIn(f"✓ Chunk activated: {chunk.sha256[:8]}...", output)

    def test_activate_command_invalid_hash(self):
        """Test /activate command with invalid hash."""

        # Create mock console

        mock_console = MockConsole()

        
        handle_slash_command("activate", "invalid_hash", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify error message
        output = mock_console.get_output()
        error_messages = [line for line in output.split("\n") if "Failed to activate chunk" in line]
        self.assertTrue(len(error_messages) > 0)

    def test_deactivate_command(self):
        """Test /deactivate command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)

        # Test deactivation
        handle_slash_command("deactivate", chunk.sha256[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk is inactive
        self.assertNotIn(chunk.sha256, self.context_mgr.active_chunks)

        # Verify success message
        self.assertIn(f"✓ Chunk deactivated: {chunk.sha256[:8]}...", mock_console.get_output())

    def test_populate_command(self):
        """Test /populate command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk without cache population
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer, populate_cache=False)

        # Test populate command
        handle_slash_command("populate", chunk.sha256[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify cache is populated
        self.assertTrue(chunk.cache_populated)

        # Verify success message
        self.assertIn(f"✓ Populated KV cache for chunk {chunk.sha256[:8]}...", mock_console.get_output())

    def test_compose_command(self):
        """Test /compose command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add chunks
        chunk1 = self.context_mgr.add_chunk("First chunk", self.tokenizer)
        chunk2 = self.context_mgr.add_chunk("Second chunk", self.tokenizer)

        # Test compose command
        args = f"{chunk1.sha256[:8]} {chunk2.sha256[:8]}"
        handle_slash_command("compose", args, self.context_mgr, self.tokenizer, console=mock_console)

        # Verify composed chunk was created (should have 3 chunks total now)
        self.assertEqual(len(self.context_mgr.chunks), 3)

        # Verify success message
        self.assertIn("✓ Composed new chunk:", mock_console.get_output())

    def test_compose_command_insufficient_chunks(self):
        """Test /compose command with insufficient chunks."""

        # Create mock console

        mock_console = MockConsole()

        
        chunk = self.context_mgr.add_chunk("Single chunk", self.tokenizer)

        # Test compose with only one chunk
        handle_slash_command("compose", chunk.sha256[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify error message
        self.assertIn("Please provide at least 2 chunk hashes to compose", mock_console.get_output())

    def test_tag_command(self):
        """Test /tag command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)

        # Test tag command
        args = f"{chunk.sha256[:8]} test-tag"
        handle_slash_command("tag", args, self.context_mgr, self.tokenizer, console=mock_console)

        # Verify tag was added
        self.assertIn("tags", chunk.metadata)
        self.assertIn("test-tag", chunk.metadata["tags"])

        # Verify success message
        self.assertIn(f"✓ Tagged chunk {chunk.sha256[:8]}... with 'test-tag'", mock_console.get_output())

    def test_save_command(self):
        """Test /save command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)

        # Test save command
        handle_slash_command("save", chunk.sha256[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify file was created
        save_path = Path(self.temp_dir) / f"{chunk.sha256}.pkl"
        self.assertTrue(save_path.exists())

        # Verify success message
        self.assertIn(f"✓ Chunk saved: {chunk.sha256[:8]}...", mock_console.get_output())

    def test_unload_command(self):
        """Test /unload command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)
        chunk_hash = chunk.sha256

        # Test unload command
        handle_slash_command("unload", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk was unloaded to RAM
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "cpu")
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)

        # Verify success message
        self.assertIn(f"✓ Chunk moved to system RAM: {chunk_hash[:8]}...", mock_console.get_output())

    def test_restore_command(self):
        """Test /restore command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add and unload a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)
        chunk_hash = chunk.sha256
        self.context_mgr.unload_chunk(chunk_hash)

        # Test restore command
        handle_slash_command("restore", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk was restored to VRAM
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "active")
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)

        # Verify success message
        self.assertIn(f"✓ Chunk restored: {chunk_hash[:8]}...", mock_console.get_output())

    def test_erase_command(self):
        """Test /erase command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add a chunk
        chunk = self.context_mgr.add_chunk("Test chunk", self.tokenizer)
        chunk_hash = chunk.sha256

        # Test erase command
        handle_slash_command("erase", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk was erased
        self.assertNotIn(chunk_hash, self.context_mgr.chunks)

        # Verify success message
        self.assertIn(f"✓ Chunk erased from all locations: {chunk_hash[:8]}...", mock_console.get_output())

    def test_clear_command(self):
        """Test /clear command."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add multiple chunks
        self.context_mgr.add_chunk("Chunk 1", self.tokenizer)
        self.context_mgr.add_chunk("Chunk 2", self.tokenizer)
        self.context_mgr.add_chunk("Chunk 3", self.tokenizer)

        # Test clear command
        handle_slash_command("clear", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify all chunks were cleared
        self.assertEqual(len(self.context_mgr.chunks), 0)

        # Verify success message
        self.assertIn("✓ All chunks cleared", mock_console.get_output())

    def test_help_command(self):
        """Test /help command."""

        # Create mock console

        mock_console = MockConsole()

        
        handle_slash_command("help", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify help information was printed
        output = mock_console.get_output()
        self.assertIn("Available Commands", output)
        self.assertIn("/load", output)
        self.assertIn("/context", output)

    def test_unknown_command(self):
        """Test unknown command."""

        # Create mock console

        mock_console = MockConsole()

        
        handle_slash_command("unknown", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify error message
        self.assertIn("Unknown command: /unknown", mock_console.get_output())
        self.assertIn("Type /help for available commands", mock_console.get_output())


class TestCLIIntegration(unittest.TestCase):
    """Test suite for CLI integration with context manager."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
        
        # Set different port to avoid conflicts
        import os
        os.environ["MASTER_PORT"] = "2335"
        
        # Initialize fixtures if not already done (for individual test runs)
        global _test_llm, _test_tokenizer, _test_context_mgr
        if _test_llm is None:
            if not cls.model_path.exists():
                raise unittest.SkipTest(f"Model not found at {cls.model_path}")
                
            from transformers import AutoTokenizer
            from nanovllm import LLM
            from nanovllm.engine.context_manager import ContextManager
            
            _test_tokenizer = AutoTokenizer.from_pretrained(
                str(cls.model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            
            _test_llm = LLM(cls.model_path, enforce_eager=True, tensor_parallel_size=1)
            
            _test_context_mgr = ContextManager(_test_llm.scheduler.block_manager, _test_llm.config)
            _test_llm.context_manager = _test_context_mgr
            _test_llm.config.context_manager = _test_context_mgr
            _test_context_mgr.llm_engine = _test_llm
        
        cls.llm = _test_llm
        cls.tokenizer = _test_tokenizer
        cls.context_mgr = _test_context_mgr

    def setUp(self):
        """Set up for each test."""
        self.context_mgr.clear_all()

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.context_mgr.disk_path = self.temp_dir

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_and_generation_workflow(self):
        """Test complete workflow: load content, generate with context."""

        # Create mock console

        mock_console = MockConsole()

        
        # Create test file
        test_file = Path(self.temp_dir) / "context.txt"
        test_content = "The capital of France is Lyon. This is fictional information for testing."
        test_file.write_text(test_content)

        # Load content using CLI command
        handle_slash_command("load", str(test_file), self.context_mgr, self.tokenizer, console=mock_console)

        # Verify chunk was created
        self.assertEqual(len(self.context_mgr.chunks), 1)
        chunk = list(self.context_mgr.chunks.values())[0]

        # Test generation with context
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = self.context_mgr.build_prompt_with_context(messages, self.tokenizer)

        # Verify context is included in prompt
        self.assertIn("Lyon", formatted_prompt)

    def test_chunk_lifecycle_commands(self):
        """Test full chunk lifecycle through CLI commands."""

        # Create mock console

        mock_console = MockConsole()

        
        # Add chunk
        chunk = self.context_mgr.add_chunk("Test lifecycle content", self.tokenizer)
        chunk_hash = chunk.sha256

        # Test deactivation
        handle_slash_command("deactivate", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)

        # Test reactivation
        handle_slash_command("activate", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)

        # Test tagging
        handle_slash_command("tag", f"{chunk_hash[:8]} lifecycle-test", self.context_mgr, self.tokenizer, console=mock_console)
        self.assertIn("lifecycle-test", chunk.metadata["tags"])

        # Test save
        handle_slash_command("save", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        save_path = Path(self.temp_dir) / f"{chunk_hash}.pkl"
        self.assertTrue(save_path.exists())

        # Test unload to RAM
        handle_slash_command("unload", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "cpu")
        self.assertNotIn(chunk_hash, self.context_mgr.active_chunks)

        # Test restoration from RAM
        handle_slash_command("restore", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        self.assertEqual(self.context_mgr.chunks[chunk_hash].status, "active")
        self.assertIn(chunk_hash, self.context_mgr.active_chunks)
        
        # Test complete erasure
        handle_slash_command("erase", chunk_hash[:8], self.context_mgr, self.tokenizer, console=mock_console)
        self.assertNotIn(chunk_hash, self.context_mgr.chunks)

    def test_multiple_chunks_management(self):
        """Test managing multiple chunks through CLI."""

        # Create mock console

        mock_console = MockConsole()

        
        # Create multiple test files
        files_content = {
            "file1.txt": "Content of first file.",
            "file2.txt": "Content of second file.",
            "file3.txt": "Content of third file."
        }

        for filename, content in files_content.items():
            file_path = Path(self.temp_dir) / filename
            file_path.write_text(content)

            # Load each file
            handle_slash_command("load", str(file_path), self.context_mgr, self.tokenizer, console=mock_console)

        # Verify all chunks were created
        self.assertEqual(len(self.context_mgr.chunks), 3)

        # Test context command with multiple chunks
        handle_slash_command("context", "", self.context_mgr, self.tokenizer, console=mock_console)

        # Verify output mentions multiple chunks
        output = mock_console.get_output()
        # Check that we have references to files or chunks
        self.assertTrue("file" in output.lower() or "chunk" in output.lower())

        # Test clearing all chunks
        handle_slash_command("clear", "", self.context_mgr, self.tokenizer, console=mock_console)

        self.assertEqual(len(self.context_mgr.chunks), 0)


if __name__ == "__main__":
    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCLICommands))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
