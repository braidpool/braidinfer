# Context Manager Test Suite

This directory contains comprehensive unit tests for the Context Manager implementation in nano-vLLM.

## Test Files

### `test_context_manager.py`
Main unit tests for Context Manager functionality:

- **TestContextManager**: Core functionality tests
  - `test_add_chunk_basic`: Basic chunk creation
  - `test_add_chunk_with_cache_population`: KV cache population
  - `test_chunk_activation_deactivation`: Activation/deactivation
  - `test_populate_chunk_cache`: Manual cache population
  - `test_chunk_composition`: Combining multiple chunks
  - `test_chunk_tagging`: Metadata tagging
  - `test_chunk_deletion`: Chunk removal
  - `test_save_and_restore_chunk`: Disk persistence
  - `test_context_info`: Information retrieval
  - `test_memory_stats`: Memory usage statistics
  - `test_build_prompt_with_context`: Context integration
  - `test_chunk_preview`: Preview generation
  - `test_hash_resolution`: Partial hash handling
  - `test_duplicate_content_handling`: Deduplication
  - `test_clear_all`: Bulk operations

- **TestContextIntegration**: Context integration with generation
  - `test_generation_without_context`: Baseline generation
  - `test_generation_with_context`: Context-aware generation
  - `test_context_deactivation_effect`: Deactivation verification
  - `test_multiple_contexts`: Multiple context handling
  - `test_output_tracking`: Output chunk creation

### `test_cli.py`
Unit tests for CLI slash commands:

- **TestCLICommands**: Individual command tests
  - `test_load_command`: File loading
  - `test_context_command`: Status display
  - `test_activate_command`: Chunk activation
  - `test_deactivate_command`: Chunk deactivation
  - `test_populate_command`: Cache population
  - `test_compose_command`: Chunk composition
  - `test_tag_command`: Tagging
  - `test_save_command`: Disk saving
  - `test_restore_command`: Disk restoration
  - `test_delete_command`: Chunk deletion
  - `test_clear_command`: Bulk clearing
  - `test_help_command`: Help display
  - `test_unknown_command`: Error handling

- **TestCLIIntegration**: End-to-end CLI workflows
  - `test_load_and_generation_workflow`: Complete usage workflow
  - `test_chunk_lifecycle_commands`: Full lifecycle management
  - `test_multiple_chunks_management`: Multi-chunk operations

### `run_tests.py`
Test runner that executes all tests with detailed reporting.

## Running Tests

### Prerequisites
1. Ensure the model is downloaded:
   ```bash
   python cli.py  # Download model if needed
   ```

2. Install test dependencies (if not already installed):
   ```bash
   pip install transformers torch
   ```

### Running All Tests
```bash
python run_tests.py
```

### Running Individual Test Files
```bash
# Context Manager tests only
python -m unittest test_context_manager -v

# CLI tests only  
python -m unittest test_cli -v

# Specific test class
python -m unittest test_context_manager.TestContextManager -v

# Specific test method
python -m unittest test_context_manager.TestContextManager.test_add_chunk_basic -v
```

### Running with Python's unittest module
```bash
# Discover and run all tests
python -m unittest discover -s . -p "test_*.py" -v
```

## Test Coverage

The test suite covers:

### Core Functionality
- ✅ Chunk creation and management
- ✅ SHA256 content addressing
- ✅ Activation/deactivation without deletion
- ✅ KV cache population and persistence
- ✅ Virtual block table operations
- ✅ Memory hierarchy (GPU/CPU/disk)
- ✅ Chunk composition and tagging
- ✅ Context integration with generation

### CLI Commands
- ✅ All slash commands (`/load`, `/context`, `/activate`, etc.)
- ✅ Error handling and validation
- ✅ File operations and persistence
- ✅ Interactive workflows

### Error Conditions
- ✅ Invalid hash handling
- ✅ Missing file handling
- ✅ Insufficient arguments
- ✅ Cache population errors
- ✅ Disk I/O errors

### Integration
- ✅ Context affects generation output
- ✅ Deactivated chunks are ignored
- ✅ Multiple contexts work together
- ✅ Output tracking creates chunks

## Test Configuration

### Model Path
Tests use the model at: `~/huggingface/Qwen3-0.6B-GPTQ-Int8`

To change the model path, update the `model_path` variable in each test file.

### Temporary Directories
Tests create temporary directories for:
- Disk persistence testing
- File loading testing
- Save/restore operations

All temporary files are automatically cleaned up after tests.

### Mocking
CLI tests use `unittest.mock` to:
- Capture print output
- Isolate command testing
- Prevent side effects

## Example Output

```
======================================================================
Context Manager Test Suite
======================================================================
✅ Loaded tests from test_context_manager
✅ Loaded tests from test_cli

🧪 Running 31 tests...
----------------------------------------------------------------------
test_add_chunk_basic (test_context_manager.TestContextManager) ... ok
test_add_chunk_with_cache_population (test_context_manager.TestContextManager) ... ok
test_chunk_activation_deactivation (test_context_manager.TestContextManager) ... ok
...
test_multiple_chunks_management (test_cli.TestCLIIntegration) ... ok

----------------------------------------------------------------------
Ran 31 tests in 45.123s

OK

======================================================================
Test Summary
======================================================================
Total Tests: 31
✅ Passed: 31

🎉 All tests passed!
======================================================================
```

## Troubleshooting

### Model Not Found
```
❌ Model not found at /home/user/huggingface/Qwen3-0.6B-GPTQ-Int8
```
**Solution**: Run `python cli.py` first to download the model.

### Import Errors
```
❌ Failed to import test_context_manager: No module named 'nanovllm'
```
**Solution**: Ensure you're running from the nano-vllm project directory.

### CUDA Errors
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `tensor_parallel_size` or use a smaller model.

### Slow Tests
The tests use `enforce_eager=True` to avoid CUDA graph compilation overhead. If tests are still slow, consider:
- Using a smaller model
- Reducing `max_tokens` in test sampling parameters
- Running individual test classes instead of the full suite

## Contributing

When adding new features to the Context Manager:

1. Add corresponding unit tests
2. Test both success and error cases
3. Include CLI command tests if applicable
4. Update this README if new test files are added
5. Ensure all tests pass before submitting