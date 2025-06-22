#!/usr/bin/env python3
"""
Test runner for Context Manager test suite.
"""

import unittest
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", message=".*packed_modules_mapping is empty.*")

def main():
    """Run all Context Manager tests."""
    print("=" * 70)
    print("Context Manager Test Suite")
    print("=" * 70)

    # Check if model exists
    model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run the CLI first to download the model or update the model path.")
        sys.exit(1)

    # Initialize shared test fixtures once
    print("🔧 Initializing shared test fixtures...")
    try:
        from transformers import AutoTokenizer
        from nanovllm import LLM
        from nanovllm.engine.context_manager import ContextManager
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Load LLM (this initializes distributed once)
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        # Initialize context manager
        context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
        llm.context_manager = context_mgr
        llm.config.context_manager = context_mgr
        context_mgr.llm_engine = llm
        
        # Set global fixtures in both test modules
        import test_context_manager
        test_context_manager._test_llm = llm
        test_context_manager._test_tokenizer = tokenizer
        test_context_manager._test_context_mgr = context_mgr
        
        import test_cli
        test_cli._test_llm = llm
        test_cli._test_tokenizer = tokenizer
        test_cli._test_context_mgr = context_mgr
        
        print("✅ Shared fixtures initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize fixtures: {e}")
        sys.exit(1)

    # Discover and run tests
    loader = unittest.TestLoader()

    # Load test modules
    test_modules = [
        'test_context_manager',
        'test_cli'
    ]

    suite = unittest.TestSuite()

    for module_name in test_modules:
        try:
            module = __import__(module_name)
            module_tests = loader.loadTestsFromModule(module)
            suite.addTests(module_tests)
            print(f"✅ Loaded tests from {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error loading tests from {module_name}: {e}")
            continue

    if suite.countTestCases() == 0:
        print("❌ No tests found to run")
        sys.exit(1)

    print(f"\n🧪 Running {suite.countTestCases()} tests...")
    print("-" * 70)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True  # Capture stdout/stderr during tests
    )

    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    if skipped > 0:
        print(f"⏭️  Skipped: {skipped}")
    if failures > 0:
        print(f"❌ Failed: {failures}")
    if errors > 0:
        print(f"💥 Errors: {errors}")

    # Print details for failures and errors
    if result.failures:
        print(f"\n📝 Failure Details:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else 'Unknown failure'}")

    if result.errors:
        print(f"\n💥 Error Details:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else 'Unknown error'}")

    # Print overall result
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed!")
        exit_code = 0
    else:
        print(f"\n❌ Some tests failed")
        exit_code = 1

    print("=" * 70)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
