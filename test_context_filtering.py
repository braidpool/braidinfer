#!/usr/bin/env python3
"""
Test script to verify Context Manager filtering works correctly
"""

import sys
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from transformers import AutoTokenizer

def test_filtering():
    """Test that deactivated chunks are actually filtered from attention"""
    print("Testing Context Manager Filtering...")
    
    # Initialize model
    model_path = Path.home() / "huggingface" / "Qwen3-4B"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    # Initialize context manager
    context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
    llm.context_manager = context_mgr
    llm.config.context_manager = context_mgr
    
    # Test 1: Add two very different chunks
    print("\n1. Creating test chunks...")
    
    chunk1 = context_mgr.add_chunk(
        "The capital of France is Paris. Paris is known for the Eiffel Tower, Louvre Museum, and excellent cuisine.",
        tokenizer,
        metadata={"topic": "France"}
    )
    print(f"   Chunk 1 (France): {chunk1.sha256[:16]}...")
    
    chunk2 = context_mgr.add_chunk(
        "Python is a high-level programming language. It emphasizes code readability and supports multiple paradigms.",
        tokenizer,
        metadata={"topic": "Programming"}
    )
    print(f"   Chunk 2 (Programming): {chunk2.sha256[:16]}...")
    
    # Test 2: Generate with both chunks active
    print("\n2. Generating with both chunks active...")
    prompt = "Tell me about"
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    
    output1 = llm.generate([prompt], sampling_params, use_tqdm=False)
    print(f"   Output: {output1[0]['text'].strip()}")
    
    # Test 3: Deactivate programming chunk
    print("\n3. Deactivating programming chunk...")
    context_mgr.deactivate_chunk(chunk2.sha256)
    
    # Check virtual block stats
    stats = context_mgr.virtual_block_table.get_statistics()
    print(f"   Active blocks: {stats['active_virtual_blocks']} / {stats['total_virtual_blocks']}")
    
    # Test 4: Generate again - should only reference France
    print("\n4. Generating with only France chunk active...")
    output2 = llm.generate([prompt], sampling_params, use_tqdm=False)
    print(f"   Output: {output2[0]['text'].strip()}")
    
    # Test 5: Deactivate France, activate programming
    print("\n5. Switching active chunks...")
    context_mgr.deactivate_chunk(chunk1.sha256)
    context_mgr.activate_chunk(chunk2.sha256)
    
    stats = context_mgr.virtual_block_table.get_statistics()
    print(f"   Active blocks: {stats['active_virtual_blocks']} / {stats['total_virtual_blocks']}")
    
    # Test 6: Generate again - should only reference programming
    print("\n6. Generating with only programming chunk active...")
    output3 = llm.generate([prompt], sampling_params, use_tqdm=False)
    print(f"   Output: {output3[0]['text'].strip()}")
    
    # Test 7: Deactivate all
    print("\n7. Deactivating all chunks...")
    context_mgr.deactivate_chunk(chunk2.sha256)
    
    stats = context_mgr.virtual_block_table.get_statistics()
    print(f"   Active blocks: {stats['active_virtual_blocks']} / {stats['total_virtual_blocks']}")
    
    # Test 8: Generate with no context
    print("\n8. Generating with no active context...")
    output4 = llm.generate([prompt], sampling_params, use_tqdm=False)
    print(f"   Output: {output4[0]['text'].strip()}")
    
    print("\n✓ Filtering test completed!")
    print("\nSummary:")
    print("- With both chunks: Should mention both topics")
    print("- With only France: Should focus on France/Paris") 
    print("- With only Programming: Should focus on Python/coding")
    print("- With no chunks: Should generate generic response")

if __name__ == "__main__":
    test_filtering()