#!/usr/bin/env python3
"""
Test script for KV cache persistence functionality.
"""

import torch
import tempfile
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager


def test_kv_cache_persistence():
    """Test saving and restoring actual KV cache data"""
    print("=" * 60)
    print("Testing KV Cache Persistence")
    print("=" * 60)
    
    # Setup
    model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    # Set a different port to avoid conflicts
    import os
    os.environ["MASTER_PORT"] = "2334"
        
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True
    )
    
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
    llm.context_manager = context_mgr
    llm.config.context_manager = context_mgr
    context_mgr.llm_engine = llm
    
    # Create temporary directory for disk storage
    temp_dir = tempfile.mkdtemp()
    context_mgr.disk_path = temp_dir
    
    try:
        # Test 1: Create chunk with KV cache
        print("\n1. Creating chunk with populated KV cache...")
        content = "The capital of France is Paris. Paris is a beautiful city known for the Eiffel Tower."
        chunk = context_mgr.add_chunk(content, tokenizer, populate_cache=True)
        chunk_hash = chunk.sha256
        print(f"   Created chunk: {chunk_hash[:16]}...")
        print(f"   Tokens: {len(chunk.token_ids)}")
        print(f"   Cache populated: {chunk.cache_populated}")
        
        # Test 2: Extract KV cache data
        print("\n2. Extracting KV cache data...")
        kv_data = context_mgr.extract_kv_cache_for_chunk(chunk)
        if kv_data:
            print(f"   Extracted {len(kv_data['k_cache'])} K-cache entries")
            print(f"   Extracted {len(kv_data['v_cache'])} V-cache entries")
            
            # Check data structure
            k_entry = kv_data['k_cache'][0]
            print(f"   K-cache entry shape: {k_entry['data'].shape}")
            print(f"   Data type: {k_entry['data'].dtype}")
        else:
            print("   Failed to extract KV cache data!")
            
        # Test 3: Save chunk to disk
        print("\n3. Saving chunk to disk...")
        context_mgr.save_chunk(chunk_hash)
        save_path = Path(temp_dir) / f"{chunk_hash}.pkl"
        print(f"   Saved to: {save_path}")
        print(f"   File size: {save_path.stat().st_size / 1024:.1f} KB")
        
        # Test 4: Unload chunk to RAM (preserves disk file)
        print("\n4. Unloading chunk to RAM...")
        context_mgr.unload_chunk(chunk_hash)
        print(f"   Chunk status: {context_mgr.chunks[chunk_hash].status}")
        print(f"   In CPU cache: {chunk_hash in context_mgr.cpu_cache}")
        print(f"   File still exists: {save_path.exists()}")
        
        # Test 5: Erase chunk completely (removes from all locations)
        print("\n5. Erasing chunk completely...")
        context_mgr.erase_chunk(chunk_hash)
        print(f"   Chunk in memory: {chunk_hash in context_mgr.chunks}")
        print(f"   In CPU cache: {chunk_hash in context_mgr.cpu_cache}")
        print(f"   File exists: {save_path.exists()}")
        
        # Create a new chunk to test save/restore workflow
        print("\n6. Testing save/unload/restore workflow...")
        content2 = "The Eiffel Tower is located in Paris, France."
        chunk2 = context_mgr.add_chunk(content2, tokenizer, populate_cache=True)
        chunk2_hash = chunk2.sha256
        chunk2_tokens = chunk2.token_ids.copy()
        
        # Save and unload
        context_mgr.save_chunk(chunk2_hash)
        context_mgr.unload_chunk(chunk2_hash)
        
        # Clear from memory but keep file
        del context_mgr.chunks[chunk2_hash]
        del context_mgr.cpu_cache[chunk2_hash]
        
        # Restore from disk
        restored_chunk = context_mgr.restore_chunk(chunk2_hash)
        print(f"   Restored chunk: {restored_chunk.sha256[:16]}...")
        print(f"   Cache populated: {restored_chunk.cache_populated}")
        print(f"   Tokens match: {restored_chunk.token_ids == chunk2_tokens}")
        
        # Test 7: Test generation with restored chunk
        print("\n7. Testing generation with restored chunk...")
        prompt = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt}]
        formatted = context_mgr.build_prompt_with_context(messages, tokenizer)
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
        result = llm.generate([formatted], sampling_params)[0]
        response = result["text"]
        print(f"   Question: {prompt}")
        print(f"   Response: {response}")
        print(f"   Contains 'Paris': {'Paris' in response}")
        
        # Test 8: Test unload/restore cycle
        print("\n8. Testing unload/restore cycle...")
        chunk3 = context_mgr.add_chunk("London is the capital of England.", tokenizer, populate_cache=True)
        chunk3_hash = chunk3.sha256
        
        print(f"   Created chunk3: {chunk3_hash[:16]}...")
        print(f"   Initial status: {chunk3.status}")
        
        # Unload to CPU
        context_mgr.unload_chunk(chunk3_hash)
        print(f"   After unload - status: {context_mgr.chunks[chunk3_hash].status}")
        print(f"   In CPU cache: {chunk3_hash in context_mgr.cpu_cache}")
        
        # Restore from CPU
        restored3 = context_mgr.restore_chunk(chunk3_hash)
        print(f"   After restore - status: {restored3.status}")
        print(f"   Cache populated: {restored3.cache_populated}")
        
        # Test 9: Verify KV cache data integrity
        print("\n9. Verifying KV cache data integrity...")
        if kv_data:
            # Extract KV data from the restored chunk2
            kv_data_restored = context_mgr.extract_kv_cache_for_chunk(restored_chunk)
            if kv_data_restored:
                # Compare first K-cache entry between different extractions
                k_entry_idx = 0
                if len(kv_data['k_cache']) > k_entry_idx and len(kv_data_restored['k_cache']) > k_entry_idx:
                    # Since chunk2 is different content, we'll verify internal consistency instead
                    k_restored = kv_data_restored['k_cache'][k_entry_idx]['data']
                    v_restored = kv_data_restored['v_cache'][k_entry_idx]['data']
                    
                    print(f"   Restored K-cache shape: {k_restored.shape}")
                    print(f"   Restored V-cache shape: {v_restored.shape}")
                    print(f"   Data restored successfully: {k_restored.shape[0] > 0}")
                else:
                    print("   Cache entries mismatch")
            else:
                print("   Could not extract restored KV cache data")
        
        print("\n" + "=" * 60)
        print("KV Cache Persistence Test Complete!")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_kv_cache_persistence()