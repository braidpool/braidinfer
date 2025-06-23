#!/usr/bin/env python3
"""Test pure context management without chat interface"""

import os
import time
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from transformers import AutoTokenizer

# Use a different port for testing
os.environ["MASTER_PORT"] = "9997"

def main():
    # Initialize
    model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    # Initialize context manager
    context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
    llm.context_manager = context_mgr
    llm.config.context_manager = context_mgr
    context_mgr.llm_engine = llm

    print("=== Pure Context Management Test ===\n")

    # Add some chunks
    print("1. Adding chunks to context:")
    
    chunk1 = context_mgr.add_chunk(
        content="The capital of France is Paris.",
        tokenizer=tokenizer,
        metadata={"source": "test", "id": 1},
        populate_cache=True
    )
    print(f"   Added chunk 1: {chunk1.sha256[:16]}... ({chunk1.size} tokens)")
    
    chunk2 = context_mgr.add_chunk(
        content="The capital of Germany is Berlin.",
        tokenizer=tokenizer,
        metadata={"source": "test", "id": 2},
        populate_cache=True
    )
    print(f"   Added chunk 2: {chunk2.sha256[:16]}... ({chunk2.size} tokens)")
    
    chunk3 = context_mgr.add_chunk(
        content="The capital of Italy is Rome.",
        tokenizer=tokenizer,
        metadata={"source": "test", "id": 3},
        populate_cache=True
    )
    print(f"   Added chunk 3: {chunk3.sha256[:16]}... ({chunk3.size} tokens)")
    
    # Show context
    print("\n2. Current context:")
    blocks, tokens = context_mgr.get_all_active_blocks()
    print(f"   Active blocks: {len(blocks)}")
    print(f"   Total tokens: {tokens}")
    
    # Run inference with additional query
    print("\n3. Running inference with query:")
    query = "What is the capital of France?"
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    print(f"   Query: '{query}' ({len(query_tokens)} tokens)")
    
    response = ""
    token_count = 0
    start_time = time.time()
    
    for token_data in llm.infer_from_blocks_stream(
        existing_blocks=blocks,
        existing_token_count=tokens,
        new_tokens=query_tokens,
        sampling_params=SamplingParams(max_tokens=50, temperature=0.1)
    ):
        if not token_data["finished"]:
            response += token_data["token"]
            token_count += 1
            print(token_data["token"], end="", flush=True)
        else:
            break
    
    duration = time.time() - start_time
    print(f"\n\n   Generated {token_count} tokens in {duration:.1f}s ({token_count/duration:.1f} tok/s)")
    
    # Add the response as a chunk
    print("\n4. Adding response as chunk:")
    response_chunk = context_mgr.add_chunk(
        content=response,
        tokenizer=tokenizer,
        metadata={"source": "inference_output", "query": query},
        populate_cache=True
    )
    print(f"   Added response: {response_chunk.sha256[:16]}... ({response_chunk.size} tokens)")
    
    # Show final context
    print("\n5. Final context:")
    blocks, tokens = context_mgr.get_all_active_blocks()
    print(f"   Active blocks: {len(blocks)}")
    print(f"   Total tokens: {tokens}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()