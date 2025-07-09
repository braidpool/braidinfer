#!/usr/bin/env python3
"""
Test correctness of chunked vs standard generation.
"""

import os
from nanovllm import LLM, ChunkedLLM, ChunkType, SamplingParams

def test_correctness():
    """Compare outputs from standard and chunked generation."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    # Test inputs
    system_prompt = "You are a helpful AI assistant. Please respond in English."
    user_query = "What is the capital of France?"
    
    print("=" * 80)
    print("TESTING CHUNKED VS STANDARD GENERATION CORRECTNESS")
    print("=" * 80)
    
    # Test 1: Standard generation
    print("\n1. STANDARD GENERATION:")
    print("-" * 40)
    
    llm = LLM(model_path, enforce_eager=True)
    
    # Build prompt using tokenizer's chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt: {prompt}")
    
    # Generate with temperature=0 for deterministic output
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate([prompt], sampling_params)
    
    standard_output = outputs[0]["text"]
    standard_tokens = outputs[0]["token_ids"]
    
    print(f"\nStandard output: {standard_output}")
    print(f"Token count: {len(standard_tokens)}")
    print(f"First 10 tokens: {standard_tokens[:10]}")
    
    # Test 2: Chunked generation
    print("\n\n2. CHUNKED GENERATION:")
    print("-" * 40)
    
    chunked_llm = ChunkedLLM(
        model_path,
        max_chunks=100,
        chunk_memory_ratio=0.5,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Register chunks
    system_chunk_id = chunked_llm.register_chunk(
        system_prompt,
        ChunkType.SYSTEM_PROMPT
    )
    
    query_chunk_id = chunked_llm.register_chunk(
        user_query,
        ChunkType.QUERY
    )
    
    print(f"System chunk ID: {system_chunk_id[:8]}...")
    print(f"Query chunk ID: {query_chunk_id[:8]}...")
    
    # Generate using chunks
    result = chunked_llm.generate_from_chunks(
        system_chunk_id=system_chunk_id,
        query_chunk_id=query_chunk_id,
        context_chunk_ids=None,
        sampling_params={"temperature": 0.0, "max_tokens": 50},
        stream=False
    )
    
    chunked_output = result["text"]
    chunked_tokens = result["token_ids"]
    
    print(f"\nChunked output: {chunked_output}")
    print(f"Token count: {len(chunked_tokens)}")
    print(f"First 10 tokens: {chunked_tokens[:10]}")
    
    # Compare results
    print("\n\n3. COMPARISON:")
    print("-" * 40)
    
    if standard_output == chunked_output:
        print("✅ PASS: Outputs are identical!")
    else:
        print("❌ FAIL: Outputs differ!")
        print(f"\nStandard: {standard_output}")
        print(f"Chunked:  {chunked_output}")
        
        # Find where they differ
        for i, (s, c) in enumerate(zip(standard_output, chunked_output)):
            if s != c:
                print(f"\nFirst difference at position {i}:")
                print(f"  Standard: '{s}' (ord={ord(s)})")
                print(f"  Chunked:  '{c}' (ord={ord(c)})")
                break
    
    # Token comparison
    if standard_tokens == chunked_tokens:
        print("\n✅ PASS: Token sequences are identical!")
    else:
        print("\n❌ FAIL: Token sequences differ!")
        print(f"Length difference: {len(standard_tokens)} vs {len(chunked_tokens)}")
        
        # Find first differing token
        for i in range(min(len(standard_tokens), len(chunked_tokens))):
            if standard_tokens[i] != chunked_tokens[i]:
                print(f"First difference at position {i}:")
                print(f"  Standard: {standard_tokens[i]}")
                print(f"  Chunked:  {chunked_tokens[i]}")
                break

if __name__ == "__main__":
    test_correctness()