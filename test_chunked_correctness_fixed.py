#!/usr/bin/env python3
"""
Test correctness of chunked vs standard generation with proper tokenization.
"""

import os
from nanovllm import LLM, ChunkedLLM, ChunkType, SamplingParams
from nanovllm.chunks import Chunk
import hashlib

class FixedChunkedLLM(ChunkedLLM):
    """ChunkedLLM with fixed tokenization that includes chat template."""
    
    def register_chunk(self, content: str, chunk_type: ChunkType, metadata=None):
        """Register a chunk with proper chat template tokenization."""
        # Create chunk ID based on content
        chunk_id = hashlib.sha256(f"{chunk_type.value}:{content}".encode()).hexdigest()
        
        # Check if already registered
        try:
            existing = self.registry.get(chunk_id)
            return chunk_id
        except:
            pass
        
        # Tokenize with chat template based on chunk type
        if chunk_type == ChunkType.SYSTEM_PROMPT:
            # System chunk: <|im_start|>system\n{content}<|im_end|>\n
            template_text = f"<|im_start|>system\n{content}<|im_end|>\n"
            token_ids = self.tokenizer.encode(template_text, add_special_tokens=False)
        elif chunk_type == ChunkType.QUERY:
            # User chunk: <|im_start|>user\n{content}<|im_end|>\n
            template_text = f"<|im_start|>user\n{content}<|im_end|>\n"
            token_ids = self.tokenizer.encode(template_text, add_special_tokens=False)
        else:
            # Context chunks - just raw content for now
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
        
        # Create chunk with proper tokens
        chunk = Chunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=content,
            token_ids=token_ids,
            token_count=len(token_ids),
            metadata=metadata or {}
        )
        
        # Register and prefill
        self.registry.register(chunk)
        if not chunk.kv_cache_allocated:
            self._prefill_chunk(chunk)
        
        return chunk_id

def test_correctness_fixed():
    """Compare outputs from standard and chunked generation with fixed tokenization."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    # Test inputs
    system_prompt = "You are a helpful AI assistant. Please respond in English."
    user_query = "What is the capital of France?"
    
    print("=" * 80)
    print("TESTING CHUNKED VS STANDARD GENERATION (WITH FIXED TOKENIZATION)")
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
    
    print(f"Prompt: {repr(prompt)}")
    
    # Tokenize to see structure
    prompt_tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
    
    # Generate with temperature=0 for deterministic output
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate([prompt], sampling_params)
    
    standard_output = outputs[0]["text"]
    standard_tokens = outputs[0]["token_ids"]
    
    print(f"\nStandard output: {standard_output}")
    print(f"Token count: {len(standard_tokens)}")
    print(f"First 10 tokens: {standard_tokens[:10]}")
    
    # Test 2: Fixed chunked generation
    print("\n\n2. FIXED CHUNKED GENERATION:")
    print("-" * 40)
    
    chunked_llm = FixedChunkedLLM(
        model_path,
        max_chunks=100,
        chunk_memory_ratio=0.5,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Register chunks with proper tokenization
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
    
    # Get the chunks to see their tokens
    system_chunk = chunked_llm.registry.get(system_chunk_id)
    query_chunk = chunked_llm.registry.get(query_chunk_id)
    print(f"System chunk tokens ({len(system_chunk.token_ids)}): {system_chunk.token_ids}")
    print(f"Query chunk tokens ({len(query_chunk.token_ids)}): {query_chunk.token_ids}")
    
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
        min_len = min(len(standard_output), len(chunked_output))
        for i in range(min_len):
            if standard_output[i] != chunked_output[i]:
                print(f"\nFirst difference at position {i}:")
                print(f"  Standard: '{standard_output[i]}' (ord={ord(standard_output[i])})")
                print(f"  Chunked:  '{chunked_output[i]}' (ord={ord(chunked_output[i])})")
                break
    
    # Token comparison
    if standard_tokens == chunked_tokens:
        print("\n✅ PASS: Token sequences are identical!")
    else:
        print("\n❌ FAIL: Token sequences differ!")
        print(f"Length difference: {len(standard_tokens)} vs {len(chunked_tokens)}")
        
        # Find first differing token
        min_len = min(len(standard_tokens), len(chunked_tokens))
        for i in range(min_len):
            if standard_tokens[i] != chunked_tokens[i]:
                print(f"First difference at position {i}:")
                print(f"  Standard: {standard_tokens[i]}")
                print(f"  Chunked:  {chunked_tokens[i]}")
                break

if __name__ == "__main__":
    test_correctness_fixed()