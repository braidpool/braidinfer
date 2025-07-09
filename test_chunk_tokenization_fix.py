#!/usr/bin/env python3
"""
Test proper tokenization for chunks with chat template.
"""

import os
from transformers import AutoTokenizer
from nanovllm.chunks import Chunk, ChunkType

def test_chunk_tokenization():
    """Test how chunks should be tokenized with chat template."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    system_content = "You are a helpful AI assistant. Please respond in English."
    user_content = "What is the capital of France?"
    
    print("=" * 80)
    print("CHUNK TOKENIZATION WITH CHAT TEMPLATE")
    print("=" * 80)
    
    # Method 1: Current approach (raw content)
    print("\n1. CURRENT APPROACH (Raw content tokens):")
    system_tokens_raw = tokenizer.encode(system_content, add_special_tokens=False)
    user_tokens_raw = tokenizer.encode(user_content, add_special_tokens=False)
    print(f"System tokens ({len(system_tokens_raw)}): {system_tokens_raw}")
    print(f"User tokens ({len(user_tokens_raw)}): {user_tokens_raw}")
    print(f"Total raw tokens: {len(system_tokens_raw) + len(user_tokens_raw)}")
    
    # Method 2: Tokens with partial template
    print("\n\n2. TOKENS WITH PARTIAL TEMPLATE:")
    
    # System chunk with template
    system_template = f"<|im_start|>system\n{system_content}<|im_end|>"
    system_tokens_template = tokenizer.encode(system_template, add_special_tokens=False)
    print(f"System with template ({len(system_tokens_template)}): {system_tokens_template}")
    
    # User chunk with template  
    user_template = f"<|im_start|>user\n{user_content}<|im_end|>"
    user_tokens_template = tokenizer.encode(user_template, add_special_tokens=False)
    print(f"User with template ({len(user_tokens_template)}): {user_tokens_template}")
    
    # Method 3: Full conversation tokens
    print("\n\n3. FULL CONVERSATION APPROACH:")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"Full conversation tokens ({len(full_tokens)}): {full_tokens}")
    
    # Decode to see structure
    full_text = tokenizer.decode(full_tokens)
    print(f"\nFull text:\n{repr(full_text)}")
    
    # Method 4: Proposed chunk structure
    print("\n\n4. PROPOSED CHUNK STRUCTURE:")
    
    # We need to figure out how to split the full template into chunks
    # that can be reused while preserving the chat structure
    
    # Option A: Each chunk includes its role markers
    print("\nOption A - Chunks with role markers:")
    chunks_with_markers = []
    
    # System chunk: <|im_start|>system\n{content}<|im_end|>\n
    system_chunk_tokens = [151644, 8948, 198] + system_tokens_raw + [151645, 198]
    chunks_with_markers.append(("system", system_chunk_tokens))
    
    # User chunk: <|im_start|>user\n{content}<|im_end|>\n
    user_chunk_tokens = [151644, 872, 198] + user_tokens_raw + [151645, 198]
    chunks_with_markers.append(("user", user_chunk_tokens))
    
    # Generation prompt: <|im_start|>assistant\n
    gen_prompt_tokens = [151644, 77091, 198]
    chunks_with_markers.append(("gen_prompt", gen_prompt_tokens))
    
    for name, tokens in chunks_with_markers:
        print(f"{name}: {tokens} ({len(tokens)} tokens)")
    
    # Verify concatenation matches
    concatenated = []
    for _, tokens in chunks_with_markers:
        concatenated.extend(tokens)
    
    print(f"\nConcatenated tokens ({len(concatenated)}): {concatenated}")
    print(f"Matches full tokens: {concatenated == full_tokens}")
    
    if concatenated != full_tokens:
        print("\nDifferences:")
        for i, (a, b) in enumerate(zip(concatenated, full_tokens)):
            if a != b:
                print(f"  Position {i}: {a} vs {b}")
                break

if __name__ == "__main__":
    test_chunk_tokenization()