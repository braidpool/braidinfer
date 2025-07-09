#!/usr/bin/env python3
"""
Test to compare tokenization between standard and chunked paths.
"""

import os
from transformers import AutoTokenizer

def test_tokenization():
    """Compare tokenization in different contexts."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    system_prompt = "You are a helpful AI assistant. Please respond in English."
    user_query = "What is the capital of France?"
    
    print("=" * 80)
    print("TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    # Test 1: Full chat template
    print("\n1. FULL CHAT TEMPLATE:")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Without generation prompt
    template_no_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(f"\nTemplate without prompt:\n{repr(template_no_prompt)}")
    
    tokens_no_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    print(f"\nTokens without prompt ({len(tokens_no_prompt)}): {tokens_no_prompt[:20]}...")
    
    # With generation prompt
    template_with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nTemplate with prompt:\n{repr(template_with_prompt)}")
    
    tokens_with_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"\nTokens with prompt ({len(tokens_with_prompt)}): {tokens_with_prompt[:25]}...")
    
    # Test 2: Individual pieces
    print("\n\n2. INDIVIDUAL TOKENIZATION:")
    
    # System prompt alone
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    print(f"\nSystem prompt tokens ({len(system_tokens)}): {system_tokens}")
    
    # User query alone
    query_tokens = tokenizer.encode(user_query, add_special_tokens=False)
    print(f"\nQuery tokens ({len(query_tokens)}): {query_tokens}")
    
    # Special tokens
    print(f"\n\n3. SPECIAL TOKENS:")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Check for special tokens in vocabulary
    special_ids = [151643, 151644, 151645, 151667, 77091, 198]
    print(f"\n\n4. DECODING SPECIAL IDS:")
    for token_id in special_ids:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"Token {token_id}: {repr(decoded)}")
        except:
            print(f"Token {token_id}: <decode error>")
    
    # Test generation prompt extraction
    print(f"\n\n5. GENERATION PROMPT EXTRACTION:")
    empty_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""}
    ]
    empty_with = tokenizer.apply_chat_template(empty_messages, tokenize=False, add_generation_prompt=True)
    empty_without = tokenizer.apply_chat_template(empty_messages, tokenize=False, add_generation_prompt=False)
    
    print(f"Empty template without prompt: {repr(empty_without)}")
    print(f"Empty template with prompt: {repr(empty_with)}")
    
    if empty_with != empty_without:
        generation_prompt = empty_with[len(empty_without):]
        print(f"Extracted generation prompt: {repr(generation_prompt)}")
        gen_tokens = tokenizer.encode(generation_prompt, add_special_tokens=False)
        print(f"Generation prompt tokens: {gen_tokens}")

if __name__ == "__main__":
    test_tokenization()