#!/usr/bin/env python3
"""
Debug the attention issue in chunked generation.
"""

import os
import torch
from transformers import AutoTokenizer

def debug_tokens():
    """Debug token issues."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("=" * 80)
    print("TOKEN DEBUGGING")
    print("=" * 80)
    
    # Check what token 151667 is
    print(f"\nToken 151667: {repr(tokenizer.decode([151667]))}")
    print(f"Token 32313: {repr(tokenizer.decode([32313]))}")
    
    # Check the standard generation prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Please respond in English."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    print(f"\nFull prompt tokens ({len(tokens)}):")
    print(tokens)
    
    print(f"\nDecoded tokens:")
    for i, token in enumerate(tokens):
        print(f"{i:2d}: {token:6d} -> {repr(tokenizer.decode([token]))}")
    
    # Check what happens after the assistant marker
    print(f"\n\nAssistant marker analysis:")
    assistant_start_idx = None
    for i, token in enumerate(tokens):
        if token == 77091:  # 'assistant'
            assistant_start_idx = i
            break
    
    if assistant_start_idx:
        print(f"Assistant token at position {assistant_start_idx}")
        print(f"Tokens after assistant marker: {tokens[assistant_start_idx:]}")
        
    # Test generation with just the assistant prompt
    print(f"\n\nGeneration prompt only:")
    gen_prompt = "<|im_start|>assistant\n"
    gen_tokens = tokenizer.encode(gen_prompt, add_special_tokens=False)
    print(f"Generation prompt tokens: {gen_tokens}")
    
    # Check if the model expects a specific token after assistant
    print(f"\n\nChecking model's special tokens:")
    print(f"Model config eos_token_id: {tokenizer.eos_token_id}")
    print(f"Model config bos_token_id: {tokenizer.bos_token_id}")
    print(f"Model config pad_token_id: {tokenizer.pad_token_id}")

if __name__ == "__main__":
    debug_tokens()