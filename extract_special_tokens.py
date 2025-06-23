#!/usr/bin/env python3
"""Extract special tokens from chat template"""

from pathlib import Path
from transformers import AutoTokenizer

model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)

print("=== Extracting Special Tokens ===\n")

# Test 1: System message
system_msg = [{"role": "system", "content": "You are a helpful assistant."}]
system_formatted = tokenizer.apply_chat_template(system_msg, tokenize=False, add_generation_prompt=False)
system_tokens = tokenizer.encode(system_formatted)

print("System message:")
print(f"  Formatted: {repr(system_formatted)}")
print(f"  Tokens: {system_tokens}")
print(f"  Decoded tokens:")
for i, token in enumerate(system_tokens):
    print(f"    {i}: {token} -> {repr(tokenizer.decode([token]))}")

# Test 2: User message  
user_msg = [{"role": "user", "content": "Hello"}]
user_formatted = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=False)
user_tokens = tokenizer.encode(user_formatted)

print("\nUser message:")
print(f"  Formatted: {repr(user_formatted)}")
print(f"  Tokens: {user_tokens}")
print(f"  Decoded tokens:")
for i, token in enumerate(user_tokens):
    print(f"    {i}: {token} -> {repr(tokenizer.decode([token]))}")

# Test 3: Assistant message
assistant_msg = [{"role": "assistant", "content": "Hi there"}]
assistant_formatted = tokenizer.apply_chat_template(assistant_msg, tokenize=False, add_generation_prompt=False)
assistant_tokens = tokenizer.encode(assistant_formatted)

print("\nAssistant message:")
print(f"  Formatted: {repr(assistant_formatted)}")
print(f"  Tokens: {assistant_tokens}")
print(f"  Decoded tokens:")
for i, token in enumerate(assistant_tokens):
    print(f"    {i}: {token} -> {repr(tokenizer.decode([token]))}")

# Test 4: Extract just the role markers
print("\n=== Extracting Role Markers ===")

# System start
system_content_start = "You are a helpful assistant."
system_start_idx = system_formatted.index(system_content_start)
system_prefix = system_formatted[:system_start_idx]
system_suffix = system_formatted[system_start_idx + len(system_content_start):]

print(f"\nSystem prefix: {repr(system_prefix)}")
system_prefix_tokens = tokenizer.encode(system_prefix, add_special_tokens=False)
print(f"  Direct encode: {system_prefix_tokens}")

# Try to find the special tokens
print("\n=== Special Token IDs ===")
print(f"<|im_start|>: {tokenizer.encode('<|im_start|>', add_special_tokens=False)}")
print(f"<|im_end|>: {tokenizer.encode('<|im_end|>', add_special_tokens=False)}")
print(f"system: {tokenizer.encode('system', add_special_tokens=False)}")
print(f"user: {tokenizer.encode('user', add_special_tokens=False)}")  
print(f"assistant: {tokenizer.encode('assistant', add_special_tokens=False)}")
print(f"\\n: {tokenizer.encode('\\n', add_special_tokens=False)}")

# Find the actual token IDs by analyzing the encoded messages
print("\n=== Identified Token Structure ===")
print(f"im_start_id: {system_tokens[0]}")
print(f"system_id: {system_tokens[1]}")
print(f"newline_id: {system_tokens[2]}")
print(f"im_end_id: {user_tokens[-2]}")  # Second to last in user message
print(f"user_id: {user_tokens[1]}")
print(f"assistant_id: {assistant_tokens[1]}")

# Verify our understanding
print("\n=== Verification ===")
im_start = system_tokens[0]
im_end = user_tokens[-2]
system_id = system_tokens[1]
user_id = user_tokens[1]
assistant_id = assistant_tokens[1]
newline = system_tokens[2]

print(f"System chunk structure: [{im_start}, {system_id}, {newline}, ...content..., {im_end}, {newline}]")
print(f"User chunk structure: [{im_start}, {user_id}, {newline}, ...content..., {im_end}, {newline}]")
print(f"Assistant prompt: [{im_start}, {assistant_id}, {newline}]")