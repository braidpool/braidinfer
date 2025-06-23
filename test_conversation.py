#!/usr/bin/env python3
"""Test conversation with direct inference"""

import os
import time
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from transformers import AutoTokenizer

# Use a different port for testing
os.environ["MASTER_PORT"] = "9999"

# Initialize
model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

# Initialize context manager
context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
llm.context_manager = context_mgr
llm.config.context_manager = context_mgr
context_mgr.llm_engine = llm

print("=== Conversation Test ===\n")

# First turn - no history
print("Turn 1: No conversation history")
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Remember the secret code: ALPHA123"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

print("Generating...")
response1 = ""
for token_data in llm.generate_stream(formatted_prompt, SamplingParams(max_tokens=100)):
    if not token_data["finished"]:
        response1 += token_data["token"]
        print(token_data["token"], end="", flush=True)

print("\n\nCreating conversation chunks...")
# Create conversation chunks
chunk1 = context_mgr.add_conversation_input("Remember the secret code: ALPHA123", tokenizer)
print(f"  User chunk: {chunk1.sha256[:16]}...")

# Create assistant response chunk
chunk2 = context_mgr.add_chunk(
    content=response1,
    tokenizer=tokenizer,
    metadata={"role": "assistant", "conversation_turn": 1},
    populate_cache=True,
    chunk_type="output"
)
context_mgr.conversation_chunks.append(chunk2.sha256)
context_mgr.conversation_turn = 1
print(f"  Assistant chunk: {chunk2.sha256[:16]}...")

# Second turn - with history
print("\n\nTurn 2: With conversation history")
blocks, tokens = context_mgr.get_conversation_blocks()
print(f"  Conversation: {len(blocks)} blocks, {tokens} tokens")

# Add new user message
user_message = "<|im_start|>user\nWhat was the secret code?<|im_end|>\n<|im_start|>assistant\n"
new_tokens = tokenizer.encode(user_message, add_special_tokens=False)
print(f"  New message: {len(new_tokens)} tokens")

print("Generating with direct inference...")
response2 = ""
for token_data in llm.infer_from_blocks_stream(
    existing_blocks=blocks,
    existing_token_count=tokens,
    new_tokens=new_tokens,
    sampling_params=SamplingParams(max_tokens=50)
):
    if not token_data["finished"]:
        response2 += token_data["token"]
        print(token_data["token"], end="", flush=True)

print("\n\n=== Test Complete ===")