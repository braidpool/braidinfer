#!/usr/bin/env python3
"""Test direct inference with conversation context"""

import os
import time
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from transformers import AutoTokenizer

# Use a different port for testing
os.environ["MASTER_PORT"] = "9998"

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

    print("=== Direct Inference Test ===\n")

    # First turn - traditional generation
    print("1. First turn using traditional generation:")
    prompt1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": "The password is ZEBRA456. Remember it."}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    response1 = ""
    for token_data in llm.generate_stream(prompt1, SamplingParams(max_tokens=30)):
        if not token_data["finished"]:
            response1 += token_data["token"]
    print(f"Response: {response1[:100]}...")
    
    # Create conversation chunks manually
    print("\n2. Creating conversation chunks:")
    user_chunk = context_mgr.add_conversation_input("The password is ZEBRA456. Remember it.", tokenizer)
    print(f"   User chunk: {user_chunk.sha256[:16]}... ({user_chunk.size} tokens)")
    
    # Create assistant response chunk
    assistant_chunk = context_mgr.add_chunk(
        content=response1,
        tokenizer=tokenizer,
        metadata={"role": "assistant", "conversation_turn": 1},
        populate_cache=True,
        chunk_type="output"
    )
    context_mgr.conversation_chunks.append(assistant_chunk.sha256)
    context_mgr.conversation_turn = 1
    print(f"   Assistant chunk: {assistant_chunk.sha256[:16]}... ({assistant_chunk.size} tokens)")
    
    # Second turn - direct inference
    print("\n3. Second turn using direct inference:")
    blocks, tokens = context_mgr.get_conversation_blocks()
    print(f"   Conversation context: {len(blocks)} blocks, {tokens} tokens")
    
    # New user query
    user_message = "<|im_start|>user\nWhat was the password?<|im_end|>\n<|im_start|>assistant\n"
    new_tokens = tokenizer.encode(user_message, add_special_tokens=False)
    print(f"   New query: {len(new_tokens)} tokens")
    
    # Direct inference
    response2 = ""
    token_count = 0
    for token_data in llm.infer_from_blocks_stream(
        existing_blocks=blocks,
        existing_token_count=tokens,
        new_tokens=new_tokens,
        sampling_params=SamplingParams(max_tokens=30)
    ):
        if not token_data["finished"]:
            response2 += token_data["token"]
            token_count += 1
            if token_count <= 30:  # Only print first 30 tokens
                print(token_data["token"], end="", flush=True)
    
    print(f"\n\n4. Checking if model remembered the password:")
    if "ZEBRA456" in response2:
        print("   ✓ SUCCESS: Model correctly recalled the password!")
    else:
        print("   ✗ FAIL: Model did not recall the password")
        print(f"   Response preview: {response2[:100]}...")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()