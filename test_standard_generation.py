#!/usr/bin/env python3
"""
Test standard generation to ensure it works correctly.
"""

import os
from nanovllm import LLM, SamplingParams

def test_standard_generation():
    """Test standard generation without chunks."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    print("=" * 80)
    print("TESTING STANDARD GENERATION")
    print("=" * 80)
    
    # Initialize LLM
    llm = LLM(model_path, enforce_eager=True)
    
    # Test cases
    test_cases = [
        {
            "system": "You are a helpful AI assistant. Please respond in English.",
            "user": "What is the capital of France?"
        },
        {
            "system": "You are a math tutor. Explain concepts clearly.",
            "user": "What is 2 + 2?"
        },
        {
            "system": "You are a helpful assistant.",
            "user": "Hello, how are you?"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n\nTest {i+1}:")
        print(f"System: {test['system']}")
        print(f"User: {test['user']}")
        print("-" * 40)
        
        # Build prompt
        messages = [
            {"role": "system", "content": test["system"]},
            {"role": "user", "content": test["user"]}
        ]
        prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
        outputs = llm.generate([prompt], sampling_params)
        
        output_text = outputs[0]["text"]
        print(f"Output: {output_text}")
        
        # Check if output is reasonable
        if len(output_text) < 10:
            print("❌ Output too short!")
        elif any(ord(c) > 127 for c in output_text[:20]):  # Check for non-ASCII in first 20 chars
            print("❌ Output contains non-ASCII characters!")
        else:
            print("✅ Output looks reasonable")

if __name__ == "__main__":
    test_standard_generation()