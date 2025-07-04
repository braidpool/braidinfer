#!/usr/bin/env python3
"""
Basic usage example for nano-vllm.

This example shows the simplest way to use nano-vllm for text generation.
"""

import os
from nanovllm import LLM, SamplingParams


def main():
    """Basic text generation example."""
    print("=== Basic nano-vllm Usage ===\n")
    
    # Model path - adjust this to your model location
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Initialize the model
    print("Loading model...")
    llm = LLM(
        model_path,
        max_model_len=2048,      # Maximum sequence length
        enforce_eager=True,       # Use eager mode (no CUDA graphs)
        num_kvcache_blocks=128    # Number of KV cache blocks
    )
    
    # Example 1: Simple generation
    print("\n1. Simple text generation:")
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"   Prompt: {prompts[0]}")
        print(f"   Output: {output['text']}")
    
    # Example 2: Multiple prompts
    print("\n2. Batch generation with multiple prompts:")
    prompts = [
        "Python is a",
        "Machine learning is",
        "The weather today is"
    ]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    
    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print(f"   Prompt: {prompt}")
        print(f"   Output: {output['text']}")
        print()
    
    # Example 3: Different sampling parameters
    print("3. Generation with different parameters:")
    prompt = "Once upon a time"
    
    # Deterministic (temperature=0)
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=50))
    print(f"   Deterministic: {outputs[0]['text'][:100]}...")
    
    # Creative (high temperature)
    outputs = llm.generate([prompt], SamplingParams(temperature=1.0, max_tokens=50))
    print(f"   Creative: {outputs[0]['text'][:100]}...")
    
    # Example 4: Chat format (if using a chat model)
    print("\n4. Chat format example:")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the speed of light?"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=100))
    print(f"   User: {messages[1]['content']}")
    print(f"   Assistant: {outputs[0]['text']}")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()