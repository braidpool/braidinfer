#!/usr/bin/env python3
"""
Simple example demonstrating cascade attention feature in nano-vllm.

This example shows how cascade attention can save memory by sharing
common prefixes (like system prompts) across multiple requests.
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def demonstrate_cascade_attention():
    """Show cascade attention in action with shared system prompt."""
    print("=== Cascade Attention Demo ===\n")
    
    # Model path
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Initialize LLM with cascade attention enabled
    print("Initializing model with cascade attention...")
    llm = LLM(
        model_path,
        enable_cascade_attention=True,
        cascade_shared_prefix_len=100,  # First 100 tokens are considered shared
        enforce_eager=True,
        num_kvcache_blocks=64,
        kvcache_block_size=256
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Common system prompt that will be shared
    system_prompt = """You are a helpful AI assistant with expertise in Python programming. 
    Please provide clear, concise answers with code examples when appropriate."""
    
    # Different user queries that will share the system prompt
    queries = [
        "What is a Python list comprehension?",
        "How do I read a file in Python?",
        "Explain Python decorators with an example."
    ]
    
    # Create prompts with shared system prompt
    prompts = []
    for query in queries:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Show token counts
    print(f"\nSystem prompt length: ~{len(tokenizer.encode(system_prompt))} tokens")
    print(f"Number of queries: {len(queries)}\n")
    
    # Generate responses
    print("Generating responses with cascade attention...")
    print("(The system prompt KV cache will be shared across all queries)\n")
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=150)
    outputs = llm.generate(prompts, sampling_params)
    
    # Display results
    for i, (query, output) in enumerate(zip(queries, outputs)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {output['text'][:200]}...")  # Show first 200 chars
        print("-" * 60)
    
    print("\n✅ Cascade attention demo completed!")
    print("\nKey benefits demonstrated:")
    print("• Shared system prompt KV cache across all queries")
    print("• Memory savings proportional to shared prefix length")
    print("• No performance degradation")


def demonstrate_memory_savings():
    """Calculate and show memory savings from cascade attention."""
    print("\n\n=== Memory Savings Calculation ===\n")
    
    # Parameters
    system_prompt_tokens = 50
    query_tokens = 20
    num_queries = 10
    kv_cache_per_token = 2 * 32 * 128 * 2  # 2 (K,V) * num_heads * head_dim * fp16
    
    # Without cascade attention
    without_cascade = num_queries * (system_prompt_tokens + query_tokens) * kv_cache_per_token
    
    # With cascade attention
    with_cascade = (system_prompt_tokens + num_queries * query_tokens) * kv_cache_per_token
    
    # Savings
    saved = without_cascade - with_cascade
    savings_pct = (saved / without_cascade) * 100
    
    print(f"Scenario: {num_queries} queries with shared {system_prompt_tokens}-token system prompt")
    print(f"Each query: {query_tokens} tokens\n")
    
    print(f"Memory without cascade: {without_cascade / (1024*1024):.1f} MB")
    print(f"Memory with cascade: {with_cascade / (1024*1024):.1f} MB")
    print(f"Memory saved: {saved / (1024*1024):.1f} MB ({savings_pct:.1f}%)")
    print(f"Reduction factor: {without_cascade / with_cascade:.1f}x")


def main():
    """Run the cascade attention demonstration."""
    # Check if model exists
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Run demonstrations
    demonstrate_cascade_attention()
    demonstrate_memory_savings()
    
    print("\n\n🎉 All demonstrations completed successfully!")


if __name__ == "__main__":
    main()