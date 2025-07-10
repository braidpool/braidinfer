#!/usr/bin/env python3
"""
Example demonstrating cascade attention using the ChunkedLLM API.

This example shows how the ChunkedLLM API automatically enables cascade
attention by sharing common chunks (like system prompts) across multiple
requests, providing both memory savings and performance benefits.
"""

import os
from braidinfer import ChunkedLLM, ChunkType


def demonstrate_cascade_with_chunks():
    """Show cascade attention in action using ChunkedLLM API."""
    print("=== Cascade Attention with ChunkedLLM API ===\n")
    
    # Model path
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Initialize ChunkedLLM (cascade attention is automatic)
    print("Initializing ChunkedLLM (cascade attention enabled automatically)...")
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        chunk_memory_ratio=0.5,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Common system prompt that will be shared across all queries
    system_prompt = """You are a helpful AI assistant with expertise in Python programming. 
Please provide clear, concise answers with code examples when appropriate."""
    
    # Register the system prompt once
    system_chunk_id = llm.register_chunk(
        system_prompt,
        ChunkType.SYSTEM_PROMPT,
        metadata={"description": "Python expert system prompt"}
    )
    print(f"System prompt registered as chunk: {system_chunk_id[:8]}...")
    
    # Different user queries that will share the system prompt
    queries = [
        "What is a Python list comprehension?",
        "How do I read a file in Python?", 
        "Explain Python decorators with an example."
    ]
    
    # Process each query
    print(f"\nProcessing {len(queries)} queries with shared system prompt...\n")
    
    results = []
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}")
        
        # Register query as a chunk
        query_chunk_id = llm.register_chunk(
            query,
            ChunkType.QUERY,
            metadata={"query_index": i}
        )
        
        # Generate response using chunks
        output = llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            query_chunk_id=query_chunk_id,
            sampling_params={"temperature": 0.7, "max_tokens": 150}
        )
        
        # Store result
        results.append({
            "query": query,
            "response": output['text'][:200] + "..." if len(output['text']) > 200 else output['text']
        })
        
        print(f"Response: {results[-1]['response']}")
        print("-" * 60)
    
    # Show chunk statistics
    stats = llm.get_chunk_stats()
    print(f"\n✅ Cascade attention demonstration completed!")
    print(f"\nChunk Statistics:")
    print(f"• Total chunks: {stats['total_chunks']}")
    print(f"• Cache hits: {stats['cache_hits']}")
    print(f"• Hit rate: {stats['hit_rate']:.1%}")
    print(f"• Memory used: {stats['memory_used_mb']:.1f} MB")
    
    print("\nKey benefits demonstrated:")
    print("• System prompt chunk shared across all queries (registered once)")
    print("• Automatic cascade attention for shared chunks")
    print("• Memory savings from chunk deduplication")
    print("• Cache hits show chunk reuse efficiency")


def demonstrate_multi_level_cascade():
    """Demonstrate multi-level cascade with system + context + query."""
    print("\n\n=== Multi-Level Cascade Attention ===\n")
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # Initialize ChunkedLLM
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Level 0: System prompt (shared across all)
    system_chunk_id = llm.register_chunk(
        "You are an AI assistant helping with document analysis.",
        ChunkType.SYSTEM_PROMPT
    )
    
    # Level 1: Document context (shared across related queries)
    document = """Python is a high-level programming language known for its simplicity 
and readability. It supports multiple programming paradigms including procedural, 
object-oriented, and functional programming. Python's extensive standard library 
and vast ecosystem of third-party packages make it suitable for various applications."""
    
    context_chunk_id = llm.register_chunk(
        document,
        ChunkType.CONTEXT,
        metadata={"document": "Python overview"}
    )
    
    # Level 2: Different queries about the same document
    queries = [
        "What programming paradigms does Python support?",
        "What makes Python suitable for various applications?",
        "Summarize the key points about Python from the document."
    ]
    
    print("Processing queries with 3-level cascade:")
    print(f"• Level 0: System prompt (shared)")
    print(f"• Level 1: Document context (shared)")
    print(f"• Level 2: Individual queries\n")
    
    for i, query in enumerate(queries):
        query_chunk_id = llm.register_chunk(query, ChunkType.QUERY)
        
        # Generate with multi-level cascade
        output = llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            context_chunk_ids=[context_chunk_id],
            query_chunk_id=query_chunk_id,
            sampling_params={"temperature": 0.5, "max_tokens": 100}
        )
        
        print(f"Q{i+1}: {query}")
        print(f"A{i+1}: {output['text'][:150]}...")
        print()
    
    # Final statistics
    stats = llm.get_chunk_stats()
    print(f"Final statistics:")
    print(f"• Total unique chunks: {stats['total_chunks']}")
    print(f"• System chunk reused: {len(queries)} times")
    print(f"• Context chunk reused: {len(queries)} times")
    print(f"• Cascade levels: 3 (system → context → query)")


def calculate_memory_savings():
    """Calculate and display memory savings from cascade attention."""
    print("\n\n=== Memory Savings Calculation ===\n")
    
    # Scenario parameters
    batch_size = 10
    system_tokens = 50
    context_tokens = 200
    query_tokens = 20
    output_tokens = 100
    
    # KV cache size estimation (simplified)
    # Assuming: 32 layers, 32 heads, 128 head_dim, fp16
    bytes_per_token = 32 * 2 * 32 * 128 * 2  # layers * (K,V) * heads * dim * fp16
    
    print(f"Scenario: {batch_size} requests with shared system prompt and context")
    print(f"• System prompt: {system_tokens} tokens")
    print(f"• Shared context: {context_tokens} tokens")
    print(f"• Each query: {query_tokens} tokens")
    print(f"• Each output: {output_tokens} tokens\n")
    
    # Without cascade (duplicate everything)
    without_cascade_tokens = batch_size * (system_tokens + context_tokens + query_tokens + output_tokens)
    without_cascade_memory = without_cascade_tokens * bytes_per_token
    
    # With cascade (share system and context)
    with_cascade_tokens = (system_tokens + context_tokens) + batch_size * (query_tokens + output_tokens)
    with_cascade_memory = with_cascade_tokens * bytes_per_token
    
    # Savings
    saved_tokens = without_cascade_tokens - with_cascade_tokens
    saved_memory = without_cascade_memory - with_cascade_memory
    savings_percent = (saved_memory / without_cascade_memory) * 100
    
    print(f"Without cascade attention:")
    print(f"• Total tokens: {without_cascade_tokens:,}")
    print(f"• Memory usage: {without_cascade_memory / (1024**2):.1f} MB\n")
    
    print(f"With cascade attention:")
    print(f"• Total tokens: {with_cascade_tokens:,}")
    print(f"• Memory usage: {with_cascade_memory / (1024**2):.1f} MB\n")
    
    print(f"Savings:")
    print(f"• Tokens saved: {saved_tokens:,}")
    print(f"• Memory saved: {saved_memory / (1024**2):.1f} MB")
    print(f"• Reduction: {savings_percent:.1f}%")
    print(f"• Efficiency factor: {without_cascade_memory / with_cascade_memory:.2f}x")


def main():
    """Run all cascade attention demonstrations."""
    # Check if model exists
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Run demonstrations
    demonstrate_cascade_with_chunks()
    demonstrate_multi_level_cascade()
    calculate_memory_savings()
    
    print("\n\n🎉 All cascade attention demonstrations completed!")
    print("\nKey takeaways:")
    print("1. ChunkedLLM automatically enables cascade attention for shared chunks")
    print("2. System prompts and contexts can be registered once and reused")
    print("3. Significant memory savings with minimal code changes")
    print("4. Cache statistics show the efficiency of chunk reuse")


if __name__ == "__main__":
    main()