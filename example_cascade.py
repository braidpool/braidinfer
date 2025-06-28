#!/usr/bin/env python3
"""
Simple example of cascade attention showing deduplication and memory savings.
"""

import os
from transformers import AutoTokenizer

from nanovllm.engine.context_chunks import ContextChunk, ChunkType, ChunkBuilder
from nanovllm.engine.chunk_registry import ChunkRegistry


def demo_chunk_composition():
    """Demonstrate creating and composing context chunks."""
    print("=== Chunk Composition Demo ===\n")
    
    # Initialize tokenizer
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Create chunk registry
    registry = ChunkRegistry(max_chunks=100)
    
    # Register a system prompt
    system_prompt = """You are an expert Python programmer who specializes in 
    performance optimization and clean code practices."""
    
    system_chunk = registry.register(
        system_prompt,
        ChunkType.SYSTEM_PROMPT,
        tokenizer=tokenizer
    )
    print(f"System chunk registered: {system_chunk.chunk_id[:8]}... ({system_chunk.seq_len} tokens)")
    
    # Register some code context
    code_context = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    code_chunk = registry.register(
        code_context,
        ChunkType.CODE,
        tokenizer=tokenizer
    )
    print(f"Code chunk registered: {code_chunk.chunk_id[:8]}... ({code_chunk.seq_len} tokens)")
    
    # Register documentation context
    doc_context = """
The Fibonacci sequence is a series of numbers where each number is the sum 
of the two preceding ones. The sequence starts with 0 and 1. This recursive 
implementation has exponential time complexity O(2^n).
"""
    
    doc_chunk = registry.register(
        doc_context,
        ChunkType.CONTEXT,
        tokenizer=tokenizer
    )
    print(f"Doc chunk registered: {doc_chunk.chunk_id[:8]}... ({doc_chunk.seq_len} tokens)")
    
    # Test deduplication
    duplicate_system = registry.register(
        system_prompt,
        ChunkType.SYSTEM_PROMPT,
        tokenizer=tokenizer
    )
    print(f"\nDuplicate detection: {duplicate_system.chunk_id == system_chunk.chunk_id}")
    print(f"Access count: {duplicate_system.access_count}")
    
    # Show registry statistics
    stats = registry.get_stats()
    print(f"\nRegistry stats: {stats}")
    
    return registry, system_chunk, code_chunk, doc_chunk


def demo_composition_patterns():
    """Demonstrate different composition patterns."""
    print("\n=== Composition Patterns ===\n")
    
    # Get registry from previous demo
    registry = ChunkRegistry(max_chunks=100)
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Create different types of chunks
    chunks = {
        "system": registry.register(
            "You are a helpful coding assistant.",
            ChunkType.SYSTEM_PROMPT,
            tokenizer=tokenizer
        ),
        "code": registry.register(
            "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
            ChunkType.CODE,
            tokenizer=tokenizer
        ),
        "docs": registry.register(
            "Factorial is the product of all positive integers less than or equal to n.",
            ChunkType.CONTEXT,
            tokenizer=tokenizer
        )
    }
    
    # Show different composition patterns
    builder = ChunkBuilder()
    
    patterns = [
        ("Basic", ["system", "code"]),
        ("With docs", ["system", "code", "docs"]),
        ("Duplicate", ["system", "code", "code"]),  # Same code twice
        ("Reordered", ["system", "docs", "code"])
    ]
    
    for name, chunk_names in patterns:
        selected_chunks = [chunks[name].content for name in chunk_names]
        composition = builder.build_composition(
            system_prompt=selected_chunks[0] if chunk_names[0] == "system" else None,
            context_chunks=selected_chunks[1:] if len(selected_chunks) > 1 else [],
            query="Explain this code"
        )
        
        levels = composition.get_cascade_levels()
        print(f"{name} pattern: {len(levels)} levels, {len(composition.chunks)} total chunks")
    
    return registry


def demo_memory_savings():
    """Demonstrate memory savings with cascade attention."""
    print("\n=== Memory Savings Demo ===\n")
    
    # Initialize
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    registry = ChunkRegistry(max_chunks=100)
    
    # Simulate a common scenario: multiple queries with shared context
    system_prompt = "You are an AI assistant specialized in Python programming."
    
    # Shared documentation context
    python_docs = """
    Python is a high-level programming language known for its simplicity.
    It supports multiple programming paradigms including procedural, 
    object-oriented, and functional programming.
    """
    
    # Different user queries
    queries = [
        "What is Python?",
        "Is Python object-oriented?",
        "What programming paradigms does Python support?",
        "Why is Python popular?"
    ]
    
    # Register shared chunks
    system_chunk = registry.register(system_prompt, ChunkType.SYSTEM_PROMPT, tokenizer)
    docs_chunk = registry.register(python_docs, ChunkType.CONTEXT, tokenizer)
    
    print(f"Shared system prompt: {system_chunk.seq_len} tokens")
    print(f"Shared documentation: {docs_chunk.seq_len} tokens")
    
    # Calculate tokens for each query
    total_without_cascade = 0
    total_with_cascade = system_chunk.seq_len + docs_chunk.seq_len
    
    for query in queries:
        query_tokens = len(tokenizer.encode(query))
        print(f"Query '{query}': {query_tokens} tokens")
        
        # Without cascade: each request stores everything
        total_without_cascade += system_chunk.seq_len + docs_chunk.seq_len + query_tokens
        
        # With cascade: only unique query tokens
        total_with_cascade += query_tokens
    
    print(f"\nWithout cascade attention: {total_without_cascade} total tokens")
    print(f"With cascade attention: {total_with_cascade} total tokens")
    print(f"Memory savings: {(1 - total_with_cascade/total_without_cascade)*100:.1f}%")
    print(f"Reduction factor: {total_without_cascade/total_with_cascade:.1f}x")


def main():
    """Run cascade attention demos."""
    print("Cascade Attention Demo\n")
    print("Demonstrating memory efficiency through chunk deduplication\n")
    
    # Run demos
    demo_chunk_composition()
    demo_composition_patterns() 
    demo_memory_savings()
    
    print("\n✅ Demo completed!")
    print("\nKey benefits:")
    print("• Content-based deduplication")
    print("• Shared context reuse")
    print("• Significant memory savings")
    print("• Multi-head attention support")


if __name__ == "__main__":
    main()