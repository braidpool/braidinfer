#!/usr/bin/env python3
"""
Example demonstrating conversation chunk reuse with ChunkedLLM.

This example shows how conversation history can be efficiently reused
across multiple conversation branches, demonstrating significant memory
savings and cache hit improvements.
"""

import os
import time
from braidinfer import ChunkedLLM, ChunkType


def simulate_conversation_branches():
    """Simulate multiple conversation branches sharing common history."""
    print("=== Conversation Chunk Reuse Demo ===\n")
    
    # Initialize ChunkedLLM
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # System prompt
    system_chunk_id = llm.register_chunk(
        "You are a helpful AI assistant engaged in a technical discussion.",
        ChunkType.SYSTEM_PROMPT
    )
    
    # Build initial conversation
    print("Building initial conversation...\n")
    
    conversation_chunks = []
    
    # First exchange
    user1_id = llm.register_chunk(
        "Can you explain what Python decorators are?",
        ChunkType.CONTEXT,
        metadata={"role": "user", "turn": 1}
    )
    conversation_chunks.append(user1_id)
    
    assistant1_id = llm.register_chunk(
        "Python decorators are a way to modify or enhance functions without changing their code. They use the @ syntax and wrap functions with additional functionality.",
        ChunkType.CONTEXT,
        metadata={"role": "assistant", "turn": 1}
    )
    conversation_chunks.append(assistant1_id)
    
    # Second exchange
    user2_id = llm.register_chunk(
        "Can you show me a simple example?",
        ChunkType.CONTEXT,
        metadata={"role": "user", "turn": 2}
    )
    conversation_chunks.append(user2_id)
    
    assistant2_id = llm.register_chunk(
        """Here's a simple decorator example:

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer_decorator
def slow_function():
    time.sleep(1)
    return "Done"
```""",
        ChunkType.CONTEXT,
        metadata={"role": "assistant", "turn": 2}
    )
    conversation_chunks.append(assistant2_id)
    
    print("Initial conversation established with 4 chunks")
    print(f"Chunks registered: {[c[:8] + '...' for c in conversation_chunks]}\n")
    
    # Now simulate different conversation branches
    print("Simulating 3 different conversation branches from the same history...\n")
    
    branches = [
        {
            "name": "Branch A: Diving deeper into decorators",
            "query": "How do decorators with arguments work?"
        },
        {
            "name": "Branch B: Switching to performance",
            "query": "Are decorators expensive performance-wise?"
        },
        {
            "name": "Branch C: Best practices",
            "query": "What are some best practices when using decorators?"
        }
    ]
    
    # Get initial stats
    initial_stats = llm.get_chunk_stats()
    
    for i, branch in enumerate(branches):
        print(f"\n{branch['name']}")
        print("-" * 50)
        
        # Register the branch-specific query
        query_id = llm.register_chunk(
            branch['query'],
            ChunkType.QUERY,
            metadata={"branch": i, "query": branch['query']}
        )
        
        # Generate response using shared conversation history
        start_time = time.time()
        output = llm.generate_from_chunks(
            system_chunk_id=system_chunk_id,
            context_chunk_ids=conversation_chunks,  # Reuse same conversation!
            query_chunk_id=query_id,
            sampling_params={"temperature": 0.7, "max_tokens": 150}
        )
        gen_time = time.time() - start_time
        
        print(f"Query: {branch['query']}")
        print(f"Response: {output['text'][:200]}...")
        print(f"Generation time: {gen_time:.2f}s")
        
        # Show cache efficiency
        current_stats = llm.get_chunk_stats()
        new_hits = current_stats['cache_hits'] - initial_stats['cache_hits']
        print(f"Cache hits for this branch: {new_hits}")
    
    # Final statistics
    print("\n\n=== Final Statistics ===")
    final_stats = llm.get_chunk_stats()
    
    print(f"Total unique chunks: {final_stats['total_chunks']}")
    print(f"Total cache hits: {final_stats['cache_hits']}")
    print(f"Hit rate: {final_stats['hit_rate']:.1%}")
    print(f"Memory used: {final_stats['memory_used_mb']:.1f} MB")
    
    # Calculate savings
    chunks_without_reuse = 1 + (4 * 4) + 3  # system + (conv * branches) + queries
    chunks_with_reuse = final_stats['total_chunks']
    savings = chunks_without_reuse - chunks_with_reuse
    
    print(f"\nMemory efficiency:")
    print(f"• Chunks without reuse: {chunks_without_reuse}")
    print(f"• Chunks with reuse: {chunks_with_reuse}")
    print(f"• Chunks saved: {savings} ({savings/chunks_without_reuse*100:.0f}% reduction)")


def demonstrate_conversation_continuation():
    """Show how conversations can be continued efficiently."""
    print("\n\n=== Conversation Continuation Demo ===\n")
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # System prompt
    system_id = llm.register_chunk(
        "You are a helpful coding assistant.",
        ChunkType.SYSTEM_PROMPT
    )
    
    # Simulate a conversation that gets interrupted and resumed
    print("Starting conversation...")
    
    conversation = []
    
    # First session
    exchanges = [
        ("What's the difference between a list and tuple in Python?",
         "Lists are mutable (can be changed) while tuples are immutable. Lists use square brackets [], tuples use parentheses ()."),
        ("Which one is faster?",
         "Tuples are generally faster than lists for iteration and access because they're immutable and Python can optimize them better.")
    ]
    
    for user_msg, assistant_msg in exchanges:
        # User turn
        user_id = llm.register_chunk(user_msg, ChunkType.CONTEXT)
        conversation.append(user_id)
        
        # Assistant turn (simulate previous responses)
        assistant_id = llm.register_chunk(assistant_msg, ChunkType.CONTEXT)
        conversation.append(assistant_id)
    
    print(f"Session 1: {len(conversation)} chunks registered")
    
    # Simulate saving conversation state
    saved_conversation = conversation.copy()
    saved_system = system_id
    
    print("\n[Simulating session end and restart...]\n")
    
    # New session - reload conversation
    print("Resuming conversation with saved chunks...")
    
    # Continue the conversation
    new_query = "Can you show me how to convert between them?"
    query_id = llm.register_chunk(new_query, ChunkType.QUERY)
    
    # Generate response using saved conversation
    stats_before = llm.get_chunk_stats()
    
    output = llm.generate_from_chunks(
        system_chunk_id=saved_system,
        context_chunk_ids=saved_conversation,
        query_chunk_id=query_id,
        sampling_params={"temperature": 0.7, "max_tokens": 200}
    )
    
    stats_after = llm.get_chunk_stats()
    
    print(f"Continuing with: {new_query}")
    print(f"Response: {output['text'][:300]}...")
    print(f"\nChunks reused from previous session: {len(saved_conversation) + 1}")
    print(f"Cache hits: {stats_after['cache_hits'] - stats_before['cache_hits']}")
    
    print("\n✅ Conversation successfully continued with full context!")


def main():
    """Run conversation reuse demonstrations."""
    # Check if model exists
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Run demonstrations
    simulate_conversation_branches()
    demonstrate_conversation_continuation()
    
    print("\n\n🎉 Conversation reuse demonstration completed!")
    print("\nKey takeaways:")
    print("1. Conversation history can be efficiently reused across branches")
    print("2. Common conversation prefixes are stored only once")
    print("3. Conversations can be saved and resumed with full context")
    print("4. Significant memory savings for multi-turn conversations")


if __name__ == "__main__":
    main()