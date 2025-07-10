#!/usr/bin/env python3
"""
Simple example of using the ChunkedLLM API.
"""

import os
from braidinfer import ChunkedLLM, ChunkType


def main():
    """Demonstrate the chunked API."""
    print("=== ChunkedLLM API Demo ===\n")
    
    # Initialize
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = ChunkedLLM(
        model_path,
        max_chunks=100,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # 1. Register reusable chunks
    print("1. Registering chunks...")
    
    # System prompts for different agents
    analyst_id = llm.register_chunk(
        "You are a data analyst. Provide insights based on the given data.",
        ChunkType.SYSTEM_PROMPT,
        metadata={"agent": "analyst"}
    )
    
    coder_id = llm.register_chunk(
        "You are an expert programmer. Write clean, efficient code.",
        ChunkType.SYSTEM_PROMPT,
        metadata={"agent": "coder"}
    )
    
    # Shared context
    data_context_id = llm.register_chunk(
        """Sales Data:
        - Q1: $1.2M
        - Q2: $1.5M  
        - Q3: $1.1M
        - Q4: $1.8M""",
        ChunkType.CONTEXT,
        metadata={"type": "sales_data"}
    )
    
    print(f"  Analyst system: {analyst_id[:8]}...")
    print(f"  Coder system: {coder_id[:8]}...")
    print(f"  Data context: {data_context_id[:8]}...")
    
    # 2. Different agents analyze same data
    print("\n2. Multiple agents analyzing same data...\n")
    
    # Analyst perspective
    analyst_query_id = llm.register_chunk(
        "What trends do you see in this sales data?",
        ChunkType.QUERY
    )
    
    analyst_output = llm.generate_from_chunks(
        system_chunk_id=analyst_id,
        context_chunk_ids=[data_context_id],
        query_chunk_id=analyst_query_id,
        sampling_params={"temperature": 0.7, "max_tokens": 100}
    )
    
    print("Analyst's response:")
    print(analyst_output['text'][:200] + "...\n")
    
    # Coder perspective  
    coder_query_id = llm.register_chunk(
        "Write a Python function to calculate quarter-over-quarter growth.",
        ChunkType.QUERY
    )
    
    coder_output = llm.generate_from_chunks(
        system_chunk_id=coder_id,
        context_chunk_ids=[data_context_id],
        query_chunk_id=coder_query_id,
        sampling_params={"temperature": 0.7, "max_tokens": 150}
    )
    
    print("Coder's response:")
    print(coder_output['text'][:200] + "...\n")
    
    # 3. Show efficiency
    stats = llm.get_chunk_stats()
    print(f"3. Efficiency stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"\nThe data context chunk was reused across both requests!")
    
    # 4. Convenience method
    print("\n4. Using convenience method...")
    
    output = llm.generate(
        system_prompt="You are a helpful assistant.",
        context=["Python is a versatile language.", "It's used for web, data science, and more."],
        query="What can I build with Python?",
        sampling_params={"temperature": 0.7, "max_tokens": 100},
        persist_chunks=False  # Don't keep these chunks
    )
    
    print(f"Response: {output['text'][:150]}...")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
    else:
        main()