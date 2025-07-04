#!/usr/bin/env python3
"""Benchmark chunked API with content deduplication."""

import time
from random import randint, seed
import torch

from nanovllm.chunked_llm import ChunkedLLM
from nanovllm.chunks import ChunkType
from common import (
    format_size, aggressive_cleanup, save_results, print_results,
    DEFAULT_CONFIG
)


def benchmark_chunked_api(model_path: str, num_seqs: int, max_input_len: int, max_output_len: int):
    """Benchmark chunked API with content deduplication."""
    print("=== Chunked API (Content-based Deduplication) ===")
    
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        chunk_memory_ratio=0.5,
        enable_deduplication=True,
        enforce_eager=True,
        max_model_len=4096
    )
    
    # Create shared system prompt and contexts
    system_prompt = "You are a helpful AI assistant. Answer questions accurately and concisely."
    
    # Create some shared context chunks that multiple queries will reference
    shared_contexts = [
        "Context A: The Earth is the third planet from the Sun and the only known planet to harbor life.",
        "Context B: Python is a high-level programming language known for its simplicity and readability.",
        "Context C: Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    ]
    
    # Register shared chunks
    system_chunk_id = llm.register_chunk(system_prompt, ChunkType.SYSTEM_PROMPT)
    context_chunk_ids = [llm.register_chunk(ctx, ChunkType.CONTEXT) for ctx in shared_contexts]
    
    # Create test queries with different context combinations
    requests = []
    
    for i in range(num_seqs):
        # Each query uses 0-2 shared contexts
        num_contexts = randint(0, min(2, len(context_chunk_ids)))
        selected_contexts = context_chunk_ids[:num_contexts] if num_contexts > 0 else None
        
        query_text = f"Question {i}: " + " ".join([f"word{randint(0, 1000)}" for _ in range(randint(20, 50))])
        query_chunk_id = llm.register_chunk(query_text, ChunkType.QUERY)
        
        requests.append({
            "system_chunk_id": system_chunk_id,
            "query_chunk_id": query_chunk_id,
            "context_chunk_ids": selected_contexts
        })
    
    sampling_params = {
        "temperature": 0.6,
        "ignore_eos": True,
        "max_tokens": max_output_len
    }
    
    # Warmup
    llm.generate(system_prompt, "Warmup query", sampling_params={"max_tokens": 10})
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()
    
    outputs = llm.batch_generate_from_chunks(requests, sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    
    # Calculate metrics
    total_time = end_time - start_time
    memory_used = end_memory - start_memory
    total_output_tokens = len(outputs) * sampling_params["max_tokens"]
    throughput = total_output_tokens / total_time
    
    # Get chunk statistics
    stats = llm.get_chunk_stats()
    
    results = {
        "method": "Chunked API",
        "num_sequences": num_seqs,
        "num_unique_chunks": stats["total_chunks"],
        "cache_hits": stats.get("cache_hits", 0),
        "total_output_tokens": total_output_tokens,
        "time_seconds": total_time,
        "throughput_tokens_per_sec": throughput,
        "memory_used_bytes": memory_used,
        "memory_used_formatted": format_size(memory_used),
        "chunk_memory_mb": stats.get("memory_used_mb", 0)
    }
    
    del llm
    aggressive_cleanup()
    
    return results


def main():
    """Run the benchmark."""
    config = DEFAULT_CONFIG
    seed(config["seed"])
    
    print(f"Model: {config['model_path']}")
    print(f"Batch sizes: {config['batch_sizes']}")
    print(f"Max input length: {config['max_input_len']}")
    print(f"Max output length: {config['max_output_len']}")
    
    all_results = []
    
    for batch_size in config["batch_sizes"]:
        print(f"\n--- Testing batch size: {batch_size} ---")
        results = benchmark_chunked_api(
            config["model_path"],
            batch_size,
            config["max_input_len"],
            config["max_output_len"]
        )
        results["batch_size"] = batch_size
        all_results.append(results)
        print_results(results)
        
        # Cleanup between batch sizes
        aggressive_cleanup()
    
    save_results(all_results, "results_chunked.json")


if __name__ == "__main__":
    main()