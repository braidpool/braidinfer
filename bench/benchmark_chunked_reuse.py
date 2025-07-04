#!/usr/bin/env python3
"""Benchmark chunked API with chunk reuse across batches."""

import time
from random import randint, seed
import torch

from nanovllm.chunked_llm import ChunkedLLM
from nanovllm.chunks import ChunkType
from common import (
    format_size, aggressive_cleanup, save_results, print_results,
    DEFAULT_CONFIG
)


def benchmark_chunked_with_reuse(model_path: str, num_batches: int, batch_size: int, max_output_len: int):
    """Benchmark chunked API with chunk reuse across batches."""
    print("=== Chunked API (Cross-batch Chunk Reuse) ===")
    
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        chunk_memory_ratio=0.7,
        enable_deduplication=True,
        enforce_eager=True,
        max_model_len=4096
    )
    
    # Create a pool of reusable chunks
    system_prompts = [
        "You are a helpful AI assistant.",
        "You are an expert in science and technology.",
        "You are a creative writing assistant."
    ]
    
    contexts = [
        "The solar system consists of the Sun and eight planets.",
        "Machine learning models can be trained on various datasets.",
        "Natural language processing enables computers to understand human language.",
        "Climate change is one of the most pressing issues of our time.",
        "Quantum computing promises to revolutionize computation."
    ]
    
    # Register all chunks
    system_chunk_ids = [llm.register_chunk(sp, ChunkType.SYSTEM_PROMPT) for sp in system_prompts]
    context_chunk_ids = [llm.register_chunk(ctx, ChunkType.CONTEXT) for ctx in contexts]
    
    total_time = 0
    total_output_tokens = 0
    all_stats = []
    
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()
    
    # Run multiple batches
    for batch_idx in range(num_batches):
        requests = []
        
        for i in range(batch_size):
            # Randomly select system prompt and contexts
            system_id = system_chunk_ids[randint(0, len(system_chunk_ids) - 1)]
            num_contexts = randint(0, 3)
            selected_contexts = [context_chunk_ids[randint(0, len(context_chunk_ids) - 1)] 
                               for _ in range(num_contexts)]
            
            # Create unique query for this request
            query_text = (f"Batch {batch_idx}, Query {i}: What is " + 
                         " ".join([f"topic{randint(0, 100)}" for _ in range(5)]) + "?")
            query_id = llm.register_chunk(query_text, ChunkType.QUERY)
            
            requests.append({
                "system_chunk_id": system_id,
                "query_chunk_id": query_id,
                "context_chunk_ids": selected_contexts if selected_contexts else None
            })
        
        sampling_params = {
            "temperature": 0.6,
            "ignore_eos": True,
            "max_tokens": max_output_len
        }
        
        # Time this batch
        torch.cuda.synchronize()
        batch_start = time.time()
        
        outputs = llm.batch_generate_from_chunks(requests, sampling_params)
        
        torch.cuda.synchronize()
        batch_time = time.time() - batch_start
        
        total_time += batch_time
        total_output_tokens += len(outputs) * sampling_params["max_tokens"]
        
        # Collect stats
        stats = llm.get_chunk_stats()
        all_stats.append(stats)
        
        print(f"  Batch {batch_idx + 1}: {batch_time:.2f}s, Cache hits: {stats.get('cache_hits', 0)}")
    
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    memory_used = end_memory - start_memory
    
    # Calculate aggregate metrics
    throughput = total_output_tokens / total_time
    final_stats = llm.get_chunk_stats()
    
    results = {
        "method": "Chunked API (Cross-batch Reuse)",
        "num_batches": num_batches,
        "batch_size": batch_size,
        "total_sequences": num_batches * batch_size,
        "num_unique_chunks": final_stats["total_chunks"],
        "total_cache_hits": final_stats.get("cache_hits", 0),
        "total_output_tokens": total_output_tokens,
        "time_seconds": total_time,
        "throughput_tokens_per_sec": throughput,
        "memory_used_bytes": memory_used,
        "memory_used_formatted": format_size(memory_used),
        "chunk_memory_mb": final_stats.get("memory_used_mb", 0),
        "batch_stats": all_stats
    }
    
    del llm
    aggressive_cleanup()
    
    return results


def main():
    """Run the benchmark."""
    config = DEFAULT_CONFIG
    seed(config["seed"])
    
    # Configuration for cross-batch reuse
    num_batches = 5
    
    print(f"Model: {config['model_path']}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch sizes: {config['batch_sizes']}")
    print(f"Max output length: {config['max_output_len']}")
    
    all_results = []
    
    for batch_size in config["batch_sizes"]:
        print(f"\n--- Testing batch size: {batch_size} ---")
        print(f"Total sequences: {num_batches * batch_size}")
        
        results = benchmark_chunked_with_reuse(
            config["model_path"],
            num_batches,
            batch_size,
            config["max_output_len"]
        )
        
        print(f"\nTotal time: {results['time_seconds']:.2f}s")
        print(f"Overall throughput: {results['throughput_tokens_per_sec']:.2f} tokens/s")
        print(f"Memory used: {results['memory_used_formatted']}")
        print(f"Total unique chunks: {results['num_unique_chunks']}")
        print(f"Total cache hits: {results['total_cache_hits']}")
        
        all_results.append(results)
        
        # Cleanup between batch sizes
        aggressive_cleanup()
    
    save_results(all_results, "results_chunked_reuse.json")


if __name__ == "__main__":
    main()