import os
import time
import json
import gc
from random import randint, seed
from typing import List, Dict, Any
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.chunked_llm import ChunkedLLM
from nanovllm.chunks import ChunkType

# Set environment variable to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def format_size(bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def aggressive_cleanup():
    """Aggressively clean up GPU memory between benchmarks."""
    # Clear all caches
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Clear cache again
    torch.cuda.empty_cache()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Give OS time to reclaim memory
    time.sleep(3)


def benchmark_standard_kv_cache(model_path: str, num_seqs: int, max_input_len: int, max_output_len: int) -> Dict[str, Any]:
    """Benchmark standard KV cache (no sharing)."""
    print("\n=== Standard KV Cache (No Sharing) ===")
    
    llm = LLM(model_path, enforce_eager=True, max_model_len=4096)
    
    # Generate test sequences
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    
    # Use same sampling params for all sequences for fair comparison
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_output_len)
    
    # Warmup
    llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=10))
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()
    
    outputs = llm.generate(prompt_token_ids, sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    
    # Calculate metrics
    total_time = end_time - start_time
    memory_used = end_memory - start_memory
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = len(outputs) * sampling_params.max_tokens
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / total_time
    
    results = {
        "method": "Standard KV Cache",
        "num_sequences": num_seqs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "time_seconds": total_time,
        "throughput_tokens_per_sec": throughput,
        "memory_used_bytes": memory_used,
        "memory_used_formatted": format_size(memory_used)
    }
    
    print(f"Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Memory used: {format_size(memory_used)}")
    
    del llm
    aggressive_cleanup()
    
    return results


def benchmark_cascade_prefix_sharing(model_path: str, num_seqs: int, max_input_len: int, max_output_len: int, shared_prefix_len: int) -> Dict[str, Any]:
    """Benchmark cascade attention with prefix sharing."""
    print(f"\n=== Cascade Attention (Shared Prefix: {shared_prefix_len} tokens) ===")
    
    llm = LLM(
        model_path,
        enable_cascade_attention=True,
        cascade_shared_prefix_len=shared_prefix_len,
        enforce_eager=True,
        max_model_len=4096
    )
    
    # Generate test sequences with shared prefix
    shared_prefix = [randint(0, 10000) for _ in range(shared_prefix_len)]
    prompt_token_ids = [shared_prefix + [randint(0, 10000) for _ in range(randint(50, max_input_len - shared_prefix_len))] for _ in range(num_seqs)]
    
    # Use same sampling params for all sequences
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_output_len)
    
    # Warmup
    llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=10))
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()
    
    outputs = llm.generate(prompt_token_ids, sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    
    # Calculate metrics
    total_time = end_time - start_time
    memory_used = end_memory - start_memory
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = len(outputs) * sampling_params.max_tokens
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / total_time
    
    # Calculate memory savings
    unique_tokens_per_seq = total_input_tokens - (shared_prefix_len * num_seqs) + shared_prefix_len
    savings_ratio = 1 - (unique_tokens_per_seq / total_input_tokens)
    
    results = {
        "method": "Cascade Attention (Prefix Sharing)",
        "shared_prefix_len": shared_prefix_len,
        "num_sequences": num_seqs,
        "total_input_tokens": total_input_tokens,
        "unique_input_tokens": unique_tokens_per_seq,
        "memory_savings_ratio": savings_ratio,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "time_seconds": total_time,
        "throughput_tokens_per_sec": throughput,
        "memory_used_bytes": memory_used,
        "memory_used_formatted": format_size(memory_used)
    }
    
    print(f"Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Memory used: {format_size(memory_used)}")
    print(f"Memory savings: {savings_ratio*100:.1f}% (shared {shared_prefix_len} tokens across {num_seqs} sequences)")
    
    del llm
    aggressive_cleanup()
    
    return results


def benchmark_chunked_api(model_path: str, num_seqs: int, max_input_len: int, max_output_len: int) -> Dict[str, Any]:
    """Benchmark chunked API with content deduplication."""
    print("\n=== Chunked API (Content-based Deduplication) ===")
    
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
    queries = []
    query_chunk_ids = []
    requests = []
    
    for i in range(num_seqs):
        # Each query uses 0-2 shared contexts
        num_contexts = randint(0, min(2, len(context_chunk_ids)))
        selected_contexts = context_chunk_ids[:num_contexts] if num_contexts > 0 else None
        
        query_text = f"Question {i}: " + " ".join([f"word{randint(0, 1000)}" for _ in range(randint(20, 50))])
        query_chunk_id = llm.register_chunk(query_text, ChunkType.QUERY)
        query_chunk_ids.append(query_chunk_id)
        
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
    total_output_tokens = len(outputs) * sampling_params["max_tokens"]  # Approximate
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
    
    print(f"Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Memory used: {format_size(memory_used)}")
    print(f"Unique chunks: {stats['total_chunks']}, Cache hits: {stats.get('cache_hits', 0)}")
    
    del llm
    aggressive_cleanup()
    
    return results


def benchmark_chunked_with_reuse(model_path: str, num_batches: int, batch_size: int, max_output_len: int) -> Dict[str, Any]:
    """Benchmark chunked API with chunk reuse across batches."""
    print("\n=== Chunked API (Cross-batch Chunk Reuse) ===")
    
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
            selected_contexts = [context_chunk_ids[randint(0, len(context_chunk_ids) - 1)] for _ in range(num_contexts)]
            
            # Create unique query for this request
            query_text = f"Batch {batch_idx}, Query {i}: What is " + " ".join([f"topic{randint(0, 100)}" for _ in range(5)]) + "?"
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
        "chunk_memory_mb": final_stats.get("memory_used_mb", 0)
    }
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Overall throughput: {throughput:.2f} tokens/s")
    print(f"Memory used: {format_size(memory_used)}")
    print(f"Total unique chunks: {final_stats['total_chunks']}, Total cache hits: {final_stats.get('cache_hits', 0)}")
    
    del llm
    aggressive_cleanup()
    
    return results


def main():
    seed(42)  # For reproducibility
    
    # Configuration
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    num_seqs = 32  # Reduced to avoid OOM
    max_input_len = 256
    max_output_len = 128
    
    print("=== NanoVLLM KV Cache Benchmark Suite ===")
    print(f"Model: {model_path}")
    print(f"Sequences: {num_seqs}")
    print(f"Max input length: {max_input_len}")
    print(f"Max output length: {max_output_len}")
    
    results = []
    
    # Benchmark 1: Standard KV cache
    results.append(benchmark_standard_kv_cache(model_path, num_seqs, max_input_len, max_output_len))
    
    # Benchmark 2: Cascade with different prefix lengths
    for prefix_len in [50, 100, 200]:
        results.append(benchmark_cascade_prefix_sharing(
            model_path, num_seqs, max_input_len, max_output_len, prefix_len
        ))
    
    # Benchmark 3: Chunked API with deduplication
    results.append(benchmark_chunked_api(model_path, num_seqs, max_input_len, max_output_len))
    
    # Benchmark 4: Chunked API with cross-batch reuse
    results.append(benchmark_chunked_with_reuse(model_path, num_batches=5, batch_size=10, max_output_len=max_output_len))
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Method':<40} {'Time (s)':<10} {'Throughput (tok/s)':<20} {'Memory':<15}")
    print("-" * 85)
    
    for result in results:
        method = result["method"]
        if "shared_prefix_len" in result:
            method += f" ({result['shared_prefix_len']} tok)"
        print(f"{method:<40} {result['time_seconds']:<10.2f} {result['throughput_tokens_per_sec']:<20.2f} {result['memory_used_formatted']:<15}")
    
    # Save detailed results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to benchmark_results.json")


if __name__ == "__main__":
    main()