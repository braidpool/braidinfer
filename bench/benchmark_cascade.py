#!/usr/bin/env python3
"""Benchmark cascade attention with prefix sharing."""

import time
from random import randint, seed
import torch

from nanovllm import LLM, SamplingParams
from common import (
    format_size, aggressive_cleanup, save_results, print_results,
    DEFAULT_CONFIG
)


def benchmark_cascade_prefix_sharing(model_path: str, num_seqs: int, max_input_len: int, 
                                   max_output_len: int, shared_prefix_len: int):
    """Benchmark cascade attention with prefix sharing."""
    print(f"=== Cascade Attention (Shared Prefix: {shared_prefix_len} tokens) ===")
    
    llm = LLM(
        model_path,
        enable_cascade_attention=True,
        cascade_shared_prefix_len=shared_prefix_len,
        enforce_eager=True,
        max_model_len=4096
    )
    
    # Generate test sequences with shared prefix
    shared_prefix = [randint(0, 10000) for _ in range(shared_prefix_len)]
    prompt_token_ids = [
        shared_prefix + [randint(0, 10000) for _ in range(randint(50, max_input_len - shared_prefix_len))] 
        for _ in range(num_seqs)
    ]
    
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
    
    del llm
    aggressive_cleanup()
    
    return results


def main():
    """Run the benchmark for different prefix lengths."""
    config = DEFAULT_CONFIG
    seed(config["seed"])
    
    print(f"Model: {config['model_path']}")
    print(f"Batch sizes: {config['batch_sizes']}")
    print(f"Max input length: {config['max_input_len']}")
    print(f"Max output length: {config['max_output_len']}")
    
    all_results = []
    
    # Test different batch sizes and prefix lengths
    for batch_size in config["batch_sizes"]:
        for prefix_len in [50, 100, 200]:
            print(f"\n--- Testing batch size: {batch_size}, prefix length: {prefix_len} ---")
            results = benchmark_cascade_prefix_sharing(
                config["model_path"],
                batch_size,
                config["max_input_len"],
                config["max_output_len"],
                prefix_len
            )
            results["batch_size"] = batch_size
            
            print_results(results)
            all_results.append(results)
            
            # Extra cleanup between runs
            aggressive_cleanup()
    
    save_results(all_results, "results_cascade.json")


if __name__ == "__main__":
    main()