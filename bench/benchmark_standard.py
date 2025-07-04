#!/usr/bin/env python3
"""Benchmark standard KV cache (no sharing)."""

import time
from random import randint, seed
import torch

from nanovllm import LLM, SamplingParams
from common import (
    format_size, aggressive_cleanup, save_results, print_results,
    DEFAULT_CONFIG
)


def benchmark_standard_kv_cache(model_path: str, num_seqs: int, max_input_len: int, max_output_len: int):
    """Benchmark standard KV cache (no sharing)."""
    print("=== Standard KV Cache (No Sharing) ===")
    
    llm = LLM(model_path, enforce_eager=True, max_model_len=4096)
    
    # Generate test sequences
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] 
                        for _ in range(num_seqs)]
    
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
        results = benchmark_standard_kv_cache(
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
    
    save_results(all_results, "results_standard.json")


if __name__ == "__main__":
    main()