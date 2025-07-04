"""Common utilities for benchmarking scripts."""

import os
import time
import gc
import torch
from typing import Dict, Any

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


def save_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to JSON file."""
    import json
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def print_results(results: Dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print(f"\nMethod: {results['method']}")
    print(f"Time: {results['time_seconds']:.2f}s")
    print(f"Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"Memory used: {results['memory_used_formatted']}")
    
    if 'memory_savings_ratio' in results:
        print(f"Memory savings: {results['memory_savings_ratio']*100:.1f}%")
    
    if 'cache_hits' in results:
        print(f"Cache hits: {results['cache_hits']}")
    
    if 'num_unique_chunks' in results:
        print(f"Unique chunks: {results['num_unique_chunks']}")


# Default configuration
DEFAULT_CONFIG = {
    "model_path": os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
    "batch_sizes": [1, 2, 4],  # Test different batch sizes
    "max_input_len": 256,
    "max_output_len": 128,
    "seed": 42
}