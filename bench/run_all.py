#!/usr/bin/env python3
"""Run all benchmarks in sequence and generate a summary report."""

import os
import json
import subprocess
import sys
from common import aggressive_cleanup


def run_benchmark(script_name):
    """Run a benchmark script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    finally:
        # Extra cleanup between benchmarks
        aggressive_cleanup()


def load_results(filename):
    """Load results from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def generate_summary():
    """Generate a summary of all benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Load all results
    results_files = {
        "Standard KV Cache": "results_standard.json",
        "Cascade Attention": "results_cascade.json",
        "Chunked API": "results_chunked.json",
        "Chunked API (Reuse)": "results_chunked_reuse.json"
    }
    
    all_results = []
    for name, filename in results_files.items():
        data = load_results(filename)
        if data:
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    
    if not all_results:
        print("No results found!")
        return
    
    # Print summary table
    print(f"\n{'Method':<40} {'Batch':<8} {'Time (s)':<12} {'Throughput':<15} {'Memory':<15}")
    print("-" * 90)
    
    for result in all_results:
        method = result["method"]
        if "shared_prefix_len" in result:
            method += f" ({result['shared_prefix_len']} tok)"
        
        batch_str = str(result.get('batch_size', result.get('num_sequences', 'N/A')))
        time_str = f"{result['time_seconds']:.2f}"
        throughput_str = f"{result['throughput_tokens_per_sec']:.1f} tok/s"
        memory_str = result.get('memory_used_formatted', 'N/A')
        
        print(f"{method:<40} {batch_str:<8} {time_str:<12} {throughput_str:<15} {memory_str:<15}")
    
    # Save combined results
    with open("benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nDetailed results saved to:")
    for filename in results_files.values():
        if os.path.exists(filename):
            print(f"  - {filename}")
    print("  - benchmark_summary.json (combined)")


def main():
    """Run all benchmarks."""
    print("NanoVLLM Benchmark Suite")
    print("========================")
    
    # Check if model exists
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # List of benchmark scripts to run
    benchmarks = [
        "benchmark_standard.py",
        "benchmark_cascade.py",
        "benchmark_chunked.py",
        "benchmark_chunked_reuse.py"
    ]
    
    # Run each benchmark
    success_count = 0
    for benchmark in benchmarks:
        if run_benchmark(benchmark):
            success_count += 1
        else:
            print(f"\nWARNING: {benchmark} failed!")
    
    # Generate summary
    print(f"\n\nCompleted {success_count}/{len(benchmarks)} benchmarks successfully.")
    generate_summary()


if __name__ == "__main__":
    main()