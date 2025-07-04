# NanoVLLM Benchmark Suite

This directory contains benchmarking scripts for evaluating different KV cache configurations in nano-vllm.

## Overview

The benchmarks are split into separate scripts to avoid out-of-memory (OOM) issues that can occur when running all configurations in a single process. Each script focuses on a specific KV cache strategy.

## Benchmark Scripts

1. **benchmark_standard.py** - Standard KV cache without any sharing
   - Baseline performance measurement
   - Each sequence has its own independent KV cache

2. **benchmark_cascade.py** - Cascade attention with prefix sharing
   - Tests with different shared prefix lengths (50, 100, 200 tokens)
   - Demonstrates memory savings from sharing common prefixes

3. **benchmark_chunked.py** - Chunked API with content deduplication
   - Content-addressed chunk management
   - Automatic deduplication of repeated content

4. **benchmark_chunked_reuse.py** - Chunked API with cross-batch reuse
   - Demonstrates chunk reuse across multiple batches
   - Shows cache hit rates and efficiency gains

## Running Benchmarks

### Run All Benchmarks
```bash
python bench/run_all.py
```

### Run Individual Benchmarks
```bash
python bench/benchmark_standard.py
python bench/benchmark_cascade.py
python bench/benchmark_chunked.py
python bench/benchmark_chunked_reuse.py
```

## Configuration

Default configuration (in `common.py`):
- Model: `~/huggingface/Qwen3-0.6B/`
- Batch sizes: [1, 2, 4] (optimized for single-user scenarios)
- Max input length: 256 tokens
- Max output length: 128 tokens

You can modify these settings in the `DEFAULT_CONFIG` dictionary in `common.py`.

Each benchmark script will test all configured batch sizes and report results for each.

## Output Files

Each benchmark creates a results JSON file:
- `results_standard.json` - Standard KV cache results
- `results_cascade.json` - Cascade attention results
- `results_chunked.json` - Chunked API results
- `results_chunked_reuse.json` - Cross-batch reuse results
- `benchmark_summary.json` - Combined results from all benchmarks

## Memory Management

The scripts include aggressive memory cleanup between benchmarks:
- GPU cache clearing
- Garbage collection
- Memory stats reset
- 3-second pause for OS memory reclamation

This helps prevent OOM errors and ensures more consistent results.

## Prerequisites

1. Download the model:
   ```bash
   huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
   ```

2. Ensure nano-vllm is installed:
   ```bash
   pip install -e ..
   ```

## Interpreting Results

Key metrics to compare:
- **Throughput**: Tokens processed per second (higher is better)
- **Memory Usage**: GPU memory consumed (lower is better)
- **Time**: Total execution time (lower is better)
- **Memory Savings**: For cascade/chunked approaches (higher is better)
- **Cache Hits**: For chunked API (higher indicates better reuse)