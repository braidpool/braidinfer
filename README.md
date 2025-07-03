# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Current Status - Cascade Attention Branch (December 2024)

### Overview
This branch contains a fully functional cascade attention feature for compositional context caching using FlashInfer. **Status: Complete and working! ✅**

### What Works
- ✅ Basic model inference with regular attention
- ✅ PagedKVCache memory management  
- ✅ Multi-level cascade attention using FlashInfer
- ✅ Proper tensor formats and API usage
- ✅ Bit-for-bit identical results to FlashInfer
- ✅ Efficient memory usage with shared workspace buffers
- ✅ Integration tests passing

### Refactoring Complete ✓
Successfully refactored to use FlashInfer correctly:
1. **Correct API Usage**: Uses FlashInfer's cascade API exactly as intended
2. **Proper Tensor Shapes**: All tensors match FlashInfer's expected formats
3. **No Custom Implementation**: Direct wrapper around FlashInfer (~320 lines total)
4. **Memory Efficient**: Fixed workspace buffer allocation to share buffers across layers

### Architecture Components (Simplified)
- **FlashInferCascadeAttention**: Simple wrapper around FlashInfer's API (~200 lines)
- **FlashInferScheduler**: Prepares cascade data in FlashInfer's format (~120 lines)
- **Total Implementation**: ~320 lines (down from ~6000)

### Usage
```python
# Enable cascade attention
llm = LLM(
    model_path,
    enable_cascade_attention=True,
    cascade_shared_prefix_len=1024  # Length of shared prefix
)

# Use normally
output = llm.generate("Hello, world!", sampling_params)
```

### Documentation
- `claude_docs/cascade_refactoring_complete.md` - Refactoring summary
- `claude_docs/flashinfer_cascade_spec.md` - FlashInfer API specification
- `test_cascade_validation.py` - Validation tests proving correctness
- `test_cascade_integration_fixed.py` - Integration tests with proper memory management

### Performance
The cascade attention feature provides memory savings for sequences with shared prefixes (like system prompts) while maintaining the same generation speed. The implementation is highly efficient with minimal overhead.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)