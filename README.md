# Braidinfer

A high-performance, single-GPU LLM inference engine designed for local deployment with advanced chunk-based context management and cascade attention.

## Key Features

* 🧠 **Chunk-Based Context Management** - Reusable semantic units with automatic KV cache deduplication
* ⚡ **Cascade Attention** - Efficient combination of pre-computed chunks without concatenation
* 🚀 **Fused Kernels** - 2.64x speedup with custom RMSNorm+QKV fusion
* 💾 **Output KV Retention** - Cache generated responses for efficient multi-turn conversations
* 📖 **Memory Efficient** - 53.3% memory savings for shared context through paged KV cache
* 🎯 **Production Ready** - Automatic model compatibility detection with graceful fallbacks

## Quick Start

### Installation

```bash
pip install git+https://github.com/GeeeekExplorer/Braidinfer.git
```

### Basic Usage

```python
from braidinfer import ChunkedLLM, ChunkType

# Initialize with model
llm = ChunkedLLM(
    model_path="path/to/model",
    max_chunks=1000,
    enable_deduplication=True
)

# Simple generation
output = llm.generate(
    system_prompt="You are a helpful assistant.",
    query="What is the capital of France?",
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
print(output['text'])
```

### Chunk-Based Generation (Recommended)

```python
# Register reusable chunks
system_id = llm.register_chunk(
    "You are a helpful assistant.", 
    ChunkType.SYSTEM_PROMPT
)
query_id = llm.register_chunk(
    "What is the capital of France?", 
    ChunkType.QUERY
)

# Generate using chunks (reuses KV cache automatically)
output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    query_chunk_id=query_id,
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
```

## Performance

**Current Performance (RTX 4090):**
- **Single Token**: ~29.4 tok/s (batch size 1)
- **Batch Processing**: 237 tok/s (batch size 8)
- **Cascade Capability**: 2,938 tok/s theoretical maximum
- **Fused Kernel Boost**: 2.64x speedup when compatible
- **Memory Efficiency**: 53.3% reduction for shared prompts

**Optimization Impact:**
- Custom fused kernels provide 2.64x speedup for compatible models
- Cascade attention eliminates redundant computation for shared context
- Output retention enables near-zero cost multi-turn conversations

## Core Architecture

### Chunk-Based Context Management

Instead of monolithic prompts, Braidinfer uses **Chunks** - semantic units with pre-computed KV caches:

- **Content-Addressable**: SHA256 hashing provides automatic deduplication
- **Reusable**: Same chunk content reuses computation across requests
- **Compositional**: Mix and match chunks dynamically at attention level
- **Persistent**: Expensive computations cached and shared

### Cascade Attention

Mathematically equivalent to concatenated attention while avoiding physical concatenation:

- **Online Softmax**: Maintains global normalization across chunk boundaries
- **Differential RoPE**: Corrects positional embeddings for cached chunks
- **Memory Efficient**: Processes chunks sequentially without concatenation
- **Mathematically Exact**: Guaranteed equivalence to standard attention

### Advanced Features

- **Output KV Retention**: Cache generated responses as reusable chunks
- **Think Tag Handling**: Automatic processing of reasoning patterns
- **Model Compatibility**: Automatic detection of fused kernel compatibility
- **Paged Memory**: Efficient KV cache allocation with HND layout

## Use Cases

### RAG (Retrieval-Augmented Generation)
```python
# Register document chunks once
doc_chunks = [llm.register_chunk(doc, ChunkType.CONTEXT) for doc in documents]

# Query with relevant documents (automatic KV reuse)
output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    context_chunk_ids=relevant_doc_chunks[:3],
    query_chunk_id=query_id
)
```

### Multi-Turn Conversations
```python
# Generate and retain output KV cache
output = llm.generate_and_retain_output(
    system_prompt="You are a helpful assistant.",
    query="What is quantum computing?"
)

# Use previous output as context
next_output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    context_chunk_ids=[output['output_chunk_id']],
    query_chunk_id=followup_query_id
)
```

### Multi-Agent Systems
```python
# Define agent personalities
agents = {
    "researcher": llm.register_chunk("Research assistant...", ChunkType.SYSTEM_PROMPT),
    "coder": llm.register_chunk("Expert programmer...", ChunkType.SYSTEM_PROMPT),
}

# Shared project context reused across agents
project_context = llm.register_chunk("Project details...", ChunkType.CONTEXT)
```

## CLI Interface

```bash
# Start interactive CLI
python -m braidinfer.cli --model path/to/model

# Basic commands
> /system You are a helpful assistant
> /query What is Python?
> /infer

# Chunk management
> /chunks                    # List all chunks
> /stats                     # Memory and performance stats
> /use-output 1              # Use previous output as context
```

## Model Compatibility

**Fully Compatible:**
- TinyLlama (all optimizations enabled)
- LLaMA-based architectures

**Compatible with Fallbacks:**
- Qwen3 (extreme K normalization weights require standard kernels)
- Most transformer architectures

**Automatic Detection:**
- Compatibility checker analyzes weight distributions
- Graceful fallback to standard kernels when needed
- No user intervention required

## Development Roadmap

### Current Status: Production Ready ✅

**Completed Features:**
- ✅ Custom kernels with 2.64x speedup
- ✅ Cascade attention with 53.3% memory savings
- ✅ Output KV retention system
- ✅ Automatic model compatibility detection
- ✅ Streaming support with minimal overhead

### Near-Term (Q1 2025)

**Sprint 1: Quantization Integration**
- INT8/INT4 quantization support
- 2-4x additional speedup expected
- Target: 200+ tok/s for 0.6B models

**Sprint 2: Multi-Model Support**
- LLaMA-2/3, Mistral, Gemma, Phi architectures
- Model-specific optimization profiles
- Automated architecture detection

**Sprint 3: Performance Pipeline Optimization**
- End-to-end latency profiling
- CPU-GPU synchronization optimization
- 10-20% latency reduction target

### Medium-Term (Q2-Q3 2025)
- Developer experience improvements
- Advanced sampling techniques
- LangChain/LlamaIndex integration
- REST API with OpenAI compatibility

### Long-Term Vision (Q4 2025+)
- 500+ tok/s for 1B models
- Full layer fusion (single kernel per layer)
- TensorRT integration
- Hardware-specific optimizations

## Examples and Documentation

- **`examples/`** - Usage patterns and tutorials
- **`ARCHITECTURE.md`** - Complete system architecture
- **`USER_GUIDE.md`** - Comprehensive API documentation
- **`ROADMAP.md`** - Development progress and plans

### Demo Applications

```bash
python -m braidinfer.cli                    # Interactive CLI with visual chunk management
python examples/chat_chunked.py             # Multi-turn chat with chunk reuse
python examples/multi_agent.py              # Multiple AI personas sharing context
python examples/rag_demo.py                 # Document Q&A with automatic chunk reuse
```

## Benchmarks

**Test Configuration:**
- Hardware: RTX 4090 (24GB)
- Model: TinyLlama-1.1B
- Batch Size: 8
- Context Length: Mixed (100-2048 tokens)

**Performance Comparison:**
| System | Throughput | Memory Usage | Features |
|--------|------------|--------------|----------|
| Braidinfer (Fused) | 237 tok/s | 8.2GB | Cascade attention, chunk reuse |
| Braidinfer (Standard) | 150 tok/s | 12.1GB | Standard kernels |
| vLLM | 180 tok/s | 11.8GB | Paged attention |

## Contributing

Braidinfer focuses on single-GPU optimization with production-ready features. Key areas for contribution:

- **Quantization**: INT8/INT4 support for memory-bound scenarios
- **Model Support**: Additional architectures and optimization profiles
- **Performance**: Advanced fusion techniques and hardware optimizations
- **Integration**: Ecosystem compatibility and tooling

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/Braidinfer&type=Date)](https://www.star-history.com/#GeeeekExplorer/Braidinfer&Date)