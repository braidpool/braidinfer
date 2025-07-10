# Braidinfer Examples

This directory contains examples demonstrating various features of Braidinfer.

## Examples

### 1. basic_usage.py
**Simple text generation example**
- Basic model initialization
- Single and batch text generation
- Different sampling parameters
- Chat format usage

```bash
python examples/basic_usage.py
```

### 2. cascade_attention.py
**Cascade attention for memory efficiency (Low-level API)**
- Manual cascade attention setup with LLM API
- Memory savings calculation
- Multi-query scenarios with shared system prompts

```bash
python examples/cascade_attention.py
```

### 3. cascade_attention_chunked.py
**Cascade attention using ChunkedLLM API (Recommended)**
- Automatic cascade attention with chunk-based API
- System prompt and context reuse
- Multi-level cascade demonstration
- Cache statistics and efficiency metrics

```bash
python examples/cascade_attention_chunked.py
```

### 4. chunked_api.py
**Content-addressed chunk management**
- Registering reusable text chunks
- Efficient KV cache reuse
- Multi-agent scenarios
- Chunk statistics and hit rates

```bash
python examples/chunked_api.py
```

### 5. conversation_reuse.py
**Efficient conversation management**
- Conversation history as reusable chunks
- Branching conversations from shared history
- Session persistence and continuation
- Memory efficiency metrics

```bash
python examples/conversation_reuse.py
```

### 6. multi_agent.py
**Multiple AI agents with shared context**
- Different agents with unique system prompts
- Shared data analysis from multiple perspectives
- Collaborative agent workflows
- Extreme memory efficiency for multi-agent systems

```bash
python examples/multi_agent.py
```

## Prerequisites

Before running the examples, ensure you have:

1. **Downloaded a model**:
   ```bash
   huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
   ```

2. **Installed Braidinfer**:
   ```bash
   pip install -e .
   ```

## Key Features Demonstrated

- **High Performance**: Optimized for single-GPU inference
- **Memory Efficiency**: Cascade attention and chunk reuse
- **Simple API**: Easy-to-use interface compatible with vLLM
- **Flexibility**: Support for various models and configurations

## Performance Tips

- Use `enforce_eager=True` for debugging
- Adjust `num_kvcache_blocks` based on your GPU memory
- Enable cascade attention for scenarios with shared prefixes
- Use the ChunkedLLM API for content that will be reused