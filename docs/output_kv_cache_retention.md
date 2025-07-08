# Output KV Cache Retention

This document describes the output KV cache retention feature that enables reusing generated outputs as context chunks in subsequent generations.

## Overview

By default, when a sequence completes generation, its KV cache is immediately deallocated to free memory. The output KV cache retention feature allows you to:

1. **Retain the KV cache** after generation completes
2. **Register outputs as reusable chunks** that can be used as context
3. **Efficiently continue conversations** without regenerating previous outputs
4. **Analyze or refine previous generations** using the cached KV states

## How It Works

### Position-Aware KV Cache

The cascade attention mechanism handles position embeddings correctly:
- Each chunk maintains its position range within the sequence
- RoPE (Rotary Position Embeddings) are applied during chunk prefilling
- Chunks can be composed at different positions while maintaining correctness

### Think Tag Handling

The system automatically handles `<think>` and `</think>` tags:
- Think tags are detected in generated output
- The stored chunk content has think tags filtered out
- The KV cache retains the full sequence (including think positions)
- Metadata tracks whether think tags were present

## API Usage

### Basic Output Retention

```python
from nanovllm import ChunkedLLM, SamplingParams

# Initialize ChunkedLLM
llm = ChunkedLLM(model_path)

# Generate with output retention
output = llm.generate_and_retain_output(
    system_prompt="You are a helpful assistant.",
    query="Explain quantum computing",
    sampling_params={"temperature": 0.7, "max_tokens": 200}
)

# Output contains:
# - text: The generated text (with think tags filtered)
# - token_ids: The generated token IDs  
# - output_chunk_id: ID of the registered output chunk
# - retained_seq_id: ID of the retained sequence
```

### Using Output as Context

```python
# First generation
output1 = llm.generate_and_retain_output(
    system_prompt="You are a helpful assistant.",
    query="What is the capital of France?",
    sampling_params={"temperature": 0.1, "max_tokens": 50}
)

# Use the output as context for next generation
output2 = llm.generate_from_chunks(
    system_chunk_id=llm.register_chunk("You are a helpful assistant.", ChunkType.SYSTEM_PROMPT),
    query_chunk_id=llm.register_chunk("What city did we just discuss?", ChunkType.QUERY),
    context_chunk_ids=[output1['output_chunk_id']],  # Use previous output as context
    sampling_params={"temperature": 0.1, "max_tokens": 50}
)
```

### Manual Control

```python
# Enable retention with standard generate
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    retain_output_cache=True  # Enable retention
)

# Access retained sequences directly
retained = llm.llm.get_retained_sequences()
for seq_id, info in retained.items():
    print(f"Sequence {seq_id}:")
    print(f"  Text: {info['text']}")  # Think tags filtered
    print(f"  Has think tags: {info['think_positions'] is not None}")

# Manually release when done
llm.llm.release_retained_sequence(seq_id)
```

## CLI Demo

The CLI (`cli.py`) includes enhanced commands for output chunks:

- `/output` - List all retained output chunks
- `/use-output <n>` - Add output chunk N as context
- `/delete-output <n>` - Delete output chunk N and release its KV cache

Example session:
```
> /system You are a helpful assistant
> /query What is 2+2?
> /infer
[Output: The answer is 4]

> /output
Output Chunks
# Preview                    Tokens  ID
1 The answer is 4           5       a1b2c3d4...

> /query Why did you say that?
> /use-output 1
> /infer
[Output: I said "The answer is 4" because you asked what 2+2 equals...]
```

## Memory Management

### Automatic Deduplication
- Output chunks are deduplicated by content hash
- Empty outputs (e.g., all think tags) create empty chunks

### Manual Deallocation
- Output chunks persist until manually deleted
- Use `delete_chunk()` or `release_retained_sequence()`
- The CLI provides `/delete-output` command

### Memory Usage
- Each retained sequence keeps its KV cache pages allocated
- Monitor memory with `get_chunk_stats()`
- Set `chunk_memory_ratio` to control memory allocation

## Technical Details

### Chunk Types
- `OUTPUT` chunks are assigned cascade level 1 (same as `CONTEXT`)
- They can be freely mixed with other context chunks
- Standard composition rules apply

### Think Tag Detection
- Token IDs: `<think>` = 151667, `</think>` = 151668
- Handles both complete and unclosed think blocks
- Filtered text is stored in chunks for clean context

### Performance Benefits
- Avoids regenerating previous outputs
- Reduces token processing for multi-turn conversations  
- Enables efficient chain-of-thought analysis
- Memory trade-off for computation savings

## Best Practices

1. **Release unused outputs** - KV cache memory is limited
2. **Use deduplication** - Enable to avoid duplicate chunks
3. **Monitor memory usage** - Check stats regularly
4. **Filter think tags** - Already handled automatically
5. **Batch related generations** - Reuse system/context chunks

## Limitations

1. **Memory constraints** - Each retained output uses KV cache memory
2. **Position limits** - Very long sequences may hit model limits
3. **Think tag handling** - Only filters at chunk storage, not KV cache
4. **No persistence** - KV cache is not saved between sessions