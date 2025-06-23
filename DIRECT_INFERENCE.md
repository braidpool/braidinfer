# Direct KV Cache Inference

This document explains the new direct inference approach that bypasses chat templates and works directly with KV cache blocks.

## Overview

The direct inference approach allows you to:
1. Run inference on arbitrary tokens already in the KV cache
2. Avoid redundant tokenization and chat template formatting
3. Maintain full control over what's in the prompt
4. Efficiently reuse cached computations

## Key Components

### VirtualSequence

A new sequence type that can reference existing blocks without reallocating them:

```python
from nanovllm.engine.virtual_sequence import VirtualSequence

seq = VirtualSequence(
    token_ids=new_tokens,           # Only new tokens to process
    sampling_params=params,
    existing_blocks=block_list,     # Blocks already in KV cache
    existing_token_count=count      # Total tokens in existing blocks
)
```

### Direct Inference API

Two new methods on LLMEngine:

```python
# Non-streaming
result = llm.infer_from_blocks(
    existing_blocks=blocks,
    existing_token_count=token_count,
    new_tokens=new_tokens,          # Optional new tokens
    sampling_params=params
)

# Streaming
for token_data in llm.infer_from_blocks_stream(...):
    print(token_data["token"], end="")
```

### Context Manager Integration

Helper methods to get blocks from chunks:

```python
# Get conversation blocks
blocks, count = context_mgr.get_conversation_blocks()

# Get all active blocks  
blocks, count = context_mgr.get_all_active_blocks()
```

## CLI Commands

New command for direct inference:

```
/infer [text]    - Run inference on active blocks, optionally append text
```

## Usage Examples

### 1. Continue Generation from Existing Context

```python
# Get existing conversation
blocks, token_count = context_mgr.get_conversation_blocks()

# Continue generation
result = llm.infer_from_blocks(
    existing_blocks=blocks,
    existing_token_count=token_count
)
```

### 2. Add New Input to Context

```python
# Tokenize new input (without chat template)
new_tokens = tokenizer.encode("Continue this thought: ")

# Run inference with context + new tokens
result = llm.infer_from_blocks(
    existing_blocks=blocks,
    existing_token_count=token_count,
    new_tokens=new_tokens
)
```

### 3. Manual Prompt Construction

```python
# Build prompt manually
prompt_parts = [
    "<|im_start|>system\nYou are helpful.\n<|im_end|>\n",
    "<|im_start|>user\nHello!\n<|im_end|>\n", 
    "<|im_start|>assistant\n"
]

# Tokenize each part
tokens = []
for part in prompt_parts:
    tokens.extend(tokenizer.encode(part, add_special_tokens=False))

# Run inference
result = llm.infer_from_blocks(
    existing_blocks=[],
    existing_token_count=0,
    new_tokens=tokens
)
```

## Benefits

1. **Full Control**: You decide exactly what tokens go into the model
2. **Efficiency**: Reuse existing KV cache entries without recomputation
3. **Flexibility**: Mix and match blocks from different sources
4. **Transparency**: See exactly what the model is processing

## Technical Details

### Block Allocation

- VirtualSequence only allocates blocks for new tokens
- Existing blocks have their reference count incremented
- On deallocation, only owned blocks are freed

### Attention Computation

- Block tables combine existing + new blocks
- Attention sees the complete context
- Position embeddings account for total sequence length

### Memory Management

- Existing blocks remain in their current memory tier
- New blocks are allocated in GPU memory
- Reference counting prevents premature deallocation

## Future Enhancements

1. **Block-level editing**: Modify tokens within existing blocks
2. **Partial block reuse**: Share common prefixes between sequences  
3. **Block compression**: Reduce memory usage for inactive blocks
4. **Cross-request caching**: Share blocks between different requests