# Nano-vLLM Chunk-Based API

## Overview

The Nano-vLLM Chunk API enables external agent applications to manage reusable context chunks that can be composed dynamically for inference. This API leverages FlashInfer's multi-level cascade attention mechanism to efficiently share KV caches across multiple requests.

## Key Concepts

### Chunks
A **chunk** is a reusable piece of text with a pre-computed KV cache. Chunks are identified by their content hash (SHA256) for automatic deduplication.

### Chunk Types
- **System Prompt**: Defines the behavior/persona of the model. Exactly one must be selected for inference.
- **Context**: Additional information (documents, examples, etc.). Zero or more can be selected for inference.
- **Query**: The user's question or request. Exactly one must be selected for inference.

### Cascade Levels
FlashInfer organizes chunks into cascade levels:
- **Level 0**: Shared across all sequences in a batch (typically system prompt)
- **Level 1+**: Unique or partially shared chunks (contexts and query)

## API Reference

### Initialization

```python
from nanovllm import ChunkedLLM, ChunkType

# Initialize the model with chunk support
llm = ChunkedLLM(
    model_path="path/to/model",
    max_chunks=1000,           # Maximum chunks to cache
    chunk_memory_ratio=0.5,    # Fraction of KV cache for chunks
    enable_deduplication=True  # Auto-deduplicate by content hash
)
```

### Chunk Management

#### Register a Chunk

```python
# Register a system prompt
system_id = llm.register_chunk(
    content="You are a helpful AI assistant specialized in Python.",
    chunk_type=ChunkType.SYSTEM_PROMPT,
    metadata={"name": "python_expert", "version": "1.0"}
)

# Register context chunks
doc_id = llm.register_chunk(
    content="Python is a high-level programming language...",
    chunk_type=ChunkType.CONTEXT,
    metadata={"source": "python_docs", "section": "intro"}
)

# Register query
query_id = llm.register_chunk(
    content="What are Python decorators?",
    chunk_type=ChunkType.QUERY
)
```

Returns: `chunk_id` (string) - SHA256 hash of content

#### List Chunks

```python
# List all chunks
chunks = llm.list_chunks()

# List by type
system_chunks = llm.list_chunks(chunk_type=ChunkType.SYSTEM_PROMPT)

# List with metadata filter
python_chunks = llm.list_chunks(metadata_filter={"source": "python_docs"})
```

Returns: List of chunk info dictionaries:
```python
[{
    "chunk_id": "a7b9c2d4e6...",
    "chunk_type": "system_prompt",
    "content_preview": "You are a helpful...",
    "token_count": 25,
    "metadata": {"name": "python_expert"},
    "created_at": "2024-01-01T00:00:00Z",
    "last_accessed": "2024-01-01T00:01:00Z",
    "access_count": 5
}]
```

#### Get Chunk Details

```python
chunk = llm.get_chunk(chunk_id)
# Returns full chunk details including content
```

#### Delete Chunk

```python
llm.delete_chunk(chunk_id)
# Removes chunk and frees KV cache memory
```

### Inference with Chunks

#### Simple Inference

```python
# Compose chunks for inference
output = llm.generate_from_chunks(
    system_chunk_id="a7b9c2d4e6...",
    context_chunk_ids=["b8c3d5e7f1...", "c9d4e6f8a2..."],
    query_chunk_id="d1e5f7a9b3...",
    sampling_params={"temperature": 0.7, "max_tokens": 200}
)
```

#### Batch Inference with Different Compositions

```python
# Multiple requests with different chunk combinations
requests = [
    {
        "system_chunk_id": "python_expert_id",
        "context_chunk_ids": ["python_docs_id"],
        "query_chunk_id": "query1_id"
    },
    {
        "system_chunk_id": "python_expert_id",  # Reuses KV cache
        "context_chunk_ids": ["django_docs_id"], # Different context
        "query_chunk_id": "query2_id"
    }
]

outputs = llm.batch_generate_from_chunks(
    requests=requests,
    sampling_params={"temperature": 0.7, "max_tokens": 200}
)
```

### Advanced Features

#### Chunk Composition Preview

```python
# Preview how chunks will be composed without running inference
composition = llm.preview_composition(
    system_chunk_id="...",
    context_chunk_ids=["...", "..."],
    query_chunk_id="..."
)

print(f"Total tokens: {composition['total_tokens']}")
print(f"Cascade levels: {composition['num_levels']}")
print(f"Memory usage: {composition['memory_bytes'] / 1024**2:.1f} MB")
```

#### Direct Text Inference with Chunk Registration

```python
# Convenience method that registers chunks and runs inference
output = llm.generate(
    system_prompt="You are a helpful assistant.",
    context=[
        "Document 1 content...",
        "Document 2 content..."
    ],
    query="What do the documents say about X?",
    sampling_params={"temperature": 0.7},
    persist_chunks=True  # Keep chunks for reuse
)

# Returns output and chunk IDs for future reuse
print(output['text'])
print(output['chunk_ids'])  # {"system": "...", "context": [...], "query": "..."}
```

#### Chunk Statistics

```python
stats = llm.get_chunk_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Memory used: {stats['memory_used_mb']} MB")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

## Memory Management

### Eviction Policy
When chunk memory is full, the least recently used (LRU) chunks are evicted automatically.

### Manual Memory Control

```python
# Clear all chunks
llm.clear_chunks()

# Clear chunks by type
llm.clear_chunks(chunk_type=ChunkType.CONTEXT)

# Clear unused chunks (not accessed in last N seconds)
llm.evict_unused_chunks(max_age_seconds=3600)
```

## Best Practices

1. **Chunk Granularity**: Create chunks at semantic boundaries (documents, sections, examples)
2. **System Prompts**: Keep them concise and register variations for different behaviors  
3. **Context Selection**: Order contexts by relevance; most important first
4. **Deduplication**: Let the system handle it automatically via content hashing
5. **Batch Requests**: Group requests sharing the same system prompt for efficiency

## Example Use Cases

### RAG System
```python
# Register document chunks
for doc in documents:
    llm.register_chunk(
        content=doc['content'],
        chunk_type=ChunkType.CONTEXT,
        metadata={"source": doc['url'], "title": doc['title']}
    )

# Query with relevant documents
relevant_docs = search_engine.search(query)
output = llm.generate_from_chunks(
    system_chunk_id=rag_system_prompt_id,
    context_chunk_ids=[doc['chunk_id'] for doc in relevant_docs],
    query_chunk_id=llm.register_chunk(query, ChunkType.QUERY)
)
```

### Multi-Turn Conversation
```python
# Register conversation history as context
history_id = llm.register_chunk(
    content=format_conversation(messages),
    chunk_type=ChunkType.CONTEXT,
    metadata={"conversation_id": conv_id}
)

# Generate response with history
output = llm.generate_from_chunks(
    system_chunk_id=assistant_prompt_id,
    context_chunk_ids=[history_id],
    query_chunk_id=llm.register_chunk(user_message, ChunkType.QUERY)
)
```

### Multi-Agent System
```python
# Different agents with different system prompts
agents = {
    "researcher": llm.register_chunk("You are a research assistant...", ChunkType.SYSTEM_PROMPT),
    "coder": llm.register_chunk("You are an expert programmer...", ChunkType.SYSTEM_PROMPT),
    "reviewer": llm.register_chunk("You are a code reviewer...", ChunkType.SYSTEM_PROMPT)
}

# Shared context across agents
shared_context_id = llm.register_chunk(project_description, ChunkType.CONTEXT)

# Each agent processes with their own prompt but shared context
for agent_name, system_id in agents.items():
    output = llm.generate_from_chunks(
        system_chunk_id=system_id,
        context_chunk_ids=[shared_context_id],
        query_chunk_id=task_query_id
    )
```

## Implementation Notes

### FlashInfer Integration
- Chunks are organized into cascade levels automatically
- Level assignment based on chunk type and sharing patterns
- KV caches are pre-allocated and reused across requests

### Performance Characteristics
- **First use**: O(n) where n is chunk length (KV cache computation)
- **Subsequent uses**: O(1) chunk lookup + standard attention cost
- **Memory**: Persistent KV cache per chunk
- **Deduplication**: Automatic via SHA256 content hashing

## Error Handling

Common errors and solutions:

```python
try:
    output = llm.generate_from_chunks(...)
except ChunkNotFoundError:
    # Chunk was evicted or doesn't exist
    pass
except IncompatibleChunksError:
    # Chunks have incompatible dimensions
    pass
except OutOfMemoryError:
    # Not enough KV cache memory
    llm.evict_unused_chunks()
```

## Migration from Standard API

```python
# Before (standard API)
output = llm.generate(
    prompt="System: You are helpful.\nUser: Hello",
    sampling_params={...}
)

# After (chunk API with persistence)
system_id = llm.register_chunk("You are helpful.", ChunkType.SYSTEM_PROMPT)
query_id = llm.register_chunk("Hello", ChunkType.QUERY)
output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    query_chunk_id=query_id,
    sampling_params={...}
)
# Subsequent calls reuse the KV caches
```