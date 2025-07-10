# Braidinfer User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Chunk API](#core-chunk-api)
3. [Advanced Features](#advanced-features)
4. [Output KV Cache Retention](#output-kv-cache-retention)
5. [CLI Interface](#cli-interface)
6. [Best Practices](#best-practices)
7. [Examples & Use Cases](#examples--use-cases)

## Quick Start

### Installation and Setup

```python
from braidinfer import ChunkedLLM, ChunkType

# Initialize with model
llm = ChunkedLLM(
    model_path="path/to/model",
    max_chunks=1000,           # Maximum chunks to cache
    chunk_memory_ratio=0.5,    # Fraction of KV cache for chunks
    enable_deduplication=True  # Auto-deduplicate by content hash
)
```

### Simple Generation

```python
# Basic text generation
output = llm.generate(
    system_prompt="You are a helpful assistant.",
    query="What is the capital of France?",
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
print(output['text'])
```

### Chunk-Based Generation

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

# Generate using chunks (reuses KV cache)
output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    query_chunk_id=query_id,
    sampling_params={"temperature": 0.7, "max_tokens": 100}
)
```

## Core Chunk API

### Chunk Concepts

**Chunks** are reusable pieces of text with pre-computed KV caches. They enable:
- **Memory efficiency**: Shared context reuses computation
- **Performance**: Avoid recomputing identical prefixes
- **Flexibility**: Mix and match context dynamically

### Chunk Types

```python
from braidinfer import ChunkType

# System Prompt: Defines model behavior/persona
ChunkType.SYSTEM_PROMPT  # Exactly one required per generation

# Context: Additional information (documents, examples)
ChunkType.CONTEXT        # Zero or more per generation

# Query: User's question or request
ChunkType.QUERY          # Exactly one required per generation

# Output: Generated response (for retention feature)
ChunkType.OUTPUT         # Created automatically when retaining outputs
```

### Chunk Management

#### Registering Chunks

```python
# System prompt with metadata
system_id = llm.register_chunk(
    content="You are a Python expert assistant.",
    chunk_type=ChunkType.SYSTEM_PROMPT,
    metadata={"domain": "programming", "version": "1.0"}
)

# Context documents
doc_id = llm.register_chunk(
    content="Python is a high-level programming language...",
    chunk_type=ChunkType.CONTEXT,
    metadata={"source": "python_docs", "section": "intro"}
)

# User query
query_id = llm.register_chunk(
    content="Explain Python decorators",
    chunk_type=ChunkType.QUERY
)
```

**Returns**: Chunk ID (SHA256 hash of content) for automatic deduplication

#### Listing and Searching Chunks

```python
# List all chunks
all_chunks = llm.list_chunks()

# Filter by type
system_chunks = llm.list_chunks(chunk_type=ChunkType.SYSTEM_PROMPT)

# Filter by metadata
python_chunks = llm.list_chunks(metadata_filter={"domain": "programming"})

# Get detailed chunk information
chunk_info = llm.get_chunk(chunk_id)
print(f"Content: {chunk_info['content']}")
print(f"Tokens: {chunk_info['token_count']}")
print(f"Created: {chunk_info['created_at']}")
```

#### Chunk Information Structure

```python
{
    "chunk_id": "a7b9c2d4e6f8...",
    "chunk_type": "system_prompt",
    "content": "You are a helpful assistant...",
    "content_preview": "You are a helpful...",  # First 50 chars
    "token_count": 25,
    "metadata": {"domain": "general"},
    "created_at": "2024-01-01T00:00:00Z",
    "last_accessed": "2024-01-01T00:01:00Z",
    "access_count": 5,
    "memory_bytes": 8192
}
```

### Generation with Chunks

#### Basic Chunk Composition

```python
# Generate with multiple context chunks
output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    context_chunk_ids=[doc1_id, doc2_id, doc3_id],
    query_chunk_id=query_id,
    sampling_params={
        "temperature": 0.7,
        "max_tokens": 200,
        "top_p": 0.9
    }
)

print(output['text'])
print(f"Total tokens: {output['usage']['total_tokens']}")
```

#### Batch Generation

```python
# Multiple requests with shared chunks
requests = [
    {
        "system_chunk_id": system_id,     # Shared system prompt
        "context_chunk_ids": [doc1_id],   # Different contexts
        "query_chunk_id": query1_id
    },
    {
        "system_chunk_id": system_id,     # Reuses KV cache
        "context_chunk_ids": [doc2_id],   
        "query_chunk_id": query2_id
    }
]

outputs = llm.batch_generate_from_chunks(
    requests=requests,
    sampling_params={"temperature": 0.7}
)
```

### Memory Management

#### Automatic Eviction

```python
# LRU eviction when memory is full
stats = llm.get_chunk_stats()
print(f"Memory used: {stats['memory_used_mb']:.1f} MB")
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

#### Manual Memory Control

```python
# Delete specific chunks
llm.delete_chunk(chunk_id)

# Clear chunks by type
llm.clear_chunks(chunk_type=ChunkType.CONTEXT)

# Evict old chunks
llm.evict_unused_chunks(max_age_seconds=3600)

# Clear all chunks
llm.clear_chunks()
```

## Advanced Features

### Composition Preview

```python
# Preview token count and memory usage before generation
composition = llm.preview_composition(
    system_chunk_id=system_id,
    context_chunk_ids=[doc1_id, doc2_id],
    query_chunk_id=query_id
)

print(f"Total tokens: {composition['total_tokens']}")
print(f"Memory usage: {composition['memory_bytes'] / 1024**2:.1f} MB")
print(f"Cascade levels: {composition['num_levels']}")
```

### Text-to-Chunks Convenience API

```python
# Automatically register chunks and generate
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

# Access chunk IDs for future reuse
chunk_ids = output['chunk_ids']
system_id = chunk_ids['system']
context_ids = chunk_ids['context']
query_id = chunk_ids['query']
```

### Streaming Generation

```python
# Stream tokens in real-time
for token in llm.generate_stream(
    system_chunk_id=system_id,
    query_chunk_id=query_id,
    sampling_params={"temperature": 0.7}
):
    print(token, end='', flush=True)
```

## Output KV Cache Retention

### Overview

Output retention allows you to cache generated responses and reuse them as context in subsequent generations, enabling efficient multi-turn conversations and chain-of-thought processing.

### Basic Output Retention

```python
# Generate and retain the output KV cache
output = llm.generate_and_retain_output(
    system_prompt="You are a helpful assistant.",
    query="Explain quantum computing",
    sampling_params={"temperature": 0.7, "max_tokens": 200}
)

print(output['text'])                    # Generated text
print(output['output_chunk_id'])         # ID of retained output chunk
print(output['retained_seq_id'])         # Internal sequence ID
```

### Using Output as Context

```python
# First generation
output1 = llm.generate_and_retain_output(
    system_prompt="You are a helpful assistant.",
    query="What is the capital of France?",
    sampling_params={"temperature": 0.1, "max_tokens": 50}
)

# Use previous output as context for next generation
system_id = llm.register_chunk("You are a helpful assistant.", ChunkType.SYSTEM_PROMPT)
query_id = llm.register_chunk("What city did we just discuss?", ChunkType.QUERY)

output2 = llm.generate_from_chunks(
    system_chunk_id=system_id,
    context_chunk_ids=[output1['output_chunk_id']],  # Previous output as context
    query_chunk_id=query_id,
    sampling_params={"temperature": 0.1, "max_tokens": 50}
)
```

### Think Tag Handling

The system automatically handles `<think>` and `</think>` tags in outputs:

```python
# If model generates: "Let me think... <think>Paris is in France</think>The answer is Paris."
# The stored chunk will contain: "Let me think... The answer is Paris."
# But the KV cache preserves the full sequence including think positions

output = llm.generate_and_retain_output(
    system_prompt="Think step by step.",
    query="What is 2+2?",
    sampling_params={"temperature": 0.1}
)

# Check if think tags were present
chunk_info = llm.get_chunk(output['output_chunk_id'])
has_think_tags = chunk_info.get('had_think_tags', False)
```

### Manual Retention Control

```python
# Enable retention with standard generate
from braidinfer import SamplingParams

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    retain_output_cache=True  # Enable output retention
)

output = llm.generate_from_chunks(
    system_chunk_id=system_id,
    query_chunk_id=query_id,
    sampling_params=sampling_params
)

# Access retained sequences directly
retained = llm.get_retained_sequences()
for seq_id, info in retained.items():
    print(f"Sequence {seq_id}: {info['text'][:50]}...")

# Release when done
llm.release_retained_sequence(seq_id)
```

### Output Chunk Management

```python
# List all retained output chunks
output_chunks = llm.list_chunks(chunk_type=ChunkType.OUTPUT)

# Delete specific output chunk and release its KV cache
llm.delete_chunk(output_chunk_id)

# Get memory usage for output chunks
stats = llm.get_chunk_stats()
output_memory = stats['memory_by_type']['output']
print(f"Output chunks using {output_memory:.1f} MB")
```

## CLI Interface

### Starting the CLI

```bash
python -m braidinfer.cli --model path/to/model
```

### Basic Commands

```bash
# Set system prompt
> /system You are a helpful assistant

# Set query
> /query What is Python?

# Run inference
> /infer

# Clear conversation
> /clear

# Exit
> /exit
```

### Chunk Management Commands

```bash
# List all chunks
> /chunks

# List chunks by type
> /chunks system
> /chunks context
> /chunks query

# Delete chunk by ID
> /delete-chunk a7b9c2d4...

# Show chunk statistics
> /stats
```

### Output Management Commands

```bash
# List retained output chunks
> /output

# Use output chunk as context
> /use-output 1

# Delete output chunk
> /delete-output 1

# Show current context composition
> /context
```

### Advanced CLI Features

```bash
# Load context from file
> /load-context document.txt

# Save conversation
> /save conversation.json

# Load previous conversation
> /load conversation.json

# Enable/disable streaming
> /stream on
> /stream off
```

## Best Practices

### Chunk Design

1. **Semantic Boundaries**: Create chunks at logical divisions (documents, sections, examples)
2. **Granularity**: Balance reusability vs. memory usage
3. **Metadata**: Use descriptive metadata for easy filtering and organization
4. **Content Normalization**: Remove unnecessary whitespace for better deduplication

### Performance Optimization

1. **Shared System Prompts**: Use same system prompt across requests for maximum KV cache reuse
2. **Batch Requests**: Group requests with shared chunks
3. **Context Ordering**: Place most relevant contexts first
4. **Memory Monitoring**: Regularly check chunk statistics and evict unused chunks

### Memory Management

1. **Set Appropriate Limits**: Configure `max_chunks` and `chunk_memory_ratio` based on available VRAM
2. **Monitor Usage**: Use `get_chunk_stats()` to track memory consumption
3. **Clean Up**: Delete output chunks when no longer needed
4. **Eviction Strategy**: Rely on LRU for automatic management, but manually evict for predictable behavior

### Error Handling

```python
from braidinfer.exceptions import ChunkNotFoundError, OutOfMemoryError

try:
    output = llm.generate_from_chunks(
        system_chunk_id=system_id,
        query_chunk_id=query_id
    )
except ChunkNotFoundError as e:
    print(f"Chunk not found: {e.chunk_id}")
    # Re-register the chunk
except OutOfMemoryError:
    print("KV cache memory full")
    # Evict unused chunks or reduce max_chunks
    llm.evict_unused_chunks()
```

## Examples & Use Cases

### RAG (Retrieval-Augmented Generation)

```python
# Setup RAG system
system_id = llm.register_chunk(
    "You are a helpful assistant that answers questions based on provided documents.",
    ChunkType.SYSTEM_PROMPT
)

# Register document chunks
document_chunks = []
for doc in knowledge_base:
    chunk_id = llm.register_chunk(
        content=doc['content'],
        chunk_type=ChunkType.CONTEXT,
        metadata={
            "source": doc['url'],
            "title": doc['title'],
            "domain": doc['domain']
        }
    )
    document_chunks.append(chunk_id)

# Query with relevant documents
def answer_question(question):
    # Retrieve relevant documents
    relevant_docs = search_engine.search(question, top_k=3)
    relevant_chunk_ids = [doc['chunk_id'] for doc in relevant_docs]
    
    query_id = llm.register_chunk(question, ChunkType.QUERY)
    
    return llm.generate_from_chunks(
        system_chunk_id=system_id,
        context_chunk_ids=relevant_chunk_ids,
        query_chunk_id=query_id,
        sampling_params={"temperature": 0.3, "max_tokens": 300}
    )
```

### Multi-Turn Conversation

```python
# Initialize conversation
system_id = llm.register_chunk(
    "You are a helpful assistant. Maintain context across the conversation.",
    ChunkType.SYSTEM_PROMPT
)

conversation_history = []

def chat_turn(user_message):
    global conversation_history
    
    # Add user message to history
    conversation_history.append(f"User: {user_message}")
    
    # Generate response with retained output
    output = llm.generate_and_retain_output(
        system_chunk_id=system_id,
        context_chunk_ids=[chunk['id'] for chunk in conversation_history[:-1]],  # Previous turns
        query=user_message,
        sampling_params={"temperature": 0.7, "max_tokens": 200}
    )
    
    # Add assistant response to history
    conversation_history.append({
        'text': f"Assistant: {output['text']}",
        'id': output['output_chunk_id']
    })
    
    return output['text']

# Example conversation
response1 = chat_turn("What's the weather like?")
response2 = chat_turn("What about tomorrow?")  # Maintains context
```

### Multi-Agent System

```python
# Define agent personalities
agents = {
    "researcher": llm.register_chunk(
        "You are a research assistant. Focus on finding and analyzing information.",
        ChunkType.SYSTEM_PROMPT
    ),
    "coder": llm.register_chunk(
        "You are an expert programmer. Write clean, efficient code.",
        ChunkType.SYSTEM_PROMPT
    ),
    "reviewer": llm.register_chunk(
        "You are a code reviewer. Focus on quality, bugs, and improvements.",
        ChunkType.SYSTEM_PROMPT
    )
}

# Shared project context
project_context = llm.register_chunk(
    "Project: Build a web scraping tool for e-commerce sites...",
    ChunkType.CONTEXT
)

def agent_task(agent_name, task_description):
    task_id = llm.register_chunk(task_description, ChunkType.QUERY)
    
    return llm.generate_from_chunks(
        system_chunk_id=agents[agent_name],
        context_chunk_ids=[project_context],
        query_chunk_id=task_id,
        sampling_params={"temperature": 0.7, "max_tokens": 500}
    )

# Example workflow
research = agent_task("researcher", "Research web scraping libraries for Python")
code = agent_task("coder", "Implement a basic web scraper based on the research")
review = agent_task("reviewer", "Review the scraper code for improvements")
```

### Chain-of-Thought with Output Retention

```python
# Step-by-step problem solving
system_id = llm.register_chunk(
    "You are a helpful assistant. Think step by step and show your reasoning.",
    ChunkType.SYSTEM_PROMPT
)

# Step 1: Initial reasoning
step1 = llm.generate_and_retain_output(
    system_chunk_id=system_id,
    query="Solve this math problem: If a train travels 120 km in 2 hours, what's its speed?",
    sampling_params={"temperature": 0.3, "max_tokens": 200}
)

# Step 2: Follow-up question using previous reasoning
step2 = llm.generate_and_retain_output(
    system_chunk_id=system_id,
    context_chunk_ids=[step1['output_chunk_id']],
    query="Now calculate how far it would travel in 5 hours at the same speed.",
    sampling_params={"temperature": 0.3, "max_tokens": 200}
)

# Step 3: Final verification
step3 = llm.generate_from_chunks(
    system_chunk_id=system_id,
    context_chunk_ids=[step1['output_chunk_id'], step2['output_chunk_id']],
    query_chunk_id=llm.register_chunk(
        "Verify your calculations are correct.",
        ChunkType.QUERY
    ),
    sampling_params={"temperature": 0.1, "max_tokens": 100}
)
```

### Performance Monitoring

```python
import time

def monitor_performance():
    start_time = time.time()
    
    # Generate with chunks
    output = llm.generate_from_chunks(
        system_chunk_id=system_id,
        query_chunk_id=query_id
    )
    
    generation_time = time.time() - start_time
    
    # Get performance stats
    stats = llm.get_chunk_stats()
    
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens/second: {output['usage']['total_tokens'] / generation_time:.1f}")
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"Memory usage: {stats['memory_used_mb']:.1f} MB")
    
    return output

# Monitor multiple generations
for i in range(10):
    monitor_performance()
```

This user guide provides comprehensive coverage of Braidinfer's chunk-based API and advanced features, enabling users to efficiently leverage the system's memory and performance optimizations for various LLM applications.