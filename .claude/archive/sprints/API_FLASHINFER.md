# API_FLASHINFER.md - FlashInfer Library Documentation

## Overview

FlashInfer is a high-performance GPU kernel library for Large Language Model (LLM) serving that provides optimized attention implementations. It focuses on efficient sparse/dense attention kernels with support for paged KV cache, which is crucial for memory-efficient LLM serving.

## Key Concepts

### 1. KV Cache Layouts

FlashInfer supports two tensor layouts for KV cache: **NHD** and **HND**.

#### Regular Tensors (non-paged):
- **NHD**: `[seq_len, num_heads, head_dim]`
- **HND**: `[num_heads, seq_len, head_dim]`

#### Paged KV Cache (5D combined format):
- **NHD**: `[num_pages, 2, page_size, num_kv_heads, head_dim]`
- **HND**: `[num_pages, 2, num_kv_heads, page_size, head_dim]`

Where:
- `num_pages`: Total number of pages across all sequences
- `2`: Combined K/V dimension (index 0 = keys, index 1 = values)
- `page_size`: Number of tokens per page
- `num_kv_heads`: Number of key/value attention heads
- `head_dim`: Dimension of each attention head

#### Paged KV Cache (4D separate format):
FlashInfer also supports separate K and V tensors:
- **NHD**: K/V shape = `[num_pages, page_size, num_kv_heads, head_dim]`
- **HND**: K/V shape = `[num_pages, num_kv_heads, page_size, head_dim]`

### 2. Plan-Run Pattern

FlashInfer uses a two-stage pattern for optimal performance:
1. **Plan Stage**: Analyzes the workload and creates an execution plan
2. **Run Stage**: Executes the attention computation using the plan

This decouples scheduling from computation, enabling better load balancing for variable-length sequences.

## Core Classes

### BatchPrefillWithPagedKVCacheWrapper

Used for prefill (processing multiple tokens) and append operations with paged KV cache.

```python
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    float_workspace_buffer,  # Pre-allocated workspace buffer
    kv_layout="NHD"         # or "HND"
)
```

#### Key Methods:

**plan()** - Prepares the wrapper for batch attention computation:
```python
wrapper.plan(
    qo_indptr,          # Query indices pointer [batch_size + 1]
    paged_kv_indptr,    # KV page indices pointer [batch_size + 1]
    paged_kv_indices,   # Page indices [total_num_pages]
    paged_kv_last_page_len,  # Last page lengths [batch_size]
    num_qo_heads,       # Number of query/output heads
    num_kv_heads,       # Number of key/value heads
    head_dim,           # Head dimension
    page_size,          # Page size
    causal=True,        # Apply causal mask
    q_data_type=torch.float16,
    kv_data_type=torch.float16
)
```

**run()** - Executes the attention computation:
```python
output = wrapper.run(q, paged_kv_cache)
# or
output, lse = wrapper.run(q, paged_kv_cache, return_lse=True)
```

### BatchDecodeWithPagedKVCacheWrapper

Used for decode operations (single token generation) with paged KV cache.

```python
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    float_workspace_buffer,
    kv_layout="NHD",        # or "HND"
    use_tensor_cores=True   # Enable tensor core acceleration
)
```

The API is similar to prefill wrapper but optimized for single-token generation.

## Index Management

### Understanding Indices

FlashInfer uses several index tensors to manage variable-length sequences:

1. **qo_indptr** (query indices pointer): Cumulative sum of query lengths
   - Shape: `[batch_size + 1]`
   - Example: For sequences of length [5, 3, 7], qo_indptr = [0, 5, 8, 15]

2. **paged_kv_indptr**: Cumulative sum of pages per sequence
   - Shape: `[batch_size + 1]`
   - Example: For 2 sequences using [1, 2] pages, paged_kv_indptr = [0, 1, 3]

3. **paged_kv_indices**: Actual page indices for each sequence
   - Shape: `[total_num_pages]`
   - Maps logical page positions to physical page indices

4. **paged_kv_last_page_len**: Number of valid tokens in the last page of each sequence
   - Shape: `[batch_size]`
   - Important for handling partially-filled pages

### Helper Functions

```python
# Get batch indices and positions for tokens
from flashinfer.page import get_batch_indices_positions

batch_indices, positions = get_batch_indices_positions(
    q_indptr,      # Query indices pointer
    seq_lens,      # Sequence lengths
    total_tokens   # Total number of tokens
)
```

## Memory Management

### Workspace Buffers

FlashInfer requires pre-allocated workspace buffers:

```python
# Recommended size: 32-128 MB depending on workload
workspace_buffer = torch.empty(
    128 * 1024 * 1024,  # 128 MB
    dtype=torch.uint8,
    device="cuda"
)
```

### Page Size Considerations

- Common page sizes: 1, 8, 16
- Larger page sizes reduce metadata overhead but may waste memory
- Page size = 1 effectively disables paging (dense attention)

## Position Encoding

FlashInfer supports various position encoding modes:

```python
# In plan() or run()
pos_encoding_mode="NONE"       # No position encoding
pos_encoding_mode="ROPE_LLAMA" # RoPE for LLaMA-style models
pos_encoding_mode="ALIBI"      # ALiBi position encoding
```

## Best Practices

1. **Reuse Wrappers**: Create wrappers once and reuse across layers
2. **Plan Once**: For the same batch configuration, plan once and run multiple times
3. **Buffer Management**: Pre-allocate workspace buffers to avoid allocation overhead
4. **Layout Choice**: 
   - NHD is the default and widely supported
   - HND may be more efficient for certain access patterns
5. **Error Handling**: Check return values and handle CUDA errors appropriately

## Common Pitfalls

1. **Layout Mismatch**: Ensure KV cache tensor layout matches the kv_layout parameter
2. **Index Errors**: Verify index tensors are correctly constructed
3. **Page Alignment**: Ensure last_page_len values are > 0 and ≤ page_size
4. **Workspace Size**: Insufficient workspace can cause failures
5. **Data Types**: Ensure q_data_type and kv_data_type match actual tensor dtypes

## Example: Complete Batch Attention

```python
import torch
import flashinfer

# Configuration
batch_size = 2
seq_lens = [10, 15]
page_size = 16
num_kv_heads = 8
num_qo_heads = 32
head_dim = 128

# Allocate pages
total_pages = sum((seq_len + page_size - 1) // page_size for seq_len in seq_lens)
kv_cache = torch.randn(
    total_pages, 2, num_kv_heads, page_size, head_dim,
    dtype=torch.float16, device="cuda"
)

# Create queries
total_tokens = sum(seq_lens)
q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")

# Build indices
q_indptr = torch.tensor([0, seq_lens[0], sum(seq_lens)], dtype=torch.int32, device="cuda")
kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
last_page_lens = torch.tensor([10, 15], dtype=torch.int32, device="cuda")

# Create and use wrapper
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace, kv_layout="HND"
)

wrapper.plan(
    q_indptr, kv_indptr, kv_indices, last_page_lens,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    causal=True, q_data_type=torch.float16, kv_data_type=torch.float16
)

output = wrapper.run(q, kv_cache)
```

## Version Notes

This documentation is based on FlashInfer v0.2.x. The library is under active development, so APIs may evolve. Always refer to the official documentation for the latest updates.