# KV Cache Isolation Issue: FlashInfer vs Flash Attention

## The Problem

When running example.py with two prompts:
1. "Please list all prime numbers between 1 and 100."
2. "introduce yourself"

The second response incorrectly references content from the first prompt, showing cross-contamination between sequences.

## Root Cause: Prefix Caching in nano-vLLM

### How BlockManager Works

nano-vLLM implements automatic prefix caching through hash-based deduplication:

1. **Block Hashing**: Each block of tokens (256 tokens) gets hashed using xxhash
2. **Hash Chain**: Each block's hash includes the previous block's hash as a prefix
3. **Deduplication**: If two sequences have blocks with matching hashes, they share the same physical KV cache block
4. **Reference Counting**: Shared blocks use ref_count to track usage

From `block_manager.py`:
```python
def allocate(self, seq: Sequence):
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)  # Chain hashes
        block_id = self.hash_to_block_id.get(h, -1)
        
        if block_id != -1:  # Found matching block
            # Reuse existing block (sequences share KV cache)
            block.ref_count += 1
```

## Why Flash Attention Doesn't Show This Issue

### Flash Attention's Block Table Design

Flash attention uses **block tables** that provide natural isolation:

1. **Indirect Access**: Each sequence has its own block_table that maps logical blocks to physical blocks
2. **Kernel-Level Isolation**: The CUDA kernel only accesses blocks specified in the sequence's block_table
3. **No Cross-Sequence Access**: Even if blocks are shared, the kernel respects sequence boundaries

```python
# Flash attention API
flash_attn_with_kvcache(
    q=q[i:i+1],  # Single sequence query
    k_cache=k_cache,
    v_cache=v_cache,
    block_table=block_table[i:i+1],  # This sequence's blocks only
    cache_seqlens=cache_seqlens[i:i+1],
)
```

The kernel only reads KV cache blocks listed in this sequence's block_table, preventing cross-contamination.

## Why FlashInfer Shows This Issue

### Our Basic Migration Approach

Our migration flattens the paged KV cache without proper isolation:

```python
# Decode path - problematic approach
k_cache_view = k_cache.view(batch_size, num_blocks, self.block_size, num_heads, head_dim)
v_cache_view = v_cache.view(batch_size, num_blocks, self.block_size, num_heads, head_dim)

# We're selecting ALL blocks for the sequence
k_seq = k_cache_view[i]  # Shape: [num_blocks, block_size, num_heads, head_dim]
v_seq = v_cache_view[i]

# Then slicing to sequence length
if not self.cuda_graph_mode:
    seq_len = context.context_lens[i].item()
    k_seq = k_seq.reshape(-1, self.num_heads, self.head_dim)[:seq_len]
    v_seq = v_seq.reshape(-1, self.num_heads, self.head_dim)[:seq_len]
```

### The Critical Difference

1. **Flash Attention**: Uses block_table to select ONLY the blocks belonging to this sequence
2. **Our FlashInfer Migration**: Selects ALL blocks in a batch position, then slices by length

When prefix caching causes block sharing:
- Sequence 1 uses blocks [0, 1, 2]
- Sequence 2 uses blocks [0, 1, 3] (blocks 0,1 are shared due to matching prefix)
- Our migration gives sequence 2 access to ALL blocks [0, 1, 2, 3, ...] then slices

This means sequence 2 can see block 2 (from sequence 1) if the slice length is long enough.

## Why This Happens Now

### Batch Position Assignment

In nano-vLLM, sequences get assigned to batch positions, and our migration assumes:
- Batch position 0 → All blocks for position 0
- Batch position 1 → All blocks for position 1

But with prefix caching, blocks are shared across sequences regardless of batch position!

### The Exposure

Our migration exposed a latent issue:
1. nano-vLLM's prefix caching shares blocks between sequences
2. Flash attention's block_table API naturally isolates sequences
3. Our basic flashinfer migration doesn't maintain this isolation
4. Result: Cross-contamination becomes visible

## The Solution

### Proper FlashInfer Integration

FlashInfer provides proper paged attention with CSR format:

```python
# Convert block tables to CSR format
kv_indices = []  # Physical block indices
kv_indptr = [0]  # Start positions for each sequence

for seq in sequences:
    kv_indices.extend(seq.block_table)
    kv_indptr.append(len(kv_indices))

# Use batch decode wrapper
output = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    kv_indices=kv_indices,
    kv_indptr=kv_indptr,
    # ... other params
)
```

This maintains sequence isolation by:
1. Each sequence specifies exactly which blocks it uses
2. No access to blocks outside the sequence's block_table
3. Proper handling of shared blocks without cross-contamination

## Summary

The issue isn't with flashinfer itself, but with our basic migration approach. Flash attention's block_table API provided automatic isolation that our simple flashinfer migration doesn't maintain. The solution is to properly implement flashinfer's paged attention using CSR format, which will restore sequence isolation while still benefiting from prefix caching.