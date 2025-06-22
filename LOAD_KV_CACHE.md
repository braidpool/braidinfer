# KV Cache Population on Load

The `/load` command now automatically populates the KV cache when loading files as context chunks. This improves performance by pre-computing the attention states for the loaded content.

## Usage

```bash
/load example_context.txt
```

Output:
```
Loading example_context.txt...
✓ KV cache populated for chunk 8ecfd9fbcf8e5d6a...
✓ Added chunk:
  Hash: 8ecfd9fbcf8e5d6a...
  Size: 47 tokens
  Blocks allocated: 1
  KV cache: populated
```

## Benefits

1. **Faster context switching**: Pre-computed KV cache eliminates the need to recompute attention during generation
2. **Immediate availability**: Loaded context is ready for use without additional processing
3. **Persistent cache**: KV cache data is saved with chunks when using `/save` command

## Implementation Details

- KV cache is populated immediately after tokenization
- Uses the optimized `populate_kv_cache_optimized()` method
- Automatically saves KV cache tensors when chunks are persisted to disk
- Cache status is shown in `/context` command output

## Example Workflow

```bash
# Load a context file with automatic KV cache population
/load example_context.txt

# Context is immediately ready for use
/context

# Generate with the loaded context
What is the capital of France?

# Save the chunk with populated KV cache for later use
/save 8ecfd9fb
```

The loaded context will be included in all subsequent generations until deactivated or erased.