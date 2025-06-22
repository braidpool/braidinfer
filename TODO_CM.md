# Context Manager Implementation Plan for nano-vLLM

## Overview
A fine-grained KV cache control system where ALL content (input and output) is tracked as persistent chunks requiring manual management. No automatic deletion or cleanup - users have full control over what stays in memory.

## Core Design Principles
1. **Everything is a chunk**: All KV cache content, including model-generated outputs, is tracked as addressable chunks
2. **Content-based addressing**: Each chunk is identified by SHA256 hash for deduplication and retrieval
3. **Manual lifecycle management**: Chunks persist until explicitly deleted by the user
4. **Selective inference**: Chunks can be activated/deactivated to control what's visible during inference
5. **Memory hierarchy**: Chunks can be moved between GPU/CPU/disk tiers

## Architecture

### 1. Chunk Management
- **Chunk Definition**: A chunk is a sequence of blocks (each block = 256 tokens) with:
  - SHA256 hash for content addressing
  - Activation state (active/inactive)
  - Memory tier (GPU/CPU/disk)
  - Metadata (source, timestamp, tags)
  - Reference count for shared blocks

### 2. Output Tracking
- **Key Innovation**: Model outputs are automatically chunked as they're generated
- Each generation creates a new chunk that must be manually managed
- Output chunks are linked to their input context for provenance
- Chunks can be composed/concatenated to form new chunks

### 3. Virtual Block Tables
- **Shadow Page Tables**: Maintain virtual->physical block mappings
- **Active Block Lists**: Track which blocks participate in attention
- **Zero-copy activation**: Toggle chunks without moving data

## Implementation Steps

### Phase 1: Core Infrastructure (Completed)
- [x] Extended Block class with activation state and SHA256 hashing
- [x] Created ContextManager class for chunk operations
- [x] Added slash commands to CLI interface
- [x] Fixed AttributeError in CLI initialization

### Phase 2: Output Chunking (TODO)
1. **Modify Generation Pipeline**:
   - Hook into token generation to capture outputs
   - Automatically create chunks for generated sequences
   - Link output chunks to input context

2. **Implement ChunkInfo Extensions**:
   ```python
   class ChunkInfo:
       hash: str
       blocks: List[int]
       token_count: int
       is_active: bool
       metadata: Dict
       parent_chunks: List[str]  # Input chunks that led to this output
       generation_params: Dict   # Temperature, top_p, etc.
   ```

3. **Add Output Management Commands**:
   - `/output list` - Show all generated output chunks
   - `/output activate <hash>` - Include output in future contexts
   - `/output delete <hash>` - Permanently remove output chunk
   - `/output tag <hash> <tag>` - Tag outputs for organization

### Phase 3: Attention Filtering (Critical)
1. **Modify Attention Mechanism**:
   - Pass active block information to attention layers
   - Implement block filtering in FlashAttention
   - Handle CUDA graph compatibility (eager mode when filtering)

2. **Virtual Memory Management**:
   ```python
   class VirtualBlockTable:
       def __init__(self):
           self.virtual_to_physical = {}
           self.active_blocks = set()
       
       def map_blocks(self, seq: Sequence) -> List[int]:
           """Map virtual blocks to physical, filtering inactive"""
           return [self.virtual_to_physical[v] 
                   for v in seq.virtual_blocks 
                   if v in self.active_blocks]
   ```

3. **Integration Points**:
   - Modify `model_runner.prepare_prefill()` to use virtual tables
   - Update `model_runner.prepare_decode()` for filtered blocks
   - Adjust block table construction in attention

### Phase 4: Memory Hierarchy
1. **Implement Tiered Storage**:
   - GPU tier: Active blocks in VRAM
   - CPU tier: Inactive blocks in system RAM
   - Disk tier: Archived blocks on storage

2. **Movement Operations**:
   ```python
   class MemoryManager:
       def move_to_cpu(self, block_id: int):
           """Move block from GPU to CPU memory"""
       
       def move_to_gpu(self, block_id: int):
           """Move block from CPU to GPU memory"""
       
       def archive_to_disk(self, chunk_hash: str):
           """Archive entire chunk to disk"""
   ```

3. **Automatic Tiering Policies** (Optional):
   - LRU eviction from GPU to CPU
   - Compression for disk storage
   - Prefetching based on activation patterns

### Phase 5: Advanced Features
1. **Chunk Composition**:
   - Merge multiple chunks into new chunks
   - Split chunks at arbitrary boundaries
   - Create "view" chunks that reference others

2. **Chunk Relationships**:
   - Track parent-child relationships
   - Build conversation trees
   - Enable branching dialogues

3. **Persistence**:
   - Save/load chunk database
   - Export/import individual chunks
   - Synchronize across sessions

## Technical Challenges & Solutions

### 1. CUDA Graph Compatibility
**Problem**: CUDA graphs require fixed tensor shapes, but filtering changes shapes
**Solution**: 
- Use eager mode when any blocks are inactive
- Cache CUDA graphs for common activation patterns
- Implement fast path for "all active" case

### 2. Position Embeddings
**Problem**: Need correct position encoding with filtered blocks
**Solution**: 
- Positions are tied to tokens, not blocks
- No adjustment needed - positions remain sequential
- RoPE embeddings work correctly with filtering

### 3. Memory Overhead
**Problem**: Tracking metadata for every chunk
**Solution**:
- Lazy loading of chunk metadata
- Bloom filters for quick existence checks
- Hierarchical chunk organization

### 4. Output Chunk Granularity
**Problem**: When to create new chunks for outputs?
**Solution**:
- Configurable: per-response, per-N-tokens, or manual
- Default: One chunk per complete response
- API for custom chunking strategies

## Interface Design

### Core Commands
```
/load <file>              - Load document as chunk
/save <hash> <file>       - Save chunk to file
/list [pattern]           - List chunks (with filtering)
/activate <hash>          - Make chunk visible to model
/deactivate <hash>        - Hide chunk from model
/delete <hash>            - Permanently remove chunk
/context                  - Show current active context
/clear                    - Deactivate all chunks
/info <hash>              - Show chunk details
/compose <hash1> <hash2>  - Merge chunks
/tag <hash> <tag>         - Add tag to chunk
```

### Python API
```python
# Basic usage
ctx = ContextManager(block_manager, scheduler)
chunk = ctx.add_chunk("Hello, world!", tokenizer)
ctx.activate_chunk(chunk.hash)

# Output tracking
with ctx.track_output() as output:
    response = model.generate(prompt)
    output_chunk = output.get_chunk()

# Chunk composition
new_chunk = ctx.compose_chunks([chunk1.hash, chunk2.hash])

# Memory management
ctx.move_to_cpu(chunk.hash)
ctx.archive_to_disk(chunk.hash)
```

## Benefits
1. **Perfect Memory**: Never lose context or outputs
2. **Reproducibility**: Exact recreation of any conversation state
3. **Efficiency**: Deduplication via content addressing
4. **Flexibility**: Fine-grained control over visible context
5. **Debugging**: Inspect exact model inputs/outputs

## Next Steps
1. Implement output chunking in generation pipeline
2. Complete attention filtering mechanism
3. Add memory hierarchy support
4. Extensive testing with various workloads
5. Performance optimization for large chunk counts