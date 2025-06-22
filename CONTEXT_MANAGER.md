# Context Manager for nano-vLLM

## Overview

The Context Manager provides fine-grained control over the KV cache in nano-vLLM, allowing individual chunks of context to be:
- Activated/deactivated without deletion
- Moved between memory tiers (GPU/CPU/disk)
- Reused across sessions via content-based addressing (SHA256)

## Features

### 1. **Content-Based Addressing**
- Each chunk is identified by its SHA256 hash
- Automatic deduplication of identical content
- Persistent addressing across sessions

### 2. **Fine-Grained Control**
- Activate/deactivate chunks without re-decoding
- Only active chunks participate in inference
- Maintain multiple contexts simultaneously

### 3. **Memory Hierarchy** (Partial Implementation)
- **GPU**: Active chunks ready for inference
- **CPU**: Inactive chunks moved to system RAM
- **Disk**: Persistent storage for long-term reuse

## Usage

### CLI Commands

Start the chat interface:
```bash
python cli.py
```

Available slash commands:
- `/load <file>` - Load a file as a context chunk
- `/context` - Show current context status
- `/activate <hash>` - Activate a chunk for inference
- `/deactivate <hash>` - Deactivate a chunk (keep in cache)
- `/save <hash>` - Save chunk to disk
- `/restore <hash>` - Restore chunk from disk
- `/erase <hash>` - Remove chunk completely
- `/clear` - Clear all chunks
- `/help` - Show all commands

### Example Session

```
> /load utils.py
✓ Added chunk:
  Hash: a3b4c5d6e7f8...
  Size: 512 tokens
  Position: 0-511

> /context
Context Status:
  Total: 65536 tokens
  Used: 512 tokens
  Free: 65024 tokens

Active Chunks:
  1. a3b4c5d6e7f8... [512 tokens]

> /deactivate a3b4c5d6e7f8
✓ Chunk deactivated: a3b4c5d6e7f8...

> What does the code do?
[Model responds without seeing utils.py]

> /activate a3b4c5d6e7f8
✓ Chunk activated: a3b4c5d6e7f8...

> What does the code do?
[Model now sees utils.py and can analyze it]
```

## Architecture

### Core Components

1. **Block Manager Extension** (`nanovllm/engine/block_manager.py`)
   - Added activation state to blocks
   - Added SHA256 hashing
   - Support for metadata

2. **Context Manager** (`nanovllm/engine/context_manager.py`)
   - Chunk lifecycle management
   - State tracking (active/inactive/cpu/disk)
   - Save/restore functionality

3. **Attention Mechanism** (`nanovllm/layers/attention.py`)
   - Modified to respect activation states
   - Filters block tables based on active blocks

4. **CLI Integration** (`cli.py`)
   - Slash command parser
   - Interactive context management
   - Visual status display

## Implementation Status

### ✅ Completed
- Block activation/deactivation
- Content-based addressing (SHA256)
- Slash command interface
- Basic save/restore to disk
- Context status tracking

### 🚧 Partial Implementation
- Attention filtering (placeholder)
- Memory hierarchy (GPU↔CPU movement)
- KV cache serialization

### 📋 Future Work
- Full attention mechanism integration
- Efficient KV cache serialization
- CPU offloading with tensor movement
- Compression for disk storage
- Context profiles and templates

## Technical Details

### Block Structure
```python
class Block:
    block_id: int
    ref_count: int
    hash: int  # xxhash for dedup
    sha256: str  # SHA256 for addressing
    token_ids: list[int]
    is_active: bool
    metadata: dict
    memory_tier: str  # gpu/cpu/disk
```

### Chunk Information
```python
@dataclass
class ChunkInfo:
    sha256: str
    blocks: List[int]
    token_ids: List[int]
    size: int
    position: Tuple[int, int]
    metadata: Dict
    status: str  # active/inactive/cpu/disk
```

## Limitations

1. **Attention Filtering**: Currently uses placeholder implementation
2. **Memory Movement**: GPU↔CPU transfer not fully implemented
3. **Persistence**: Only saves metadata, not actual KV tensors
4. **Context Ordering**: Chunks maintain fixed positions

## Testing

Run the test script:
```bash
python test_context_manager.py
```

This will test:
- Adding/removing chunks
- Activation/deactivation
- Save/restore operations
- Context status tracking