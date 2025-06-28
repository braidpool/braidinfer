# Cascade Attention Integration Guide

## Overview

This guide outlines the steps needed to integrate the cascade attention implementation into the main nano-vllm inference pipeline.

## Integration Points

### 1. Update Model Architecture

Replace the standard `Attention` layer with `CascadeAttention` in the model definition:

```python
# In nanovllm/models/qwen3.py
from nanovllm.layers.cascade_attention import CascadeAttention

# Replace Attention with CascadeAttention
self.attn = CascadeAttention(
    num_heads=self.num_heads,
    head_dim=self.head_dim,
    scale=self.scale,
    num_kv_heads=self.num_kv_heads,
    layer_idx=layer_idx
)
```

### 2. Update ModelRunner

Modify ModelRunner to use CascadePageManager and CascadeWrapperManager:

```python
# In model_runner.py
from nanovllm.engine.cascade_page_manager import CascadePageManager
from nanovllm.engine.cascade_wrapper_manager import CascadeWrapperManager

# Replace PageManager with CascadePageManager
self.page_manager = CascadePageManager(...)

# Replace WrapperManager with CascadeWrapperManager
self.wrapper_manager = CascadeWrapperManager(...)
```

### 3. Update InferenceContext

Add cascade configuration to InferenceContext:

```python
# In inference_context.py
from nanovllm.layers.cascade_attention import CascadeConfig

@dataclass
class InferenceContext:
    # ... existing fields ...
    cascade_config: Optional[CascadeConfig] = None
    
    def get_workspace_buffer(self, size: int) -> torch.Tensor:
        """Get workspace buffer for cascade attention."""
        # Implementation to provide workspace buffer
```

### 4. Update Scheduler

The scheduler needs the most work to support cascade-aware batching:

```python
# In scheduler.py
def schedule(self):
    # Group sequences by shared chunks
    chunk_groups = self._group_by_chunks(sequences)
    
    # Build cascade configuration
    cascade_config = self._build_cascade_config(chunk_groups)
    
    # Create context with cascade info
    context.cascade_config = cascade_config
```

### 5. API Extensions

Add chunk management to the LLM API:

```python
# In llm_engine.py
class LLMEngine:
    def register_chunk(self, content: str, chunk_type: str):
        """Register a reusable context chunk."""
        return self.chunk_registry.register(content, ChunkType[chunk_type])
    
    def generate_with_chunks(self, prompts, chunk_ids, sampling_params):
        """Generate with specified context chunks."""
        # Implementation
```

## Example Integration Flow

1. **Startup**: Initialize cascade components
   ```python
   chunk_registry = ChunkRegistry()
   page_manager = CascadePageManager(...)
   page_manager.chunk_registry = chunk_registry
   ```

2. **Chunk Registration**: Pre-register common chunks
   ```python
   system_chunk = engine.register_chunk(
       "You are a helpful assistant",
       "SYSTEM_PROMPT"
   )
   ```

3. **Request Processing**: Use chunks in generation
   ```python
   outputs = engine.generate_with_chunks(
       prompts=["Explain this code"],
       chunk_ids=[system_chunk.chunk_id, code_chunk.chunk_id],
       sampling_params=params
   )
   ```

## Configuration Options

Add to Config class:
```python
class Config:
    # Cascade attention settings
    enable_cascade_attention: bool = True
    chunk_page_ratio: float = 0.5
    max_cascade_levels: int = 3
    chunk_registry_size: int = 1000
    chunk_persistence_dir: Optional[str] = None
```

## Performance Considerations

1. **Batch Composition**: Group sequences with similar chunks
2. **Level Assignment**: Put most-shared chunks at level 0
3. **Page Allocation**: Monitor chunk vs dynamic page usage
4. **Cache Management**: Set appropriate eviction policies

## Testing Strategy

1. **Unit Tests**: Test each component individually (✓ done)
2. **Integration Tests**: Test full pipeline with cascade
3. **Performance Tests**: Benchmark vs baseline attention
4. **Stress Tests**: Many concurrent requests with shared chunks

## Rollout Plan

1. **Phase 1**: Replace attention layer, test with cascade_config=None
2. **Phase 2**: Enable cascade for specific models/requests
3. **Phase 3**: Full cascade support with chunk management API
4. **Phase 4**: Production deployment with monitoring

## Monitoring

Add metrics for:
- Chunk cache hit rate
- Page allocation distribution
- Cascade level utilization
- Attention state merge overhead
- Memory savings from deduplication

This phased approach allows gradual integration while maintaining stability.