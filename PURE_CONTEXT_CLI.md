# Pure Context Management CLI

## Overview

The Nano-vLLM CLI has been transformed from a turn-based chat interface into a pure KV cache management system. This provides direct control over what goes into the KV cache and when inference is performed.

## Key Changes

### 1. No Automatic Inference
- Text typed at the prompt is added as a chunk to the context
- No inference is triggered automatically
- Complete control over context building

### 2. Manual Inference with `/infer`
- Only the `/infer` command triggers model inference
- Can optionally append text: `/infer What does this mean?`
- Streams output in real-time with statistics

### 3. Removed Chat Interface
- No conversation tracking
- No chat templates or formatting
- No turn-based interaction model

## Usage Examples

### Basic Usage
```
> The capital of France is Paris.
✓ Added chunk: abc123def456... (7 tokens)

> The capital of Germany is Berlin.
✓ Added chunk: 789xyz012345... (7 tokens)

> /infer What are these facts about?
Running inference on 2 blocks (14 tokens) + 6 new tokens...
These facts are about European capitals...

Generated 15 tokens in 0.8s (18.8 tok/s)
✓ Output saved as chunk: def456abc789... (15 tokens)
```

### Building Complex Context
```
> /load document.txt
✓ Added chunk: 123abc456def... (500 tokens)

> Additional context about the document topic
✓ Added chunk: 456def789xyz... (8 tokens)

> /show 123abc
[Shows content of document.txt chunk]

> /deactivate 456def
✓ Chunk deactivated: 456def789xyz...

> /infer
Running inference on 1 blocks (500 tokens)...
[Model generates based only on document.txt]
```

### Context Management Commands

| Command | Description |
|---------|-------------|
| `/load <file>` | Load a file as a context chunk |
| `/context` | Show current context status |
| `/show <hash>` | Display chunk content |
| `/activate <hash>` | Activate a chunk for inference |
| `/deactivate <hash>` | Deactivate a chunk |
| `/populate <hash>` | Pre-populate KV cache for chunk |
| `/compose <h1> <h2>...` | Compose multiple chunks |
| `/tag <hash> <tag>` | Add a tag to a chunk |
| `/save <hash>` | Save chunk to disk |
| `/unload <hash>` | Move chunk to system RAM |
| `/restore <hash>` | Restore chunk to VRAM |
| `/erase <hash>` | Remove chunk completely |
| `/infer [text]` | Run inference on active blocks |
| `/clear` | Clear all chunks |
| `/help` | Show available commands |

## Benefits

1. **Full Control**: You decide exactly what goes into the KV cache
2. **No Hidden Formatting**: No chat templates or role formatting
3. **Flexible Context Building**: Add, remove, and reorder chunks as needed
4. **Efficient Memory Use**: Only populate KV cache for chunks you need
5. **Reproducible**: Same context always produces same inference

## Implementation Details

### Text Input Behavior
When you type text at the prompt (not a command):
1. Text is tokenized using the model's tokenizer
2. A new chunk is created with the tokens
3. The chunk is added to the context manager
4. KV cache is populated for the chunk
5. Chunk hash and token count are displayed

### Inference Behavior
When you run `/infer [optional text]`:
1. All active chunks are gathered in chronological order
2. Optional text is tokenized and appended
3. Direct inference runs on the combined context
4. Output streams to console with syntax highlighting
5. Generated text is saved as a new chunk

### Chunk Lifecycle
1. **Creation**: Text input or file load creates chunks
2. **Activation**: Chunks are active by default, ready for inference
3. **Deactivation**: Temporarily exclude chunks from inference
4. **Memory Management**: Move chunks between GPU/CPU/disk
5. **Deletion**: Permanently remove chunks with `/erase`

## Advanced Usage

### Building a Knowledge Base
```
> /load facts/geography.txt
> /load facts/history.txt
> /load facts/science.txt
> /tag <geo_hash> geography
> /tag <hist_hash> history
> /tag <sci_hash> science

> /deactivate <hist_hash>
> /deactivate <sci_hash>

> /infer Tell me about world capitals
[Inference uses only geography context]
```

### Iterative Context Refinement
```
> Initial prompt for the task
✓ Added chunk: aaa111... (10 tokens)

> /infer
[Generate response]
✓ Output saved as chunk: bbb222... (50 tokens)

> Additional clarification 
✓ Added chunk: ccc333... (8 tokens)

> /infer Continue with more detail
[Model sees all previous context plus new instruction]
```

### Memory-Efficient Long Context
```
> /load chapter1.txt
> /load chapter2.txt
> /load chapter3.txt

> /unload <chapter1_hash>  # Move to CPU
> /unload <chapter2_hash>  # Move to CPU

> /infer Summarize chapter 3
[Only chapter 3 is in GPU memory]

> /restore <chapter1_hash>  # Bring back to GPU
> /infer Compare chapters 1 and 3
```

## Tips

1. Use `/context` frequently to understand your active context
2. Tag chunks meaningfully for easy organization
3. Deactivate rather than erase chunks you might need later
4. Save important chunks to disk for persistence across sessions
5. Use `/show` to verify chunk contents before inference