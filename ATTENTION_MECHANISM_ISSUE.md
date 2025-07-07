# Attention Mechanism Issue with Custom Kernels

## Summary

The custom fused RMSNorm+QKV kernels produce gibberish output during generation, despite computing correctly in isolation. The issue is in the integration with the attention/KV cache system.

## Symptoms

When using custom kernels, the model generates repetitive nonsense:
- Chat format: " context context context..."
- Plain text: "email,email,email..." or "nalnalnal..."
- Different prompts produce different repetitive patterns

## Root Cause

1. **The fused kernel works correctly**: When tested in isolation, it produces outputs that match PyTorch with <0.002 difference
2. **The issue is in attention computation**: The custom attention path (Qwen3AttentionFused) still uses the standard attention module which expects a proper InferenceContext
3. **Missing KV cache integration**: During generation, the attention module requires:
   - A valid InferenceContext object
   - Proper page_manager for KV cache handling
   - Correct sequence tracking

Without these, the attention computation fails and produces garbage.

## Technical Details

### Working Path (Standard Kernels)
```
Input → Embeddings → LayerNorm → QKV projection → Q/K norm → RoPE → Attention (with KV cache) → Output
```

### Broken Path (Custom Kernels)
```
Input → Embeddings → Fused(LayerNorm+QKV) → Q/K norm → RoPE → Attention (fails without context) → Garbage
```

The attention module (in `nanovllm/layers/attention.py`) has this check:
```python
if context.page_manager is not None:
    context.page_manager.append_kv_to_cache(...)
```

When context is None (as in direct forward passes), this crashes. During generation, the context exists but the integration is somehow broken.

## Evidence

1. **Isolated kernel test**: Perfect match with PyTorch (0.000000 difference)
2. **Simple attention test**: Only 0.002 max difference between standard and custom paths
3. **Generation test**: Completely different outputs
4. **Extreme K norm weights**: Up to 96.5x amplification makes any small error catastrophic

## Next Steps

To fix this issue:

1. **Debug the attention integration**: Trace through how InferenceContext is passed and used
2. **Check KV cache handling**: Ensure the custom path properly manages KV cache
3. **Verify sequence tracking**: Make sure sequence positions are correctly tracked
4. **Consider FlashInfer integration**: The issue might be in how FlashInfer handles the custom tensor layouts

## Workaround

For now, disable custom kernels when using Qwen3 models:
```python
llm = LLM(model="Qwen/Qwen3-0.6B", model_kwargs={"use_custom_kernels": False})
```