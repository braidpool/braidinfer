# SPRINT.md - Find Compatible Models for Fused Kernels

## Sprint Goal
Systematically test alternative models to find ones that work well with fused kernels, starting with TinyLlama and ERNIE, and implement support for these model types in the engine.

## Background
Qwen3-0.6B is incompatible with fused kernels due to extreme K normalization weights (up to 96.5x). We need to find models with more moderate weight distributions that can benefit from kernel fusion optimizations.

## Target Models
1. **TinyLlama-1.1B**: Located at `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/`
2. **ERNIE-4.5-0.3B**: Located at `~/.cache/huggingface/hub/models--baidu--ERNIE-4.5-0.3B-PT/snapshots/50f620af62e77797fc297d1512628e861537e1de/`

## Sprint Tasks

### Task 1: Implement LLaMA Model Support
- [ ] Create `nanovllm/models/llama.py` for LLaMA architecture
- [ ] Implement `LlamaForCausalLM` class with proper attention mechanism
- [ ] Add support for RMSNorm and SwiGLU activation
- [ ] Integrate with existing attention implementations (standard and fused)
- [ ] Update ModelLoader to recognize "llama" model type
- [ ] Test loading TinyLlama model

### Task 2: Analyze TinyLlama Compatibility
- [ ] Run compatibility checker on TinyLlama
- [ ] Document weight distributions (K/Q norm weights)
- [ ] Compare with Qwen3 weight patterns
- [ ] Test generation with standard kernels
- [ ] Test generation with fused kernels if compatible
- [ ] Benchmark performance differences

### Task 3: Implement ERNIE Model Support
- [ ] Create `nanovllm/models/ernie.py` for ERNIE architecture
- [ ] Implement `ERNIE45ForCausalLM` class
- [ ] Handle ERNIE-specific components (if any)
- [ ] Update ModelLoader to recognize "ernie4_5" model type
- [ ] Test loading ERNIE-4.5 model

### Task 4: Analyze ERNIE Compatibility
- [ ] Run compatibility checker on ERNIE-4.5
- [ ] Document weight distributions
- [ ] Compare with both Qwen3 and TinyLlama
- [ ] Test generation with standard kernels
- [ ] Test generation with fused kernels if compatible
- [ ] Benchmark performance differences

### Task 5: Create Model Comparison Report
- [ ] Create comprehensive weight analysis table
- [ ] Document compatibility scores for all models
- [ ] Compare generation quality across models
- [ ] Benchmark performance with/without fused kernels
- [ ] Identify characteristics of compatible models

### Task 6: Update Compatibility System
- [ ] Add model-specific thresholds if needed
- [ ] Create compatibility profiles for different architectures
- [ ] Update documentation with findings
- [ ] Add new models to test suite

### Task 7: Sprint Review
- [ ] Summarize which models work with fused kernels
- [ ] Document performance improvements achieved
- [ ] Create recommendations for model selection
- [ ] Plan next models to test

## Technical Approach

### LLaMA Architecture Implementation
```python
class LlamaForCausalLM(nn.Module):
    """LLaMA model implementation compatible with nano-vllm."""
    
    def __init__(self, config, use_custom_kernels=False):
        # RMSNorm instead of LayerNorm
        # SwiGLU activation in MLP
        # Rotary embeddings (RoPE)
        # Grouped Query Attention (if applicable)
```

### Model Loading Integration
```python
# In ModelLoader.load_model():
if model_type == "llama":
    model = LlamaForCausalLM(hf_config, use_custom_kernels=use_custom_kernels)
elif model_type == "ernie4_5":
    model = ERNIE45ForCausalLM(hf_config, use_custom_kernels=use_custom_kernels)
```

### Expected Compatibility Analysis
```
Model Weight Analysis:
===================
Model         | Max K Weight | K Ratio | Compatibility | Fused Performance
--------------|--------------|---------|---------------|------------------
Qwen3-0.6B    | 96.5        | 42.3    | INCOMPATIBLE  | N/A
TinyLlama-1.1B| ?           | ?       | ?             | ?
ERNIE-4.5-0.3B| ?           | ?       | ?             | ?
```

## Success Criteria

1. **Model Support**
   - Successfully load and run inference with TinyLlama
   - Successfully load and run inference with ERNIE-4.5
   - Both models produce coherent text output

2. **Compatibility Analysis**
   - Clear compatibility scores for both models
   - At least one model shows better compatibility than Qwen3

3. **Performance Testing**
   - Measure tok/s with standard kernels
   - Measure tok/s with fused kernels (if compatible)
   - Document any performance improvements

4. **Documentation**
   - Model implementation guide
   - Compatibility analysis report
   - Performance benchmark results

## Risk Mitigation

1. **Unknown Architecture Details**
   - Study model configs carefully
   - Reference transformers library implementations
   - Test incrementally with small inputs

2. **Compatibility Issues**
   - Use compatibility checker before enabling fused kernels
   - Have fallback to standard kernels ready
   - Monitor for numerical instabilities

3. **Performance Variations**
   - Test with consistent prompts
   - Use same hardware/settings for all tests
   - Run multiple iterations for reliable metrics

## Expected Outcomes

1. Find at least one model that works well with fused kernels
2. Understand what makes models compatible vs incompatible
3. Expand nano-vllm's model support beyond Qwen3
4. Create a foundation for testing more models in the future