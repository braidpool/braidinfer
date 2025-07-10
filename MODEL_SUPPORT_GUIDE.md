# Model Support Guide: Braidinfer Compatibility and Performance

## Executive Summary

Braidinfer supports various transformer model architectures with varying levels of compatibility and performance. This guide provides a definitive reference for model selection, compatibility requirements, and performance expectations.

## 1. Supported Model Architectures

### 1.1 Core Architecture Families

#### LLaMA-Based Models ✅ RECOMMENDED
- **Architecture**: Standard transformer with RMSNorm, RoPE, GQA
- **Compatibility**: Excellent
- **Examples**: TinyLlama, Mistral, Gemma (untested but expected compatible)

#### Qwen Family ⚠️ LIMITED SUPPORT
- **Architecture**: LLaMA-like with Q/K layer normalization
- **Compatibility**: Model-dependent, many incompatible
- **Examples**: Qwen3-0.6B (incompatible), Qwen3-1.8B (untested)

#### ERNIE Models 🔧 IMPLEMENTATION ISSUES
- **Architecture**: LLaMA-like with variations
- **Compatibility**: Compatible weights but implementation gaps
- **Examples**: ERNIE-4.5-0.3B (needs implementation fixes)

### 1.2 Architecture Requirements

#### Supported Features
- RMSNorm layer normalization
- Rotary Position Embedding (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation functions
- Standard linear projections

#### Unsupported/Problematic Features
- Extreme K normalization weights (>10x)
- LayerNorm (instead of RMSNorm)
- Absolute position embeddings
- Complex attention variants (sparse, local, etc.)

## 2. Performance Characteristics by Model

### 2.1 Benchmark Results

| Model | Size | Performance | Compatibility | Notes |
|-------|------|-------------|--------------|-------|
| TinyLlama-1.1B | 1.1B | ~29.4 tok/s | ✅ Excellent | Reference implementation |
| Qwen3-0.6B | 0.6B | ~29.4 tok/s | ❌ Incompatible | Fused kernels fail |
| ERNIE-4.5-0.3B | 0.3B | N/A | ⚠️ Partial | Implementation issues |

### 2.2 Performance Analysis

#### Standard Performance Expectations
- **Single GPU**: 25-35 tok/s for 0.5-1B models
- **Batch Size 1**: Typical interactive usage
- **No Quantization**: Full precision inference

#### Performance Scaling
- **Batch Size 8**: Up to 237 tok/s (8x improvement)
- **Quantization**: 2-4x improvement potential
- **Fused Kernels**: Currently slower (-7% performance)

#### Memory Usage
- **0.6B Model**: ~1.2GB VRAM
- **1.1B Model**: ~2.2GB VRAM
- **KV Cache**: ~50MB per 1000 tokens

## 3. Compatibility Matrix

### 3.1 Model Compatibility Assessment

#### ✅ COMPATIBLE Models
**TinyLlama-1.1B**
- Compatibility Score: 1.00
- Max K Weight: <5.0
- Status: Full support
- Test Result: Coherent output with all kernel types

#### ❌ INCOMPATIBLE Models
**Qwen3-0.6B**
- Compatibility Score: 0.00
- Max K Weight: 96.5
- Status: Fused kernels incompatible
- Test Result: Gibberish output with custom kernels

#### ⚠️ PARTIAL SUPPORT Models
**ERNIE-4.5-0.3B**
- Compatibility Score: 1.00 (weight compatibility)
- Max K Weight: <5.0
- Status: Implementation gaps
- Test Result: Gibberish with current implementation

### 3.2 Compatibility Criteria

#### Automatic Detection Rules
```python
def assess_compatibility(model):
    criteria = {
        'k_norm_ratio': max_k_weight / mean_k_weight < 20.0,
        'max_k_weight': max_k_weight < 10.0,
        'architecture': has_standard_llama_structure(),
        'weight_distribution': no_extreme_outliers()
    }
    return all(criteria.values())
```

#### Manual Verification Checklist
1. **Architecture Type**: LLaMA-based preferred
2. **Normalization**: RMSNorm required
3. **Position Encoding**: RoPE supported
4. **Weight Analysis**: No extreme values (>10x)
5. **Test Generation**: Verify coherent output

## 4. Known Issues and Workarounds

### 4.1 Critical Issues

#### Qwen3 Extreme K Normalization Weights
**Issue**: K normalization weights up to 96.5x cause numerical instability
**Root Cause**: Small differences amplified through 28 layers
**Workaround**: Use standard kernels only
```python
# Force standard kernels for Qwen3
if 'qwen3' in model_name.lower():
    config.use_fused_kernels = False
```

#### Fused Kernel Performance
**Issue**: Custom Triton kernels 15x slower than PyTorch
**Root Cause**: Poor parallelization and memory access patterns
**Workaround**: Disable custom kernels
```python
config.use_custom_kernels = False  # Default behavior
```

#### ERNIE Implementation Gaps
**Issue**: Gibberish output despite compatible weights
**Root Cause**: Missing architecture-specific implementations
**Workaround**: Under investigation

### 4.2 Configuration Issues

#### Embedding Scaling
**Issue**: Models may require specific embedding scaling
**Solution**: Qwen3 needs `1/sqrt(hidden_size)` scaling
```python
if model_type == 'qwen3':
    embeddings *= (1.0 / math.sqrt(config.hidden_size))
```

#### RoPE Theta Values
**Issue**: Non-standard theta values cause divergence
**Solution**: Use model-specific theta values
```python
rope_theta_values = {
    'qwen3': 1000000.0,
    'ernie': 500000.0,
    'llama': 10000.0
}
```

#### Corrupted Bias Values
**Issue**: Checkpoints may contain corrupted bias tensors
**Solution**: Validate and ignore when config specifies no bias
```python
if not config.attention_bias:
    # Ignore bias values from checkpoint
    bias = None
```

### 4.3 Numerical Stability

#### Precision Requirements
- **Critical Operations**: Use float32 for accumulation
- **Storage**: Keep weights in original precision
- **Conversion Points**: Match PyTorch behavior exactly

#### Error Amplification Detection
```python
def detect_amplification_risk(model):
    max_weights = []
    for layer in model.layers:
        if hasattr(layer.self_attn, 'k_norm'):
            max_weight = layer.self_attn.k_norm.weight.abs().max()
            max_weights.append(max_weight)
    
    return max(max_weights) if max_weights else 0.0
```

## 5. Recommendations for Model Selection

### 5.1 Production Deployment

#### Tier 1: Recommended (Production Ready)
- **TinyLlama-1.1B**: Proven compatibility, stable performance
- **LLaMA-2/3 variants**: Expected compatibility (untested)
- **Mistral models**: Expected compatibility (untested)

#### Tier 2: Supported with Limitations
- **Qwen3 models**: Standard kernels only, no fusion
- **Code LLaMA**: Expected compatibility (untested)

#### Tier 3: Experimental/Unsupported
- **ERNIE models**: Implementation work required
- **Specialized architectures**: Case-by-case evaluation

### 5.2 Performance Optimization Strategy

#### For Interactive Use (Batch Size 1)
1. Choose smaller models (0.5-1B parameters)
2. Use standard kernels (avoid custom implementations)
3. Consider quantization for memory-constrained environments
4. Enable streaming for better user experience

#### For Batch Processing
1. Use larger batch sizes (4-8)
2. Enable all available optimizations
3. Consider larger models with better batch efficiency
4. Implement CUDA graphs for reduced overhead

#### For Development/Testing
1. Use TinyLlama-1.1B as reference
2. Test compatibility before deployment
3. Run numerical validation scripts
4. Monitor for performance regressions

### 5.3 Selection Criteria

#### Model Size Considerations
- **<1B parameters**: Best for single GPU, interactive use
- **1-3B parameters**: Good balance of capability and performance
- **>3B parameters**: Requires multiple GPUs or quantization

#### Architecture Preferences
1. **LLaMA-based**: Highest compatibility
2. **Standard components**: RMSNorm, RoPE, GQA
3. **Proven implementations**: Well-tested in llama.cpp
4. **Moderate weight distributions**: No extreme values

### 5.4 Validation Workflow

#### Pre-Deployment Checklist
1. **Architecture Analysis**: Verify supported components
2. **Weight Validation**: Check for extreme values
3. **Compatibility Test**: Run automated assessment
4. **Output Quality**: Generate test samples
5. **Performance Benchmark**: Measure actual throughput

#### Continuous Monitoring
1. **Numerical Stability**: Monitor for drift
2. **Performance Tracking**: Compare with baselines
3. **Error Logging**: Capture and analyze failures
4. **User Feedback**: Track quality issues

## 6. Future Development Roadmap

### 6.1 Short-term Improvements
- Fix ERNIE model implementation gaps
- Add support for Mistral/Gemma architectures
- Improve custom kernel performance
- Implement quantization support

### 6.2 Medium-term Goals
- Automatic model compatibility detection
- Model-specific optimization profiles
- Better error handling and diagnostics
- Enhanced streaming capabilities

### 6.3 Long-term Vision
- Universal model architecture support
- Adaptive precision optimization
- Advanced quantization techniques
- Multi-GPU scaling support

## Conclusion

Braidinfer provides solid support for LLaMA-based model architectures with excellent compatibility and reasonable performance. The key to successful deployment is careful model selection, proper compatibility assessment, and realistic performance expectations. Focus on proven architectures like TinyLlama for production use, and thoroughly test any new model before deployment.

For optimal results:
- Use TinyLlama-1.1B for reference implementations
- Avoid models with extreme weight distributions
- Prioritize standard kernels over experimental optimizations
- Plan for batch processing to achieve higher throughput
- Implement proper validation and monitoring workflows