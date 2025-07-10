# Guide to Fixing Numerical Instability in Qwen3 Models

This guide is for developers who are experiencing numerical instability (e.g., exploding norms, NaN values) when implementing the Qwen3-0.6B model in a new environment. These issues are common when porting models, and they almost always stem from small, easy-to-miss architectural details.

The `llama.cpp` implementation of this model is working correctly, which confirms that the model weights are valid. The instability in other projects is therefore due to implementation differences. Here are the most likely culprits and a step-by-step plan to fix them.

## The Root of the Problem: Missed Architectural Details

Modern transformer models have many small but critical details. Missing even one of these can lead to catastrophic numerical instability. For Qwen3, there are two particularly important details that are often overlooked:

1.  **Embedding Scaling:** The input embeddings must be scaled by `1 / sqrt(hidden_size)`.
2.  **RoPE Theta Value:** The Rotary Position Embedding (RoPE) uses a non-standard `theta` value of `1,000,000`.

## Action Plan for All Interns

Both the `Braidinfer` and `dynamic_kv_cache` projects are likely suffering from one or both of these issues. Here is a clear, actionable plan to resolve them.

### Step 1: Fix the Embedding Scaling (High-Impact)

This is the most likely cause of the ~10x norm difference reported by the `dynamic_kv_cache` intern.

*   **Action:** In your embedding lookup function (e.g., `embedding_lookup` in `huggingface_model.cu` or the equivalent in `Braidinfer`), after you have retrieved the embedding vectors from the weight matrix, you **must** scale them.

    *   Add a simple CUDA kernel that multiplies each element of the embedding tensor by `1.0f / sqrtf(hidden_size)`.
    *   The `hidden_size` for Qwen3-0.6B is `1024`, so you will be scaling by `1.0f / 32.0f = 0.03125`.

*   **Verification:** After implementing this, the norm of the initial `hidden_states` tensor should be much smaller and should closely match the norm of the PyTorch reference implementation.

### Step 2: Verify the RoPE Theta Value

An incorrect `theta` value will cause the model to diverge after a few layers.

*   **Action:** Find your RoPE implementation (e.g., in `attention_kernels.cu` or `layers/attention.py`) and ensure that the `theta` value is set to `1000000.0f`. Do not use the default value of `10000.0f`.

### Step 3: Systematic Layer-by-Layer Verification

If the issues persist after fixing the two items above, you must perform a rigorous, layer-by-layer comparison against a known-good implementation (like the one in `llama.cpp` or a `transformers`-based PyTorch script).

1.  **Create a Reference Implementation:** Write a simple Python script using the `transformers` library to load the Qwen3-0.6B model. Add hooks to the model to save the output of each operation (input norm, Q/K/V projections, attention output, FFN output, etc.) for a single, fixed input sequence.

2.  **Instrument Your Code:** Add similar hooks to your C++ or Python code to save the output of the same operations.

3.  **Compare the Tensors:** Write a script to load the corresponding tensors from both implementations and use `torch.allclose` (or a similar function) to find the *first* operation where the outputs diverge. This is the source of your bug.

### Common Points of Failure

When you find the point of divergence, investigate these common areas:

*   **Weight Transposition:** Are your weight matrices being loaded and used in the correct orientation? Remember that cuBLAS expects column-major matrices, while PyTorch uses row-major.
*   **Activation Functions:** Is your `SiLU` or `SwiGLU` implementation identical to the one used in the reference model?
*   **Bias Terms:** Are you correctly handling bias terms (or the lack thereof) in all linear projections?
*   **Data Types and Precision:** Are you maintaining sufficient precision throughout the forward pass? Avoid unnecessary casts to lower-precision formats.

## Additional Findings: Extreme K Normalization Weights

Recent investigation has revealed that the Qwen3-0.6B model has extreme K normalization weights (up to 96.5x in layer 0, with many layers having values > 20x). This creates exceptional numerical sensitivity that requires special handling.

### Critical Float32 Precision Points

When implementing fused kernels or optimizations, the following operations **must** use float32 precision:

1. **RMS Variance Accumulation**: The sum of squares for computing variance must use float32 accumulators
2. **RMS Normalization Division**: The operation `input / rms * norm_weight` must be computed in float32
3. **Matrix Multiplication Accumulators**: Dot product reductions need float32 accumulation
4. **Output Storage**: For layers with extreme normalization weights, consider storing intermediate outputs in float32

### Implementation Strategy

Based on analysis of llama.cpp's fusion-qwen.cu:
- Keep matrix data in float16/bfloat16 to minimize memory bandwidth
- Use float32 only for critical accumulator operations
- The entire RMS normalization must stay in float32 to avoid precision loss

Example approach:
```python
# Bad: Converting too early loses precision
normalized = (input.to(float16) / rms.to(float16)) * norm_weight.to(float16)

# Good: Keep normalization in float32
normalized_f32 = input.to(float32) / rms * norm_weight.to(float32)
normalized = normalized_f32.to(float16)  # Convert only after normalization
```

### Critical Discovery: BFloat16 Conversion Point Must Match PyTorch

A subtle but critical precision issue has been identified that causes gibberish output even when kernels appear to compute correctly. The issue is the **exact point** where float32 is converted to bfloat16.

#### The Problem

When implementing fused kernels, there are two possible conversion strategies:
1. **Kernel approach (incorrect)**: Compute everything in float32, convert to bfloat16 at the end
2. **PyTorch approach (correct)**: Convert to bfloat16 after normalization but before matrix multiplication

This difference creates small rounding errors (typically ~0.0078) that get amplified 96x by Qwen3's extreme K normalization weights, resulting in completely different model outputs.

#### The Solution

Fused kernels must match PyTorch's conversion behavior exactly:

```python
# Incorrect: Convert after matmul
normalized_f32 = (input_f32 / rms) * norm_weight_f32
output_f32 = matmul(normalized_f32, weight_f32)
output = output_f32.to(bfloat16)  # Different rounding than PyTorch!

# Correct: Convert before matmul (matching PyTorch)
normalized_f32 = (input_f32 / rms) * norm_weight_f32
normalized = normalized_f32.to(bfloat16)  # Convert here!
output = matmul(normalized, weight.to(bfloat16))  # Matmul in bfloat16
```

#### Verification

A correctly implemented kernel should show:
- Exact match with PyTorch (0.000000 difference)
- Stable values even after 96.5x K normalization
- No divergence across multiple layers

#### Key Insight

The numerical differences come from different rounding points, not from computation errors. Even though both approaches are "correct" mathematically, they produce different results due to floating-point rounding. Since the model was trained with PyTorch's specific rounding behavior, kernels must match it exactly.

### Error Amplification

With extreme K normalization weights, even tiny errors get amplified exponentially:
- 0.001 initial error → 0.0965 after K norm (96.5x)
- Through 28 layers: error grows to >10^6

This means that standard float16/bfloat16 computation is insufficient for this model.

### Recommendations

1. **For Custom Kernels**: Always use the minimal float32 approach described above
2. **For Production**: Consider using models without extreme normalization weights
3. **For Debugging**: If you see values exploding at layer 1 or 2, check your precision handling

## Critical Discovery: Corrupted Bias Values in Model Checkpoint

A major source of numerical instability has been identified in the Qwen3-0.6B model checkpoint: **corrupted attention bias values**.

### The Problem

1. **Configuration Mismatch**: The Qwen3 model configuration specifies `attention_bias: False`
2. **Corrupted Values**: Despite this, the model checkpoint contains bias values with extreme magnitudes (10^29 to 10^34) in layers 1-4
3. **Symptom**: When these corrupted bias values are used, layer outputs explode to infinity immediately

### Root Cause Analysis

The issue occurs because:
- The model was likely trained without attention bias (as per config)
- The checkpoint still contains uninitialized or corrupted bias tensors
- Loading code may still load these bias values even though they shouldn't be used
- Fused kernels that apply these corrupted biases cause immediate numerical explosion

### The Solution

When implementing Qwen3, **always ignore attention bias values from the checkpoint**:

```python
# Bad: Using bias from checkpoint even though config says no bias
q, k, v = fused_kernel(hidden_states, norm_weight, qkv_weight, qkv_bias)

# Good: Pass None for bias when config specifies attention_bias=False
q, k, v = fused_kernel(hidden_states, norm_weight, qkv_weight, None)
```

### Implementation Checklist

1. **Check Model Config**: Verify if `attention_bias` is False in the model configuration
2. **Ignore Checkpoint Bias**: If bias is disabled in config, always pass `None` or zero bias
3. **Validate Loaded Weights**: Add checks to ensure bias values are reasonable if they should be used
4. **Test Layer Outputs**: Monitor layer outputs for explosion to infinity (a clear sign of bias corruption)

### Diagnostic Code

To check for bias corruption in your implementation:

```python
# Check if bias values are corrupted
for i, layer in enumerate(model.layers):
    if hasattr(layer.self_attn, 'qkv_proj') and layer.self_attn.qkv_proj.bias is not None:
        bias = layer.self_attn.qkv_proj.bias
        if not torch.isfinite(bias).all():
            print(f"Layer {i} has non-finite bias values!")
        elif bias.abs().max() > 1e6:
            print(f"Layer {i} has extreme bias values: max={bias.abs().max()}")
```

### Key Takeaway

Even if a model's configuration says it doesn't use certain parameters, the checkpoint may still contain corrupted values for those parameters. Always validate that loaded values are reasonable and match the model's configuration. When in doubt, trust the configuration over the checkpoint values.

By following this guide, you should be able to quickly identify and fix the source of the numerical instability. The key is to be systematic, validate each component against a known-good reference, pay special attention to numerical precision in the presence of extreme normalization weights, and be aware of potential corruption in model checkpoints.
