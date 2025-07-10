# Kernel Compatibility Guide

## Overview

Some models are incompatible with fused kernels due to extreme weight values that amplify small numerical differences. This guide explains how to check if your model is compatible with Braidinfer's optimized kernels.

## Quick Start

Check your model's compatibility:

```bash
python -m braidinfer.utils.check_compatibility_cli <model_path>
```

Example:
```bash
python -m braidinfer.utils.check_compatibility_cli ~/models/qwen3-0.6b

# With detailed analysis
python -m braidinfer.utils.check_compatibility_cli ~/models/qwen3-0.6b --verbose

# Export report
python -m braidinfer.utils.check_compatibility_cli ~/models/qwen3-0.6b --export report.txt
```

## Understanding Compatibility

### Status Levels

1. **COMPATIBLE** (Score > 0.8)
   - Safe to use fused kernels
   - Expected performance improvements

2. **WARNING** (Score 0.5-0.8)
   - Fused kernels may work but test thoroughly
   - Monitor output quality

3. **INCOMPATIBLE** (Score < 0.5)
   - Do not use fused kernels
   - Model will produce gibberish output

### What Makes a Model Incompatible?

Models are incompatible when they have:
- Extreme normalization weights (>20x median)
- Many layers with extreme weights (>50%)
- High error amplification potential

Example: Qwen3-0.6B has K normalization weights up to 96.5x, which amplifies tiny numerical differences into completely different outputs.

## Integration with Model Loading

### Automatic Detection

```python
from nanovllm import LLM

# Automatic compatibility checking
llm = LLM(
    "path/to/model",
    model_kwargs={"use_custom_kernels": True}  # Will fall back if incompatible
)
```

### Manual Override (Testing Only)

```python
# Force custom kernels (may produce gibberish!)
llm = LLM(
    "path/to/model",
    model_kwargs={
        "use_custom_kernels": True,
        "force_custom_kernels": True  # Bypass compatibility check
    }
)
```

## Programmatic Usage

```python
from nanovllm.utils.kernel_compatibility import (
    FusedKernelCompatibilityChecker,
    check_model_compatibility
)

# Quick check
can_use, result = check_model_compatibility(model, "model_name")

# Detailed analysis
checker = FusedKernelCompatibilityChecker()
result = checker.check_model(model)

print(f"Status: {result.status}")
print(f"Score: {result.score:.2f}")
print(f"Reason: {result.reason}")
```

## Technical Details

### Metrics Analyzed

1. **Weight Ratios**: max_weight / median_weight
2. **Extreme Layer Count**: Layers with weights > 10x
3. **Error Amplification**: Cumulative effect through layers

### Thresholds

- Safe weight ratio: < 20x
- Warning weight ratio: 15-20x
- Extreme weight threshold: > 10x
- Max safe amplification: < 10x

## Troubleshooting

### Model Shows as Incompatible

If your model is marked incompatible:
1. Use standard kernels (automatic fallback)
2. Consider quantization to reduce weight magnitudes
3. Check if a different model checkpoint is available

### Performance Without Fused Kernels

Standard kernels are still optimized and will provide good performance. The fused kernels offer additional optimization but require numerical stability.

## Future Improvements

- Model-specific kernel tuning
- Adaptive precision based on weight analysis
- Post-training weight regularization tools