#!/usr/bin/env python3
"""Test RMSNorm behavior with extreme weights."""

import torch
from transformers import AutoModelForCausalLM
import os

model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/')

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
layer0 = model.model.layers[0]
k_norm_hf = layer0.self_attn.k_norm

print("K-norm weight statistics from HuggingFace:")
print(f"  Shape: {k_norm_hf.weight.shape}")
print(f"  Mean: {k_norm_hf.weight.mean():.4f}")
print(f"  Std: {k_norm_hf.weight.std():.4f}")
print(f"  Max: {k_norm_hf.weight.max():.2f}")
print(f"  Min: {k_norm_hf.weight.min():.2f}")

# Test with a sample input
test_input = torch.randn(1, 8, 128, dtype=torch.bfloat16) * 0.2
print(f"\nTest input: shape={test_input.shape}, mean={test_input.mean():.6f}, std={test_input.std():.6f}")

# Apply HuggingFace's k_norm
with torch.no_grad():
    hf_output = k_norm_hf(test_input)
print(f"HF output: mean={hf_output.mean():.6f}, std={hf_output.std():.6f}, max={hf_output.max():.6f}")

# Now let's manually compute what RMSNorm should do
print("\nManual RMSNorm computation:")
# Convert to float32 for computation
x = test_input.float()
# Compute RMS
var = x.pow(2).mean(dim=-1, keepdim=True)
print(f"  Variance: mean={var.mean():.6f}")
# Normalize
normalized = x * torch.rsqrt(var + 1e-6)
print(f"  After normalization: mean={normalized.mean():.6f}, std={normalized.std():.6f}")
# Apply weight
weighted = normalized * k_norm_hf.weight.float()
print(f"  After weighting: mean={weighted.mean():.6f}, std={weighted.std():.6f}, max={weighted.max():.6f}")

# Check if they match
print(f"\nDo they match? {torch.allclose(hf_output.float(), weighted, atol=1e-3)}")

# Now let's see what happens with extreme values
print("\n" + "="*60)
print("Testing with values similar to what we see in Layer 1:")
test_extreme = torch.randn(1, 8, 128, dtype=torch.float32) * 0.2
print(f"Input: mean={test_extreme.mean():.6f}, std={test_extreme.std():.6f}")

# Manual computation
var = test_extreme.pow(2).mean(dim=-1, keepdim=True)
normalized = test_extreme * torch.rsqrt(var + 1e-6)
weighted = normalized * k_norm_hf.weight.float()
print(f"Output: mean={weighted.mean():.6f}, std={weighted.std():.6f}, max={weighted.abs().max():.2f}")

# Check individual head behavior
print("\n" + "="*60)
print("Per-head analysis:")
for head_idx in range(8):
    head_weights = k_norm_hf.weight
    head_input = test_extreme[0, head_idx, :]
    
    # RMSNorm on this head
    var = head_input.pow(2).mean()
    normalized = head_input * torch.rsqrt(var + 1e-6)
    weighted = normalized * head_weights
    
    print(f"\nHead {head_idx}:")
    print(f"  Input std: {head_input.std():.6f}")
    print(f"  Normalized std: {normalized.std():.6f}")
    print(f"  Output std: {weighted.std():.6f}")
    print(f"  Max weight impact: {head_weights.abs().max():.2f}")
    
    # Find which dimensions have extreme weights
    extreme_dims = torch.where(head_weights.abs() > 10)[0]
    if len(extreme_dims) > 0:
        print(f"  Extreme weight dims: {extreme_dims[:5].tolist()}")
        print(f"  Extreme weight values: {head_weights[extreme_dims[:5]].tolist()}")
        print(f"  Output at extreme dims: {weighted[extreme_dims[:5]].tolist()}")