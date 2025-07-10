#!/usr/bin/env python3
"""Layer-by-layer comparison between standard and custom kernels."""

import os
import torch
from braidinfer import LLM, SamplingParams

model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/')

# Store outputs from both models
standard_outputs = {}
custom_outputs = {}

# First, collect outputs from standard model
print("Collecting outputs from standard model...")
llm_std = LLM(model=model_path, device='cuda', model_kwargs={'use_custom_kernels': False})

# Hook all layers
for i in range(3):  # First 3 layers
    layer = llm_std.model_runner.model.model.layers[i]
    
    def make_hook(layer_idx, output_dict):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            output_dict[f'layer_{layer_idx}'] = output.clone()
        return hook
    
    layer.register_forward_hook(make_hook(i, standard_outputs))

# Also hook the attention modules
for i in range(3):
    layer = llm_std.model_runner.model.model.layers[i]
    attn = layer.self_attn
    
    def make_attn_hook(layer_idx, output_dict):
        def hook(module, input, output):
            output_dict[f'attn_{layer_idx}'] = output.clone()
        return hook
    
    attn.register_forward_hook(make_attn_hook(i, standard_outputs))

# Generate with standard model
outputs = llm_std.generate("Hi", SamplingParams(temperature=0.0, max_tokens=1))
standard_token = outputs[0]['token_ids'][0]

# Now collect from custom model
print("\nCollecting outputs from custom model...")
llm_custom = LLM(model=model_path, device='cuda', model_kwargs={'use_custom_kernels': True})

# Hook all layers
for i in range(3):
    layer = llm_custom.model_runner.model.model.layers[i]
    layer.register_forward_hook(make_hook(i, custom_outputs))

# Hook attention modules
for i in range(3):
    layer = llm_custom.model_runner.model.model.layers[i]
    attn = layer.self_attn
    attn.register_forward_hook(make_attn_hook(i, custom_outputs))

# Generate with custom model
outputs = llm_custom.generate("Hi", SamplingParams(temperature=0.0, max_tokens=1))
custom_token = outputs[0]['token_ids'][0]

# Compare outputs
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(f"Generated tokens: standard={standard_token}, custom={custom_token}")

# Find first divergence
first_divergence = None
for key in sorted(standard_outputs.keys()):
    if key in custom_outputs:
        std_out = standard_outputs[key]
        cust_out = custom_outputs[key]
        
        # Calculate statistics
        std_mean = std_out.mean().item()
        std_std = std_out.std().item()
        cust_mean = cust_out.mean().item()
        cust_std = cust_out.std().item()
        
        # Check if they're close
        mean_close = abs(std_mean - cust_mean) < 0.01
        std_close = abs(std_std - cust_std) / (std_std + 1e-6) < 0.1
        
        print(f"\n{key}:")
        print(f"  Standard: mean={std_mean:8.6f}, std={std_std:8.6f}")
        print(f"  Custom:   mean={cust_mean:8.6f}, std={cust_std:8.6f}")
        
        if not (mean_close and std_close):
            print(f"  DIVERGENCE: mean_diff={abs(std_mean - cust_mean):.6f}, std_ratio={cust_std/(std_std+1e-6):.2f}")
            if first_divergence is None:
                first_divergence = key
                
                # For attention outputs, check intermediate values
                if key.startswith('attn_'):
                    layer_idx = int(key.split('_')[1])
                    print(f"\n  Investigating {key} divergence...")
                    
                    # Check shapes
                    print(f"  Shapes: standard={std_out.shape}, custom={cust_out.shape}")
                    
                    # Check for NaN/Inf
                    print(f"  Standard has NaN: {torch.isnan(std_out).any().item()}")
                    print(f"  Custom has NaN: {torch.isnan(cust_out).any().item()}")
                    print(f"  Standard has Inf: {torch.isinf(std_out).any().item()}")
                    print(f"  Custom has Inf: {torch.isinf(cust_out).any().item()}")

if first_divergence:
    print(f"\n⚠️  First divergence found at: {first_divergence}")
else:
    print("\n✓ No significant divergence found!")