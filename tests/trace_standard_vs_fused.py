#!/usr/bin/env python3
"""Trace through standard vs fused paths to find the difference."""

import os
import torch
from nanovllm import LLM

model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/')

# Test input
test_ids = torch.tensor([14582], device='cuda')  # "Hello"
positions = torch.tensor([0], device='cuda')

print("=== STANDARD PATH ===")
llm_std = LLM(model=model_path, device='cuda', model_kwargs={'use_custom_kernels': False})

# Trace Layer 1
layer1_std = llm_std.model_runner.model.model.layers[1]
embed_out = llm_std.model_runner.model.model.embed_tokens(test_ids)
embed_scaled = embed_out * (1.0 / (1024 ** 0.5))

# Through Layer 0
hidden = embed_scaled
for i in range(1):
    hidden = llm_std.model_runner.model.model.layers[i](positions, hidden, None)

print(f"Layer 0 output: mean={hidden.mean():.6f}, std={hidden.std():.6f}")

# Now Layer 1 step by step
residual = hidden
normed = layer1_std.input_layernorm(hidden)
print(f"After input norm: mean={normed.mean():.6f}, std={normed.std():.6f}")

# Through attention
attn_out = layer1_std.self_attn(positions, normed, None)
print(f"After attention: mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")

hidden = residual + attn_out
print(f"After residual: mean={hidden.mean():.6f}, std={hidden.std():.6f}")

print("\n=== FUSED PATH ===")
llm_fused = LLM(model=model_path, device='cuda', model_kwargs={'use_custom_kernels': True})

# Trace Layer 1
layer1_fused = llm_fused.model_runner.model.model.layers[1]
embed_out = llm_fused.model_runner.model.model.embed_tokens(test_ids)
embed_scaled = embed_out * (1.0 / (1024 ** 0.5))

# Through Layer 0
hidden = embed_scaled
for i in range(1):
    hidden = llm_fused.model_runner.model.model.layers[i](positions, hidden, None)

print(f"Layer 0 output: mean={hidden.mean():.6f}, std={hidden.std():.6f}")

# Now Layer 1 with fused kernel
# The fused path calls self_attn with additional parameters
residual = hidden
try:
    # This is what happens in the fused path
    attn_out = layer1_fused.self_attn(
        positions, 
        hidden,  # Note: unnormalized hidden states
        None,  # context
        residual,  # residual for fused output
        layer1_fused.input_layernorm.weight  # layernorm weight
    )
    print(f"After fused attention: mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")
    
    # Check if extreme values
    if attn_out.abs().max() > 1000:
        print(f"WARNING: Extreme values detected! max={attn_out.max():.2e}")
        
        # Let's trace inside the attention
        print("\nDebugging inside fused attention...")
        # The issue is likely in how K values are processed after the extreme normalization
        
except Exception as e:
    print(f"Error in fused path: {e}")
    import traceback
    traceback.print_exc()

# Key insight check
print("\n=== KEY INSIGHT ===")
print("The issue is that both standard and fused paths apply RMSNorm to K values.")
print("The extreme K norm weights (up to 96.5) cause massive amplification.")
print("However, the standard path seems to handle this better.")
print("\nLet's check if there's a difference in precision or order of operations...")