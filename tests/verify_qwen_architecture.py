#!/usr/bin/env python3
"""Verify the Qwen3 architecture to understand K normalization."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import inspect

model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/')

print("Loading Qwen3 model from transformers...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

print("\nModel architecture:")
print(model)

# Find the attention module
print("\n" + "="*60)
print("Examining attention layer structure...")

# Get first layer's attention module
first_layer = model.model.layers[0]
attention = first_layer.self_attn

print(f"\nAttention module type: {type(attention)}")
print(f"Attention module attributes: {[attr for attr in dir(attention) if not attr.startswith('_')]}")

# Check for q_norm and k_norm
if hasattr(attention, 'q_layernorm'):
    print(f"\nFound q_layernorm: {type(attention.q_layernorm)}")
    print(f"  Weight shape: {attention.q_layernorm.weight.shape}")
    print(f"  Weight stats: mean={attention.q_layernorm.weight.mean():.4f}, std={attention.q_layernorm.weight.std():.4f}")

if hasattr(attention, 'k_layernorm'):
    print(f"\nFound k_layernorm: {type(attention.k_layernorm)}")
    print(f"  Weight shape: {attention.k_layernorm.weight.shape}")
    print(f"  Weight stats: mean={attention.k_layernorm.weight.mean():.4f}, std={attention.k_layernorm.weight.std():.4f}")
    print(f"  Weight max: {attention.k_layernorm.weight.max():.2f}")

# Get the forward method source
print("\n" + "="*60)
print("Attention forward method source code:")
print("="*60)

try:
    # Get the source code of the forward method
    forward_source = inspect.getsource(attention.forward)
    # Print relevant lines (looking for q_layernorm and k_layernorm usage)
    lines = forward_source.split('\n')
    for i, line in enumerate(lines):
        if 'q_layernorm' in line or 'k_layernorm' in line or 'query_states' in line or 'key_states' in line:
            # Print context around the line
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                prefix = ">>>" if j == i else "   "
                print(f"{prefix} {lines[j]}")
            print()
except Exception as e:
    print(f"Could not get source code: {e}")
    print("\nTrying alternative approach...")
    
    # Try to trace through a forward pass
    print("\nTracing forward pass...")
    with torch.no_grad():
        # Create dummy inputs
        batch_size = 1
        seq_len = 1
        hidden_states = torch.randn(batch_size, seq_len, model.config.hidden_size, dtype=torch.bfloat16)
        
        # Get projections
        print("\n1. QKV projection:")
        qkv = attention.qkv_proj(hidden_states)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        print(f"   Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        
        # Check how q_layernorm is applied
        print("\n2. Checking q_layernorm application:")
        q_reshaped = q.view(batch_size, seq_len, attention.num_heads, attention.head_dim)
        print(f"   Q reshaped: {q_reshaped.shape}")
        
        if hasattr(attention, 'q_layernorm'):
            q_normed = attention.q_layernorm(q_reshaped)
            print(f"   Q after layernorm: {q_normed.shape}")
            print(f"   Q norm type: {type(attention.q_layernorm)}")
            
        # Check how k_layernorm is applied
        print("\n3. Checking k_layernorm application:")
        k_reshaped = k.view(batch_size, seq_len, attention.num_kv_heads, attention.head_dim)
        print(f"   K reshaped: {k_reshaped.shape}")
        
        if hasattr(attention, 'k_layernorm'):
            k_normed = attention.k_layernorm(k_reshaped)
            print(f"   K after layernorm: {k_normed.shape}")
            print(f"   K norm type: {type(attention.k_layernorm)}")

# Check if it's actually RMSNorm or something else
print("\n" + "="*60)
print("Checking normalization type...")

if hasattr(attention, 'k_layernorm'):
    k_norm = attention.k_layernorm
    print(f"\nk_layernorm class: {k_norm.__class__}")
    print(f"k_layernorm base classes: {k_norm.__class__.__bases__}")
    
    # Check if it has the typical RMSNorm attributes
    if hasattr(k_norm, 'eps'):
        print(f"Has eps attribute: {k_norm.eps}")
    if hasattr(k_norm, 'normalized_shape'):
        print(f"Has normalized_shape: {k_norm.normalized_shape}")
    
    # Check the actual implementation
    print("\nChecking if it's RMSNorm or LayerNorm...")
    print(f"Module type: {type(k_norm).__name__}")
    print(f"Has variance_epsilon: {hasattr(k_norm, 'variance_epsilon')}")
    print(f"Has elementwise_affine: {hasattr(k_norm, 'elementwise_affine')}")