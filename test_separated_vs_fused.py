"""
Direct comparison of fused vs separated implementations.

This tests specifically for numerical differences.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import Qwen2Config
from nanovllm.models.qwen3 import Qwen3AttentionFused
from nanovllm.models.qwen3_separated import Qwen3AttentionSeparated
from nanovllm.kernels.fused_rmsnorm_qkv_minimal_f32 import FusedRMSNormQKVMinimalF32
from nanovllm.kernels.rmsnorm_f32 import RMSNormF32
from nanovllm.kernels.qkv_rope_simple import QKVRoPESimple


def test_kernel_comparison():
    """Compare fused vs separated kernel outputs."""
    print("=== Kernel-Level Comparison ===\n")
    
    device = 'cuda'
    seq_len = 8
    hidden_dim = 256
    num_heads = 8
    num_kv_heads = 2
    head_dim = hidden_dim // num_heads
    
    # Create test inputs
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
    layernorm_weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=device)
    
    # Create QKV weight with proper shape
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_weight = torch.randn(qkv_dim, hidden_dim, dtype=torch.bfloat16, device=device) * 0.1
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Input norm: {torch.norm(hidden_states).item():.4f}")
    
    # Test 1: Fused kernel
    print("\n1. Fused RMSNorm+QKV kernel:")
    q_fused, k_fused, v_fused = FusedRMSNormQKVMinimalF32.forward(
        hidden_states,
        layernorm_weight,
        qkv_weight,
        num_heads,
        num_kv_heads,
        eps=1e-6
    )
    print(f"   Q shape: {q_fused.shape}, norm: {torch.norm(q_fused).item():.4f}")
    print(f"   K shape: {k_fused.shape}, norm: {torch.norm(k_fused).item():.4f}")
    print(f"   V shape: {v_fused.shape}, norm: {torch.norm(v_fused).item():.4f}")
    
    # Test 2: Separated kernels
    print("\n2. Separated RMSNorm then QKV:")
    
    # Step 1: RMSNorm
    normalized = RMSNormF32.forward(hidden_states, layernorm_weight, eps=1e-6)
    print(f"   After RMSNorm: shape {normalized.shape}, norm: {torch.norm(normalized).item():.4f}")
    
    # Step 2: QKV projection (without RoPE for now)
    # Simple matmul for comparison
    qkv = torch.matmul(normalized, qkv_weight.t().to(torch.float32))
    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    q_sep = qkv[:, :q_size].view(seq_len, num_heads, head_dim)
    k_sep = qkv[:, q_size:q_size + k_size].view(seq_len, num_kv_heads, head_dim)
    v_sep = qkv[:, q_size + k_size:].view(seq_len, num_kv_heads, head_dim)
    
    print(f"   Q shape: {q_sep.shape}, norm: {torch.norm(q_sep).item():.4f}")
    print(f"   K shape: {k_sep.shape}, norm: {torch.norm(k_sep).item():.4f}")
    print(f"   V shape: {v_sep.shape}, norm: {torch.norm(v_sep).item():.4f}")
    
    # Compare outputs
    print("\n3. Comparison:")
    q_diff = torch.max(torch.abs(q_fused - q_sep)).item()
    k_diff = torch.max(torch.abs(k_fused - k_sep)).item()
    v_diff = torch.max(torch.abs(v_fused - v_sep)).item()
    
    print(f"   Q max diff: {q_diff:.2e}")
    print(f"   K max diff: {k_diff:.2e}")
    print(f"   V max diff: {v_diff:.2e}")
    
    # Test with extreme K norm weights
    print("\n4. Testing with extreme K normalization:")
    k_norm_weight = torch.ones(head_dim, device=device)
    k_norm_weight[0] = 96.5  # Extreme weight like Qwen3-0.6B
    
    # Apply K normalization to both
    from nanovllm.layers.layernorm import RMSNorm
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device)
    k_norm.weight.data = k_norm_weight
    
    k_fused_normed = k_norm(k_fused)
    k_sep_normed = k_norm(k_sep)
    
    print(f"   K norm weight max: {k_norm_weight.max().item():.1f}")
    print(f"   Fused K after norm: norm={torch.norm(k_fused_normed).item():.4f}")
    print(f"   Separated K after norm: norm={torch.norm(k_sep_normed).item():.4f}")
    
    k_norm_diff = torch.max(torch.abs(k_fused_normed - k_sep_normed)).item()
    print(f"   K diff after normalization: {k_norm_diff:.2e}")
    
    # Check relative error
    k_rel_error = torch.max(torch.abs((k_fused_normed - k_sep_normed) / (k_sep_normed + 1e-8))).item()
    print(f"   K relative error: {k_rel_error:.2%}")
    
    return k_rel_error < 0.01  # Less than 1% error


def test_attention_layer_comparison():
    """Compare full attention layers."""
    print("\n\n=== Attention Layer Comparison ===\n")
    
    device = 'cuda'
    config = Qwen2Config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    
    # Create layers
    fused_layer = Qwen3AttentionFused(
        layer_idx=0,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device)
    
    separated_layer = Qwen3AttentionSeparated(
        layer_idx=0,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
    ).to(device)
    
    # Copy weights
    separated_layer.qkv_proj.weight.data = fused_layer.qkv_proj.weight.data.clone()
    separated_layer.o_proj.weight.data = fused_layer.o_proj.weight.data.clone()
    separated_layer.q_norm.weight.data = fused_layer.q_norm.weight.data.clone()
    separated_layer.k_norm.weight.data = fused_layer.k_norm.weight.data.clone()
    
    # Test input
    seq_len = 16
    hidden_states = torch.randn(seq_len, config.hidden_size, device=device)
    positions = torch.arange(seq_len, device=device)
    layernorm_weight = torch.ones(config.hidden_size, device=device)
    
    # Set extreme K norm weight
    with torch.no_grad():
        fused_layer.k_norm.weight[0] = 96.5
        separated_layer.k_norm.weight[0] = 96.5
    
    print(f"Input norm: {torch.norm(hidden_states).item():.4f}")
    print(f"K norm max weight: {fused_layer.k_norm.weight.max().item():.1f}")
    
    # Forward pass
    with torch.no_grad():
        fused_output = fused_layer(positions, hidden_states, layernorm_weight=layernorm_weight)
        separated_output = separated_layer(positions, hidden_states, layernorm_weight=layernorm_weight)
    
    print(f"\nFused output norm: {torch.norm(fused_output).item():.4f}")
    print(f"Separated output norm: {torch.norm(separated_output).item():.4f}")
    
    # Compare
    max_diff = torch.max(torch.abs(fused_output - separated_output)).item()
    rel_diff = torch.max(torch.abs((fused_output - separated_output) / (separated_output + 1e-8))).item()
    
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2%}")
    
    # Check if outputs are finite
    fused_finite = torch.all(torch.isfinite(fused_output))
    separated_finite = torch.all(torch.isfinite(separated_output))
    
    print(f"\nFused output finite: {fused_finite}")
    print(f"Separated output finite: {separated_finite}")
    
    return separated_finite and rel_diff < 0.1


def main():
    """Run all tests."""
    print("Testing Fused vs Separated Implementation\n")
    print("="*50 + "\n")
    
    # Test 1: Kernel comparison
    kernel_ok = test_kernel_comparison()
    
    # Test 2: Full layer comparison
    layer_ok = test_attention_layer_comparison()
    
    print("\n" + "="*50)
    print("\nSummary:")
    print(f"  Kernel test: {'✅ PASSED' if kernel_ok else '❌ FAILED'}")
    print(f"  Layer test: {'✅ PASSED' if layer_ok else '❌ FAILED'}")
    
    if kernel_ok and layer_ok:
        print("\n✅ All tests passed! Separated implementation is numerically stable.")
    else:
        print("\n⚠️ Some tests failed. Further investigation needed.")


if __name__ == "__main__":
    main()