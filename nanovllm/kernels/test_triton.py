"""
Test basic Triton functionality with a simple kernel.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def simple_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Very simple attention kernel for testing."""
    # Get program IDs
    head_idx = tl.program_id(0)
    
    # Map to KV head
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load full query vector
    q_offset = head_idx * head_dim
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < head_dim)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = -float('inf')
    sum_exp = 0.0
    
    # Simple loop over sequence
    for pos in range(seq_len):
        # Load K
        k_offset = pos * num_kv_heads * head_dim + kv_head_idx * head_dim
        k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < head_dim)
        
        # Compute score
        score = tl.sum(q * k, axis=0) / tl.sqrt(float(head_dim))
        
        # Update max for numerical stability
        max_score = tl.maximum(max_score, score)
    
    # Second pass for softmax
    for pos in range(seq_len):
        # Load K again
        k_offset = pos * num_kv_heads * head_dim + kv_head_idx * head_dim
        k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < head_dim)
        
        # Compute score
        score = tl.sum(q * k, axis=0) / tl.sqrt(float(head_dim))
        
        # Compute exp(score - max)
        exp_score = tl.exp(score - max_score)
        sum_exp += exp_score
        
        # Load V and accumulate
        v_offset = pos * num_kv_heads * head_dim + kv_head_idx * head_dim
        v = tl.load(v_ptr + v_offset + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < head_dim)
        
        acc += exp_score * v
    
    # Normalize
    acc = acc / sum_exp
    
    # Store output
    out_offset = head_idx * head_dim
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_D), acc, mask=tl.arange(0, BLOCK_D) < head_dim)


def test_simple_kernel():
    """Test the simple kernel."""
    # Small test case
    seq_len = 128
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    
    # Create test tensors
    q = torch.randn(num_heads, head_dim, dtype=torch.float32, device='cuda')
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float32, device='cuda')
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float32, device='cuda')
    output = torch.empty(num_heads, head_dim, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (num_heads,)
    
    simple_attention_kernel[grid](
        q, k, v, output,
        seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_H=1,
        BLOCK_D=BLOCK_D,
    )
    
    print("Kernel executed successfully!")
    
    # Verify with PyTorch
    # Expand K/V for comparison
    k_expanded = k.unsqueeze(1).repeat(1, num_heads // num_kv_heads, 1, 1).reshape(seq_len, num_heads, head_dim)
    v_expanded = v.unsqueeze(1).repeat(1, num_heads // num_kv_heads, 1, 1).reshape(seq_len, num_heads, head_dim)
    
    # Compute attention in PyTorch
    scores = torch.matmul(q.unsqueeze(1), k_expanded.transpose(0, 1).transpose(1, 2)) / math.sqrt(head_dim)
    attn_weights = torch.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(attn_weights, v_expanded.transpose(0, 1)).squeeze(1)
    
    # Check difference
    max_diff = torch.max(torch.abs(output - pytorch_output)).item()
    mean_diff = torch.mean(torch.abs(output - pytorch_output)).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    num_iters = 1000
    start.record()
    for _ in range(num_iters):
        simple_attention_kernel[grid](
            q, k, v, output,
            seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_H=1,
            BLOCK_D=BLOCK_D,
        )
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters
    
    print(f"\nKernel time: {elapsed_ms:.3f} ms")
    print(f"Theoretical throughput: {1000/elapsed_ms:.1f} tok/s")


if __name__ == "__main__":
    test_simple_kernel()