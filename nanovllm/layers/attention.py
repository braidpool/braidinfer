import torch
from torch import nn
import triton
import triton.language as tl

import flashinfer
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.block_size = 256  # Default block size from nano-vllm

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # Use flashinfer's single prefill for now (will migrate to batch later)
            # Reshape tensors to match flashinfer's expected format
            total_q = q.shape[0]
            total_kv = k.shape[0]
            
            # For prefill, flashinfer expects flattened tensors with cumulative lengths
            o = flashinfer.single_prefill_with_kv_cache(
                q=q,  # [total_q, num_heads, head_dim]
                k=k,  # [total_kv, num_kv_heads, head_dim]
                v=v,  # [total_kv, num_kv_heads, head_dim]
                kv_layout="NHD",  # Using NHD layout to match current tensor format
                pos_encoding_mode="NONE",  # Will add RoPE support later
                sm_scale=self.scale,
                causal=True,
            )
        else:    # decode
            # For decode, we need to reshape the paged KV cache to match flashinfer's expectations
            # k_cache and v_cache are [num_blocks, block_size, num_kv_heads, head_dim]
            # We need to flatten them to [total_kv_len, num_kv_heads, head_dim]
            
            # Get the actual KV length from context
            if hasattr(context, 'context_lens') and context.context_lens.numel() > 0:
                kv_len = context.context_lens[0].item()
                # Calculate how many blocks are actually used
                num_blocks_used = (kv_len + self.block_size - 1) // self.block_size
                
                # Reshape cache to contiguous format for now (will optimize with paged later)
                k_flat = k_cache[:num_blocks_used].reshape(-1, self.num_kv_heads, self.head_dim)[:kv_len]
                v_flat = v_cache[:num_blocks_used].reshape(-1, self.num_kv_heads, self.head_dim)[:kv_len]
            else:
                # Fallback: use all available cache
                k_flat = k_cache.reshape(-1, self.num_kv_heads, self.head_dim)
                v_flat = v_cache.reshape(-1, self.num_kv_heads, self.head_dim)
            
            # Process all queries in the batch
            outputs = []
            for i in range(q.shape[0]):
                q_single = q[i]  # [num_heads, head_dim]
                
                # Use flashinfer's single decode
                o_single = flashinfer.single_decode_with_kv_cache(
                    q=q_single,  # [num_heads, head_dim]
                    k=k_flat,    # [kv_len, num_kv_heads, head_dim]
                    v=v_flat,    # [kv_len, num_kv_heads, head_dim]
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    sm_scale=self.scale,
                )
                outputs.append(o_single)
            
            # Stack outputs
            o = torch.stack(outputs, dim=0)
                
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
