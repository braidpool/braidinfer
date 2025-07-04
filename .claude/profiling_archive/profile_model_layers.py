#!/usr/bin/env python3
"""Profile individual model layers to find the bottleneck."""

import os
import time
import torch
from nanovllm import LLM, SamplingParams

def profile_model_layers():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    print("=== Model Layer Profiling ===")
    
    llm = LLM(
        model_path,
        enforce_eager=False,
        enable_cuda_graph=False,
        max_model_len=1024
    )
    
    # Warmup
    print("\nWarming up...")
    llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
    
    # Get a sequence for testing
    from nanovllm.engine.sequence import Sequence
    test_seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=1))
    for i in range(10):
        test_seq.append_token(100 + i)
    llm.scheduler.page_manager.allocate(test_seq)
    sequences = [test_seq]
    
    # Prepare inputs
    input_ids, positions = llm.model_runner.prepare_decode(sequences)
    
    # Profile each component
    print("\n=== Component Timing (100 iterations) ===")
    
    # 1. Embeddings
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        hidden = llm.model_runner.model.model.embed_tokens(input_ids)
    torch.cuda.synchronize()
    embed_time = (time.time() - start) / 100 * 1000
    print(f"Embeddings: {embed_time:.3f} ms")
    
    # 2. Create context
    from nanovllm.engine.inference_context import InferenceContext
    context = InferenceContext(
        sequences=sequences,
        page_manager=llm.model_runner.page_manager,
        wrapper=llm.model_runner.decode_wrapper,
        is_prefill=False,
        cu_seqlens_q=None,
        cascade_data=None
    )
    
    # 3. Single layer timing
    layer = llm.model_runner.model.model.layers[0]
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = layer(positions, hidden, context)
    torch.cuda.synchronize()
    single_layer_time = (time.time() - start) / 100 * 1000
    print(f"Single transformer layer: {single_layer_time:.3f} ms")
    print(f"All 28 layers (estimated): {single_layer_time * 28:.3f} ms")
    
    # 4. Profile attention specifically
    print("\n=== Attention Breakdown ===")
    
    # Get attention module
    attn = layer.self_attn
    
    # Profile QKV projection
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        q = attn.q(hidden)
        k = attn.k(hidden) 
        v = attn.v(hidden)
    torch.cuda.synchronize()
    qkv_time = (time.time() - start) / 100 * 1000
    print(f"QKV projection: {qkv_time:.3f} ms")
    
    # Profile rotary embedding
    q = q.view(1, attn.num_heads, attn.head_dim)
    k = k.view(1, attn.num_kv_heads, attn.head_dim)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        q_rot = attn.rotary_emb(q, positions)
        k_rot = attn.rotary_emb(k, positions)
    torch.cuda.synchronize()
    rope_time = (time.time() - start) / 100 * 1000
    print(f"Rotary embeddings: {rope_time:.3f} ms")
    
    # Profile attention computation
    v = v.view(1, attn.num_kv_heads, attn.head_dim)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):  # Fewer iterations as this includes FlashInfer
        _ = attn.attn(q_rot, k_rot, v, context)
    torch.cuda.synchronize()
    attn_time = (time.time() - start) / 10 * 1000
    print(f"Attention (FlashInfer): {attn_time:.3f} ms")
    
    # 5. Profile MLP
    mlp = layer.mlp
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = mlp(hidden)
    torch.cuda.synchronize()
    mlp_time = (time.time() - start) / 100 * 1000
    print(f"MLP: {mlp_time:.3f} ms")
    
    # 6. Profile layer norm
    ln = layer.input_layernorm
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = ln(hidden)
    torch.cuda.synchronize()
    ln_time = (time.time() - start) / 100 * 1000
    print(f"LayerNorm: {ln_time:.3f} ms")
    
    # Total breakdown
    print("\n=== Total Breakdown ===")
    layer_total = qkv_time + rope_time + attn_time + mlp_time + ln_time * 2  # 2 layer norms
    print(f"Components sum: {layer_total:.3f} ms per layer")
    print(f"Measured: {single_layer_time:.3f} ms per layer")
    print(f"Overhead: {single_layer_time - layer_total:.3f} ms ({(single_layer_time - layer_total)/single_layer_time*100:.1f}%)")
    
    # 7. Test with torch.compile
    print("\n=== Testing torch.compile ===")
    if not llm.config.enforce_eager:
        print("Model is already compiled, skipping...")
    else:
        print("Testing with compiled layer...")
        compiled_layer = torch.compile(layer)
        
        # Warmup
        for _ in range(5):
            _ = compiled_layer(positions, hidden, context)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = compiled_layer(positions, hidden, context)
        torch.cuda.synchronize()
        compiled_time = (time.time() - start) / 100 * 1000
        print(f"Compiled layer: {compiled_time:.3f} ms")
        print(f"Speedup: {single_layer_time/compiled_time:.2f}x")

if __name__ == "__main__":
    profile_model_layers()