#!/usr/bin/env python3
"""Profile FlashInfer decode performance specifically."""

import os
import time
import torch
import flashinfer
from braidinfer import LLM, SamplingParams

def profile_flashinfer_decode():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    print("=== FlashInfer Decode Profiling ===")
    
    llm = LLM(
        model_path,
        enforce_eager=False,
        enable_cuda_graph=False,
        max_model_len=1024
    )
    
    # Warmup
    print("\nWarming up...")
    llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
    
    print("\n=== Checking FlashInfer wrapper state ===")
    
    # Check if decode wrapper is properly initialized
    decode_wrapper = llm.model_runner.decode_wrapper
    print(f"Decode wrapper type: {type(decode_wrapper)}")
    print(f"Workspace buffer size: {llm.model_runner.workspace_buffer.shape[0] / 1024 / 1024:.1f} MB")
    
    # Generate a sequence to get the model into decode state
    outputs = llm.generate(["The weather is"], SamplingParams(max_tokens=1, temperature=0.0))
    
    # Now profile the decode operation
    print("\n=== Profiling decode operation ===")
    
    # Get a running sequence
    if llm.scheduler.running:
        sequences = llm.scheduler.running[:1]
        
        # Profile prepare_decode
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            input_ids, positions = llm.model_runner.prepare_decode(sequences)
        torch.cuda.synchronize()
        prepare_time = (time.time() - start) / 100 * 1000
        print(f"prepare_decode: {prepare_time:.3f} ms")
        
        # Check if wrapper is planned
        print(f"\nWrapper cache state:")
        print(f"  Last batch size: {llm.model_runner._last_decode_batch_size}")
        print(f"  Last seq ids: {llm.model_runner._last_decode_seq_ids}")
        print(f"  Last num pages: {llm.model_runner._last_decode_num_pages}")
        
        # Profile the full model forward
        print("\n=== Full model forward timing ===")
        
        # Time with torch profiler
        import torch.profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True
        ) as prof:
            with torch.profiler.record_function("model_forward"):
                torch.cuda.synchronize()
                start = time.time()
                logits = llm.model_runner.run_model(
                    input_ids, positions, sequences, 
                    is_prefill=False, cu_seqlens_q=None, cascade_data=None
                )
                torch.cuda.synchronize()
                forward_time = (time.time() - start) * 1000
        
        print(f"Model forward: {forward_time:.3f} ms")
        
        # Print top operations
        print("\nTop CUDA operations:")
        events = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
        print(events)
        
        # Test raw FlashInfer decode
        print("\n=== Testing raw FlashInfer decode ===")
        
        # Get KV cache info
        kv_cache = llm.model_runner.page_manager.kv_cache
        print(f"KV cache shape: {kv_cache.shape}")
        print(f"KV cache dtype: {kv_cache.dtype}")
        
        # Create a simple test
        batch_size = 1
        num_heads = 16
        head_dim = 64
        
        # Create query tensor
        q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        
        # Get KV indices from page manager
        kv_indices, kv_indptr, last_page_lens = llm.model_runner.page_manager.build_indices_for_sequences(
            sequences, for_prefill=False
        )
        
        # Time the raw decode operation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            output = decode_wrapper.run(q, kv_cache)
        torch.cuda.synchronize()
        raw_time = (time.time() - start) / 100 * 1000
        
        print(f"Raw FlashInfer decode.run(): {raw_time:.3f} ms per call")
        print(f"For 28 layers: {raw_time * 28:.3f} ms")
        
        # Compare with full model
        print(f"\nOverhead: {forward_time - raw_time * 28:.3f} ms ({(forward_time - raw_time * 28) / forward_time * 100:.1f}%)")

if __name__ == "__main__":
    profile_flashinfer_decode()