#!/usr/bin/env python3
"""Trace CPU overhead in detail."""

import os
import time
import torch
import torch.utils.benchmark as benchmark
from nanovllm import LLM, SamplingParams

def trace_cpu_overhead():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TORCH_LOGS'] = '+dynamo'
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    print("=== Tracing CPU Overhead ===")
    
    # Test with torch.compile OFF first
    print("\n1. Testing with enforce_eager=True (no torch.compile)")
    
    llm = LLM(
        model_path,
        enforce_eager=True,
        enable_cuda_graph=False,
        max_model_len=1024
    )
    
    # Warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
    
    # Time generation
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(["The weather is"], SamplingParams(max_tokens=20, temperature=0.0))
    torch.cuda.synchronize()
    eager_time = time.time() - start
    print(f"Eager mode: {20/eager_time:.1f} tok/s")
    
    del llm
    torch.cuda.empty_cache()
    
    # Now test with torch.compile ON
    print("\n2. Testing with enforce_eager=False (torch.compile enabled)")
    
    llm = LLM(
        model_path,
        enforce_eager=False,
        enable_cuda_graph=False,
        max_model_len=1024
    )
    
    # Warmup - this will trigger compilation
    print("Warming up (compiling)...")
    warmup_start = time.time()
    llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
    print(f"Compilation took: {time.time() - warmup_start:.1f}s")
    
    # Time generation after compilation
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(["The weather is"], SamplingParams(max_tokens=20, temperature=0.0))
    torch.cuda.synchronize()
    compiled_time = time.time() - start
    print(f"Compiled mode: {20/compiled_time:.1f} tok/s")
    
    # Check if the model is actually compiled
    print("\n3. Checking compilation status")
    model = llm.model_runner.model
    
    # Look for compiled modules
    compiled_count = 0
    for name, module in model.named_modules():
        if hasattr(module, '_compiled'):
            compiled_count += 1
            print(f"Compiled: {name}")
    
    print(f"Total compiled modules: {compiled_count}")
    
    # Profile a single forward pass
    print("\n4. Benchmarking single forward pass")
    
    # Get test inputs
    from nanovllm.engine.sequence import Sequence
    test_seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=1))
    for i in range(10):
        test_seq.append_token(100 + i)
    llm.scheduler.page_manager.allocate(test_seq)
    sequences = [test_seq]
    
    input_ids, positions = llm.model_runner.prepare_decode(sequences)
    
    # Create context
    from nanovllm.engine.inference_context import InferenceContext
    context = InferenceContext(
        sequences=sequences,
        page_manager=llm.model_runner.page_manager,
        wrapper=llm.model_runner.decode_wrapper,
        is_prefill=False,
        cu_seqlens_q=None,
        cascade_data=None
    )
    
    # Benchmark the forward pass
    def forward_fn():
        with torch.no_grad():
            hidden = model(input_ids, positions, context)
            logits = model.compute_logits(hidden, context)
        return logits
    
    # Run benchmark
    timer = benchmark.Timer(
        stmt='forward_fn()',
        globals={'forward_fn': forward_fn},
        num_threads=1
    )
    
    result = timer.timeit(100)
    print(f"Forward pass: {result.mean * 1000:.2f} ms (±{result.std * 1000:.2f} ms)")
    
    # Test without any wrappers
    print("\n5. Testing raw model layers")
    
    # Direct layer test
    hidden = model.model.embed_tokens(input_ids)
    layer = model.model.layers[0]
    
    def layer_fn():
        with torch.no_grad():
            return layer(positions, hidden, context)
    
    timer = benchmark.Timer(
        stmt='layer_fn()',
        globals={'layer_fn': layer_fn},
        num_threads=1
    )
    
    result = timer.timeit(100)
    print(f"Single layer: {result.mean * 1000:.2f} ms (±{result.std * 1000:.2f} ms)")
    print(f"28 layers would be: {result.mean * 28 * 1000:.2f} ms")
    
    # Check if we're hitting Python overhead
    print("\n6. Checking Python vs CUDA time")
    
    # Use CUDA events for precise GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Time GPU only
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        
        hidden = model(input_ids, positions, context)
        logits = model.compute_logits(hidden, context)
        
        end_event.record()
        torch.cuda.synchronize()
    
    gpu_time = start_event.elapsed_time(end_event)
    
    # Time wall clock
    torch.cuda.synchronize()
    wall_start = time.time()
    
    with torch.no_grad():
        hidden = model(input_ids, positions, context)
        logits = model.compute_logits(hidden, context)
    
    torch.cuda.synchronize()
    wall_time = (time.time() - wall_start) * 1000
    
    print(f"GPU time: {gpu_time:.2f} ms")
    print(f"Wall time: {wall_time:.2f} ms")
    print(f"CPU overhead: {wall_time - gpu_time:.2f} ms ({(wall_time - gpu_time)/wall_time*100:.1f}%)")

if __name__ == "__main__":
    trace_cpu_overhead()