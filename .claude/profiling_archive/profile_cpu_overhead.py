#!/usr/bin/env python3
"""Profile CPU overhead in the generation loop."""

import os
import time
import torch
import cProfile
import pstats
from io import StringIO
from braidinfer import LLM, SamplingParams

def profile_cpu_overhead():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    print("=== CPU Overhead Profiling ===")
    
    # Create LLM instance
    llm = LLM(
        model_path,
        enforce_eager=False,
        enable_cuda_graph=False,
        max_model_len=1024
    )
    
    # Warmup
    print("\nWarming up...")
    llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
    
    # Profile the generation loop
    print("\nProfiling generation loop...")
    
    profiler = cProfile.Profile()
    
    # Start profiling
    torch.cuda.synchronize()
    start_time = time.time()
    profiler.enable()
    
    # Generate tokens
    outputs = llm.generate(
        ["The weather today is"],
        SamplingParams(max_tokens=20, temperature=0.0)
    )
    
    profiler.disable()
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Print timing
    elapsed = end_time - start_time
    print(f"\nGeneration time: {elapsed:.3f}s")
    print(f"Throughput: {20/elapsed:.2f} tokens/s")
    
    # Print profiling results
    print("\n=== Top 30 Functions by Cumulative Time ===")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Print profiling results by total time
    print("\n=== Top 30 Functions by Total Time ===")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Save detailed profile
    profiler.dump_stats('cpu_profile.prof')
    print("\nDetailed profile saved to cpu_profile.prof")
    
    # Manual timing of key components
    print("\n=== Manual Component Timing ===")
    
    # Get references to key methods
    scheduler = llm.scheduler
    model_runner = llm.model_runner
    
    # Time scheduler step
    num_iterations = 100
    
    # Create a request for timing
    from braidinfer.engine.llm_request import LLMRequest
    request = LLMRequest(
        request_id="test",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=1)
    )
    
    # Add request
    scheduler.add_request(request)
    
    # Time scheduler steps
    scheduler_times = []
    for _ in range(num_iterations):
        start = time.time()
        scheduler.schedule()
        scheduler_times.append(time.time() - start)
    
    print(f"Scheduler.schedule(): {sum(scheduler_times)/len(scheduler_times)*1000:.3f} ms avg")
    
    # Time other components if we have sequences
    if scheduler.running:
        sequences = scheduler.running[:1]
        
        # Time prepare_decode
        prepare_times = []
        for _ in range(num_iterations):
            start = time.time()
            input_ids, positions = model_runner.prepare_decode(sequences)
            prepare_times.append(time.time() - start)
        
        print(f"prepare_decode(): {sum(prepare_times)/len(prepare_times)*1000:.3f} ms avg")
        
        # Time run_model (fewer iterations due to GPU work)
        model_times = []
        torch.cuda.synchronize()
        for _ in range(10):
            start = time.time()
            logits = model_runner.run_model(input_ids, positions, sequences, is_prefill=False)
            torch.cuda.synchronize()
            model_times.append(time.time() - start)
        
        print(f"run_model(): {sum(model_times)/len(model_times)*1000:.3f} ms avg")

if __name__ == "__main__":
    profile_cpu_overhead()