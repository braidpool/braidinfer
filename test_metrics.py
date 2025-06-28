#!/usr/bin/env python3
"""Test metrics collection in nano-vllm."""

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
import json

def main():
    # Initialize model
    import os
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    llm = LLMEngine(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )
    
    # Generate some text
    prompts = [
        "Tell me a short joke",
        "What is 2+2?",
    ]
    
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Print outputs
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {output['text']}")
    
    # Get and print metrics
    print("\n" + "="*50)
    print("Performance Metrics:")
    print("="*50)
    metrics = llm.get_metrics()
    print(json.dumps(metrics, indent=2))
    
    llm.exit()

if __name__ == "__main__":
    main()