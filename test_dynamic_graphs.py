#!/usr/bin/env python3
"""
Test dynamic CUDA graph recomputation.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager

def test_dynamic_graphs():
    """Test dynamic CUDA graph recomputation with context"""
    print("Testing dynamic CUDA graph recomputation...")
    
    model_path = Path.home() / "huggingface" / "Qwen3-0.6B"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    # Set different port to avoid conflicts
    os.environ["MASTER_PORT"] = "2338"
    
    print("Loading model with CUDA graphs enabled...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True
    )
    
    # Important: Use enforce_eager=False to enable CUDA graphs
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
    llm.context_manager = context_mgr
    llm.config.context_manager = context_mgr
    context_mgr.llm_engine = llm
    
    print("\n=== Step 1: Normal generation (should use existing CUDA graphs) ===")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    result = llm.generate(["Hello world"], sampling_params)
    print(f"Generated: {result[0]['text'][:50]}...")
    
    print("\n=== Step 2: Load context and generate (should trigger recomputation) ===")
    # Add a context chunk
    context_content = "Paris is the capital city of France. It is known for the Eiffel Tower and the Louvre Museum."
    chunk = context_mgr.add_chunk(
        content=context_content,
        tokenizer=tokenizer,
        populate_cache=True
    )
    print(f"Added context chunk: {chunk.sha256[:16]}...")
    
    # Generate with context
    context_prompt = context_mgr.build_prompt_with_context(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenizer
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    result = llm.generate([context_prompt], sampling_params)
    print(f"Generated with context: {result[0]['text'][:100]}...")
    
    print("\n=== Step 3: Generate again (should use cached graph) ===")
    result = llm.generate([context_prompt], sampling_params)
    print(f"Generated with context (2nd time): {result[0]['text'][:100]}...")
    
    print("\n=== Graph Statistics ===")
    llm.model_runner.print_graph_stats()

if __name__ == "__main__":
    test_dynamic_graphs()