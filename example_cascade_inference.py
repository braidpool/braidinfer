#!/usr/bin/env python3
"""
Example of cascade attention with actual inference to verify coherent output.
Uses randomly generated names/places that the model wouldn't know.
"""

import os
import torch
from transformers import AutoTokenizer

from nanovllm.engine.context_chunks import ContextChunk, ChunkType, ChunkBuilder
from nanovllm.engine.chunk_registry import ChunkRegistry
from nanovllm.engine.cascade_scheduler import CascadeSequence
from nanovllm.sampling_params import SamplingParams
from nanovllm import LLM


def create_test_context():
    """Create test context with fictional information."""
    
    # Fictional company information
    company_context = """
    Zephyrix Technologies is a company founded in 2019 by Dr. Elara Moonwhisper 
    in the city of Crystalton, located in the fictional country of Nebulonia. 
    The company specializes in quantum-enhanced tea brewing systems. Their main 
    product, the BrewMaster 3000, uses quantum entanglement to achieve perfect 
    tea temperature.
    """
    
    # Fictional person bio
    person_context = """
    Dr. Elara Moonwhisper graduated from the University of Stardust Falls with 
    a PhD in Theoretical Beverage Physics. She has published 47 papers on the 
    quantum properties of tea leaves. Her breakthrough came when she discovered 
    that Earl Grey tea exhibits superposition at exactly 87.3 degrees Celsius.
    """
    
    # Irrelevant context (should not affect answers)
    irrelevant_context = """
    The ancient city of Atlantis was known for its advanced plumbing systems.
    Dolphins are highly intelligent marine mammals. The speed of light is 
    approximately 299,792,458 meters per second.
    """
    
    # System prompt
    system_prompt = """You are a helpful AI assistant. Answer questions based 
    only on the provided context. If the information is not in the context, 
    say "I don't have that information in the provided context."
    """
    
    return {
        "system": system_prompt,
        "company": company_context,
        "person": person_context,
        "irrelevant": irrelevant_context
    }


def test_simple_inference():
    """Test basic inference with cascade attention."""
    print("=== Simple Cascade Inference Test ===\n")
    
    # Initialize
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download with:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Create LLM with cascade support
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cascade_attention=True,  # Enable cascade
        chunk_page_ratio=0.5,
        max_num_seqs=4
    )
    
    # Get contexts
    contexts = create_test_context()
    
    # Test questions that require specific context
    test_cases = [
        {
            "question": "Who founded Zephyrix Technologies?",
            "contexts": ["system", "company"],
            "expected_info": "Dr. Elara Moonwhisper"
        },
        {
            "question": "What temperature makes Earl Grey tea exhibit superposition?",
            "contexts": ["system", "person"],
            "expected_info": "87.3 degrees Celsius"
        },
        {
            "question": "Where is Crystalton located?",
            "contexts": ["system", "company", "irrelevant"],  # Include irrelevant
            "expected_info": "Nebulonia"
        },
        {
            "question": "What is the main product of Zephyrix Technologies?",
            "contexts": ["system", "company", "company"],  # Duplicate context
            "expected_info": "BrewMaster 3000"
        }
    ]
    
    # Test each case
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['question']}")
        print(f"Contexts: {test['contexts']}")
        print(f"Expected to mention: {test['expected_info']}")
        
        # Build prompt with contexts
        prompt_parts = []
        for ctx_name in test['contexts']:
            prompt_parts.append(f"Context:\n{contexts[ctx_name]}\n")
        prompt_parts.append(f"Question: {test['question']}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response with more tokens
        sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistency
            max_tokens=200
        )
        
        outputs = llm.generate([full_prompt], sampling_params)
        response = outputs[0]["text"]
        
        # Handle thinking tags if present
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:]
        
        print(f"Response: {response.strip()}")
        
        # Check if response contains expected information
        if test['expected_info'].lower() in response.lower():
            print("✓ Response contains expected information")
        else:
            print("✗ Response missing expected information")
    
    print("\n=== Test Complete ===")


def test_with_chunk_registry():
    """Test using the chunk registry for deduplication."""
    print("\n=== Chunk Registry Inference Test ===\n")
    
    # Initialize
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Create registry
    registry = ChunkRegistry(max_chunks=100)
    
    # Get contexts
    contexts = create_test_context()
    
    # Register chunks
    chunks = {}
    for name, content in contexts.items():
        chunk_type = ChunkType.SYSTEM_PROMPT if name == "system" else ChunkType.CONTEXT
        chunk = registry.register(content, chunk_type, tokenizer=tokenizer)
        chunks[name] = chunk
        print(f"Registered {name}: {chunk.chunk_id[:8]}... ({chunk.seq_len} tokens)")
    
    # Test with duplicate registration
    duplicate_company = registry.register(contexts["company"], ChunkType.CONTEXT, tokenizer=tokenizer)
    print(f"\nDuplicate company chunk has same ID: {duplicate_company.chunk_id == chunks['company'].chunk_id}")
    print(f"Access count: {duplicate_company.access_count}")
    
    # Show registry stats
    stats = registry.get_stats()
    print(f"\nRegistry stats: Hits={stats['hits']}, Misses={stats['misses']}")
    print(f"Memory efficiency: {stats['hit_rate']*100:.1f}% hit rate")


def main():
    """Run cascade inference tests."""
    print("Cascade Attention Inference Tests\n")
    print("Testing coherent output generation with fictional context\n")
    
    # Run simple inference test
    test_simple_inference()
    
    # Run chunk registry test
    test_with_chunk_registry()


if __name__ == "__main__":
    main()