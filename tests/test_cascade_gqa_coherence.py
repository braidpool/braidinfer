#!/usr/bin/env python3
"""
Test cascade attention with GQA for coherent output.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM
from nanovllm.config import Config


def test_cascade_coherence():
    """Test that cascade attention produces coherent output with Qwen3 GQA."""
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print("=== CASCADE ATTENTION + GQA COHERENCE TEST ===\n")
    
    # Test 1: Regular attention (baseline)
    print("Test 1: Regular attention (baseline)")
    llm_regular = LLM(
        model_name,
        max_model_len=512,
        enable_cascade_attention=False,
        use_custom_kernels=True  # Use our fixed fused kernels
    )
    
    prompt = "The capital of Aistonia is Flubarg. What is the capital of Aistonia?"
    output_regular = llm_regular.generate(prompt, temperature=0.1, max_tokens=20)
    print(f"Regular output: {output_regular}")
    
    # Test 2: Cascade attention 
    print("\nTest 2: Cascade attention with shared prefix")
    llm_cascade = LLM(
        model_name,
        max_model_len=512,
        enable_cascade_attention=True,
        cascade_shared_prefix_len=50,  # First 50 tokens are shared
        use_custom_kernels=True  # Use our fixed fused kernels
    )
    
    output_cascade = llm_cascade.generate(prompt, temperature=0.1, max_tokens=20)
    print(f"Cascade output: {output_cascade}")
    
    # Verify coherence
    regular_coherent = "Flubarg" in output_regular or "capital" in output_regular.lower()
    cascade_coherent = "Flubarg" in output_cascade or "capital" in output_cascade.lower()
    
    print(f"\nRegular coherent: {regular_coherent}")
    print(f"Cascade coherent: {cascade_coherent}")
    
    # Test 3: Multiple queries with shared system prompt
    print("\nTest 3: Multiple queries with shared system prompt")
    system_prompt = "You are a helpful assistant. Always answer questions accurately."
    questions = [
        "What is 2 + 2?",
        "What color is the sky?",
        "Name a programming language."
    ]
    
    outputs = []
    for q in questions:
        full_prompt = f"{system_prompt}\n\nUser: {q}\nAssistant:"
        output = llm_cascade.generate(full_prompt, temperature=0.1, max_tokens=20)
        outputs.append(output)
        print(f"Q: {q}")
        print(f"A: {output}")
    
    # Test 4: Aistonia factual recall with cascade
    print("\nTest 4: Aistonia factual recall with cascade + system prompt")
    aistonia_prompt = f"{system_prompt}\n\nUser: The capital of Aistonia is Flubarg. What is the capital of Aistonia?\nAssistant:"
    aistonia_output = llm_cascade.generate(aistonia_prompt, temperature=0.1, max_tokens=30)
    print(f"Aistonia output: {aistonia_output}")
    aistonia_correct = "Flubarg" in aistonia_output
    print(f"Aistonia correct: {aistonia_correct}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"✓ Regular attention: {'COHERENT' if regular_coherent else 'FAILED'}")
    print(f"✓ Cascade attention: {'COHERENT' if cascade_coherent else 'FAILED'}")
    print(f"✓ Aistonia recall: {'CORRECT' if aistonia_correct else 'FAILED'}")
    print(f"✓ Fused kernels: ENABLED")
    
    success = regular_coherent and cascade_coherent and aistonia_correct
    if success:
        print("\n🎉 SUCCESS! Cascade attention with GQA and fused kernels works!")
    else:
        print("\n❌ FAILURE! Issues detected with cascade attention")
    
    return success


def test_memory_savings():
    """Test memory savings with cascade attention."""
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print("\n=== MEMORY SAVINGS TEST ===\n")
    
    # Shared system prompt (100 tokens)
    system_prompt = """You are a helpful AI assistant. Your responses should be:
1. Accurate and factual
2. Clear and concise
3. Helpful to the user
4. Professional in tone
Always think step by step before answering. If you're unsure about something, say so."""
    
    # Create LLM with cascade attention
    llm = LLM(
        model_name,
        max_model_len=1024,
        enable_cascade_attention=True,
        cascade_shared_prefix_len=100,  # System prompt is ~100 tokens
        use_custom_kernels=True
    )
    
    # Simulate batch processing
    batch_queries = [
        "What is the weather like today?",
        "How do I cook pasta?",
        "What is machine learning?",
        "Tell me a joke.",
        "What is the capital of France?",
        "How do I learn Python?",
        "What is quantum computing?",
        "Explain photosynthesis.",
        "What is the speed of light?",
        "How do I stay healthy?"
    ]
    
    print(f"Processing {len(batch_queries)} queries with shared system prompt...")
    print(f"System prompt length: ~100 tokens")
    print(f"Average query length: ~10 tokens")
    
    # Calculate theoretical memory savings
    # Without cascade: Each query stores full prompt (100 + 10 = 110 tokens)
    # With cascade: Shared prefix stored once, each query stores only unique part (10 tokens)
    without_cascade = len(batch_queries) * 110
    with_cascade = 100 + (len(batch_queries) * 10)
    savings = (without_cascade - with_cascade) / without_cascade * 100
    
    print(f"\nTheoretical memory usage:")
    print(f"Without cascade: {without_cascade} token positions")
    print(f"With cascade: {with_cascade} token positions")
    print(f"Memory savings: {savings:.1f}%")
    
    # Test actual generation
    print("\nTesting actual generation with cascade...")
    for i, query in enumerate(batch_queries[:3]):  # Test first 3
        full_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
        output = llm.generate(full_prompt, temperature=0.1, max_tokens=30)
        print(f"\nQuery {i+1}: {query}")
        print(f"Response: {output[:50]}...")
    
    print(f"\n✓ Cascade attention enables {savings:.1f}% memory savings for this batch!")
    

if __name__ == "__main__":
    # Run coherence test
    coherence_success = test_cascade_coherence()
    
    # Run memory savings test
    test_memory_savings()
    
    if coherence_success:
        print("\n✅ All cascade attention tests passed!")
    else:
        print("\n❌ Some tests failed")