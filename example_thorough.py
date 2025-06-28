#!/usr/bin/env python3
"""
Thorough test of cascade attention with context rearrangement.
Tests various orderings, duplications, and interference patterns.
"""

import os
import random
import itertools
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer

from nanovllm.engine.context_chunks import ContextChunk, ChunkType, ChunkBuilder
from nanovllm.engine.chunk_registry import ChunkRegistry
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def create_comprehensive_test_contexts():
    """Create a comprehensive set of test contexts with fictional information."""
    
    contexts = {
        # Primary context about a fictional research facility
        "facility": """
        The Meridian Research Institute is located in the underground city of 
        Subterra, 3 kilometers beneath Mount Ethereal. Founded in 2157 by 
        Professor Zara Nightingale, the institute employs exactly 342 researchers. 
        The facility's main project, codenamed "Project Aurora", studies the 
        effects of crystallized moonlight on plant growth.
        """,
        
        # Context about a fictional discovery
        "discovery": """
        In 2159, Dr. Marcus Stormwind discovered that exposure to crystallized 
        moonlight increases tomato growth rate by 847%. This discovery was made 
        accidentally when his lunch sandwich was left in the Aurora Chamber for 
        72 hours. The tomatoes had grown to the size of basketballs.
        """,
        
        # Conflicting information (different numbers)
        "conflict": """
        Some reports claim the Meridian Research Institute employs 500 researchers,
        but official records from Subterra's Department of Underground Affairs 
        confirm the actual number is 342 researchers as of January 2160.
        """,
        
        # Technical details
        "technical": """
        The Aurora Chamber operates at a temperature of -127.5°C and requires 
        exactly 2.7 gigawatts of power. The crystallized moonlight is stored 
        in containers made from Nebulite, a fictional metal with a melting 
        point of 4,892°C. Each container holds 17.3 kilograms of crystals.
        """,
        
        # Irrelevant context #1
        "irrelevant1": """
        The Pacific Ocean is the largest ocean on Earth. Penguins are flightless 
        birds that live primarily in the Southern Hemisphere. The chemical formula 
        for water is H2O.
        """,
        
        # Irrelevant context #2
        "irrelevant2": """
        Shakespeare wrote 37 plays and 154 sonnets. The Great Wall of China is 
        visible from space. Pizza was invented in Naples, Italy.
        """,
        
        # Misleading but related context
        "misleading": """
        The Solar Research Institute, a completely different organization, is 
        located in the city of Skyholm and studies the effects of concentrated 
        sunlight on metal alloys. It employs 178 researchers and was founded 
        by Dr. Luna Dawnbringer in 2145.
        """,
        
        # System prompts with different instructions
        "system_strict": """
        You are a precise AI assistant. Answer questions using ONLY information 
        from the provided context. Be exact with numbers and names. If multiple 
        conflicting values exist, mention the conflict.
        """,
        
        "system_casual": """
        You are a helpful assistant. Answer questions based on the context 
        provided, but feel free to elaborate or explain in a conversational way.
        """,
        
        "system_brief": """
        You are a concise assistant. Give short, direct answers based on the 
        provided context. Maximum 20 words per response.
        """
    }
    
    return contexts


def create_test_questions():
    """Create questions that test specific aspects of the context."""
    
    questions = [
        {
            "id": "basic_fact",
            "question": "How many researchers work at the Meridian Research Institute?",
            "expected": "342",
            "required_contexts": ["facility"]
        },
        {
            "id": "location",
            "question": "Where is the Meridian Research Institute located?",
            "expected": "Subterra, 3 kilometers beneath Mount Ethereal",
            "required_contexts": ["facility"]
        },
        {
            "id": "discovery_detail",
            "question": "By what percentage does crystallized moonlight increase tomato growth?",
            "expected": "847%",
            "required_contexts": ["discovery"]
        },
        {
            "id": "technical_spec",
            "question": "What temperature does the Aurora Chamber operate at?",
            "expected": "-127.5°C",
            "required_contexts": ["technical"]
        },
        {
            "id": "cross_context",
            "question": "Who discovered the tomato growth effect and where do they work?",
            "expected": "Dr. Marcus Stormwind at Meridian Research Institute",
            "required_contexts": ["facility", "discovery"]
        },
        {
            "id": "conflict_resolution",
            "question": "Is there any disagreement about the number of researchers at Meridian?",
            "expected": "conflict between 500 and 342",
            "required_contexts": ["facility", "conflict"]
        },
        {
            "id": "avoid_confusion",
            "question": "Who founded the Meridian Research Institute?",
            "expected": "Professor Zara Nightingale",
            "required_contexts": ["facility"],
            "misleading_contexts": ["misleading"]  # Should not confuse with Luna Dawnbringer
        }
    ]
    
    return questions


def test_context_arrangements(llm: LLM, contexts: Dict[str, str], questions: List[Dict]):
    """Test various arrangements of context."""
    
    print("=== Testing Context Arrangements ===\n")
    
    # Test configurations
    test_configs = [
        {
            "name": "Normal Order",
            "arrangement": lambda q: ["system_strict"] + q.get("required_contexts", [])
        },
        {
            "name": "Reversed Order",
            "arrangement": lambda q: ["system_strict"] + list(reversed(q.get("required_contexts", [])))
        },
        {
            "name": "With Irrelevant Start",
            "arrangement": lambda q: ["system_strict", "irrelevant1"] + q.get("required_contexts", [])
        },
        {
            "name": "With Irrelevant Middle",
            "arrangement": lambda q: ["system_strict"] + \
                         q.get("required_contexts", [])[:1] + \
                         ["irrelevant2"] + \
                         q.get("required_contexts", [])[1:]
        },
        {
            "name": "Duplicate Context",
            "arrangement": lambda q: ["system_strict"] + \
                         q.get("required_contexts", []) + \
                         q.get("required_contexts", [])  # Duplicate
        },
        {
            "name": "With Misleading",
            "arrangement": lambda q: ["system_strict"] + \
                         q.get("misleading_contexts", []) + \
                         q.get("required_contexts", [])
        }
    ]
    
    results = []
    
    for question in questions[:3]:  # Test first 3 questions with all arrangements
        print(f"\nQuestion: {question['question']}")
        print(f"Expected answer should contain: {question['expected']}")
        print("-" * 50)
        
        question_results = []
        
        for config in test_configs:
            # Build context list
            context_keys = config["arrangement"](question)
            
            # Skip if this arrangement doesn't make sense for this question
            if len(context_keys) < 2:
                continue
            
            # Build prompt
            prompt_parts = []
            for key in context_keys:
                if key in contexts:
                    prompt_parts.append(f"{contexts[key]}\n")
            
            prompt_parts.append(f"Question: {question['question']}")
            full_prompt = "\n".join(prompt_parts)
            
            # Generate response with more tokens
            sampling_params = SamplingParams(temperature=0.1, max_tokens=300)
            outputs = llm.generate([full_prompt], sampling_params)
            response = outputs[0]["text"].strip()
            
            # Handle thinking tags if present
            if "<think>" in response and "</think>" in response:
                # Extract content after </think>
                think_end = response.find("</think>")
                if think_end != -1:
                    response = response[think_end + 8:].strip()
            
            # Check if response contains expected information
            success = question['expected'].lower() in response.lower()
            
            # Store result
            result = {
                "config": config["name"],
                "success": success,
                "response": response[:100] + "..." if len(response) > 100 else response
            }
            question_results.append(result)
            
            print(f"{config['name']:25} {'✓' if success else '✗'} {result['response']}")
        
        results.append({
            "question": question["question"],
            "results": question_results
        })
    
    return results


def test_batch_coherence(llm: LLM, contexts: Dict[str, str], questions: List[Dict]):
    """Test batch processing with mixed contexts."""
    
    print("\n=== Testing Batch Coherence ===\n")
    
    # Create multiple prompts with different context combinations
    prompts = []
    expected_answers = []
    
    # Use different system prompts and context orders
    system_prompts = ["system_strict", "system_casual", "system_brief"]
    
    for i, question in enumerate(questions[:6]):
        # Rotate through system prompts
        system = system_prompts[i % len(system_prompts)]
        
        # Build prompt with required contexts
        prompt_parts = [contexts[system]]
        
        # Add contexts in different orders
        if i % 2 == 0:
            # Normal order
            for ctx in question.get("required_contexts", []):
                prompt_parts.append(contexts[ctx])
        else:
            # Reversed order
            for ctx in reversed(question.get("required_contexts", [])):
                prompt_parts.append(contexts[ctx])
        
        # Add some irrelevant context randomly
        if i % 3 == 0:
            prompt_parts.insert(1, contexts["irrelevant1"])
        
        prompt_parts.append(f"Question: {question['question']}")
        prompts.append("\n".join(prompt_parts))
        expected_answers.append(question['expected'])
    
    # Generate all responses in batch with more tokens
    sampling_params = SamplingParams(temperature=0.1, max_tokens=300)
    outputs = llm.generate(prompts, sampling_params)
    
    # Check results
    print("Batch processing results:")
    print("-" * 80)
    
    all_correct = True
    for i, (output, expected) in enumerate(zip(outputs, expected_answers)):
        response = output["text"].strip()
        
        # Handle thinking tags if present
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()
        
        success = expected.lower() in response.lower()
        all_correct &= success
        
        print(f"Q{i+1}: {'✓' if success else '✗'} Expected '{expected}' in response")
        print(f"     Response: {response[:80]}...")
        print()
    
    print(f"Overall batch coherence: {'PASS' if all_correct else 'FAIL'}")
    
    return all_correct


def test_cascade_memory_efficiency(contexts: Dict[str, str]):
    """Test memory efficiency of cascade attention."""
    
    print("\n=== Testing Memory Efficiency ===\n")
    
    # Create registry
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    registry = ChunkRegistry(max_chunks=100)
    
    # Simulate multiple requests with overlapping contexts
    num_requests = 20
    context_keys = list(contexts.keys())
    
    total_tokens_naive = 0
    chunks_registered = set()
    
    for i in range(num_requests):
        # Each request uses 3-5 random contexts
        num_contexts = random.randint(3, 5)
        selected = random.sample(context_keys, num_contexts)
        
        request_tokens = 0
        for key in selected:
            content = contexts[key]
            
            # Register chunk
            chunk = registry.register(
                content,
                ChunkType.SYSTEM_PROMPT if "system" in key else ChunkType.CONTEXT,
                tokenizer=tokenizer
            )
            
            chunks_registered.add(chunk.chunk_id)
            request_tokens += chunk.seq_len
        
        total_tokens_naive += request_tokens
    
    # Calculate actual unique tokens
    unique_tokens = sum(
        registry.get(chunk_id).seq_len 
        for chunk_id in chunks_registered
    )
    
    stats = registry.get_stats()
    
    print(f"Simulated {num_requests} requests")
    print(f"Total token count (naive): {total_tokens_naive}")
    print(f"Unique tokens stored: {unique_tokens}")
    print(f"Memory savings: {(1 - unique_tokens/total_tokens_naive)*100:.1f}%")
    print(f"Cache hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"Number of unique chunks: {len(chunks_registered)}")


def main():
    """Run thorough cascade attention tests."""
    print("Thorough Cascade Attention Testing\n")
    
    # Initialize
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Create LLM with cascade support
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cascade_attention=True,
        chunk_page_ratio=0.5,
        max_num_seqs=8  # Support batch processing
    )
    
    # Get test data
    contexts = create_comprehensive_test_contexts()
    questions = create_test_questions()
    
    # Run tests
    # 1. Test various context arrangements
    arrangement_results = test_context_arrangements(llm, contexts, questions)
    
    # 2. Test batch coherence
    batch_success = test_batch_coherence(llm, contexts, questions)
    
    # 3. Test memory efficiency
    test_cascade_memory_efficiency(contexts)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Context arrangement tests: Completed")
    print(f"Batch coherence: {'PASS' if batch_success else 'FAIL'}")
    print("Memory efficiency: Demonstrated")
    
    print("\n✅ Thorough testing complete!")


if __name__ == "__main__":
    main()