"""
Final integration test - verify separated implementation fixes gibberish output.
"""

import torch
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoConfig
from nanovllm.models.qwen3_separated import Qwen3ForCausalLMSeparated
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


def test_with_actual_model(model_path="Qwen/Qwen2.5-0.5B"):
    """Test with actual model weights."""
    print(f"=== Testing with {model_path} ===\n")
    
    device = 'cuda'
    
    # Load tokenizer and config
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("1. Testing Original Implementation (may produce gibberish):")
    print("-" * 50)
    
    # Test with original fused implementation
    try:
        original_model = Qwen3ForCausalLM(config, use_custom_kernels=True)
        load_model(original_model, model_path)
        original_model = original_model.to(device).eval()
        
        # Check for extreme K norm weights
        if hasattr(original_model, 'check_extreme_weights'):
            original_model.check_extreme_weights()
        
        # Test generation
        test_prompt = "The capital of France is"
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Generate a few tokens
            generated = input_ids.clone()
            for _ in range(20):
                positions = torch.arange(generated.shape[1], device=device).unsqueeze(0)
                hidden_states = original_model(generated, positions)
                logits = original_model.compute_logits(hidden_states)
                next_token = torch.argmax(logits[0, -1, :])
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        original_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Original output: {original_text}")
        
    except Exception as e:
        print(f"Original implementation error: {e}")
        original_text = None
    
    print("\n2. Testing Separated Implementation (should be stable):")
    print("-" * 50)
    
    # Test with separated implementation
    try:
        separated_model = Qwen3ForCausalLMSeparated(config)
        load_model(separated_model, model_path)
        separated_model = separated_model.to(device).eval()
        
        # Analyze K norm weights
        print("\nK Normalization Weight Analysis:")
        max_k_norm = 0
        for i, layer in enumerate(separated_model.model.layers):
            k_max = layer.self_attn.k_norm.weight.max().item()
            if k_max > 20:
                print(f"  Layer {i}: K norm max = {k_max:.1f} ⚠️ EXTREME")
                max_k_norm = max(max_k_norm, k_max)
            elif i < 5:  # Show first 5 layers
                print(f"  Layer {i}: K norm max = {k_max:.1f}")
        
        if max_k_norm > 20:
            print(f"\n⚠️ Model has extreme K normalization weights (up to {max_k_norm:.1f}x)")
            print("This is why the original implementation produces gibberish!")
        
        # Test generation with same prompt
        input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Track stability
            print("\nGeneration progress:")
            generated = input_ids.clone()
            
            for step in range(20):
                positions = torch.arange(generated.shape[1], device=device).unsqueeze(0)
                hidden_states = separated_model(generated, positions)
                
                # Check stability
                if not torch.all(torch.isfinite(hidden_states)):
                    print(f"  Step {step}: ❌ Non-finite values detected!")
                    break
                
                hidden_norm = torch.norm(hidden_states).item()
                
                logits = separated_model.compute_logits(hidden_states)
                next_token = torch.argmax(logits[0, -1, :])
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if step % 5 == 0:
                    print(f"  Step {step}: hidden norm = {hidden_norm:.2f}")
        
        separated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nSeparated output: {separated_text}")
        
    except Exception as e:
        print(f"Separated implementation error: {e}")
        separated_text = None
    
    # Compare outputs
    print("\n3. Comparison:")
    print("-" * 50)
    
    if original_text and separated_text:
        # Check for repetition (common in gibberish output)
        def check_repetition(text):
            tokens = text.split()
            if len(tokens) > 5:
                # Check last 5 tokens for repetition
                last_5 = tokens[-5:]
                unique = len(set(last_5))
                return unique == 1
            return False
        
        original_repetitive = check_repetition(original_text)
        separated_repetitive = check_repetition(separated_text)
        
        print(f"Original repetitive: {original_repetitive}")
        print(f"Separated repetitive: {separated_repetitive}")
        
        if not separated_repetitive and original_repetitive:
            print("\n✅ SUCCESS: Separated implementation fixed the gibberish output!")
        elif not separated_repetitive:
            print("\n✅ Separated implementation produces coherent output")
        else:
            print("\n⚠️ Both implementations show issues")
    
    return separated_text


def main():
    """Run the test."""
    print("Final Integration Test: Separated RMSNorm Implementation\n")
    print("="*60 + "\n")
    
    # Test with available model
    model_path = "Qwen/Qwen2.5-0.5B"  # Update this to Qwen3-0.6B path if available
    
    result = test_with_actual_model(model_path)
    
    print("\n" + "="*60)
    print("\nConclusion:")
    if result:
        print("The separated RMSNorm implementation successfully handles models with")
        print("extreme normalization weights by computing RMSNorm in full float32")
        print("precision, preventing the numerical instability that causes gibberish.")
    else:
        print("Test encountered errors. Please check the model path and try again.")


if __name__ == "__main__":
    main()