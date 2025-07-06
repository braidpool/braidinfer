"""
Simple test to verify the separated implementation produces coherent output.
"""

import torch
from transformers import AutoTokenizer, Qwen2Config
from nanovllm.models.qwen3_separated import Qwen3ForCausalLMSeparated


def create_test_model():
    """Create a small test model."""
    config = Qwen2Config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=512,
        num_hidden_layers=4,
        vocab_size=1000,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    
    model = Qwen3ForCausalLMSeparated(config)
    
    # Set some extreme K norm weights to simulate Qwen3-0.6B
    with torch.no_grad():
        model.model.layers[0].self_attn.k_norm.weight[0] = 96.5
        model.model.layers[1].self_attn.k_norm.weight[0] = 50.0
    
    return model.cuda().eval()


def test_generation():
    """Test basic generation."""
    print("=== Testing Separated Implementation ===\n")
    
    model = create_test_model()
    
    # Simple vocabulary mapping for testing
    vocab = {i: f"token_{i}" for i in range(1000)}
    vocab[1] = "Hello"
    vocab[2] = "world"
    vocab[3] = "!"
    
    # Test input
    input_ids = torch.tensor([[1, 2]], device='cuda')  # "Hello world"
    
    print("Input tokens:", input_ids.tolist())
    
    generated_ids = input_ids.clone()
    
    # Generate 10 tokens
    with torch.no_grad():
        for step in range(10):
            positions = torch.arange(generated_ids.shape[1], device='cuda').unsqueeze(0)
            
            # Forward pass
            hidden_states = model(generated_ids, positions)
            
            # Check stability
            if not torch.all(torch.isfinite(hidden_states)):
                print(f"❌ Non-finite values at step {step}")
                break
            
            hidden_norm = torch.norm(hidden_states).item()
            print(f"Step {step}: hidden norm = {hidden_norm:.2f}")
            
            # Get logits
            logits = model.compute_logits(hidden_states)
            
            # Simple sampling (take most probable)
            next_token = torch.argmax(logits[0, -1, :])
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check for extreme values
            if hidden_norm > 1e6:
                print(f"⚠️ Hidden states exploding at step {step}")
                break
    
    print(f"\nGenerated token IDs: {generated_ids[0].tolist()}")
    
    # Check diversity
    unique_tokens = len(set(generated_ids[0].tolist()))
    print(f"Unique tokens: {unique_tokens}/{generated_ids.shape[1]}")
    
    if unique_tokens > 3:
        print("✅ Generation produced diverse output")
    else:
        print("❌ Generation is repetitive")


def test_layer_by_layer():
    """Test layer-by-layer computation."""
    print("\n\n=== Layer-by-Layer Analysis ===\n")
    
    model = create_test_model()
    
    # Test input
    input_ids = torch.tensor([[1, 2, 3]], device='cuda')
    positions = torch.arange(3, device='cuda').unsqueeze(0)
    
    with torch.no_grad():
        # Get embeddings
        hidden_states = model.model.embed_tokens(input_ids)
        print(f"Raw embedding norm: {torch.norm(hidden_states).item():.4f}")
        hidden_states = hidden_states * (1.0 / (model.config.hidden_size ** 0.5))
        print(f"Scaled embedding norm: {torch.norm(hidden_states).item():.4f}")
        
        print(f"Embedding norm: {torch.norm(hidden_states).item():.4f}")
        
        # Track through each layer
        for i, layer in enumerate(model.model.layers):
            # Store old state
            old_norm = torch.norm(hidden_states).item()
            
            # Forward through layer
            hidden_states = layer(positions, hidden_states)
            
            # New norm
            new_norm = torch.norm(hidden_states).item()
            ratio = new_norm / old_norm if old_norm > 0 else 0
            
            # Check K norm weight
            k_norm_max = layer.self_attn.k_norm.weight.max().item()
            
            print(f"Layer {i}: {old_norm:.4f} -> {new_norm:.4f} "
                  f"(ratio: {ratio:.2f}), K norm max: {k_norm_max:.1f}")
            
            # Check stability
            if not torch.all(torch.isfinite(hidden_states)):
                print(f"❌ Non-finite values in layer {i}")
                return False
    
    print("\n✅ All layers produced finite outputs")
    return True


def main():
    """Run tests."""
    print("Testing Qwen3 Separated Implementation\n")
    print("="*50 + "\n")
    
    # Test 1: Layer stability
    stable = test_layer_by_layer()
    
    # Test 2: Generation
    if stable:
        test_generation()
    
    print("\n" + "="*50)
    print("\nTests completed.")


if __name__ == "__main__":
    main()