# Model Testing Plan - Simple to Complex

## 1. **GPT-2 Small (124M)** - Simplest
- **Why Start Here**: 
  - No RoPE, uses learned positional embeddings
  - Simple MHA (Multi-Head Attention) without GQA
  - Well-understood architecture
  - Small size makes debugging easier
- **Key Differences**:
  - `num_heads = num_kv_heads = 12` (no GQA)
  - No layer normalization variations
  - Standard attention mechanism

## 2. **OPT-125M** - Slightly More Complex
- **Why Next**: 
  - Similar size to GPT-2
  - Still no RoPE
  - Introduces some Meta optimizations
  - Good baseline for Facebook/Meta models
- **Key Differences**:
  - Different activation functions
  - Slightly different layer norm placement

## 3. **Pythia-160M** - Introduces Modern Features
- **Why Good Intermediate**:
  - Uses RoPE (like Qwen)
  - Still has `num_heads = num_kv_heads` (no GQA)
  - EleutherAI's clean implementation
  - Good for testing RoPE implementation
- **Key Differences**:
  - RoPE instead of learned embeddings
  - More similar to modern models

## 4. **LLaMA-2-7B** or **Mistral-7B** - Full Modern Architecture
- **Why Before Qwen**:
  - Has GQA (Group Query Attention)
  - Uses RoPE
  - Well-documented architecture
  - Many reference implementations
- **Key Differences**:
  - `num_kv_heads < num_heads` (GQA)
  - SwiGLU activation
  - RMSNorm

## 5. **Qwen2-0.5B** - Target Model
- **Most Complex**:
  - GQA with specific ratios
  - RoPE with potential scaling
  - QKV bias options
  - Additional normalization on Q and K

## Testing Strategy

```python
# Test each model with:
1. Simple completion: "The capital of France is"
2. Basic math: "2 + 2 ="
3. Simple instruction: "Count to 5:"

# For each model check:
- Weights load correctly
- KV cache fills properly  
- Attention scores are reasonable
- Output is coherent
```

## Quick Model Links
- GPT-2: `gpt2` (HuggingFace built-in)
- OPT: `facebook/opt-125m`
- Pythia: `EleutherAI/pythia-160m`
- Mistral: `mistralai/Mistral-7B-v0.1` (might be too large)
- TinyLlama: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (good alternative)

## Debugging Order
1. Start with GPT-2 - get it working perfectly
2. Move to Pythia - debug RoPE
3. Try TinyLlama - debug GQA
4. Finally tackle Qwen3 with all features combined