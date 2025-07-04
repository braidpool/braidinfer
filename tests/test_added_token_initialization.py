#!/usr/bin/env python3
"""Test for proper initialization of added token weights."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nanovllm.config import Config
from nanovllm.engine.model_loader import ModelLoader


class TestAddedTokenInitialization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def test_reinitialize_added_tokens(self):
        """Test reinitializing added token weights to fix extreme logits."""
        config = Config(
            model=self.model_path,
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        model = ModelLoader.load_model(config)
        
        # Get vocab size boundaries
        base_vocab_size = self.tokenizer.vocab_size  # 151643
        total_vocab_size = model.lm_head.weight.shape[0]  # 151936
        
        # Check initial state
        regular_norms = model.lm_head.weight[:base_vocab_size].norm(dim=1)
        added_norms = model.lm_head.weight[base_vocab_size:].norm(dim=1)
        
        print(f"\nBefore reinitialization:")
        print(f"Regular tokens mean norm: {regular_norms.mean():.4f}")
        print(f"Added tokens mean norm: {added_norms.mean():.4f}")
        
        # Reinitialize added token weights using same distribution as regular tokens
        with torch.no_grad():
            # Use statistics from regular tokens
            regular_weights = model.lm_head.weight[:base_vocab_size]
            target_std = regular_weights.std()
            
            # Reinitialize added tokens
            num_added = total_vocab_size - base_vocab_size
            device = model.lm_head.weight.device
            dtype = model.lm_head.weight.dtype
            
            new_weights = torch.randn(num_added, model.lm_head.weight.shape[1], device=device, dtype=dtype) * target_std
            model.lm_head.weight[base_vocab_size:] = new_weights
            
            # Also reinitialize embeddings
            embed_regular = model.model.embed_tokens.weight[:base_vocab_size]
            embed_std = embed_regular.std()
            embed_device = model.model.embed_tokens.weight.device
            embed_dtype = model.model.embed_tokens.weight.dtype
            
            new_embeds = torch.randn(num_added, model.model.embed_tokens.weight.shape[1], device=embed_device, dtype=embed_dtype) * embed_std
            model.model.embed_tokens.weight[base_vocab_size:] = new_embeds
        
        # Check after reinitialization
        regular_norms_after = model.lm_head.weight[:base_vocab_size].norm(dim=1)
        added_norms_after = model.lm_head.weight[base_vocab_size:].norm(dim=1)
        
        print(f"\nAfter reinitialization:")
        print(f"Regular tokens mean norm: {regular_norms_after.mean():.4f}")
        print(f"Added tokens mean norm: {added_norms_after.mean():.4f}")
        
        # Test that logits are now reasonable
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tokens = torch.tensor(self.tokenizer.encode(text), device="cuda")
        
        # Simple forward pass - just test embeddings and direct lm_head
        with torch.no_grad():
            hidden = model.model.embed_tokens(tokens.to("cuda"))
            
            # Skip transformer layers, just test norm + lm_head
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden[-1:])  # Last token only
            
            # Check logit statistics
            print(f"\nLogit statistics after reinitialization:")
            print(f"Mean: {logits.mean():.4f}")
            print(f"Std: {logits.std():.4f}")
            print(f"Max: {logits.max():.4f}")
            
            # Check think token
            think_token = 151667
            think_logit = logits[0, think_token].item()
            probs = F.softmax(logits[0], dim=0)
            think_prob = probs[think_token].item()
            
            print(f"\n<think> token after fix:")
            print(f"Logit: {think_logit:.4f}")
            print(f"Probability: {think_prob:.4f}")
            
            # Should be much more reasonable now
            self.assertLess(think_prob, 0.5, "Think token probability should be reasonable after reinitialization")
            
            # Check top predictions
            top_probs, top_indices = torch.topk(probs, 5)
            print(f"\nTop 5 predictions after fix:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = self.tokenizer.decode([idx.item()])
                print(f"  {i+1}. Token {idx.item()} ('{token}'): prob={prob.item():.4f}")


if __name__ == '__main__':
    unittest.main()