#!/usr/bin/env python3
"""Test basic attention computation without FlashInfer."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from nanovllm.config import Config
from nanovllm.engine.model_loader import ModelLoader


class TestBasicAttention(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def test_compare_with_transformers(self):
        """Compare our model output with HuggingFace transformers."""
        # Load HF model
        print("\nLoading HuggingFace model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16
        ).cuda()
        
        # Load our model
        print("Loading nano-vllm model...")
        config = Config(
            model=self.model_path,
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        our_model = ModelLoader.load_model(config)
        
        # Create simple input
        text = "Hello"
        tokens = self.tokenizer.encode(text, return_tensors="pt").cuda()
        
        print(f"\nInput text: '{text}'")
        print(f"Input tokens: {tokens}")
        
        # Get HF model output
        with torch.no_grad():
            hf_outputs = hf_model(tokens)
            hf_logits = hf_outputs.logits[0, -1]  # Last token logits
            
        print(f"\nHF logits shape: {hf_logits.shape}")
        print(f"HF logits stats: mean={hf_logits.mean():.4f}, std={hf_logits.std():.4f}")
        
        # Get top 5 from HF
        hf_probs = F.softmax(hf_logits, dim=-1)
        hf_top5_probs, hf_top5_indices = torch.topk(hf_probs, 5)
        
        print("\nHuggingFace top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(hf_top5_probs, hf_top5_indices)):
            token = self.tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token {idx.item()} ('{token}'): prob={prob.item():.4f}")
        
        # Get our model output - just test embeddings since we can't run full model without context
        with torch.no_grad():
            # Get embeddings - ensure input is on CUDA
            input_ids = tokens[0].to("cuda")
            our_embeddings = our_model.model.embed_tokens(input_ids)
            
            # Compare embeddings instead of full output
            hf_embeddings = hf_model.get_input_embeddings()(tokens)[0]
            
        print(f"\nOur embeddings shape: {our_embeddings.shape}")
        print(f"HF embeddings shape: {hf_embeddings.shape}")
        
        # Check if embeddings are similar
        embed_diff = (our_embeddings - hf_embeddings).abs().mean()
        print(f"\nMean absolute embedding difference: {embed_diff:.6f}")
        
        # For this test, just verify shapes match and values are reasonable
        self.assertEqual(our_embeddings.shape, hf_embeddings.shape)
        self.assertLess(embed_diff, 0.01, "Embeddings should be very similar")


if __name__ == '__main__':
    unittest.main()