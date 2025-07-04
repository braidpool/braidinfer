#!/usr/bin/env python3
"""Test for Qwen3 logit bias issue where <think> token gets extremely high probability."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.page_manager import PageManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestQwen3LogitBias(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.config = Config(
            model=self.model_path,
            max_model_len=512,
            kvcache_block_size=16,
            enforce_eager=True
        )
        
        self.model_runner = ModelRunner(self.config)
        self.page_manager = PageManager(
            num_pages=100,
            page_size=self.config.kvcache_block_size,
            num_layers=self.config.hf_config.num_hidden_layers,
            num_kv_heads=self.config.hf_config.num_key_value_heads,
            head_dim=self.config.hf_config.head_dim,
            dtype=self.config.hf_config.torch_dtype
        )
        self.model_runner.set_page_manager(self.page_manager)

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_logit_distribution(self):
        """Test that logits have reasonable distribution."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tokens = self.tokenizer.encode(text)
        
        # Create sequence with temperature > 0 as required by Qwen3
        sampling_params = SamplingParams(max_tokens=1, temperature=0.6)
        seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
        self.page_manager.allocate(seq)
        
        # Capture logits
        captured_logits = []
        
        def capture_logits(module, inputs, outputs):
            if torch.is_tensor(outputs):
                captured_logits.append(outputs.detach().cpu().float())
        
        # Register hook on lm_head
        self.model_runner.model.lm_head.register_forward_hook(capture_logits)
        
        # Run prefill
        with torch.no_grad():
            next_tokens = self.model_runner.run([seq], is_prefill=True)
        
        self.assertTrue(len(captured_logits) > 0, "No logits captured")
        
        logits = captured_logits[-1]
        if logits.dim() == 2:
            last_logits = logits[-1]
        else:
            last_logits = logits
            
        # Check logit statistics
        logit_mean = last_logits.mean().item()
        logit_std = last_logits.std().item()
        
        # Logits should have reasonable variance
        self.assertGreater(logit_std, 1.0, f"Logit std too low: {logit_std}")
        
        # Check for extreme values
        max_logit = last_logits.max().item()
        self.assertLess(max_logit, 50.0, f"Maximum logit too high: {max_logit}")
        
        # Check think token specifically
        think_token = 151667
        if think_token < len(last_logits):
            think_logit = last_logits[think_token].item()
            think_prob = torch.softmax(last_logits, dim=-1)[think_token].item()
            
            # Think token should not dominate completely
            self.assertLess(think_prob, 0.95, 
                           f"<think> token probability too high: {think_prob}")
            
            # Log values for debugging
            print(f"\nLogit stats: mean={logit_mean:.4f}, std={logit_std:.4f}")
            print(f"<think> token: logit={think_logit:.4f}, prob={think_prob:.4f}")
            
            # Get top 5 predictions
            top_k = 5
            top_values, top_indices = torch.topk(last_logits, top_k)
            print(f"\nTop {top_k} predictions:")
            for i, (val, idx) in enumerate(zip(top_values, top_indices)):
                token = self.tokenizer.decode([idx.item()])
                prob = torch.softmax(last_logits, dim=-1)[idx].item()
                print(f"  {i+1}. Token {idx.item()} ('{token}'): "
                      f"logit={val.item():.4f}, prob={prob:.4f}")

    def test_model_weights_initialized(self):
        """Test that model weights are properly initialized."""
        # Check embedding
        embed_weight = self.model_runner.model.model.embed_tokens.weight
        embed_mean = embed_weight.mean().item()
        embed_std = embed_weight.std().item()
        
        # Embeddings should be initialized with reasonable values
        self.assertAlmostEqual(embed_mean, 0.0, places=2)
        self.assertGreater(embed_std, 0.01)
        self.assertLess(embed_std, 0.1)
        
        # Check first layer attention
        first_attn = self.model_runner.model.model.layers[0].self_attn
        qkv_weight = first_attn.qkv_proj.weight
        qkv_mean = qkv_weight.mean().item()
        qkv_std = qkv_weight.std().item()
        
        self.assertAlmostEqual(qkv_mean, 0.0, places=2)
        self.assertGreater(qkv_std, 0.01)
        self.assertLess(qkv_std, 0.1)
        
        # Check lm_head
        lm_head_weight = self.model_runner.model.lm_head.weight
        lm_head_mean = lm_head_weight.mean().item()
        lm_head_std = lm_head_weight.std().item()
        
        self.assertAlmostEqual(lm_head_mean, 0.0, places=2)
        self.assertGreater(lm_head_std, 0.01)
        self.assertLess(lm_head_std, 0.1)

    def test_different_prompts(self):
        """Test logit distribution with different prompts."""
        test_prompts = [
            "Hi",
            "2+2=",
            "The sky is",
            "/no_think Hello"
        ]
        
        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                tokens = self.tokenizer.encode(text)
                
                sampling_params = SamplingParams(max_tokens=1, temperature=0.6)
                seq = Sequence(token_ids=tokens, sampling_params=sampling_params)
                self.page_manager.allocate(seq)
                
                with torch.no_grad():
                    next_token = self.model_runner.run([seq], is_prefill=True)[0]
                
                # Should generate different tokens for different prompts
                print(f"\nPrompt: '{prompt}' -> Token: {next_token} "
                      f"('{self.tokenizer.decode([next_token])}')")


if __name__ == '__main__':
    unittest.main()