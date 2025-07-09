#!/usr/bin/env python3
"""
Debug KV cache access in chunked generation.
"""

import os
import torch
from nanovllm import LLM, ChunkedLLM, ChunkType, SamplingParams
from nanovllm.chunks import Chunk
import hashlib

class DebugAttention:
    """Debug wrapper for attention to see what's happening."""
    
    def __init__(self, original_attn):
        self.original_attn = original_attn
        self.call_count = 0
    
    def __call__(self, q, k, v, context):
        self.call_count += 1
        print(f"\n[DEBUG Attention Call {self.call_count}]")
        print(f"  Q shape: {q.shape}")
        print(f"  K shape: {k.shape}")
        print(f"  V shape: {v.shape}")
        print(f"  Is prefill: {context.is_prefill}")
        
        if hasattr(context.sequences[0], 'active_chunks'):
            chunks = context.sequences[0].active_chunks
            if chunks:
                print(f"  Active chunks: {len(chunks)}")
                for i, chunk in enumerate(chunks):
                    print(f"    Chunk {i}: {chunk.chunk_type}, {chunk.kv_length} tokens")
        
        # Call original attention
        result = self.original_attn(q, k, v, context)
        print(f"  Output shape: {result.shape}")
        
        return result

def debug_chunked_generation():
    """Debug chunked generation with detailed tracing."""
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    
    # Test inputs
    system_prompt = "You are a helpful AI assistant. Please respond in English."
    user_query = "What is the capital of France?"
    
    print("=" * 80)
    print("DEBUGGING CHUNKED GENERATION")
    print("=" * 80)
    
    # Create chunked LLM with fixed tokenization
    class DebugChunkedLLM(ChunkedLLM):
        """ChunkedLLM with debugging."""
        
        def register_chunk(self, content: str, chunk_type: ChunkType, metadata=None):
            """Register a chunk with proper chat template tokenization."""
            chunk_id = hashlib.sha256(f"{chunk_type.value}:{content}".encode()).hexdigest()
            
            try:
                existing = self.registry.get(chunk_id)
                return chunk_id
            except:
                pass
            
            # Tokenize with chat template
            if chunk_type == ChunkType.SYSTEM_PROMPT:
                template_text = f"<|im_start|>system\n{content}<|im_end|>\n"
                token_ids = self.tokenizer.encode(template_text, add_special_tokens=False)
            elif chunk_type == ChunkType.QUERY:
                template_text = f"<|im_start|>user\n{content}<|im_end|>\n"
                token_ids = self.tokenizer.encode(template_text, add_special_tokens=False)
            else:
                token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                content=content,
                token_ids=token_ids,
                token_count=len(token_ids),
                metadata=metadata or {}
            )
            
            self.registry.register(chunk)
            if not chunk.kv_cache_allocated:
                self._prefill_chunk(chunk)
            
            return chunk_id
    
    chunked_llm = DebugChunkedLLM(
        model_path,
        max_chunks=100,
        chunk_memory_ratio=0.5,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Wrap attention layers with debug
    model = chunked_llm.llm.model_runner.model
    for name, module in model.named_modules():
        if hasattr(module, 'attn') and hasattr(module.attn, 'forward'):
            # Wrap the forward method
            original_forward = module.attn.forward
            module.attn.forward = DebugAttention(original_forward)
    
    # Register chunks
    system_chunk_id = chunked_llm.register_chunk(system_prompt, ChunkType.SYSTEM_PROMPT)
    query_chunk_id = chunked_llm.register_chunk(user_query, ChunkType.QUERY)
    
    print(f"\nSystem chunk: {system_chunk_id[:8]}...")
    print(f"Query chunk: {query_chunk_id[:8]}...")
    
    # Get chunks to inspect
    system_chunk = chunked_llm.registry.get(system_chunk_id)
    query_chunk = chunked_llm.registry.get(query_chunk_id)
    
    print(f"\nSystem chunk:")
    print(f"  Tokens ({len(system_chunk.token_ids)}): {system_chunk.token_ids}")
    print(f"  Page table: {system_chunk.page_table}")
    print(f"  KV length: {system_chunk.kv_length}")
    
    print(f"\nQuery chunk:")
    print(f"  Tokens ({len(query_chunk.token_ids)}): {query_chunk.token_ids}")
    print(f"  Page table: {query_chunk.page_table}")
    print(f"  KV length: {query_chunk.kv_length}")
    
    # Generate with limited tokens
    print(f"\n\nGenerating response...")
    result = chunked_llm.generate_from_chunks(
        system_chunk_id=system_chunk_id,
        query_chunk_id=query_chunk_id,
        context_chunk_ids=None,
        sampling_params={"temperature": 0.0, "max_tokens": 5},
        stream=False
    )
    
    print(f"\n\nGenerated output: {repr(result['text'])}")
    print(f"Tokens: {result['token_ids']}")

if __name__ == "__main__":
    debug_chunked_generation()