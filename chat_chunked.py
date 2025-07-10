#!/usr/bin/env python3
"""
Fast chat interface using nano-vllm with ChunkedLLM API for context reuse.
Provides real-time streaming output with conversation chunk management.
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple, Dict, Any

from nanovllm import ChunkedLLM, ChunkType, SamplingParams


class ChunkedFastChat:
    """Fast chat interface with streaming and chunk-based context management."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_conversation_chunks: int = 20,  # Keep last 10 exchanges
    ):
        """Initialize the chat interface with ChunkedLLM."""
        print("Loading model with ChunkedLLM...", end="", flush=True)
        start_time = time.time()
        
        # Create ChunkedLLM instance
        expanded_path = os.path.expanduser(model_path)
        
        # Model kwargs no longer needed - custom kernels are always used
        model_kwargs = {}
        
        self.llm = ChunkedLLM(
            expanded_path,
            max_chunks=1000,
            chunk_memory_ratio=0.5,
            enable_deduplication=True,
            enforce_eager=True,
            model_kwargs=model_kwargs
        )
        
        load_time = time.time() - start_time
        print(f" Done! ({load_time:.1f}s)")
        print(f"Using ChunkedLLM with content-based deduplication and custom paged kernel")
        print()
        
        # Conversation management
        self.max_conversation_chunks = max_conversation_chunks
        self.conversation_chunk_ids = []  # List of (role, chunk_id) tuples
        self.system_chunk_id = None
        
        # Statistics
        self.generation_times = []
        self.total_tokens_generated = 0
        self.chunk_stats_history = []
        
        # Set default system prompt with language hint
        self._set_system_prompt("You are a helpful AI assistant. Please respond in English.")
    
    def _set_system_prompt(self, prompt: str):
        """Set or update the system prompt chunk."""
        if self.system_chunk_id:
            # System prompt changed, register new one
            # Old one may still be cached for other conversations
            pass
        
        # Format system prompt using the model's chat template
        messages = [{"role": "system", "content": prompt}]
        formatted_system_prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        self.system_chunk_id = self.llm.register_chunk(
            formatted_system_prompt,
            ChunkType.SYSTEM_PROMPT,
            metadata={"role": "system", "timestamp": time.time()}
        )
        print(f"System prompt registered (chunk: {self.system_chunk_id[:8]}...)")
    
    def _filter_think_tags(self, text: str) -> str:
        """Remove <think>...</think> sections from text."""
        filtered_parts = []
        current_pos = 0
        
        while True:
            # Find next <think> tag
            think_start = text.find("<think>", current_pos)
            if think_start == -1:
                # No more think tags
                filtered_parts.append(text[current_pos:])
                break
            
            # Add text before <think>
            filtered_parts.append(text[current_pos:think_start])
            
            # Find matching </think>
            think_end = text.find("</think>", think_start)
            if think_end == -1:
                # Unclosed think tag - include it
                filtered_parts.append(text[think_start:])
                break
            
            think_end += len("</think>")
            current_pos = think_end
        
        return "".join(filtered_parts).strip()
    
    def _build_context_chunks(self) -> List[str]:
        """Build list of context chunk IDs from conversation history."""
        # Use the last N conversation chunks as context
        context_ids = []
        for role, chunk_id in self.conversation_chunk_ids[-self.max_conversation_chunks:]:
            context_ids.append(chunk_id)
        return context_ids
    
    def generate_response(self, user_input: str) -> None:
        """Generate and stream response to user input using chunks."""
        # Get context from previous turns.
        context_chunk_ids = self._build_context_chunks()

        # Format the user's input using the chat template for consistency
        messages = [{"role": "user", "content": user_input}]
        formatted_user_input = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # The engine handles the generation prompt
        )
        
        # The user's input is the new query.
        query_chunk_id = self.llm.register_chunk(
            formatted_user_input,
            ChunkType.QUERY,
            metadata={"role": "user", "timestamp": time.time()}
        )
        
        # Show chunk usage info
        stats_before = self.llm.get_chunk_stats()
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        
        start_time = time.time()
        token_count = 0
        first_token_time = None
        
        try:
            # Generate with chunks and streaming
            cumulative_text = ""
            generated_text = ""
            
            for output in self.llm.generate_from_chunks(
                system_chunk_id=self.system_chunk_id,
                query_chunk_id=query_chunk_id,
                context_chunk_ids=context_chunk_ids if context_chunk_ids else None,
                sampling_params={
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "ignore_eos": False
                },
                stream=True
            ):
                if not output["finished"]:
                    # Stream new text including think tags
                    new_text = output["text"]
                    print(new_text, end="", flush=True)
                    cumulative_text = output.get("cumulative_text", "")
                    
                    # Track timing
                    if first_token_time is None and len(new_text.strip()) > 0:
                        first_token_time = time.time() - start_time
                    
                    if "token_ids" in output:
                        token_count += len(output["token_ids"])
                else:
                    # Final output
                    generated_text = output["final_output"]["text"]
                    token_count = len(output["final_output"]["token_ids"])
            
            # Track timing
            total_time = time.time() - start_time
            
            # Filter think tags for conversation history
            filtered_text = self._filter_think_tags(generated_text)

            # Add user's message to history using the chat template
            user_messages = [{"role": "user", "content": user_input}]
            formatted_user_context = self.llm.tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            )
            user_context_chunk_id = self.llm.register_chunk(
                formatted_user_context,
                ChunkType.CONTEXT,
                metadata={"role": "user", "timestamp": time.time()}
            )
            self.conversation_chunk_ids.append(("user", user_context_chunk_id))
            
            # Register assistant response as a chunk using the chat template
            assistant_messages = [{"role": "assistant", "content": filtered_text}]
            formatted_assistant_response = self.llm.tokenizer.apply_chat_template(
                assistant_messages, tokenize=False, add_generation_prompt=False
            )
            assistant_chunk_id = self.llm.register_chunk(
                formatted_assistant_response,
                ChunkType.CONTEXT,
                metadata={"role": "assistant", "timestamp": time.time()}
            )
            self.conversation_chunk_ids.append(("assistant", assistant_chunk_id))
        
        except KeyboardInterrupt:
            print("\n[Generation interrupted]")
            return
        
        print()  # New line after response
        
        # Calculate performance metrics
        if token_count > 0:
            tokens_per_second = token_count / total_time
            self.generation_times.append((token_count, total_time))
            self.total_tokens_generated += token_count
            
            print(f"\n[Generated {token_count} tokens in {total_time:.2f}s = {tokens_per_second:.1f} tok/s]")
            if first_token_time:
                print(f"[Time to first token: {first_token_time:.3f}s]")
        
        # Show chunk statistics
        stats_after = self.llm.get_chunk_stats()
        new_hits = stats_after['cache_hits'] - stats_before['cache_hits']
        if new_hits > 0:
            print(f"[Cache hits this generation: {new_hits}, Total hit rate: {stats_after['hit_rate']:.1%}]")
        
        # Trim conversation if needed
        if len(self.conversation_chunk_ids) > self.max_conversation_chunks * 2:
            # Keep only the last max_conversation_chunks exchanges
            self.conversation_chunk_ids = self.conversation_chunk_ids[-self.max_conversation_chunks * 2:]
    
    def print_stats(self):
        """Print performance and chunk statistics."""
        if not self.generation_times:
            return
        
        print("\n=== Performance Statistics ===")
        print(f"Total generations: {len(self.generation_times)}")
        print(f"Total tokens generated: {self.total_tokens_generated}")
        
        total_time = sum(t[1] for t in self.generation_times)
        avg_speed = self.total_tokens_generated / total_time if total_time > 0 else 0
        
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Average speed: {avg_speed:.1f} tok/s")
        
        # Find best/worst performance
        speeds = [(t[0]/t[1], t[0], t[1]) for t in self.generation_times if t[1] > 0]
        if speeds:
            best_speed = max(speeds, key=lambda x: x[0])
            worst_speed = min(speeds, key=lambda x: x[0])
            
            print(f"Best run: {best_speed[0]:.1f} tok/s ({best_speed[1]} tokens in {best_speed[2]:.2f}s)")
            print(f"Worst run: {worst_speed[0]:.1f} tok/s ({worst_speed[1]} tokens in {worst_speed[2]:.2f}s)")
        
        # Chunk statistics
        print("\n=== Chunk Statistics ===")
        stats = self.llm.get_chunk_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Memory used: {stats['memory_used_mb']:.1f} MB")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Evictions: {stats['evictions']}")
        
        # Conversation chunks
        print(f"\nConversation chunks: {len(self.conversation_chunk_ids)}")
        print(f"Unique chunks in registry: {stats['total_chunks']}")
    
    def clear_history(self):
        """Clear conversation history chunks."""
        self.conversation_chunk_ids = []
        print("Conversation history cleared. (Chunks remain in cache for potential reuse)")
    
    def clear_cache(self):
        """Clear the entire chunk cache."""
        # Note: ChunkedLLM doesn't have a clear_cache method yet
        # This would need to be implemented
        print("Cache clearing not yet implemented in ChunkedLLM.")
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt."""
        self._set_system_prompt(new_prompt)
        print(f"System prompt updated!")
    
    def run(self):
        """Run the chat REPL loop."""
        print("Fast Chat Interface with ChunkedLLM (nano-vllm)")
        print("=============================================")
        print("Commands:")
        print("  'exit' or Ctrl+C - quit")
        print("  '/stats' - show performance and chunk statistics")
        print("  '/clear' - clear conversation history")
        print("  '/cache' - show cache statistics")
        print("  '/system <prompt>' - update system prompt")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == '/stats':
                    self.print_stats()
                    continue
                elif user_input.lower() == '/clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == '/cache':
                    stats = self.llm.get_chunk_stats()
                    print(f"\nCache: {stats['total_chunks']} chunks, {stats['memory_used_mb']:.1f} MB")
                    print(f"Hit rate: {stats['hit_rate']:.1%} ({stats['cache_hits']} hits)")
                    continue
                elif user_input.lower().startswith('/system '):
                    new_prompt = user_input[8:].strip()
                    if new_prompt:
                        self.update_system_prompt(new_prompt)
                    else:
                        print("Please provide a system prompt.")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                self.generate_response(user_input)
                
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit or press Ctrl+C again.")
                try:
                    # Give user a chance to continue
                    time.sleep(1)
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fast chat interface with nano-vllm ChunkedLLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python chat_chunked.py
  
  # Use a different model
  python chat_chunked.py --model meta-llama/Llama-2-7b-chat-hf
  
  # Adjust conversation memory
  python chat_chunked.py --max-chunks 30
"""
    )
    
    parser.add_argument(
        "--model",
        default="~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/",
        help="Model to use (HuggingFace model name or local path)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Maximum conversation chunks to keep as context (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Show configuration
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Max conversation chunks: {args.max_chunks}")
    print(f"  Attention mode: custom paged kernel")
    print(f"  Using ChunkedLLM with content deduplication")
    print()
    
    # Create and run chat interface
    chat = ChunkedFastChat(
        model_path=args.model,
        max_conversation_chunks=args.max_chunks,
    )
    
    try:
        chat.run()
    finally:
        chat.print_stats()


if __name__ == "__main__":
    main()