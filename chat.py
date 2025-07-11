#!/usr/bin/env python3
"""
Fast chat interface using nano-vllm with all optimizations.
Provides real-time streaming output with conversation context.
"""

import argparse
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

from braidinfer import LLM, SamplingParams


class FastChat:
    """Fast chat interface with streaming and context management."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        debug_tokens: bool = False,  # Show token IDs
    ):
        """Initialize the chat interface."""
        print("Loading model...", end="", flush=True)
        start_time = time.time()

        # Create LLM instance with optimizations
        # Expand ~ in path
        expanded_path = os.path.expanduser(model_path)

        self.llm = LLM(
            model=expanded_path
        )

        load_time = time.time() - start_time
        print(f" Done! ({load_time:.1f}s)")
        print(f"Using custom kernels")
        print()

        # Conversation history
        self.messages = []
        self.generation_times = []
        self.total_tokens_generated = 0
        self.debug_tokens = debug_tokens
        self.should_exit = False

        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl-C signal for graceful exit."""
        self.should_exit = True
        os._exit(0)

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

        return "".join(filtered_parts)

    def _format_messages(self) -> str:
        """Format conversation history for the model using its chat template."""
        # Use the tokenizer's chat template if available
        if hasattr(self.llm.tokenizer, 'apply_chat_template'):
            # Use the tokenizer's chat template with add_generation_prompt=True
            # This adds the appropriate assistant start without empty think blocks
            prompt = self.llm.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # The template already handles formatting correctly, no need to strip
        else:
            # Fallback to Qwen format if no chat template
            prompt = ""
            for msg in self.messages:
                if msg["role"] == "user":
                    prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

            # Add start of assistant response
            prompt += "<|im_start|>assistant\n"

        return prompt

    def generate_response(self, user_input: str) -> None:
        """Generate and stream response to user input."""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        # Create prompt with full conversation history
        prompt = self._format_messages()

        # Show the exact prompt being sent to the model
        print("\n[Prompt sent to model:]")
        print(repr(prompt))

        if self.debug_tokens:
            # Show token IDs
            prompt_tokens = self.llm.tokenizer.encode(prompt)
            print(f"\n[Prompt token IDs ({len(prompt_tokens)} tokens):]")
            print(prompt_tokens[:50], "..." if len(prompt_tokens) > 50 else "")

            # Decode special tokens
            print("\n[Special tokens in prompt:]")
            special_tokens = {
                151643: "<|endoftext|>",
                151644: "<|im_start|>",
                151645: "<|im_end|>",
                151667: "<think>",
                151668: "</think>"
            }
            for i, tid in enumerate(prompt_tokens[:50]):
                if tid in special_tokens:
                    print(f"  Position {i}: {special_tokens[tid]} (ID: {tid})")

        print()

        # Generate response with streaming
        print("Assistant: ", end="", flush=True)

        start_time = time.time()
        generated_text = ""
        token_count = 0
        first_token_time = None

        try:
            # Create sampling params
            # The model's EOS token <|im_end|> (151645) is handled by ignore_eos=False
            # Note: If model generates <|endoftext|> (151643), that's abnormal - it's the BOS/PAD token
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=512,
                ignore_eos=False
                # Don't stop on additional tokens - let the model complete naturally
            )

            # Generate response with streaming
            cumulative_text = ""
            for output in self.llm.generate(prompt, sampling_params, stream=True):
                if not output["finished"]:
                    # Stream new text
                    new_text = output["text"]
                    print(new_text, end="", flush=True)
                    cumulative_text = output["cumulative_text"]

                    # Track timing
                    if first_token_time is None and len(new_text.strip()) > 0:
                        first_token_time = time.time() - start_time

                    token_count = len(output["token_ids"]) + token_count
                else:
                    # Final output
                    generated_text = output["final_output"]["text"]
                    token_count = len(output["final_output"]["token_ids"])
                    # Save token IDs for debug
                    self._last_token_ids = output["final_output"]["token_ids"]

        except KeyboardInterrupt:
            print("\n[Generation interrupted]")

        print()  # New line after response

        # Calculate performance metrics
        total_time = time.time() - start_time

        if token_count > 0:
            tokens_per_second = token_count / total_time
            self.generation_times.append((token_count, total_time))
            self.total_tokens_generated += token_count

            print(f"\n[Generated {token_count} tokens in {total_time:.2f}s = {tokens_per_second:.1f} tok/s]")
            if first_token_time:
                print(f"[Time to first token: {first_token_time:.3f}s]")

        # Show raw output with all tokens
        print("\n[Raw output with all tokens:]")
        print(repr(generated_text))

        if self.debug_tokens and token_count > 0:
            print(f"\n[Generated token IDs ({token_count} tokens):]")
            # Get the full token list from the last output
            if hasattr(self, '_last_token_ids'):
                print(self._last_token_ids[:50], "..." if len(self._last_token_ids) > 50 else "")

                # Check for special tokens in output
                print("\n[Special tokens in output:]")
                special_tokens = {
                    151643: "<|endoftext|>",
                    151644: "<|im_start|>",
                    151645: "<|im_end|>",
                    151667: "<think>",
                    151668: "</think>"
                }
                for i, tid in enumerate(self._last_token_ids):
                    if tid in special_tokens:
                        print(f"  Position {i}: {special_tokens[tid]} (ID: {tid})")

        # Filter think tags and add to history
        filtered_response = self._filter_think_tags(generated_text)
        self.messages.append({"role": "assistant", "content": filtered_response})

        # Trim conversation history if it gets too long
        # Keep last 10 exchanges (20 messages)
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def print_stats(self):
        """Print performance statistics."""
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

        # Model info
        if hasattr(self.llm, 'config'):
            print(f"\nModel: {self.llm.config.model}")
        if hasattr(self.llm, 'model_runner') and hasattr(self.llm.model_runner, 'model'):
            model = self.llm.model_runner.model

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        print("Conversation history cleared.")

    def run(self):
        """Run the chat REPL loop."""
        print("Fast Chat Interface (nano-vllm)")
        print("===============================")
        print("Commands:")
        print("  'exit' or Ctrl+C - quit")
        print("  'stats' - show performance statistics")
        print("  'clear' - clear conversation history")
        print()

        while not self.should_exit:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check for exit flag
                if self.should_exit:
                    break

                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    self.print_stats()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif not user_input:
                    continue

                # Generate response
                self.generate_response(user_input)

            except KeyboardInterrupt:
                # Ctrl-C during input - exit gracefully
                print("\nGoodbye!")
                break
            except EOFError:
                # Ctrl-D or EOF - exit gracefully
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast chat interface with nano-vllm optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python chat.py

  # Run with custom kernels for 2.64x speedup
  python chat.py --custom-kernels

  # Debug mode - show all tokens
  python chat.py --debug-tokens

  # Use a different model
  python chat.py --model meta-llama/Llama-2-7b-chat-hf
"""
    )

    parser.add_argument(
        "--model",
        default="~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/",
        help="Model to use (HuggingFace model name or local path)"
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Show token IDs and special tokens in prompts and outputs"
    )

    args = parser.parse_args()

    # Show configuration
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Custom kernels: enabled")
    print()

    # Create and run chat interface
    chat = FastChat(
        model_path=args.model,
        debug_tokens=args.debug_tokens,
    )

    try:
        chat.run()
    finally:
        chat.print_stats()


if __name__ == "__main__":
    main()
