import os
import sys
import re
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.text import Text
    from rich.status import Status
    HAS_RICH = True
except ImportError:
    print("Rich is not installed. Install it with `pip install rich` for a better experience.")
    HAS_RICH = False

def ensure_weights(repo_id: str, local_dir: Path):
    """
    Download the repo from Hugging Face into local_dir if no .bin weights are present.
    """
    # Check for any .bin files in the target folder
    if not local_dir.exists() or not any(local_dir.glob("*.bin")):
        print(f"⏬ Downloading weights for {repo_id} into {local_dir} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            cache_dir=str(local_dir)
        )
    else:
        print(f"✅ Weights already present in {local_dir}, skipping download.")

def parse_thinking_tags(text):
    """Parse and extract thinking tags from text."""
    thinking_pattern = r'<think>(.*?)</think>'
    matches = re.finditer(thinking_pattern, text, re.DOTALL)

    parts = []
    last_end = 0

    for match in matches:
        if match.start() > last_end:
            parts.append(('text', text[last_end:match.start()]))
        parts.append(('thinking', match.group(1).strip()))
        last_end = match.end()

    if last_end < len(text):
        parts.append(('text', text[last_end:]))

    return parts

def render_response(text, console=None):
    """Render response with markdown and thinking tags."""
    if not console:
        console = Console() if HAS_RICH else None

    parts = parse_thinking_tags(text)

    if HAS_RICH and console:
        for part_type, content in parts:
            if part_type == 'thinking':
                thinking_text = Text(f"💭 {content}", style="dim cyan italic")
                console.print(Panel(thinking_text, title="Thinking", border_style="dim"))
            else:
                try:
                    md = Markdown(content)
                    console.print(md)
                except Exception:
                    console.print(content)
    else:
        for part_type, content in parts:
            if part_type == 'thinking':
                print(f"\n💭 [THINKING] {content}\n")
            else:
                print(content, end="")


def main():
    repo_id = "Qwen/Qwen3-30B-A3B-FP8"
    path = Path.home() / "huggingface" / "Qwen3-30B-A3B-FP8"

    console = Console() if HAS_RICH else None

    ensure_weights(repo_id, path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True
    )
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    if HAS_RICH and console:
        console.print("[bold green]🤖 Nano-vLLM Chat Interface[/bold green]")
        console.print("[dim]Type 'quit', 'exit', or press Ctrl+C to exit[/dim]\n")
    else:
        print("🤖 Nano-vLLM Chat Interface")
        print("Type 'quit', 'exit', or press Ctrl+C to exit\n")

    last_tokens_per_sec = 0.0

    try:
        while True:
            try:
                if last_tokens_per_sec > 0:
                    if HAS_RICH and console:
                        console.print(f"[dim]{last_tokens_per_sec:.1f} tok/s[/dim]")
                    else:
                        print(f"{last_tokens_per_sec:.1f} tok/s")

                if HAS_RICH and console:
                    user_input = Prompt.ask("[bold blue]>[/bold blue]")
                else:
                    user_input = input("> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input.strip():
                    continue

                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_input}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )

                if HAS_RICH and console:
                    console.print("\n[bold green]Assistant[/bold green]:")
                else:
                    print("\nAssistant:")

                token_count = 0
                start_time = time.time()

                # State for thinking popup and streaming
                in_thinking = False
                thinking_status = None
                accumulated_text = ""
                display_buffer = ""

                if HAS_RICH and console:
                    # Use Live display for real-time markdown rendering
                    with Live("", refresh_per_second=10, console=console) as live_display:
                        for token_data in llm.generate_stream(formatted_prompt, sampling_params):
                            if not token_data["finished"]:
                                token = token_data["token"]
                                accumulated_text += token
                                token_count += 1

                                # Check for thinking tags
                                if "<think>" in accumulated_text and not in_thinking:
                                    in_thinking = True
                                    # Show thinking indicator in the live display instead of separate status
                                    live_display.update("[dim cyan]💭 Thinking...[/dim cyan]")

                                if in_thinking:
                                    # Update thinking display with content preview
                                    think_start = accumulated_text.rfind("<think>")
                                    if think_start != -1:
                                        content_start = think_start + 7
                                        thinking_content = accumulated_text[content_start:]
                                        preview = thinking_content.strip()[:50]
                                        live_display.update(f"[dim cyan]💭 {preview}...[/dim cyan]")

                                # Check for end of thinking
                                if "</think>" in accumulated_text and in_thinking:
                                    in_thinking = False

                                    # Clean accumulated text and continue
                                    accumulated_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
                                    display_buffer = accumulated_text
                                    continue

                                # Update display with visible content
                                if not in_thinking and "<think" not in token and "</think>" not in token:
                                    display_buffer += token
                                    try:
                                        md = Markdown(display_buffer)
                                        live_display.update(md)
                                    except Exception:
                                        live_display.update(display_buffer)
                            else:
                                break
                else:
                    # Plain text streaming for non-rich console
                    for token_data in llm.generate_stream(formatted_prompt, sampling_params):
                        if not token_data["finished"]:
                            token = token_data["token"]
                            accumulated_text += token
                            token_count += 1

                            # Handle thinking tags for plain text
                            if "<think>" in accumulated_text and not in_thinking:
                                in_thinking = True
                                print("\n💭 [THINKING...]", end="", flush=True)

                            if "</think>" in accumulated_text and in_thinking:
                                in_thinking = False
                                print("\r" + " " * 50 + "\r", end="", flush=True)
                                accumulated_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
                                continue

                            # Print visible tokens
                            if not in_thinking and "<think" not in token and "</think>" not in token:
                                print(token, end="", flush=True)
                        else:
                            break

                end_time = time.time()
                duration = end_time - start_time
                last_tokens_per_sec = token_count / duration if duration > 0 else 0

                print()

            except KeyboardInterrupt:
                if HAS_RICH and console:
                    console.print("\n[dim]Interrupted by user[/dim]")
                else:
                    print("\nInterrupted by user")
                continue
            except EOFError:
                break

    except KeyboardInterrupt:
        pass

    if HAS_RICH and console:
        console.print("\n[dim]👋 Goodbye![/dim]")
    else:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
