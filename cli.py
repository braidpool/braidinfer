import os
import sys
import re
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
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


def handle_slash_command(command: str, args: str, context_mgr: ContextManager, tokenizer, console=None):
    """Handle slash commands for context management"""

    # Define color codes
    GREEN = '\033[92m' if not HAS_RICH else None
    RED = '\033[91m' if not HAS_RICH else None
    YELLOW = '\033[93m' if not HAS_RICH else None
    BLUE = '\033[94m' if not HAS_RICH else None
    GRAY = '\033[90m' if not HAS_RICH else None
    RESET = '\033[0m' if not HAS_RICH else None
    BOLD = '\033[1m' if not HAS_RICH else None

    def print_msg(msg, color=None):
        if HAS_RICH and console:
            console.print(msg)
        else:
            if color:
                print(f"{color}{msg}{RESET}")
            else:
                print(msg)

    if command == "load":
        if not args:
            print_msg("Usage: /load <filename>", RED)
            return

        filename = os.path.expanduser(args.strip())
        if not os.path.exists(filename):
            print_msg(f"File not found: {filename}", RED)
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            print_msg(f"Loading {filename}...")
            chunk = context_mgr.add_chunk(
                content=content,
                tokenizer=tokenizer,
                metadata={"source": filename}
            )

            print_msg(f"✓ Added chunk:", GREEN)
            print_msg(f"  Hash: {chunk.sha256}")
            print_msg(f"  Size: {chunk.size} tokens")
            print_msg(f"  Position: {chunk.position[0]}-{chunk.position[1]}")

        except Exception as e:
            print_msg(f"Error loading file: {e}", RED)

    elif command == "context":
        info = context_mgr.get_context_info()

        if HAS_RICH and console:
            console.print(f"\n[bold]Context Status:[/bold]")
        else:
            print(f"\n{BOLD}Context Status:{RESET}")

        print_msg(f"  Total: {info['total_context']} tokens")
        print_msg(f"  Used: {info['used_context']} tokens")
        print_msg(f"  Free: {info['free_context']} tokens")

        chunks_by_status = info.get('chunks_by_status', {})

        # Show active chunks
        active_chunks = chunks_by_status.get('active', [])
        if active_chunks:
            if HAS_RICH and console:
                console.print(f"\n[bold green]Active Chunks:[/bold green]")
            else:
                print(f"\n{BOLD}{GREEN}Active Chunks:{RESET}")

            for i, chunk in enumerate(active_chunks, 1):
                print_msg(f"  {i}. {chunk['hash'][:16]}... [{chunk['size']} tokens]")

        # Show inactive chunks
        inactive_chunks = chunks_by_status.get('inactive', [])
        if inactive_chunks:
            if HAS_RICH and console:
                console.print(f"\n[bold yellow]Inactive Chunks:[/bold yellow]")
            else:
                print(f"\n{BOLD}{YELLOW}Inactive Chunks:{RESET}")

            for i, chunk in enumerate(inactive_chunks, 1):
                print_msg(f"  {i}. {chunk['hash'][:16]}... [{chunk['size']} tokens]")

    elif command == "activate":
        if not args:
            print_msg("Usage: /activate <hash>", RED)
            return

        try:
            context_mgr.activate_chunk(args.strip())
            print_msg(f"✓ Chunk activated: {args.strip()[:16]}...", GREEN)
        except Exception as e:
            print_msg(f"Failed to activate chunk: {e}", RED)

    elif command == "deactivate":
        if not args:
            print_msg("Usage: /deactivate <hash>", RED)
            return

        try:
            context_mgr.deactivate_chunk(args.strip())
            print_msg(f"✓ Chunk deactivated: {args.strip()[:16]}...", GREEN)
        except Exception as e:
            print_msg(f"Failed to deactivate chunk: {e}", RED)

    elif command == "save":
        if not args:
            print_msg("Usage: /save <hash>", RED)
            return

        try:
            context_mgr.save_chunk(args.strip())
            print_msg(f"✓ Chunk saved: {args.strip()[:16]}...", GREEN)
        except Exception as e:
            print_msg(f"Failed to save chunk: {e}", RED)

    elif command == "restore":
        if not args:
            print_msg("Usage: /restore <hash>", RED)
            return

        try:
            chunk = context_mgr.restore_chunk(args.strip())
            print_msg(f"✓ Chunk restored: {args.strip()[:16]}...", GREEN)
            print_msg(f"  Size: {chunk.size} tokens")
        except Exception as e:
            print_msg(f"Failed to restore chunk: {e}", RED)

    elif command == "erase":
        if not args:
            print_msg("Usage: /erase <hash>", RED)
            return

        try:
            context_mgr.erase_chunk(args.strip())
            print_msg(f"✓ Chunk erased: {args.strip()[:16]}...", GREEN)
        except Exception as e:
            print_msg(f"Failed to erase chunk: {e}", RED)

    elif command == "clear":
        context_mgr.clear_all()
        print_msg("✓ All chunks cleared", GREEN)

    elif command == "help":
        if HAS_RICH and console:
            console.print("\n[bold]Available Commands:[/bold]")
        else:
            print(f"\n{BOLD}Available Commands:{RESET}")

        commands = [
            ("/load <file>", "Load a file as a context chunk"),
            ("/context", "Show current context status"),
            ("/activate <hash>", "Activate a chunk for inference"),
            ("/deactivate <hash>", "Deactivate a chunk"),
            ("/save <hash>", "Save chunk to disk"),
            ("/restore <hash>", "Restore chunk from disk"),
            ("/erase <hash>", "Remove chunk completely"),
            ("/clear", "Clear all chunks"),
            ("/help", "Show this help message")
        ]

        for cmd, desc in commands:
            if HAS_RICH and console:
                console.print(f"  [bold blue]{cmd}[/bold blue] - {desc}")
            else:
                print(f"  {BLUE}{cmd}{RESET} - {desc}")

    else:
        print_msg(f"Unknown command: /{command}", RED)
        print_msg("Type /help for available commands")


def main():
    repo_id = "Qwen/Qwen3-0.6B"
    path = Path.home() / "huggingface" / "Qwen3-0.6B"

    console = Console() if HAS_RICH else None

    ensure_weights(repo_id, path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True
    )
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    # Initialize context manager
    context_mgr = ContextManager(llm.scheduler.block_manager, llm.config)
    llm.context_manager = context_mgr  # Set reference in engine

    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    if HAS_RICH and console:
        console.print("[bold green]🤖 Nano-vLLM Chat Interface with Context Manager[/bold green]")
        console.print("[dim]Type 'quit', 'exit', or press Ctrl+C to exit[/dim]")
        console.print("[dim]Type '/help' for context management commands[/dim]\n")
    else:
        print("🤖 Nano-vLLM Chat Interface with Context Manager")
        print("Type 'quit', 'exit', or press Ctrl+C to exit")
        print("Type '/help' for context management commands\n")

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

                # Check for slash commands
                if user_input.startswith('/'):
                    parts = user_input.split(None, 1)
                    command = parts[0][1:]  # Remove the leading /
                    args = parts[1] if len(parts) > 1 else ''
                    handle_slash_command(command, args, context_mgr, tokenizer, console)
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
