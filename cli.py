#!/usr/bin/env python3

import os
import sys
import re
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from nanovllm.engine.context_manager import ContextManager
from transformers import AutoTokenizer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from rich.status import Status
from rich.color import Color

def ensure_weights(repo_id: str, local_dir: Path):
    """
    Download the repo from Hugging Face into local_dir if model files are not present.
    """
    # Check if model directory exists and has required files
    if not local_dir.exists():
        needs_download = True
    else:
        # Check for model weight files (.safetensors or .bin)
        has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
        # Check for config file
        has_config = (local_dir / "config.json").exists()
        needs_download = not (has_weights and has_config)
    
    if needs_download:
        print(f"⏬ Downloading weights for {repo_id} into {local_dir} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            cache_dir=str(local_dir),
            resume_download=True
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
        console = Console()

    parts = parse_thinking_tags(text)

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


def handle_slash_command(command: str, args: str, context_mgr: ContextManager, tokenizer, console=None):
    """Handle slash commands for context management"""
    
    if not console:
        console = Console()

    def print_msg(msg, color=None):
        if color:
            console.print(f"[{color}]{msg}[/{color}]")
        else:
            console.print(msg)

    if command == "load":
        if not args:
            print_msg("Usage: /load <filename>", "red")
            return

        filename = os.path.expanduser(args.strip())
        if not os.path.exists(filename):
            print_msg(f"File not found: {filename}", "red")
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

            print_msg(f"✓ Added chunk:", "green")
            print_msg(f"  Hash: {chunk.sha256[:16]}...")
            print_msg(f"  Size: {chunk.size} tokens")
            print_msg(f"  Blocks allocated: {len(chunk.blocks)}")
            print_msg(f"  Note: Chunk is ready for use. KV cache will be populated during generation.", "yellow")

        except Exception as e:
            print_msg(f"Error loading file: {e}", "red")

    elif command == "context":
        info = context_mgr.get_context_info()
        memory_stats = context_mgr.get_memory_stats()

        from rich.table import Table
        from rich.box import ROUNDED

        # Create table for chunks
        table = Table(title="Active Context", box=ROUNDED)
        table.add_column("Hash", style="cyan", width=10)
        table.add_column("Type", style="magenta", width=6)
        table.add_column("Status", style="green", width=8)
        table.add_column("Cache", style="yellow", width=7)
        table.add_column("Tokens", justify="right", style="yellow", width=7)
        table.add_column("Memory", justify="right", style="blue", width=10)
        table.add_column("Preview", style="white", width=25)

        # Get all chunks sorted by creation time
        all_chunks = []
        for status, chunks in info['chunks_by_status'].items():
            for chunk in chunks:
                chunk['actual_status'] = status
                all_chunks.append(chunk)

        all_chunks.sort(key=lambda x: x.get('created_at', 0))

        # Add rows to table
        for chunk in all_chunks:
            hash_str = chunk['hash'][:8]
            chunk_type = chunk.get('type', 'input')
            status = chunk['actual_status']
            tokens = str(chunk['size'])
            memory = f"{chunk.get('memory_bytes', 0) / 1024 / 1024:.1f} MB"

            # Get preview
            if 'token_ids' in chunk:
                preview = context_mgr.get_preview(chunk['token_ids'], tokenizer)
            else:
                # Try to get chunk directly for preview
                try:
                    chunk_obj = context_mgr.chunks.get(chunk['hash'])
                    if chunk_obj:
                        preview = context_mgr.get_preview(chunk_obj.token_ids, tokenizer)
                    else:
                        preview = "<no preview available>"
                except:
                    preview = "<no preview available>"

            # Style based on status
            if status == 'active':
                status_style = "[green]active[/green]"
            elif status == 'inactive':
                status_style = "[yellow]inactive[/yellow]"
            elif status == 'cpu':
                status_style = "[blue]cpu[/blue]"
            else:
                status_style = "[dim]disk[/dim]"

            # Style based on type
            if chunk_type == 'output':
                type_style = "[magenta]output[/magenta]"
            else:
                type_style = "[cyan]input[/cyan]"

            # Cache status
            cache_status = "✓" if chunk.get('cache_populated', False) else "○"
            cache_style = "[green]✓[/green]" if chunk.get('cache_populated', False) else "[dim]○[/dim]"

            table.add_row(hash_str, type_style, status_style, cache_style, tokens, memory, preview)

        console.print(table)

        # Memory usage summary
        console.print(f"\n[bold]Memory Usage:[/bold]")
        for tier, stats in memory_stats.items():
            if stats['total'] > 0:
                console.print(f"  {tier.upper()}: {stats['total'] / 1024 / 1024:.1f} MB ({stats['count']} chunks) - Input: {stats['input'] / 1024 / 1024:.1f} MB, Output: {stats['output'] / 1024 / 1024:.1f} MB")

        # Token summary
        active_tokens = sum(c['size'] for c in all_chunks if c['actual_status'] == 'active')
        total_tokens = sum(c['size'] for c in all_chunks)
        console.print(f"\n[bold]Token Summary:[/bold]")
        console.print(f"  Active Tokens: {active_tokens:,}")
        console.print(f"  Total Tokens: {total_tokens:,}")
        console.print(f"  Free Blocks: {info['free_blocks']} ({info['free_context']} tokens)")

    elif command == "activate":
        if not args:
            print_msg("Usage: /activate <hash>", "red")
            return

        try:
            context_mgr.activate_chunk(args.strip())
            print_msg(f"✓ Chunk activated: {args.strip()[:16]}...", "green")
        except Exception as e:
            print_msg(f"Failed to activate chunk: {e}", "red")

    elif command == "deactivate":
        if not args:
            print_msg("Usage: /deactivate <hash>", "red")
            return

        try:
            context_mgr.deactivate_chunk(args.strip())
            print_msg(f"✓ Chunk deactivated: {args.strip()[:16]}...", "green")
        except Exception as e:
            print_msg(f"Failed to deactivate chunk: {e}", "red")

    elif command == "save":
        if not args:
            print_msg("Usage: /save <hash>", "red")
            return

        try:
            context_mgr.save_chunk(args.strip())
            print_msg(f"✓ Chunk saved: {args.strip()[:16]}...", "green")
        except Exception as e:
            print_msg(f"Failed to save chunk: {e}", "red")

    elif command == "restore":
        if not args:
            print_msg("Usage: /restore <hash>", "red")
            return

        try:
            chunk = context_mgr.restore_chunk(args.strip())
            print_msg(f"✓ Chunk restored: {args.strip()[:16]}...", "green")
            print_msg(f"  Size: {chunk.size} tokens")
        except Exception as e:
            print_msg(f"Failed to restore chunk: {e}", "red")

    elif command == "erase":
        if not args:
            print_msg("Usage: /erase <hash>", "red")
            return

        try:
            context_mgr.erase_chunk(args.strip())
            print_msg(f"✓ Chunk erased from all locations: {args.strip()[:16]}...", "green")
        except Exception as e:
            print_msg(f"Failed to erase chunk: {e}", "red")

    elif command == "unload":
        if not args:
            print_msg("Usage: /unload <hash>", "red")
            return

        try:
            context_mgr.unload_chunk(args.strip())
            print_msg(f"✓ Chunk moved to system RAM: {args.strip()[:16]}...", "green")
        except Exception as e:
            print_msg(f"Failed to unload chunk: {e}", "red")

    elif command == "compose":
        if not args:
            print_msg("Usage: /compose <hash1> <hash2> ...", "red")
            return

        try:
            hashes = args.strip().split()
            if len(hashes) < 2:
                print_msg("Please provide at least 2 chunk hashes to compose", "red")
                return

            chunk = context_mgr.compose_chunks(hashes, tokenizer)
            print_msg(f"✓ Composed new chunk:", "green")
            print_msg(f"  Hash: {chunk.sha256[:16]}...")
            print_msg(f"  Size: {chunk.size} tokens")
            print_msg(f"  From: {len(hashes)} chunks")
        except Exception as e:
            print_msg(f"Failed to compose chunks: {e}", "red")

    elif command == "tag":
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            print_msg("Usage: /tag <hash> <tag>", "red")
            return

        hash_arg, tag = parts
        try:
            context_mgr.tag_chunk(hash_arg, tag)
            print_msg(f"✓ Tagged chunk {hash_arg[:16]}... with '{tag}'", "green")
        except Exception as e:
            print_msg(f"Failed to tag chunk: {e}", "red")

    elif command == "populate":
        if not args:
            print_msg("Usage: /populate <hash>", "red")
            return

        try:
            populated = context_mgr.populate_chunk_cache(args.strip())
            if populated:
                print_msg(f"✓ Populated KV cache for chunk {args.strip()[:16]}...", "green")
            else:
                print_msg(f"Chunk {args.strip()[:16]}... already has populated cache", "yellow")
        except Exception as e:
            print_msg(f"Failed to populate cache: {e}", "red")

    elif command == "clear":
        context_mgr.clear_all()
        print_msg("✓ All chunks cleared", "green")

    elif command == "help":
        console.print("\n[bold]Available Commands:[/bold]")

        commands = [
            ("/load <file>", "Load a file as a context chunk"),
            ("/context", "Show current context status with detailed table"),
            ("/activate <hash>", "Activate a chunk for inference"),
            ("/deactivate <hash>", "Deactivate a chunk"),
            ("/populate <hash>", "Pre-populate KV cache for a chunk"),
            ("/compose <h1> <h2>...", "Compose multiple chunks into one"),
            ("/tag <hash> <tag>", "Add a tag to a chunk"),
            ("/save <hash>", "Save chunk to disk"),
            ("/unload <hash>", "Move chunk from VRAM to system RAM"),
            ("/restore <hash>", "Move chunk from RAM/disk to VRAM"),
            ("/erase <hash>", "Remove chunk from all locations"),
            ("/clear", "Clear all chunks"),
            ("/help", "Show this help message")
        ]

        for cmd, desc in commands:
            console.print(f"  [bold blue]{cmd}[/bold blue] - {desc}")

    else:
        print_msg(f"Unknown command: /{command}", "red")
        print_msg("Type /help for available commands")


def main():
    name = "Qwen3-0.6B"
    repo_id = "Qwen/" + name
    path = Path.home() / "huggingface" / name

    console = Console()

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
    llm.config.context_manager = context_mgr  # Make accessible to model runner
    context_mgr.llm_engine = llm  # Give context manager access to LLM for cache population

    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    console.print("[bold green]🤖 Nano-vLLM Chat Interface with Context Manager[/bold green]")
    console.print("[dim]Type 'quit', 'exit', or press Ctrl+C to exit[/dim]")
    console.print("[dim]Type '/help' for context management commands[/dim]\n")

    last_tokens_per_sec = 0.0

    try:
        while True:
            try:
                if last_tokens_per_sec > 0:
                    console.print(f"[dim]{last_tokens_per_sec:.1f} tok/s[/dim]")

                user_input = Prompt.ask("[bold blue]>[/bold blue]")

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

                # Use context manager to build prompt with active chunks
                if context_mgr and len(context_mgr.active_chunks) > 0:
                    # Set up the generation context first
                    context_mgr.setup_generation_context()
                    formatted_prompt = context_mgr.build_prompt_with_context(
                        [{"role": "user", "content": user_input}],
                        tokenizer
                    )
                else:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_input}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )

                console.print("\n[bold green]Assistant[/bold green]:")

                token_count = 0
                start_time = time.time()

                # State for thinking popup and streaming
                in_thinking = False
                thinking_status = None
                accumulated_text = ""
                display_buffer = ""

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
                            # Check if output chunk was created
                            if 'output_chunk' in token_data:
                                output_chunk = token_data['output_chunk']
                                console.print(f"\n[dim green]✓ Output saved as chunk: {output_chunk.sha256[:16]}... ({output_chunk.size} tokens)[/dim green]")
                            break

                end_time = time.time()
                duration = end_time - start_time
                last_tokens_per_sec = token_count / duration if duration > 0 else 0

                print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted by user[/dim]")
                continue
            except EOFError:
                break

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]👋 Goodbye![/dim]")


if __name__ == "__main__":
    main()