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


def handle_slash_command(command: str, args: str, context_mgr: ContextManager, tokenizer, llm, console=None):
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
            
            # Format with user role using chat template
            formatted_content = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=False
            )
            
            chunk = context_mgr.add_chunk(
                content=formatted_content,
                tokenizer=tokenizer,
                metadata={"source": filename},
                populate_cache=True  # Populate KV cache immediately on load
            )

            print_msg(f"✓ Added chunk:", "green")
            print_msg(f"  Hash: {chunk.sha256[:16]}...")
            print_msg(f"  Size: {chunk.size} tokens")
            print_msg(f"  Blocks allocated: {len(chunk.blocks)}")
            print_msg(f"  KV cache: {'populated' if chunk.cache_populated else 'not populated'}", "green" if chunk.cache_populated else "yellow")

        except Exception as e:
            print_msg(f"Error loading file: {e}", "red")

    elif command == "system":
        if not args:
            print_msg("Usage: /system <filename>", "yellow")
            return
        
        try:
            file_path = Path(args.strip()).expanduser()
            if not file_path.exists():
                print_msg(f"File not found: {file_path}", "red")
                return
                
            with open(file_path, 'r') as f:
                system_content = f.read()
            
            # Format with system role using chat template
            system_formatted = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_content}],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Add as a chunk with proper tokenization
            chunk = context_mgr.add_chunk(
                content=system_formatted,
                tokenizer=tokenizer,
                metadata={"source": "system_prompt", "filename": str(file_path)},
                populate_cache=True
            )
            
            print_msg(f"✓ Loaded system prompt: {chunk.sha256[:16]}... ({chunk.size} tokens)", "green")
            
        except Exception as e:
            print_msg(f"Failed to load system prompt: {e}", "red")

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
                        preview = context_mgr.get_preview(chunk_obj.get_token_ids(), tokenizer)
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

    elif command == "show":
        if not args:
            print_msg("Usage: /show <hash>", "red")
            return

        try:
            hash_input = args.strip()
            from nanovllm.engine.context_manager_utils import resolve_chunk_hash

            # Resolve the hash (handles partial hashes)
            full_hash = resolve_chunk_hash(hash_input, context_mgr.chunks)
            chunk = context_mgr.chunks[full_hash]

            # Decode tokens to text
            try:
                text = tokenizer.decode(chunk.get_token_ids(), skip_special_tokens=False)

                # Create a panel to display the chunk content
                from rich.panel import Panel
                from rich.text import Text

                # Prepare header with chunk info
                header = f"Chunk {chunk.sha256[:16]}... ({chunk.size} tokens, {chunk.status})"
                if chunk.metadata:
                    if 'source' in chunk.metadata:
                        header += f" from {chunk.metadata['source']}"

                # Create content text with proper styling
                content_text = Text(text)

                # Create panel with content
                panel = Panel(
                    content_text,
                    title=header,
                    title_align="left",
                    border_style="blue",
                    expand=False
                )

                console.print(panel)

            except Exception as decode_error:
                print_msg(f"Error decoding chunk tokens: {decode_error}", "red")
                token_ids = chunk.get_token_ids()
                print_msg(f"Raw token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")

        except Exception as e:
            print_msg(f"Failed to show chunk: {e}", "red")


    elif command == "infer":
        # Direct inference from active blocks
        try:
            blocks, token_count = context_mgr.get_all_active_blocks()
            if not blocks:
                print_msg("No active blocks to infer from", "yellow")
                return

            # Parse optional new tokens from args
            new_tokens = None
            extra_info = ""
            if args:
                # User provided additional text - format as user input with assistant prompt
                formatted_args = tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.strip()}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                new_tokens = tokenizer.encode(formatted_args)
                extra_info = f" + {len(new_tokens)} new tokens"
            else:
                # Just add assistant prompt - get it by comparing with and without prompt
                dummy_msg = [{"role": "user", "content": ""}]
                without_prompt = tokenizer.apply_chat_template(dummy_msg, tokenize=False, add_generation_prompt=False)
                with_prompt = tokenizer.apply_chat_template(dummy_msg, tokenize=False, add_generation_prompt=True)
                
                # Extract just the assistant prompt part
                assistant_prompt = with_prompt[len(without_prompt):]
                # We need to encode the full 'with_prompt' and then extract the tokens
                # to ensure special tokens are properly handled
                without_tokens = tokenizer.encode(without_prompt)
                with_tokens = tokenizer.encode(with_prompt)
                new_tokens = with_tokens[len(without_tokens):]
                extra_info = " (with generation prompt)"

            print_msg(f"Running inference on {len(blocks)} blocks ({token_count} tokens){extra_info}...", "blue")

            # Run streaming inference
            console.print()

            token_count_generated = 0
            start_time = time.time()
            accumulated_text = ""

            # Use streaming API
            for token_data in llm.infer_from_blocks_stream(
                existing_blocks=blocks,
                existing_token_count=token_count,
                new_tokens=new_tokens,
                sampling_params=SamplingParams(temperature=0.6, max_tokens=512)
            ):
                if not token_data["finished"]:
                    token = token_data["token"]
                    accumulated_text += token
                    token_count_generated += 1
                    console.print(token, end="", style="green")
                else:
                    break

            # Calculate and display statistics
            end_time = time.time()
            duration = end_time - start_time
            tokens_per_sec = token_count_generated / duration if duration > 0 else 0

            console.print(f"\n\n[dim]Generated {token_count_generated} tokens in {duration:.1f}s ({tokens_per_sec:.1f} tok/s)[/dim]")

            # Optionally save generated output as a chunk
            if accumulated_text.strip():
                try:
                    output_chunk = context_mgr.add_chunk(
                        content=accumulated_text,
                        tokenizer=tokenizer,
                        metadata={"source": "inference_output", "timestamp": time.time()},
                        populate_cache=True
                    )
                    console.print(f"[dim green]✓ Output saved as chunk: {output_chunk.sha256[:16]}... ({output_chunk.size} tokens)[/dim green]")
                except Exception as e:
                    console.print(f"[dim yellow]Warning: Failed to save output as chunk: {e}[/dim yellow]")

        except Exception as e:
            import traceback
            print_msg(f"Failed to run inference: {e}", "red")
            if console:
                console.print(f"[dim red]{traceback.format_exc()}[/dim red]")

    elif command == "clear":
        context_mgr.clear_all()
        print_msg("✓ All chunks cleared", "green")

    elif command == "help":
        console.print("\n[bold]Available Commands:[/bold]")

        commands = [
            ("/load <file>", "Load a file as a context chunk with user role"),
            ("/system <file>", "Load a file as system prompt"),
            ("/context", "Show current context status with detailed table"),
            ("/show <hash>", "Display the text content of a chunk"),
            ("/activate <hash>", "Activate a chunk for inference"),
            ("/deactivate <hash>", "Deactivate a chunk"),
            ("/populate <hash>", "Pre-populate KV cache for a chunk"),
            ("/compose <h1> <h2>...", "Compose multiple chunks into one"),
            ("/tag <hash> <tag>", "Add a tag to a chunk"),
            ("/save <hash>", "Save chunk to disk"),
            ("/unload <hash>", "Move chunk from VRAM to system RAM"),
            ("/restore <hash>", "Move chunk from RAM/disk to VRAM"),
            ("/erase <hash>", "Remove chunk from all locations"),
            ("/infer [text]", "Run inference on active blocks (optional: append text)"),
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

    console.print("[bold green]🤖 Nano-vLLM Context Manager[/bold green]")
    console.print("[dim]Type text to add it as a chunk to the context[/dim]")
    console.print("[dim]Type '/infer' to run inference on the active context[/dim]")
    console.print("[dim]Type '/help' for all commands[/dim]")
    console.print("[dim]Type 'quit', 'exit', or press Ctrl+C to exit[/dim]\n")

    try:
        while True:
            try:
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
                    handle_slash_command(command, args, context_mgr, tokenizer, llm, console)
                    continue

                # Add user input as a chunk with user role formatting (no inference)
                try:
                    # Format with user role using chat template
                    formatted_input = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_input}],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    chunk = context_mgr.add_chunk(
                        content=formatted_input,
                        tokenizer=tokenizer,
                        metadata={"source": "user_input", "timestamp": time.time()},
                        populate_cache=True
                    )

                    console.print(f"[green]✓ Added chunk: {chunk.sha256[:16]}... ({chunk.size} tokens)[/green]")

                except Exception as e:
                    console.print(f"[red]Failed to add chunk: {e}[/red]")

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
