#!/usr/bin/env python3
"""
Rich CLI interface for nano-vllm with chunk-based cascade attention.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.align import Align
from rich.rule import Rule
from rich import box

from nanovllm import ChunkedLLM, ChunkType, SamplingParams


@dataclass
class CLIState:
    """Current state of the CLI."""
    system_chunk_id: Optional[str] = None
    context_chunk_ids: List[str] = field(default_factory=list)
    query_chunk_id: Optional[str] = None
    
    # Generation state
    last_output: Optional[str] = None
    generation_time: float = 0.0
    tokens_generated: int = 0
    
    # Output chunks
    output_chunk_ids: List[str] = field(default_factory=list)


class CascadeCLI:
    """Interactive CLI for nano-vllm with chunk-based cascade attention."""
    
    def __init__(self, model_path: str):
        """Initialize the CLI with a model."""
        self.console = Console()
        self.model_path = os.path.expanduser(model_path)
        self.state = CLIState()
        
        # Initialize model with chunk support
        self.console.print("[bold cyan]Initializing ChunkedLLM with cascade attention...[/bold cyan]")
        
        try:
            self.llm = ChunkedLLM(
                self.model_path,
                max_chunks=1000,
                chunk_memory_ratio=0.5,
                enable_deduplication=True,
                enforce_eager=True,
                num_kvcache_blocks=128,
                kvcache_block_size=256
            )
            
            self.console.print("[bold green]✓ Model initialized successfully![/bold green]")
            
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize model: {e}[/bold red]")
            sys.exit(1)
    
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
    
    def render_header(self) -> Panel:
        """Render the header panel."""
        model_name = Path(self.model_path).name
        header_text = Text()
        header_text.append("NanoVLLM Cascade CLI\n", style="bold cyan")
        header_text.append(f"Model: {model_name} | ", style="white")
        header_text.append("Chunk-Based API", style="green")
        
        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="bold blue"
        )
    
    def render_chunks_table(self) -> Panel:
        """Render the current chunks table."""
        table = Table(title="Active Chunks", box=box.ROUNDED)
        
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Content Preview", width=40)
        table.add_column("Tokens", justify="right", style="yellow")
        table.add_column("ID", style="dim")
        table.add_column("Status", style="green")
        
        # Add system chunk
        if self.state.system_chunk_id:
            try:
                chunk = self.llm.get_chunk(self.state.system_chunk_id)
                table.add_row(
                    "System",
                    chunk['content'][:40] + "..." if len(chunk['content']) > 40 else chunk['content'],
                    str(chunk['token_count']),
                    chunk['chunk_id'][:8] + "...",
                    "Active"
                )
            except:
                table.add_row("System", "[Error loading chunk]", "?", "?", "Error")
        else:
            table.add_row("System", "[Not set]", "-", "-", "-")
        
        # Add context chunks
        if self.state.context_chunk_ids:
            for ctx_id in self.state.context_chunk_ids:
                try:
                    chunk = self.llm.get_chunk(ctx_id)
                    table.add_row(
                        "Context",
                        chunk['content'][:40] + "..." if len(chunk['content']) > 40 else chunk['content'],
                        str(chunk['token_count']),
                        chunk['chunk_id'][:8] + "...",
                        "Active"
                    )
                except:
                    table.add_row("Context", "[Error loading chunk]", "?", "?", "Error")
        else:
            table.add_row("Context", "[None added]", "-", "-", "-")
        
        # Add query
        if self.state.query_chunk_id:
            try:
                chunk = self.llm.get_chunk(self.state.query_chunk_id)
                table.add_row(
                    "Query",
                    chunk['content'][:40] + "..." if len(chunk['content']) > 40 else chunk['content'],
                    str(chunk['token_count']),
                    chunk['chunk_id'][:8] + "...",
                    "Active"
                )
            except:
                table.add_row("Query", "[Error loading chunk]", "?", "?", "Error")
        else:
            table.add_row("Query", "[Not set]", "-", "-", "-")
        
        # Add output chunks
        if self.state.output_chunk_ids:
            for i, output_id in enumerate(self.state.output_chunk_ids):
                try:
                    chunk = self.llm.get_chunk(output_id)
                    table.add_row(
                        f"Output {i+1}",
                        chunk['content'][:40] + "..." if len(chunk['content']) > 40 else chunk['content'],
                        str(chunk['token_count']),
                        chunk['chunk_id'][:8] + "...",
                        "Retained"
                    )
                except:
                    table.add_row(f"Output {i+1}", "[Error loading chunk]", "?", "?", "Error")
        
        return Panel(table, box=box.ROUNDED)
    
    def render_registry_stats(self) -> Panel:
        """Render chunk registry statistics."""
        stats = self.llm.get_chunk_stats()
        
        stats_text = Text()
        stats_text.append(f"Total Chunks: {stats['total_chunks']}\n")
        stats_text.append(f"Memory Used: {stats['memory_used_mb']:.1f} MB\n")
        stats_text.append(f"Cache Hits: {stats['cache_hits']} ")
        stats_text.append(f"({stats['hit_rate']:.1%} hit rate)\n", style="green")
        stats_text.append(f"Total Tokens: {stats['total_tokens']}")
        
        return Panel(
            stats_text,
            title="Registry Statistics",
            box=box.ROUNDED
        )
    
    def render_generation(self) -> Panel:
        """Render the last generation result."""
        if not self.state.last_output:
            content = "[dim]No generation yet. Use /infer to generate.[/dim]"
        else:
            # Show output
            output_text = self.state.last_output
            if len(output_text) > 300:
                output_text = output_text[:300] + "..."
            
            content = Text(output_text)
            content.append(f"\n\n[dim]Generated {self.state.tokens_generated} tokens in {self.state.generation_time:.2f}s[/dim]")
        
        return Panel(
            content,
            title="Last Generation",
            box=box.ROUNDED,
            style="green"
        )
    
    def render_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(self.render_header(), size=4),
            Layout(self.render_chunks_table(), size=12),
            Layout(self.render_registry_stats(), size=6),
            Layout(self.render_generation(), size=10)
        )
        
        return layout
    
    def show_help(self):
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[yellow]/system <text>[/yellow] - Set system prompt (creates/replaces system chunk)
[yellow]/context <text>[/yellow] - Add context chunk (can have multiple)
[yellow]/query <text>[/yellow] - Set query (creates/replaces query chunk)
[yellow]/infer[/yellow] - Run inference with current chunks
[yellow]/clear[/yellow] - Clear all active chunks
[yellow]/clear system|context|query[/yellow] - Clear specific type
[yellow]/list[/yellow] - List all chunks in registry
[yellow]/stats[/yellow] - Show detailed statistics
[yellow]/output[/yellow] - List all output chunks
[yellow]/use-output <n>[/yellow] - Add output chunk N as context
[yellow]/delete-output <n>[/yellow] - Delete output chunk N
[yellow]/help[/yellow] - Show this help
[yellow]/exit[/yellow] or [yellow]/quit[/yellow] - Exit the CLI

[bold cyan]Tips:[/bold cyan]
- Chunks are automatically deduplicated by content
- System and query are required for inference
- Context chunks are optional
- Output chunks are retained after generation for reuse
- Use the chunk-based API for efficient memory usage
        """
        self.console.print(help_text)
    
    def handle_system_prompt(self, args: str):
        """Handle /system command."""
        if not args.strip():
            self.console.print("[red]Please provide a system prompt.[/red]")
            return
        
        # Register new system chunk
        chunk_id = self.llm.register_chunk(
            args.strip(),
            ChunkType.SYSTEM_PROMPT,
            metadata={"source": "cli"}
        )
        
        self.state.system_chunk_id = chunk_id
        self.console.print(f"[green]System prompt set (chunk: {chunk_id[:8]}...)[/green]")
    
    def handle_context(self, args: str):
        """Handle /context command."""
        if not args.strip():
            self.console.print("[red]Please provide context content.[/red]")
            return
        
        # Register new context chunk
        chunk_id = self.llm.register_chunk(
            args.strip(),
            ChunkType.CONTEXT,
            metadata={"source": "cli"}
        )
        
        self.state.context_chunk_ids.append(chunk_id)
        self.console.print(f"[green]Context chunk added (chunk: {chunk_id[:8]}...)[/green]")
    
    def handle_query(self, args: str):
        """Handle /query command."""
        if not args.strip():
            self.console.print("[red]Please provide a query.[/red]")
            return
        
        # Register new query chunk
        chunk_id = self.llm.register_chunk(
            args.strip(),
            ChunkType.QUERY,
            metadata={"source": "cli"}
        )
        
        self.state.query_chunk_id = chunk_id
        self.console.print(f"[green]Query set (chunk: {chunk_id[:8]}...)[/green]")
    
    def handle_clear(self, args: str):
        """Handle /clear command."""
        args = args.strip().lower()
        
        if not args:
            # Clear everything
            self.state.system_chunk_id = None
            self.state.context_chunk_ids = []
            self.state.query_chunk_id = None
            self.console.print("[green]All active chunks cleared.[/green]")
        elif args == "system":
            self.state.system_chunk_id = None
            self.console.print("[green]System chunk cleared.[/green]")
        elif args == "context":
            self.state.context_chunk_ids = []
            self.console.print("[green]Context chunks cleared.[/green]")
        elif args == "query":
            self.state.query_chunk_id = None
            self.console.print("[green]Query chunk cleared.[/green]")
        else:
            self.console.print(f"[red]Unknown clear target: {args}[/red]")
    
    def handle_list(self):
        """Handle /list command."""
        chunks = self.llm.list_chunks()
        
        if not chunks:
            self.console.print("[yellow]No chunks in registry.[/yellow]")
            return
        
        table = Table(title="All Registered Chunks", box=box.SIMPLE)
        table.add_column("Type", style="cyan")
        table.add_column("Preview", width=50)
        table.add_column("Tokens", justify="right")
        table.add_column("ID", style="dim")
        table.add_column("Hits", justify="right")
        
        for chunk in chunks:
            table.add_row(
                chunk['chunk_type'],
                chunk['content_preview'],
                str(chunk['token_count']),
                chunk['chunk_id'][:8] + "...",
                str(chunk['access_count'])
            )
        
        self.console.print(table)
    
    def perform_inference(self):
        """Perform inference with the current chunks."""
        # Validate we have required chunks
        if not self.state.system_chunk_id:
            self.console.print("[red]Error: System prompt is required. Use /system <text>[/red]")
            return
        
        if not self.state.query_chunk_id:
            self.console.print("[red]Error: Query is required. Use /query <text>[/red]")
            return
        
        # Run inference
        self.console.print("\n[bold cyan]Running inference...[/bold cyan]")
        
        try:
            start_time = time.time()
            
            # Use generate_and_retain_output to keep the output KV cache
            output = self.llm.generate_and_retain_output(
                system_prompt=self.llm.get_chunk(self.state.system_chunk_id)['content'],
                query=self.llm.get_chunk(self.state.query_chunk_id)['content'],
                context=[self.llm.get_chunk(cid)['content'] for cid in self.state.context_chunk_ids] if self.state.context_chunk_ids else None,
                sampling_params={"temperature": 0.7, "max_tokens": 512},
                persist_chunks=False  # Don't persist the intermediate chunks
            )
            
            elapsed = time.time() - start_time
            
            # Filter think tags from output
            filtered_output = self._filter_think_tags(output['text'])
            
            # Update state with filtered output
            self.state.last_output = filtered_output
            self.state.generation_time = elapsed
            self.state.tokens_generated = len(output.get('token_ids', []))
            
            # Add output chunk ID to our list if retained
            if 'output_chunk_id' in output:
                self.state.output_chunk_ids.append(output['output_chunk_id'])
            
            # Show output
            self.console.print("\n[bold green]Generation complete![/bold green]")
            self.console.print(Rule())
            self.console.print(filtered_output)
            self.console.print(Rule())
            self.console.print(f"\n[dim]Generated {self.state.tokens_generated} tokens in {elapsed:.2f}s[/dim]")
            
        except Exception as e:
            self.console.print(f"[bold red]Inference error: {e}[/bold red]")
    
    def show_stats(self):
        """Show detailed statistics."""
        stats = self.llm.get_chunk_stats()
        
        self.console.print("\n[bold cyan]Detailed Statistics[/bold cyan]")
        self.console.print(Rule())
        
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        
        table.add_row("Total Chunks", str(stats['total_chunks']))
        table.add_row("Max Chunks", str(stats['max_chunks']))
        table.add_row("Memory Used", f"{stats['memory_used_mb']:.1f} MB")
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Cache Hits", str(stats['cache_hits']))
        table.add_row("Cache Misses", str(stats['cache_misses']))
        table.add_row("Hit Rate", f"{stats['hit_rate']:.1%}")
        table.add_row("Evictions", str(stats['evictions']))
        
        self.console.print(table)
    
    def handle_output_list(self):
        """Handle /output command to list output chunks."""
        if not self.state.output_chunk_ids:
            self.console.print("[yellow]No output chunks available.[/yellow]")
            return
        
        table = Table(title="Output Chunks", box=box.SIMPLE)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Preview", width=50)
        table.add_column("Tokens", justify="right")
        table.add_column("ID", style="dim")
        
        for i, output_id in enumerate(self.state.output_chunk_ids):
            try:
                chunk = self.llm.get_chunk(output_id)
                table.add_row(
                    str(i + 1),
                    chunk['content'][:50] + "..." if len(chunk['content']) > 50 else chunk['content'],
                    str(chunk['token_count']),
                    chunk['chunk_id'][:8] + "..."
                )
            except:
                table.add_row(str(i + 1), "[Error loading chunk]", "?", "?")
        
        self.console.print(table)
    
    def handle_use_output(self, args: str):
        """Handle /use-output command to add output chunk as context."""
        try:
            output_num = int(args.strip()) - 1
            if output_num < 0 or output_num >= len(self.state.output_chunk_ids):
                self.console.print(f"[red]Invalid output number. Choose 1-{len(self.state.output_chunk_ids)}[/red]")
                return
            
            output_id = self.state.output_chunk_ids[output_num]
            self.state.context_chunk_ids.append(output_id)
            self.console.print(f"[green]Output {output_num + 1} added as context.[/green]")
        except ValueError:
            self.console.print("[red]Please provide a valid output number.[/red]")
    
    def handle_delete_output(self, args: str):
        """Handle /delete-output command to delete an output chunk."""
        try:
            output_num = int(args.strip()) - 1
            if output_num < 0 or output_num >= len(self.state.output_chunk_ids):
                self.console.print(f"[red]Invalid output number. Choose 1-{len(self.state.output_chunk_ids)}[/red]")
                return
            
            output_id = self.state.output_chunk_ids[output_num]
            
            # Delete the chunk (this will also release its KV cache)
            if self.llm.delete_chunk(output_id):
                self.state.output_chunk_ids.pop(output_num)
                self.console.print(f"[green]Output {output_num + 1} deleted.[/green]")
            else:
                self.console.print("[red]Failed to delete output chunk.[/red]")
        except ValueError:
            self.console.print("[red]Please provide a valid output number.[/red]")
    
    def run(self):
        """Run the interactive CLI."""
        self.console.clear()
        self.show_help()
        self.console.print("\n[dim]Press Enter to start[/dim]")
        input()
        
        # Main loop
        while True:
            try:
                # Render current state
                self.console.clear()
                layout = self.render_layout()
                self.console.print(layout)
                
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip()
                
                # Parse commands
                if user_input.startswith('/'):
                    parts = user_input.split(None, 1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command in ['/exit', '/quit']:
                        break
                    elif command == '/help':
                        self.show_help()
                        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                    elif command == '/system':
                        self.handle_system_prompt(args)
                        time.sleep(0.5)  # Brief pause to see confirmation
                    elif command == '/context':
                        self.handle_context(args)
                        time.sleep(0.5)
                    elif command == '/query':
                        self.handle_query(args)
                        time.sleep(0.5)
                    elif command == '/infer':
                        self.perform_inference()
                        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                    elif command == '/clear':
                        self.handle_clear(args)
                        time.sleep(0.5)
                    elif command == '/list':
                        self.handle_list()
                        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                    elif command == '/stats':
                        self.show_stats()
                        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                    elif command == '/output':
                        self.handle_output_list()
                        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                    elif command == '/use-output':
                        self.handle_use_output(args)
                        time.sleep(0.5)
                    elif command == '/delete-output':
                        self.handle_delete_output(args)
                        time.sleep(0.5)
                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                        time.sleep(1)
                else:
                    # Treat non-command input as query
                    if user_input:
                        self.handle_query(user_input)
                        time.sleep(0.5)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
        
        self.console.print("\n[bold cyan]Goodbye![/bold cyan]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NanoVLLM Cascade CLI with Chunk-Based API")
    parser.add_argument(
        "--model",
        type=str,
        default="~/huggingface/Qwen3-0.6B/",
        help="Path to the model directory"
    )
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = CascadeCLI(args.model)
    cli.run()


if __name__ == "__main__":
    main()