"""
LLM Engine for single-GPU nano-vllm.
"""

import atexit
from dataclasses import fields
from time import perf_counter
from typing import List, Optional, Tuple, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """LLM Engine for single-GPU inference."""
    
    def __init__(self, model, **kwargs):
        # Handle model_kwargs separately
        model_kwargs = kwargs.pop('model_kwargs', {})
        
        # Extract config parameters
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        
        # Add model-specific kwargs to config
        if 'use_custom_kernels' in model_kwargs:
            config_kwargs['use_custom_kernels'] = model_kwargs['use_custom_kernels']
        if 'use_custom_chunk_kernel' in model_kwargs:
            config_kwargs['use_custom_chunk_kernel'] = model_kwargs['use_custom_chunk_kernel']
        
        config = Config(model, **config_kwargs)
        
        # For single GPU, tensor_parallel_size should be 1
        if hasattr(config, 'tensor_parallel_size') and config.tensor_parallel_size != 1:
            print(f"Warning: tensor_parallel_size={config.tensor_parallel_size} ignored for single-GPU mode")
        
        # Estimate KV cache blocks if not specified
        if config.num_kvcache_blocks == -1:
            from nanovllm.engine.model_loader import ModelLoader
            config.num_kvcache_blocks = ModelLoader.calculate_kvcache_blocks(
                config, config.hf_config, 1, config.kvcache_block_size
            )
        
        # Initialize model runner directly (no multiprocessing)
        self.model_runner = ModelRunner(config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        
        # Set token IDs from model config first, then tokenizer as fallback
        # Use model config as the authoritative source
        if hasattr(config.hf_config, 'eos_token_id'):
            config.eos = config.hf_config.eos_token_id
        else:
            config.eos = self.tokenizer.eos_token_id
            
        if hasattr(config.hf_config, 'bos_token_id'):
            config.bos = config.hf_config.bos_token_id
        else:
            config.bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else -1
        
        # Verify tokenizer matches model config
        if self.tokenizer.eos_token_id != config.eos:
            print(f"Note: Setting EOS to model config value {config.eos} (tokenizer had {self.tokenizer.eos_token_id})")
        if self.tokenizer.bos_token_id != config.bos and config.bos != -1:
            print(f"Note: Setting BOS to model config value {config.bos} (tokenizer had {self.tokenizer.bos_token_id})")
        
        # Initialize scheduler
        self.scheduler = Scheduler(config)
        
        # Connect page manager to model runner
        self.model_runner.set_page_manager(self.scheduler.page_manager)
        
        # Store cascade setting (deprecated - always False now)
        self.cascade_enabled = False
        
        # Store config for later reference
        self.config = config
        
        # Initialize timing stats
        self._timing_stats = []
        
        # Track retained output sequences
        self._retained_sequences: Dict[int, Tuple[Sequence, Dict[str, Any]]] = {}
        
        # Initialize sequence counter for chunk generation
        self._seq_counter = 10000  # Start from a high number to avoid conflicts
        
        # Register cleanup
        atexit.register(self.exit)
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.exit()

    def exit(self):
        """Clean up resources."""
        # Release all retained sequences first
        if hasattr(self, '_retained_sequences'):
            self.release_all_retained_sequences()
        
        if hasattr(self, 'model_runner'):
            # Clean up model runner resources
            if hasattr(self.model_runner, 'model'):
                del self.model_runner.model
            if hasattr(self.model_runner, 'wrapper_manager'):
                del self.model_runner.wrapper_manager
            del self.model_runner
        
        if hasattr(self, 'scheduler'):
            del self.scheduler
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Force cleanup
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a new request to the scheduler."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # Process stop sequences if provided
        if sampling_params.stop is not None:
            if sampling_params.stop_token_ids is None:
                sampling_params.stop_token_ids = []
            
            # Convert stop sequences to token IDs
            stop_sequences = sampling_params.stop if isinstance(sampling_params.stop, list) else [sampling_params.stop]
            for stop_seq in stop_sequences:
                # Tokenize without special tokens to get the exact sequence
                tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if len(tokens) == 1:
                    # Single token stop sequence
                    if tokens[0] not in sampling_params.stop_token_ids:
                        sampling_params.stop_token_ids.append(tokens[0])
                # Note: Multi-token stop sequences would need more complex handling
        
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    
    # Removed add_cascade_request - use add_request instead

    def step(self) -> list[Sequence]:
        """Run one step of inference."""
        # Scheduler now always returns 2-tuple
        seqs, is_prefill = self.scheduler.schedule()
        
        finished_seqs = []
        
        if seqs:
            start = perf_counter()
            
            # Check if any sequence has active_chunks (for custom chunk attention)
            # For chunk-based generation, we need active_chunks during both prefill and decode
            active_chunks = None
            for seq in seqs:
                if hasattr(seq, 'active_chunks') and seq.active_chunks is not None:
                    active_chunks = seq.active_chunks
                    # Debug output disabled for cleaner output
                    pass
                    break
            
            # Pass active_chunks to model runner if available
            if active_chunks is not None:
                token_ids = self.model_runner.run(seqs, is_prefill, active_chunks=active_chunks)
            else:
                token_ids = self.model_runner.run(seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            self._timing_stats.append(perf_counter() - start)
            
            # Debug: Show generated tokens (only first few and EOS)
            if token_ids and hasattr(self, '_debug_generate_from_chunks'):
                for seq, token_id in zip(seqs, token_ids):
                    if len(seq.token_ids) <= 5 or token_id == self.config.eos:
                        decoded = self.tokenizer.decode([token_id]) if token_id != self.config.eos else "<EOS>"
                        print(f"[DEBUG] Token {len(seq.token_ids)}: {token_id} ('{decoded}')")
                        if token_id == self.config.eos:
                            print(f"[DEBUG] Generated EOS token, sequence will finish")
            
            # Collect finished sequences
            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED:
                    finished_seqs.append(seq)
                    
                    # If output cache should be retained, store it
                    if seq.retain_output_cache:
                        cache_info = self.scheduler.page_manager.get_sequence_cache_info(seq)
                        if cache_info:
                            self._retained_sequences[seq.seq_id] = (seq, cache_info)
        
        return finished_seqs

    def generate(self, prompts: str | list[int] | list[str] | list[list[int]], 
                 sampling_params: SamplingParams, stream: bool = False) -> list[dict] | list:
        """Generate completions for prompts."""
        # Normalize inputs
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, list) and prompts and isinstance(prompts[0], int):
            prompts = [prompts]
        
        # Reset timing stats
        self._timing_stats = []
        
        # Add all requests
        for prompt in prompts:
            self.add_request(prompt, sampling_params)
        
        if stream:
            # Streaming mode - yield results as they're generated
            return self._generate_stream(len(prompts))
        else:
            # Non-streaming mode - collect all results
            results = []
            pbar = tqdm(total=len(prompts), desc="Generating", disable=len(prompts) == 1,
                        bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
            
            while len(results) < len(prompts):
                finished = self.step()
                results.extend(finished)
                pbar.update(len(finished))
                
                # Update progress bar stats
                if self._timing_stats:
                    avg_time = sum(self._timing_stats) / len(self._timing_stats)
                    if self.scheduler.running:
                        # For now, use a simple heuristic: sequences with only prompt tokens are in prefill
                        is_prefill = any(seq.num_completion_tokens == 0 for seq in self.scheduler.running)
                        phase = "Prefill" if is_prefill else "Decode"
                        tokens_per_sec = len(self.scheduler.running) / avg_time if avg_time > 0 else 0
                        pbar.set_postfix_str(f"{phase}={tokens_per_sec:.0f}tok/s")
            
            pbar.close()
            
            # Format results
            outputs = []
            for seq in results:
                output = {
                    "text": self.tokenizer.decode(seq.completion_token_ids),
                    "token_ids": seq.completion_token_ids
                }
                outputs.append(output)
            
            return outputs
    
    def _generate_stream(self, num_prompts: int):
        """Generate completions with streaming."""
        # Debug for chunk generation
        if hasattr(self, '_debug_generate_from_chunks'):
            print(f"[DEBUG] _generate_stream called with {num_prompts} prompts")
            print(f"[DEBUG] Scheduler state: {len(self.scheduler.waiting)} waiting, {len(self.scheduler.running)} running")
            
        completed_seqs = []
        seq_outputs = {}  # Track per-sequence outputs
        
        # Initial check - this should execute when generator is first consumed
        print(f"[DEBUG] _generate_stream body executing, num_prompts={num_prompts}")
        if hasattr(self, '_debug_generate_from_chunks'):
            print(f"[DEBUG] Debug mode is active")
        print(f"[DEBUG] completed_seqs={len(completed_seqs)}, num_prompts={num_prompts}")
        print(f"[DEBUG] Scheduler: waiting={len(self.scheduler.waiting)}, running={len(self.scheduler.running)}")
            
        while len(completed_seqs) < num_prompts:
            # Get running sequences before step
            running_seqs = list(self.scheduler.running) if self.scheduler.running else []
            
            # Take a step
            finished = self.step()
            
            # Debug - only print every 10th step to reduce noise
            if hasattr(self, '_debug_generate_from_chunks') and len(completed_seqs) == 0:
                # Only debug until we get first completion
                pass
            
            # Check for new tokens in running sequences
            for seq in running_seqs:
                if seq.status == SequenceStatus.RUNNING and seq.completion_token_ids:
                    # Get new tokens since last yield
                    seq_id = id(seq)
                    prev_len = seq_outputs.get(seq_id, 0)
                    current_tokens = seq.completion_token_ids
                    
                    if len(current_tokens) > prev_len:
                        # New tokens generated
                        new_tokens = current_tokens[prev_len:]
                        new_text = self.tokenizer.decode(new_tokens)
                        
                        yield {
                            "text": new_text,
                            "token_ids": new_tokens,
                            "cumulative_text": self.tokenizer.decode(current_tokens),
                            "finished": False
                        }
                        
                        seq_outputs[seq_id] = len(current_tokens)
            
            # Handle finished sequences
            for seq in finished:
                completed_seqs.append(seq)
                seq_id = id(seq)
                
                # Debug
                if hasattr(self, '_debug_generate_from_chunks'):
                    print(f"[DEBUG] Finished seq {seq.seq_id}: total={len(seq.token_ids)}, "
                          f"prompt={seq.num_prompt_tokens}, completion={len(seq.completion_token_ids)}")
                
                # Yield final result
                yield {
                    "text": "",  # No new text, just marking as finished
                    "token_ids": [],
                    "cumulative_text": self.tokenizer.decode(seq.completion_token_ids),
                    "finished": True,
                    "final_output": {
                        "text": self.tokenizer.decode(seq.completion_token_ids),
                        "token_ids": seq.completion_token_ids
                    }
                }
                
                # Clean up tracking
                if seq_id in seq_outputs:
                    del seq_outputs[seq_id]
                    
        # Disable debug mode if it was enabled
        if hasattr(self, '_debug_generate_from_chunks'):
            self._debug_generate_from_chunks = False
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        return self.model_runner.get_metrics()
    
    def _find_think_tag_positions(self, token_ids: List[int]) -> Optional[Tuple[int, int]]:
        """Find the positions of <think> and </think> tags in token list.
        
        Returns:
            Tuple of (think_start_pos, think_end_pos) or None if no complete think tags found.
            Positions are inclusive of the tags themselves.
            If only opening tag is found, returns (think_start, len(token_ids)-1).
        """
        # Token IDs for Qwen3 model - these should be configurable per model
        THINK_TOKEN_ID = 151667  # <think>
        THINK_END_TOKEN_ID = 151668  # </think>
        
        think_start = None
        think_end = None
        
        for i, token_id in enumerate(token_ids):
            if token_id == THINK_TOKEN_ID and think_start is None:
                think_start = i
            elif token_id == THINK_END_TOKEN_ID and think_start is not None:
                think_end = i
                break
        
        if think_start is not None:
            if think_end is not None:
                return (think_start, think_end)
            else:
                # Unclosed think tag - treat rest of output as think content
                return (think_start, len(token_ids) - 1)
        return None
    
    def get_retained_sequences(self) -> Dict[int, Dict[str, Any]]:
        """Get information about retained output sequences.
        
        Returns:
            Dictionary mapping sequence ID to info dict containing:
            - text: The generated text (without think tags if present)
            - token_ids: The token IDs
            - think_positions: Optional tuple of (start, end) positions of think tags
            - cache_info: KV cache information
        """
        result = {}
        for seq_id, (seq, cache_info) in self._retained_sequences.items():
            # Get completion tokens
            completion_tokens = seq.completion_token_ids
            
            # Find think tag positions if present
            think_positions = self._find_think_tag_positions(completion_tokens)
            
            # Decode text
            if think_positions:
                # Decode without think tags
                start_pos, end_pos = think_positions
                tokens_without_think = (
                    completion_tokens[:start_pos] + 
                    completion_tokens[end_pos + 1:]
                )
                text = self.tokenizer.decode(tokens_without_think)
            else:
                text = self.tokenizer.decode(completion_tokens)
            
            result[seq_id] = {
                "text": text,
                "token_ids": completion_tokens,
                "think_positions": think_positions,
                "cache_info": cache_info,
                "prompt_length": seq.num_prompt_tokens,
                "completion_length": seq.num_completion_tokens
            }
        
        return result
    
    def release_retained_sequence(self, seq_id: int) -> bool:
        """Manually release a retained sequence's KV cache.
        
        Returns:
            True if released, False if not found.
        """
        if seq_id not in self._retained_sequences:
            return False
        
        # Deallocate the KV cache
        self.scheduler.page_manager.deallocate_by_seq_id(seq_id)
        
        # Remove from retained sequences
        del self._retained_sequences[seq_id]
        
        return True
    
    def release_all_retained_sequences(self):
        """Release all retained sequences' KV cache."""
        seq_ids = list(self._retained_sequences.keys())
        for seq_id in seq_ids:
            self.release_retained_sequence(seq_id)
    
    def generate_from_chunks(self, composition: Dict[str, Any], 
                           sampling_params: SamplingParams,
                           stream: bool = False,
                           use_custom_kernel: bool = True):
        """
        Generate output from pre-filled chunk composition.
        
        This method creates a pre-filled sequence with KV cache from chunks
        and uses the standard generation loop to handle new token generation.
        
        Args:
            composition: Dictionary with system_chunk, context_chunks, query_chunk
            sampling_params: Generation parameters
            stream: Whether to stream output
            use_custom_kernel: Whether to use custom chunk attention kernel
            
        Returns:
            Generated output (streaming or complete)
        """
        # Extract chunks from composition
        system_chunk = composition['system_chunk']
        context_chunks = composition.get('context_chunks', [])
        query_chunk = composition['query_chunk']
        
        # Prepare all chunks in order
        all_chunks = [system_chunk] + context_chunks + [query_chunk]
        
        # Collect all token IDs from chunks (skip empty chunks)
        all_token_ids = []
        for chunk in all_chunks:
            if chunk.token_ids:  # Skip empty chunks
                all_token_ids.extend(chunk.token_ids)
        
        # Track how many tokens are from chunks (and thus in KV cache)
        chunk_token_count = len(all_token_ids)
        
        # CRITICAL: Add generation prompt tokens
        # This tells the model it's the assistant's turn to respond
        # First, build a minimal message list to get the generation prompt
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""}
        ]
        template_with_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        template_without_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Find the generation prompt by comparing the two
        generation_prompt_tokens = []
        if template_with_prompt != template_without_prompt:
            # Extract just the generation prompt part
            generation_prompt = template_with_prompt[len(template_without_prompt):]
            generation_prompt_tokens = self.tokenizer.encode(generation_prompt, add_special_tokens=False)
            all_token_ids.extend(generation_prompt_tokens)
        
        # Create a sequence that represents what needs to be processed
        from nanovllm.engine.sequence import Sequence
        
        # If we have no tokens at all, we can't generate
        if not all_token_ids:
            yield {"text": "", "token_ids": []}
            return
        
        # The key insight: Only include tokens that need prefill (generation prompt)
        # The chunk tokens are already in KV cache
        if generation_prompt_tokens:
            # Create sequence with just the generation prompt tokens
            seq = Sequence(
                token_ids=generation_prompt_tokens,
                sampling_params=sampling_params
            )
            # Track the full context length for position calculation
            seq._full_context_length = len(all_token_ids)
            seq._chunk_token_count = chunk_token_count
        else:
            # No generation prompt, create minimal sequence
            # Use a dummy token to trigger generation
            seq = Sequence(
                token_ids=[self.tokenizer.eos_token_id],
                sampling_params=sampling_params
            )
            seq._full_context_length = len(all_token_ids) + 1
            seq._chunk_token_count = chunk_token_count
        
        # Debug: Check what tokens we're processing (commented out for cleaner output)
        # print(f"[DEBUG] Sequence tokens to prefill: {len(seq.token_ids)} tokens: {seq.token_ids}")
        # print(f"[DEBUG] Full context: {len(all_token_ids)} tokens (chunks: {chunk_token_count}, prompt: {len(generation_prompt_tokens)})")
        
        # Debug chunk contents (commented out for cleaner output)
        # print(f"[DEBUG] Chunk contents:")
        # for i, chunk in enumerate(all_chunks):
        #     print(f"  Chunk {i} ({chunk.chunk_type}): '{chunk.content[:50]}...' ({len(chunk.token_ids)} tokens)")
        
        # Now we need to set up the sequence to use the pre-existing KV cache from chunks
        if use_custom_kernel:
            # For custom kernel path: create a combined page table from all chunks
            combined_page_table = []
            total_kv_length = 0
            
            for chunk in all_chunks:
                if chunk.page_table is not None and chunk.token_ids:  # Skip empty chunks
                    combined_page_table.extend(chunk.page_table)
                    total_kv_length += chunk.kv_length
            
            # Set the sequence's page table to the combined one
            seq.block_table = combined_page_table
            
            # No tokens are cached in this sequence - it only contains tokens to prefill
            seq.num_cached_tokens = 0
            
            # Update page manager to track this sequence
            if hasattr(self.scheduler.page_manager, 'seq_page_tables'):
                # Pre-register the sequence with its page table
                # This prevents the scheduler from trying to allocate new pages
                self.scheduler.page_manager.seq_page_tables[seq.seq_id] = combined_page_table
                # Set the sequence length to the total KV cache length from chunks
                self.scheduler.page_manager.seq_lengths[seq.seq_id] = total_kv_length
                
                # print(f"[DEBUG] Pre-registered seq {seq.seq_id} with {len(combined_page_table)} pages")
                pass
            
            # Debug info (commented out)
            # print(f"[DEBUG] KV cached length: {total_kv_length}, num_cached_tokens: {seq.num_cached_tokens}")
            # print(f"[DEBUG] EOS token: {self.config.eos}")
                
            # Store active chunks in sequence for custom kernel to use during decode
            seq.active_chunks = all_chunks
            
            # Also need to handle the fact that after prefill, positions will continue from full context
            if hasattr(seq, '_full_context_length'):
                # This will be used for decode position calculation
                seq._position_offset_for_decode = seq._full_context_length
        else:
            # Cascade attention path no longer supported
            raise RuntimeError("Non-custom kernel path is no longer supported. This code path should not be reached.")
        
        # Add sequence to scheduler normally
        # print(f"[DEBUG] Before scheduler.add: seq {seq.seq_id} has {len(seq.token_ids)} tokens, status={seq.status}")
        self.scheduler.add(seq)
        # print(f"[DEBUG] After scheduler.add: waiting={len(self.scheduler.waiting)}, running={len(self.scheduler.running)}")
        
        if stream:
            # Use the standard streaming generation method
            # print(f"[DEBUG] Starting streaming generation")
            self._debug_generate_from_chunks = True
            
            # Yield from the streaming generator
            yield from self._generate_stream(1)
            
            self._debug_generate_from_chunks = False
        else:
            # Use standard non-streaming generation
            self._debug_generate_from_chunks = True
            results = []
            max_steps = sampling_params.max_tokens + 10  # Safety limit
            step_count = 0
            while len(results) < 1 and step_count < max_steps:
                finished = self.step()
                results.extend(finished)
                step_count += 1
                
                # Debug: Check if we're stuck
                if step_count <= 3 and hasattr(self, '_debug_generate_from_chunks'):
                    print(f"[DEBUG] Step {step_count}: {len(finished)} finished sequences")
                    if hasattr(self.scheduler, 'running') and self.scheduler.running:
                        for seq in self.scheduler.running:
                            print(f"[DEBUG] Running seq {seq.seq_id}: {len(seq.token_ids)} tokens, "
                                  f"prompt_tokens={seq.num_prompt_tokens}, status={seq.status}")
                    if hasattr(self.scheduler, 'waiting') and self.scheduler.waiting:
                        print(f"[DEBUG] {len(self.scheduler.waiting)} sequences waiting")
            
            # Extract the completion tokens from the finished sequence
            if results:
                seq = results[0]
                # Get only the newly generated tokens
                # The sequence started with just generation prompt tokens
                # Everything after that is new generation
                num_prompt_tokens = len(generation_prompt_tokens) if generation_prompt_tokens else 1
                
                if len(seq.token_ids) > num_prompt_tokens:
                    completion_tokens = seq.token_ids[num_prompt_tokens:]
                else:
                    # No new tokens generated
                    completion_tokens = []
                
                result = {
                    "text": self.tokenizer.decode(completion_tokens) if completion_tokens else "",
                    "token_ids": completion_tokens
                }
                
                # Disable debug mode
                self._debug_generate_from_chunks = False
                
                yield result
            else:
                # Disable debug mode
                self._debug_generate_from_chunks = False
                yield {"text": "", "token_ids": []}