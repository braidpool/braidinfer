"""
LLM Engine for single-GPU nano-vllm.
"""

import atexit
from dataclasses import fields
from time import perf_counter
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
        config.eos = self.tokenizer.eos_token_id
        
        # Initialize scheduler
        if getattr(config, 'enable_cascade_attention', False):
            from nanovllm.engine.flashinfer_scheduler import FlashInferScheduler
            self.scheduler = FlashInferScheduler(config)
            print("Using FlashInferScheduler for cascade attention")
        else:
            self.scheduler = Scheduler(config)
        
        # Connect page manager to model runner
        self.model_runner.set_page_manager(self.scheduler.page_manager)
        
        # Store cascade setting
        self.cascade_enabled = getattr(config, 'enable_cascade_attention', False)
        
        # Store config for later reference
        self.config = config
        
        # Initialize timing stats
        self._timing_stats = []
        
        # Register cleanup
        atexit.register(self.exit)
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.exit()

    def exit(self):
        """Clean up resources."""
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
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    
    # Removed add_cascade_request - use add_request instead

    def step(self) -> list[Sequence]:
        """Run one step of inference."""
        # Handle both 2-tuple and 3-tuple returns from scheduler
        schedule_result = self.scheduler.schedule()
        if len(schedule_result) == 3:
            seqs, is_prefill, cascade_data = schedule_result
        else:
            seqs, is_prefill = schedule_result
            cascade_data = None
        
        finished_seqs = []
        
        if seqs:
            start = perf_counter()
            # Pass cascade data to model runner if available
            if cascade_data is not None:
                token_ids = self.model_runner.run(seqs, is_prefill, cascade_data)
            else:
                token_ids = self.model_runner.run(seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            self._timing_stats.append(perf_counter() - start)
            
            # Collect finished sequences
            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED:
                    finished_seqs.append(seq)
        
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
        completed_seqs = []
        seq_outputs = {}  # Track per-sequence outputs
        
        while len(completed_seqs) < num_prompts:
            # Get running sequences before step
            running_seqs = list(self.scheduler.running) if self.scheduler.running else []
            
            # Take a step
            finished = self.step()
            
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
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        return self.model_runner.get_metrics()