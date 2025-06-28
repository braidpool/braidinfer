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
        # Extract config parameters
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # For single GPU, tensor_parallel_size should be 1
        if hasattr(config, 'tensor_parallel_size') and config.tensor_parallel_size != 1:
            print(f"Warning: tensor_parallel_size={config.tensor_parallel_size} ignored for single-GPU mode")
        
        # Initialize model runner directly (no multiprocessing)
        self.model_runner = ModelRunner(config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # Initialize scheduler
        self.scheduler = Scheduler(config)
        
        # Connect page manager to model runner
        self.model_runner.set_page_manager(self.scheduler.page_manager)
        
        # Register cleanup
        atexit.register(self.exit)

    def exit(self):
        """Clean up resources."""
        if hasattr(self, 'model_runner'):
            # No distributed cleanup needed
            del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a new request to the scheduler."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> list[Sequence]:
        """Run one step of inference."""
        seqs, is_prefill = self.scheduler.schedule()
        finished_seqs = []
        
        if seqs:
            start = perf_counter()
            token_ids = self.model_runner.run(seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            self._timing_stats.append(perf_counter() - start)
            
            # Collect finished sequences
            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED:
                    finished_seqs.append(seq)
        
        return finished_seqs

    def generate(self, prompts: str | list[int] | list[str] | list[list[int]], 
                 sampling_params: SamplingParams) -> list[dict]:
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
        
        # Process until all complete
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
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        return self.model_runner.get_metrics()