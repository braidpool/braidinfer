import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)
        self.config = config  # Store config for access
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.context_manager = None  # Will be set externally
        
        # Make context manager accessible through config for model runner
        config.context_manager = None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # For now, don't prepend context automatically - it breaks chat formatting
        # Context should be managed through proper system messages or RAG patterns
        
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate_stream(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Generator that yields tokens as they are produced for a single prompt."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # For now, don't prepend context automatically - it breaks chat formatting
        # Context should be managed through proper system messages or RAG patterns
        
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        
        last_token_count = 0
        
        # Start tracking output if context manager is available
        output_tracker = None
        output_context = None
        if self.context_manager:
            output_context = self.context_manager.track_output()
            output_tracker = output_context.__enter__()
        
        try:
            while not seq.is_finished:
                seqs, is_prefill = self.scheduler.schedule()
                
                if seq in seqs:
                    token_ids = self.model_runner.call("run", seqs, is_prefill)
                    self.scheduler.postprocess(seqs, token_ids)
                    
                    if not is_prefill and len(seq.completion_token_ids) > last_token_count:
                        new_tokens = seq.completion_token_ids[last_token_count:]
                        
                        # Track generated tokens
                        if output_tracker:
                            output_tracker.add_tokens(new_tokens)
                        
                        for token_id in new_tokens:
                            token_text = self.tokenizer.decode([token_id])
                            yield {
                                "token": token_text,
                                "token_id": token_id,
                                "finished": False,
                                "text": self.tokenizer.decode(seq.completion_token_ids)
                            }
                        last_token_count = len(seq.completion_token_ids)
            
            # Finalize output chunk
            if output_tracker and seq.completion_token_ids:
                output_chunk = output_tracker.finalize(self.tokenizer)
                if output_chunk:
                    # Include chunk info in final yield
                    yield {
                        "token": "",
                        "token_id": None,
                        "finished": True,
                        "text": self.tokenizer.decode(seq.completion_token_ids),
                        "output_chunk": output_chunk
                    }
                else:
                    yield {
                        "token": "",
                        "token_id": None,
                        "finished": True,
                        "text": self.tokenizer.decode(seq.completion_token_ids)
                    }
            else:
                yield {
                    "token": "",
                    "token_id": None,
                    "finished": True,
                    "text": self.tokenizer.decode(seq.completion_token_ids)
                }
                
        finally:
            # Clean up output tracker
            if output_context:
                output_context.__exit__(None, None, None)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Track outputs for each prompt if context manager available
        output_trackers = {}
        output_contexts = {}
        
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            if self.context_manager:
                context = self.context_manager.track_output()
                tracker = context.__enter__()
                output_contexts[i] = context
                output_trackers[i] = tracker
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        try:
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    # Track tokens for this sequence
                    if seq_id in output_trackers:
                        output_trackers[seq_id].add_tokens(token_ids)
                    if use_tqdm:
                        pbar.update(1)
                        
            # Finalize output chunks
            outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
            results = []
            for i, token_ids in enumerate(outputs):
                result = {
                    "text": self.tokenizer.decode(token_ids), 
                    "token_ids": token_ids
                }
                
                # Add output chunk info if available
                if i in output_trackers:
                    output_chunk = output_trackers[i].finalize(self.tokenizer)
                    if output_chunk:
                        result["output_chunk"] = output_chunk
                        
                results.append(result)
                
            if use_tqdm:
                pbar.close()
            return results
            
        finally:
            # Clean up output trackers
            for context in output_contexts.values():
                context.__exit__(None, None, None)
