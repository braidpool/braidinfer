from collections import deque

from braidinfer.config import Config
from braidinfer.engine.sequence import Sequence, SequenceStatus
from braidinfer.engine.page_manager import PageManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        
        # Create page manager
        hf_config = config.hf_config
        
        # Calculate head_dim if not present
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        else:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            
        self.page_manager = PageManager(
            num_pages=config.num_kvcache_blocks,
            page_size=config.kvcache_block_size,
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=hf_config.torch_dtype
        )
        
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.page_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.page_manager.allocate(seq)
            num_batched_tokens += len(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        if not self.running:
            # No sequences to decode
            return [], False
            
        scheduled_seqs = []
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.page_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.page_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
            # All sequences were preempted
            return [], False
            
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.page_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # Check if this token is a stop token BEFORE appending
            is_stop_token = (
                (not seq.ignore_eos and token_id == self.eos) or
                (token_id in seq.stop_token_ids)
            )
            
            # Only append the token if it's not a stop token
            # This prevents stop tokens from appearing in the output
            if not is_stop_token:
                seq.append_token(token_id)
            
            # Check if we should stop (including max tokens)
            should_stop = is_stop_token or (seq.num_completion_tokens == seq.max_tokens)
            
            if should_stop:
                seq.status = SequenceStatus.FINISHED
                # Only deallocate if not retaining output cache
                if not seq.retain_output_cache:
                    self.page_manager.deallocate(seq)
                self.running.remove(seq)
