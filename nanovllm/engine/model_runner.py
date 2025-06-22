import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.virtual_sequence import VirtualSequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        import os
        master_port = os.environ.get("MASTER_PORT", "2333")
        dist.init_process_group("nccl", f"tcp://localhost:{master_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, gpu_memory_utilization):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * gpu_memory_utilization - used) // block_bytes
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]

        # Check if we need to filter blocks based on context manager
        if hasattr(self.config, 'context_manager') and self.config.context_manager:
            context_mgr = self.config.context_manager
            if context_mgr.has_inactive_blocks():
                # Use virtual block table for filtering
                raw_tables = [seq.block_table for seq in seqs]
                return context_mgr.get_filtered_block_table(raw_tables, filter_inactive=True).cuda(non_blocking=True)

        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if len(input_ids) != len(slot_mapping):
            print(f"[DEBUG] prepare_prefill mismatch:")
            print(f"  input_ids length: {len(input_ids)}")
            print(f"  slot_mapping length: {len(slot_mapping)}")
            for i, seq in enumerate(seqs):
                print(f"  Seq {i}: len={len(seq)}, cached={seq.num_cached_tokens}, blocks={seq.num_blocks}, cached_blocks={seq.num_cached_blocks}")
        assert len(input_ids) == len(slot_mapping)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Get active blocks if context manager is available
        active_blocks = None
        if hasattr(self.config, 'context_manager') and self.config.context_manager:
            active_blocks = self.config.context_manager.get_active_blocks()

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, active_blocks)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            if not seq.block_table:
                raise RuntimeError(f"Sequence has empty block_table in prepare_decode")
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # Get active blocks if context manager is available
        active_blocks = None
        if hasattr(self.config, 'context_manager') and self.config.context_manager:
            active_blocks = self.config.context_manager.get_active_blocks()

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, active_blocks=active_blocks)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        # Check if we need eager mode due to context manager
        force_eager = False
        if hasattr(self.config, 'context_manager') and self.config.context_manager:
            # Only force eager mode when we have inactive blocks (filtering required)
            # Active chunks can use CUDA graphs as long as block tables are compatible
            if self.config.context_manager.has_inactive_blocks():
                force_eager = True  # CUDA graphs don't support dynamic filtering

        if is_prefill or self.enforce_eager or force_eager or input_ids.size(0) > 512:
            # Debug: print when using eager mode
            if hasattr(self.config, 'context_manager') and self.config.context_manager and len(self.config.context_manager.active_chunks) > 0:
                reason = []
                if is_prefill: reason.append("prefill")
                if self.enforce_eager: reason.append("enforce_eager")
                if force_eager: reason.append("force_eager")
                if input_ids.size(0) > 512: reason.append("large_batch")
                print(f"[DEBUG] Using eager mode with context. Reasons: {', '.join(reason)}")
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()

            # For basic inference without context, use the original graphs
            if not hasattr(self.config, 'context_manager') or not self.config.context_manager or len(self.config.context_manager.active_chunks) == 0:
                # Check if block tables fit in original graphs
                if context.block_tables is not None and context.block_tables.size(1) > self.graph_vars["block_tables"].size(1):
                    # Block table too large, fall back to eager mode
                    return self.model.compute_logits(self.model(input_ids, positions))

                # Use original single graph set
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
                self.reset_graph_vars()
                self.graph_vars["input_ids"][:bs] = input_ids
                self.graph_vars["positions"][:bs] = positions
                self.graph_vars["slot_mapping"][:bs] = context.slot_mapping
                self.graph_vars["context_lens"][:bs] = context.context_lens
                if context.block_tables is not None:
                    self.graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                graph.replay()
                return self.model.compute_logits(self.graph_vars["outputs"][:bs])

            # Determine required block table size for context manager inference
            required_block_size = self.current_max_block_size
            if context.block_tables is not None:
                required_block_size = max(required_block_size, context.block_tables.size(1))

            # Get appropriate graph set
            try:
                graph_set = self.get_or_create_graph_set(required_block_size)
                graphs = graph_set["graphs"]
                graph_vars = graph_set["graph_vars"]

                # Reset graph vars
                graph_vars["input_ids"].zero_()
                graph_vars["positions"].zero_()
                graph_vars["slot_mapping"].zero_()
                graph_vars["context_lens"].zero_()
                graph_vars["block_tables"].zero_()

                # Find appropriate batch size graph
                graph = graphs[next(x for x in self.graph_bs if x >= bs)]

                # Set inputs
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"][:bs] = context.context_lens
                if context.block_tables is not None:
                    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

                # Debug: print when using CUDA graph with context
                #if hasattr(self.config, 'context_manager') and self.config.context_manager and len(self.config.context_manager.active_chunks) > 0:
                #    print(f"[DEBUG] Using CUDA graph with context ({len(self.config.context_manager.active_chunks)} active chunks, block_size={graph_set['max_block_size']})")

                graph.replay()

                # Periodic cleanup
                if self.graph_recomputation_stats["recomputations"] % 5 == 0:
                    self.cleanup_unused_graphs()

                return self.model.compute_logits(graph_vars["outputs"][:bs])

            except Exception as e:
                # Fall back to eager mode if graph operations fail
                print(f"[WARNING] CUDA graph execution failed ({e}), falling back to eager mode")
                self.graph_recomputation_stats["fallbacks"] += 1
                return self.model.compute_logits(self.model(input_ids, positions))

    def reset_graph_vars(self):
        """Reset graph variables for basic inference"""
        if hasattr(self, 'graph_vars') and self.graph_vars is not None:
            self.graph_vars["input_ids"].zero_()
            self.graph_vars["positions"].zero_()
            self.graph_vars["slot_mapping"].zero_()
            self.graph_vars["context_lens"].zero_()
            self.graph_vars["block_tables"].zero_()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Dynamic graph recomputation infrastructure
        self.graph_sets = {}  # Dict[(max_block_table_size,): {bs: graph, ...}]
        self.current_max_block_size = max_num_blocks
        self.graph_recomputation_stats = {"recomputations": 0, "cache_hits": 0, "fallbacks": 0}

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], active_blocks=None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # Store initial graph set
        self.graph_sets[max_num_blocks] = {
            "graphs": self.graphs.copy(),
            "graph_vars": {k: v.clone() for k, v in self.graph_vars.items()},
            "max_block_size": max_num_blocks
        }

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state

    @torch.inference_mode()
    def recapture_cudagraph_with_size(self, max_block_table_size: int):
        """Recapture CUDA graphs with a specific block table size"""
        print(f"[INFO] Recomputing CUDA graphs for block table size {max_block_table_size}")
        self.graph_recomputation_stats["recomputations"] += 1

        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        # Set device context for graph capture
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.config.hf_config.torch_dtype)
        torch.set_default_device("cuda")

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)

        # Round up to reasonable increment to reduce recomputation frequency
        rounded_size = max(max_block_table_size, ((max_block_table_size + 15) // 16) * 16)

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, rounded_size, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        graphs = {}

        # Use existing graph pool to save memory
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], active_blocks=None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # Store the new graph set
        self.graph_sets[rounded_size] = {
            "graphs": graphs,
            "graph_vars": graph_vars,
            "max_block_size": rounded_size
        }

        # Restore device context
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state

        return rounded_size

    def get_or_create_graph_set(self, required_block_size: int):
        """Get existing graph set or create new one for required block table size"""
        # Check if we have a suitable graph set
        for size, graph_set in self.graph_sets.items():
            if size >= required_block_size:
                self.graph_recomputation_stats["cache_hits"] += 1
                return graph_set

        # Need to create new graph set
        actual_size = self.recapture_cudagraph_with_size(required_block_size)
        return self.graph_sets[actual_size]

    def cleanup_unused_graphs(self, keep_recent: int = 2):
        """Clean up unused graph sets to manage memory"""
        if len(self.graph_sets) <= keep_recent:
            return

        # Keep the most recent graph sets based on block size
        sorted_sizes = sorted(self.graph_sets.keys(), reverse=True)
        sizes_to_remove = sorted_sizes[keep_recent:]

        for size in sizes_to_remove:
            print(f"[INFO] Cleaning up graph set for block size {size}")
            del self.graph_sets[size]

    def print_graph_stats(self):
        """Print CUDA graph recomputation statistics"""
        stats = self.graph_recomputation_stats
        graph_sets_info = {size: data["max_block_size"] for size, data in self.graph_sets.items()}
        print(f"[GRAPH STATS] Recomputations: {stats['recomputations']}, "
              f"Cache hits: {stats['cache_hits']}, Fallbacks: {stats['fallbacks']}")
        print(f"[GRAPH STATS] Active graph sets: {graph_sets_info}")
