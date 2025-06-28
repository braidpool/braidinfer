"""
Distributed communication management for nano-vllm.
"""

import os
import socket
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, List, Any, Tuple


class DistributedManager:
    """Manages distributed communication for model runners."""
    
    def __init__(self, world_size: int, rank: int, events: Optional[Event | List[Event]] = None):
        """Initialize distributed manager.
        
        Args:
            world_size: Total number of processes
            rank: Rank of this process
            events: Event(s) for synchronization
        """
        self.world_size = world_size
        self.rank = rank
        self.events = events
        self.shm: Optional[SharedMemory] = None
        
        # Initialize process group
        self._init_process_group()
        
        # Setup shared memory for multi-process communication
        if self.world_size > 1:
            self._setup_shared_memory()
    
    def _init_process_group(self):
        """Initialize PyTorch distributed process group."""
        # Use environment variable or find a free port
        if "MASTER_PORT" not in os.environ:
            # Find a free port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            os.environ["MASTER_PORT"] = str(port)
        
        dist.init_process_group(
            "nccl", 
            f"tcp://localhost:{os.environ['MASTER_PORT']}", 
            world_size=self.world_size, 
            rank=self.rank
        )
    
    def _setup_shared_memory(self):
        """Setup shared memory for inter-process communication."""
        if self.rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
    
    def read_from_shared_memory(self) -> Tuple[str, List[Any]]:
        """Read method call from shared memory.
        
        Returns:
            Tuple of (method_name, args)
        """
        assert self.world_size > 1 and self.rank > 0
        assert self.events is not None
        
        # Wait for signal
        if isinstance(self.events, list):
            self.events[0].wait()
            self.events[0].clear()
        else:
            self.events.wait()
            self.events.clear()
        
        # Read data
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        
        return method_name, args
    
    def write_to_shared_memory(self, method_name: str, *args):
        """Write method call to shared memory.
        
        Args:
            method_name: Name of method to call
            *args: Arguments for the method
        """
        assert self.world_size > 1 and self.rank == 0
        assert self.events is not None
        
        # Serialize data
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        # Write to shared memory
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        
        # Signal other processes
        if isinstance(self.events, list):
            for event in self.events:
                event.set()
        else:
            self.events.set()
    
    def broadcast_method_call(self, obj: Any, method_name: str, *args) -> Any:
        """Broadcast method call to all processes.
        
        Args:
            obj: Object to call method on
            method_name: Name of method to call
            *args: Arguments for the method
            
        Returns:
            Result from method call
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_to_shared_memory(method_name, *args)
        
        method = getattr(obj, method_name, None)
        if method is None:
            raise AttributeError(f"Object has no method {method_name}")
        
        return method(*args)
    
    def cleanup(self):
        """Clean up distributed resources."""
        if self.world_size > 1 and self.shm is not None:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        
        torch.cuda.synchronize()
        dist.destroy_process_group()