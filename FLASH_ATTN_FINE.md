# Fine-Grained Memory Control Plan for Flash Attention

## Executive Summary

This document outlines a comprehensive plan to implement fine-grained memory control for flash_attn, enabling precise tracking, management, and manipulation of GPU memory allocations. The plan includes memory introspection, CPU-GPU transfer capabilities, advanced eviction policies, and detailed allocation tracking.

## Core Objectives

1. **Full Memory Visibility**: Track every allocation with metadata
2. **Manual Memory Management**: Direct control over cache eviction and placement
3. **CPU-GPU Mobility**: Seamless transfer of cache blocks between host and device
4. **Advanced Eviction Policies**: LRU, LFU, and custom policies
5. **Memory Profiling**: Detailed metrics and analysis capabilities

## Architecture Design

### 1. Memory Allocation Tracker

#### Core Components

```python
class MemoryAllocation:
    """Tracks individual memory allocations"""
    allocation_id: str
    tensor_ptr: int  # CUDA memory pointer
    size_bytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    purpose: str  # "kv_cache", "workspace", "buffer", etc.
    created_at: float
    last_accessed: float
    access_count: int
    metadata: Dict[str, Any]

class MemoryTracker:
    """Global memory tracking system"""
    allocations: Dict[str, MemoryAllocation]
    total_allocated: int
    peak_allocated: int
    allocation_history: List[AllocationEvent]
    
    def register_allocation(self, tensor: torch.Tensor, purpose: str, **metadata) -> str:
        """Register new allocation with tracking"""
        
    def update_access(self, allocation_id: str):
        """Update access statistics"""
        
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get detailed allocation breakdown"""
```

#### Implementation Strategy

1. **Hook Integration**: Instrument flash_attn allocation points
2. **Tensor Wrapping**: Custom tensor class with tracking metadata
3. **CUDA Event Tracking**: Use CUDA events for precise timing
4. **Memory Mapping**: Maintain pointer-to-allocation mapping

### 2. Enhanced Cache Block Manager

#### Extended Block Metadata

```python
class CacheBlock:
    """Enhanced cache block with detailed tracking"""
    block_id: int
    allocation_id: str  # Link to MemoryTracker
    token_ids: torch.Tensor
    hash_value: int
    
    # Usage tracking
    creation_time: float
    last_access_time: float
    access_count: int
    access_pattern: List[float]  # Access timestamps
    
    # Memory location
    device_ptr: int
    is_on_cpu: bool
    cpu_tensor: Optional[torch.Tensor]
    
    # Relationships
    dependent_blocks: Set[int]
    reference_count: int
    sequence_ids: Set[int]
    
    # Priority metrics
    priority_score: float
    eviction_cost: float
    recompute_cost: float
```

#### Advanced Block Operations

```python
class EnhancedBlockManager:
    def __init__(self):
        self.blocks: Dict[int, CacheBlock] = {}
        self.memory_tracker = MemoryTracker()
        self.eviction_policy: EvictionPolicy = LRUPolicy()
        
    def allocate_block_tracked(self, purpose: str = "kv_cache") -> CacheBlock:
        """Allocate with full tracking"""
        
    def copy_to_cpu(self, block_id: int) -> torch.Tensor:
        """Copy block to system memory"""
        
    def copy_from_cpu(self, block_id: int, cpu_tensor: torch.Tensor):
        """Restore block from system memory"""
        
    def evict_manual(self, block_ids: List[int]):
        """Manually evict specific blocks"""
        
    def get_memory_map(self) -> Dict[str, Any]:
        """Get detailed memory layout"""
```

### 3. CPU-GPU Memory Transfer System

#### Transfer Manager Design

```python
class MemoryTransferManager:
    """Manages CPU-GPU memory transfers"""
    
    def __init__(self, pinned_memory_pool_size: int = 1024 * 1024 * 1024):  # 1GB
        self.pinned_pool = self._allocate_pinned_pool(pinned_memory_pool_size)
        self.transfer_queue = asyncio.Queue()
        self.active_transfers: Dict[str, TransferOperation] = {}
        
    async def async_to_cpu(self, tensor: torch.Tensor, 
                          priority: int = 0) -> TransferHandle:
        """Asynchronous GPU->CPU transfer"""
        
    async def async_to_gpu(self, cpu_tensor: torch.Tensor, 
                          device: torch.device,
                          priority: int = 0) -> TransferHandle:
        """Asynchronous CPU->GPU transfer"""
        
    def bulk_transfer(self, operations: List[TransferOp]) -> List[TransferHandle]:
        """Batch multiple transfers efficiently"""
```

#### Transfer Strategies

1. **Pinned Memory Pool**: Pre-allocated pinned memory for fast transfers
2. **Async Streams**: Multiple CUDA streams for concurrent transfers
3. **Priority Queue**: High-priority transfers for critical path
4. **Compression**: Optional compression for CPU storage

### 4. Advanced Eviction Policies

#### Policy Interface

```python
class EvictionPolicy(ABC):
    """Base class for eviction policies"""
    
    @abstractmethod
    def score_block(self, block: CacheBlock) -> float:
        """Calculate eviction score (higher = more likely to evict)"""
        
    @abstractmethod
    def update_on_access(self, block: CacheBlock):
        """Update policy state on block access"""
        
    @abstractmethod
    def select_victims(self, target_bytes: int, 
                      blocks: List[CacheBlock]) -> List[int]:
        """Select blocks to evict"""
```

#### LRU Implementation

```python
class LRUPolicy(EvictionPolicy):
    def __init__(self):
        self.access_order = OrderedDict()
        
    def score_block(self, block: CacheBlock) -> float:
        time_since_access = time.time() - block.last_access_time
        return time_since_access
        
    def update_on_access(self, block: CacheBlock):
        # Move to end (most recent)
        if block.block_id in self.access_order:
            self.access_order.move_to_end(block.block_id)
        else:
            self.access_order[block.block_id] = time.time()
```

#### Advanced Policies

```python
class AdaptiveLRUPolicy(EvictionPolicy):
    """LRU with frequency weighting"""
    def score_block(self, block: CacheBlock) -> float:
        recency_score = time.time() - block.last_access_time
        frequency_penalty = 1.0 / (1.0 + block.access_count)
        size_factor = block.size_bytes / (1024 * 1024)  # MB
        return recency_score * frequency_penalty * size_factor

class CostAwarePolicy(EvictionPolicy):
    """Consider recomputation cost"""
    def score_block(self, block: CacheBlock) -> float:
        eviction_cost = self.estimate_recompute_cost(block)
        memory_value = block.size_bytes / self.total_memory
        access_pattern_score = self.analyze_access_pattern(block)
        return eviction_cost - memory_value - access_pattern_score

class MLPolicy(EvictionPolicy):
    """Machine learning based prediction"""
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.feature_extractor = FeatureExtractor()
        
    def score_block(self, block: CacheBlock) -> float:
        features = self.feature_extractor.extract(block)
        return self.model.predict_eviction_score(features)
```

### 5. Memory Profiling and Analytics

#### Profiling Infrastructure

```python
class MemoryProfiler:
    """Comprehensive memory profiling"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.allocation_trace: List[AllocationEvent] = []
        self.access_heatmap: Dict[int, List[float]] = {}
        
    def take_snapshot(self) -> MemorySnapshot:
        """Capture current memory state"""
        return MemorySnapshot(
            timestamp=time.time(),
            total_allocated=self.get_total_allocated(),
            block_distribution=self.get_block_distribution(),
            fragmentation_ratio=self.calculate_fragmentation(),
            cache_hit_rate=self.get_cache_hit_rate()
        )
        
    def generate_report(self) -> MemoryReport:
        """Generate detailed profiling report"""
        return MemoryReport(
            allocation_timeline=self.build_timeline(),
            memory_heatmap=self.build_heatmap(),
            eviction_analysis=self.analyze_evictions(),
            optimization_suggestions=self.suggest_optimizations()
        )
```

#### Visualization Tools

```python
class MemoryVisualizer:
    """Real-time memory visualization"""
    
    def plot_allocation_timeline(self):
        """Timeline of allocations/deallocations"""
        
    def plot_memory_heatmap(self):
        """Heatmap of memory access patterns"""
        
    def plot_fragmentation(self):
        """Fragmentation over time"""
        
    def export_trace(self, format: str = "chrome"):
        """Export for external profilers"""
```

### 6. API Extensions

#### Enhanced Flash Attention Interface

```python
class ManagedFlashAttention:
    """Flash attention with fine-grained memory control"""
    
    def __init__(self, 
                 memory_tracker: MemoryTracker,
                 block_manager: EnhancedBlockManager,
                 transfer_manager: MemoryTransferManager,
                 eviction_policy: EvictionPolicy):
        self.memory_tracker = memory_tracker
        self.block_manager = block_manager
        self.transfer_manager = transfer_manager
        self.eviction_policy = eviction_policy
        
    def forward_with_control(self,
                           q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor,
                           memory_budget: Optional[int] = None,
                           cpu_offload: bool = False,
                           profile: bool = False) -> Tuple[torch.Tensor, MemoryStats]:
        """Forward pass with memory control"""
        
    def inspect_cache_state(self) -> CacheInspection:
        """Detailed cache inspection"""
        
    def manual_cache_management(self,
                              evict_blocks: List[int] = None,
                              pin_blocks: List[int] = None,
                              offload_blocks: List[int] = None):
        """Direct cache manipulation"""
```

#### User-Facing Control Interface

```python
class MemoryController:
    """High-level memory control interface"""
    
    def set_memory_budget(self, bytes: int):
        """Set total memory budget"""
        
    def set_eviction_policy(self, policy: Union[str, EvictionPolicy]):
        """Change eviction policy"""
        
    def enable_cpu_offloading(self, 
                            offload_threshold: float = 0.9,
                            offload_ratio: float = 0.2):
        """Enable automatic CPU offloading"""
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        
    def optimize_memory_layout(self):
        """Defragment and optimize memory"""
```

### 7. Implementation Phases

#### Phase 1: Core Tracking (Weeks 1-2)
- Implement MemoryAllocation and MemoryTracker
- Instrument flash_attn allocation points
- Basic allocation reporting

#### Phase 2: Enhanced Block Manager (Weeks 3-4)
- Extend CacheBlock with tracking metadata
- Implement manual eviction interface
- Add block inspection capabilities

#### Phase 3: CPU-GPU Transfers (Weeks 5-6)
- Implement MemoryTransferManager
- Add async transfer support
- Create pinned memory pool

#### Phase 4: Eviction Policies (Weeks 7-8)
- Implement LRU and adaptive policies
- Add policy selection interface
- Benchmark policy effectiveness

#### Phase 5: Profiling Tools (Weeks 9-10)
- Build MemoryProfiler
- Create visualization tools
- Generate optimization reports

#### Phase 6: Integration & Testing (Weeks 11-12)
- Integrate all components
- Performance optimization
- Comprehensive testing

## Usage Examples

### Example 1: Manual Cache Control

```python
# Initialize with custom configuration
memory_ctrl = MemoryController()
memory_ctrl.set_memory_budget(8 * 1024**3)  # 8GB
memory_ctrl.set_eviction_policy("adaptive_lru")

# Run inference with monitoring
with memory_ctrl.profile() as prof:
    output = model.generate(prompt, max_length=1000)
    
# Inspect memory state
stats = memory_ctrl.get_memory_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Memory fragmentation: {stats['fragmentation']:.2%}")

# Manual cache management
memory_ctrl.manual_cache_management(
    evict_blocks=[1, 5, 7],  # Evict specific blocks
    pin_blocks=[10, 11, 12],  # Pin important blocks
    offload_blocks=[20, 21]   # Offload to CPU
)
```

### Example 2: Custom Eviction Policy

```python
class PrefixAwarePolicy(EvictionPolicy):
    """Keep common prefixes in cache"""
    
    def score_block(self, block: CacheBlock) -> float:
        if block.is_prefix_block:
            return -1.0  # Never evict
        return super().score_block(block)

memory_ctrl.set_eviction_policy(PrefixAwarePolicy())
```

### Example 3: Memory Profiling

```python
profiler = MemoryProfiler()

# Run workload
for batch in dataloader:
    profiler.start_iteration()
    output = model(batch)
    profiler.end_iteration()
    
# Generate report
report = profiler.generate_report()
report.save_html("memory_profile.html")
report.export_trace("trace.json")  # For Chrome Tracing
```

## Performance Considerations

### Overhead Management
- Tracking overhead: ~2-5% expected
- Use sampling for high-frequency operations
- Compile tracking code with Triton/CUDA

### Optimization Strategies
- Batch metadata updates
- Async profiling to avoid blocking
- Configurable tracking granularity

## Future Extensions

1. **Distributed Memory Management**: Coordinate across multiple GPUs
2. **Predictive Prefetching**: ML-based cache warming
3. **Memory Compression**: Automatic compression for cold blocks
4. **Persistent Cache**: Save/restore cache state across runs
5. **Integration with CUDA Graph**: Memory-aware graph optimization

## Conclusion

This plan provides comprehensive fine-grained control over flash_attn memory management, enabling:
- Complete visibility into memory allocations
- Manual control over cache eviction
- Flexible CPU-GPU memory movement
- Advanced eviction policies including LRU
- Detailed profiling and optimization tools

The modular design allows incremental implementation while maintaining compatibility with existing flash_attn usage.