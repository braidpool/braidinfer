# Braidinfer Architecture Complete

## Table of Contents

1. [System Overview and Philosophy](#1-system-overview-and-philosophy)
2. [Core Components and Interactions](#2-core-components-and-interactions)
3. [Cascade Attention Implementation](#3-cascade-attention-implementation)
4. [Kernel Integration Architecture](#4-kernel-integration-architecture)
5. [Memory Management Design](#5-memory-management-design)
6. [Performance Characteristics](#6-performance-characteristics)

## 1. System Overview and Philosophy

### 1.1 Project Vision

Braidinfer is a high-performance, single-GPU LLM inference engine designed for local deployment. The core design philosophy focuses on maximizing throughput and minimizing latency through chunk-based context management, advanced memory optimization, and intelligent kernel fusion.

### 1.2 Core Philosophy: Chunk-Based Context Management

The central architectural concept in Braidinfer is the **Chunk**. Instead of representing prompts as monolithic strings, context is broken down into logical, reusable pieces:

- **Chunk Abstraction**: A `Chunk` is a semantic unit of text (system prompt, document, user query) with its own pre-computed and cached Key-Value (KV) state
- **Dynamic Composition**: Inference requests are composed by referencing chunk sets, assembled on-the-fly at the attention level
- **Content-Addressable Storage**: SHA256 hashing provides automatic deduplication of identical content
- **Persistent Caching**: Expensive computations for shared context are performed once and reused across requests

### 1.3 Design Principles

1. **Single-GPU Optimization**: Focused architecture without distributed complexity
2. **Memory Efficiency**: Paged memory management and intelligent caching
3. **Numerical Stability**: Careful precision management for model compatibility
4. **Performance First**: Kernel fusion and optimized memory access patterns
5. **Extensibility**: Modular design supporting multiple model architectures

## 2. Core Components and Interactions

### 2.1 System Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChunkedLLM    │───▶│   LLMEngine     │───▶│  ModelRunner    │
│   (API Layer)   │    │  (Orchestrator) │    │  (Execution)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │ ChunkRegistry   │    │ FlashInferSched │
         │              │ (Content Mgmt)  │    │ (Attention Ops) │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PageManager    │    │  LRU Cache      │    │ Fused Kernels   │
│ (Memory Mgmt)   │    │ (Eviction)      │    │ (Performance)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Component Responsibilities

#### ChunkedLLM (API Layer)
- **Purpose**: High-level interface for chunk-based inference
- **Key Methods**: `register_chunk()`, `generate_from_chunks()`, `chat()`
- **Responsibilities**: 
  - User request handling
  - Chunk lifecycle management
  - Position calculation coordination

#### LLMEngine (Orchestrator)
- **Purpose**: Central coordination of inference operations
- **Responsibilities**:
  - Request scheduling and batching
  - Model state management
  - Performance monitoring
  - Resource allocation

#### ModelRunner (Execution Engine)
- **Purpose**: Direct model execution with FlashInfer integration
- **Key Features**:
  - Fused kernel dispatch
  - Memory layout optimization
  - Numerical precision management
  - Multi-model support

#### PageManager (Memory Management)
- **Purpose**: Paged KV cache allocation and management
- **Layout**: HND format [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
- **Features**:
  - Pre-allocated memory pools
  - Page-level allocation/deallocation
  - Memory fragmentation prevention

#### ChunkRegistry (Content Management)
- **Purpose**: Chunk storage and retrieval with automatic deduplication
- **Key Features**:
  - SHA256 content hashing
  - LRU eviction policy
  - Thread-safe access
  - Metadata management

#### FlashInferScheduler (Attention Operations)
- **Purpose**: Cascade attention coordination
- **Features**:
  - Plan-run operation pattern
  - Multi-chunk attention batching
  - Position offset management
  - Online softmax coordination

## 3. Cascade Attention Implementation

### 3.1 Theoretical Foundation

Cascade attention enables efficient combination of pre-computed KV chunks without physical concatenation. The system processes chunks sequentially while maintaining mathematical equivalence to single-sequence attention.

### 3.2 Online Softmax Algorithm

The core innovation is the online softmax algorithm that maintains global normalization across chunk boundaries:

```
Standard: softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in all_tokens)
Online:   Iteratively update running statistics (m_i, l_i, acc_i)
```

#### Algorithm Steps
1. **Score Computation**: `s_j = dot(query, k_j)`
2. **Maximum Update**: `m_new = max(m_i, s_j)`
3. **Accumulator Renormalization**: `acc_new = acc_i * exp(m_i - m_new)`
4. **Sum Renormalization**: `l_new = l_i * exp(m_i - m_new)`
5. **Value Addition**: `acc_new += exp(s_j - m_new) * v_j`
6. **Sum Update**: `l_new += exp(s_j - m_new)`

#### High-Performance Triton Implementation

The online softmax algorithm is implemented using a high-performance Triton kernel (`braidinfer/kernels/online_softmax.py`) that replaces the inefficient Python token-by-token loop:

**Key Features:**
- **Correct Parallelization**: Parallelizes over queries (heads × batch), processes tokens sequentially
- **Causal Masking**: Integrated causal masking based on global positions
- **Race-Condition Free**: Avoids data races by processing each query's tokens sequentially
- **Register Optimization**: Keeps running state in fast GPU registers during token loop
- **Memory Efficiency**: Single read/write of state per query, optimal memory access patterns

**Performance Benefits:**
- Eliminates Python for-loop overhead
- GPU-optimized memory access patterns  
- Proper utilization of GPU parallelism
- Maintains numerical correctness with causal masking

This ensures mathematically identical results to concatenated sequence processing.

### 3.3 Differential RoPE for Positional Correctness

Rotary Position Embeddings require careful handling when combining chunks cached at different positions.

#### Mathematical Foundation
- **Property**: `R_{a+b} = R_a * R_b` (rotation matrix composition)
- **Goal**: Transform `k_cached = R_{m_local} * k_raw` to `k_global = R_{m_global} * k_raw`
- **Solution**: `k_global = R_{m_global - m_local} * k_cached`

#### Implementation
1. **Delta Calculation**: `delta = chunk.global_position_start + token_local_position - chunk.cached_position_start`
2. **Trigonometric Lookup**: Retrieve `cos(delta * theta)` and `sin(delta * theta)`
3. **Complex Rotation**:
   ```
   k_global_real = k_real * cos(delta*theta) - k_imag * sin(delta*theta)
   k_global_imag = k_real * sin(delta*theta) + k_imag * cos(delta*theta)
   ```

### 3.4 Equivalence Guarantee

The combination of online softmax and differential RoPE guarantees that:
```
attention(Q, combine([chunk_A, chunk_B])) ≡ attention(Q, concat([chunk_A, chunk_B]))
```

This mathematical equivalence enables transparent chunk reuse without accuracy loss.

### 3.5 Paged Chunk Attention Kernel

The `paged_chunk_attention_kernel` implements the complete cascade attention algorithm:

```cuda
// Pseudo-code structure
for each chunk in active_chunks:
    for each page in chunk.pages:
        for each token in page:
            k_cached = load_key(page, token)
            k_global = apply_differential_rope(k_cached, position_delta)
            k_normalized = apply_k_normalization(k_global)
            score = dot_product(query, k_normalized)
            update_online_softmax(score, value, &m_i, &l_i, &acc_i)
```

## 4. Kernel Integration Architecture

### 4.1 Fused Kernel Strategy

Braidinfer employs aggressive kernel fusion to reduce memory bandwidth and launch overhead:

#### Primary Fused Operations
1. **RMSNorm + QKV Projection**: `fused_rmsnorm_qkv_mixed_precision`
2. **Attention Output + Residual**: Planned optimization
3. **MLP Gate + Up + Down**: Future development

### 4.2 Numerical Stability Architecture

The most critical aspect of kernel integration is maintaining numerical stability across different model architectures.

#### Mixed Precision Strategy
```python
# Critical precision points for fused RMSNorm+QKV:
1. Variance accumulation: float32
2. RMS normalization: float32 → bfloat16
3. Weight multiplication: bfloat16
4. Matrix multiplication: bfloat16 inputs, float32 accumulation
```

#### Conversion Point Criticality
The exact point of float32→bfloat16 conversion determines numerical behavior:

```python
# Incorrect (kernel approach):
normalized_f32 = (input_f32 / rms) * norm_weight_f32
output = matmul(normalized_f32, weight_f32).to(bfloat16)

# Correct (PyTorch-matching approach):
normalized_f32 = (input_f32 / rms) * norm_weight_f32
normalized_bf16 = normalized_f32.to(bfloat16)
output = matmul(normalized_bf16, weight_bf16)
```

### 4.3 Model Compatibility System

#### Compatibility Detection Framework
```python
class FusedKernelCompatibilityChecker:
    def analyze_weight_distribution(self, model):
        # Detect extreme normalization weights
        # Calculate amplification potential
        # Assess numerical stability risk
        
    def compatibility_score(self, metrics):
        # Weight ratio analysis
        # Layer count evaluation  
        # Error amplification estimation
```

#### Compatibility Thresholds
- **Compatible**: Score > 0.8, max weight < 10x
- **Warning**: Score 0.5-0.8, requires testing
- **Incompatible**: Score < 0.5, extreme weights detected

#### Automatic Fallback Mechanism
```python
if not compatibility_checker.is_compatible(model):
    logger.warning("Falling back to standard kernels")
    config.use_custom_kernels = False
```

### 4.4 Model-Specific Adaptations

#### Qwen3 Architecture Handling
- **Challenge**: K normalization weights up to 96.5x
- **Solution**: Exact PyTorch precision matching
- **Status**: Compatible with standard kernels only

#### TinyLlama Architecture
- **Challenge**: Standard LLaMA architecture
- **Solution**: Full fused kernel support
- **Status**: Fully compatible

#### ERNIE Architecture  
- **Challenge**: Implementation gaps
- **Solution**: Under development
- **Status**: Partial support

### 4.5 Kernel Dispatch Architecture

```python
class KernelDispatcher:
    def select_kernel(self, operation, model_type, compatibility_score):
        if compatibility_score > 0.8 and operation in FUSED_OPERATIONS:
            return self.fused_kernels[operation]
        else:
            return self.standard_kernels[operation]
```

## 5. Memory Management Design

### 5.1 Paged Memory Architecture

#### Design Principles
- **Pre-allocation**: Single large memory pool divided into fixed-size pages
- **HND Layout**: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
- **Zero-copy Operations**: Direct page manipulation without data movement
- **Fragmentation Prevention**: Fixed page sizes minimize memory fragmentation

#### Page Allocation Strategy
```python
class PageManager:
    def __init__(self, total_memory, page_size):
        self.kv_cache_pool = torch.empty(
            (num_pages, 2, num_kv_heads, page_size, head_dim),
            dtype=torch.float16, device="cuda"
        )
        self.free_pages = list(range(num_pages))
        self.allocated_pages = {}
```

### 5.2 Chunk Memory Lifecycle

#### Allocation Phase
1. **Content Hash**: Generate SHA256 hash of chunk content
2. **Deduplication Check**: Query existing chunks with same hash
3. **Page Request**: Allocate required pages from free pool
4. **KV Computation**: Populate pages with chunk's KV cache
5. **Registration**: Add to ChunkRegistry with metadata

#### Access Phase
1. **Lookup**: Retrieve chunk by ID from registry
2. **LRU Update**: Mark chunk as recently accessed
3. **Page Access**: Direct memory access to allocated pages
4. **Position Mapping**: Calculate global positions for tokens

#### Eviction Phase
1. **LRU Selection**: Identify least recently used chunks
2. **Reference Check**: Ensure no active inference using chunk
3. **Page Release**: Return pages to free pool
4. **Registry Cleanup**: Remove chunk metadata

### 5.3 Memory Layout Optimization

#### HND vs NHD Trade-offs
- **HND Benefits**: Better cache locality for cascade attention
- **NHD Benefits**: Standard transformer memory layout
- **Decision**: HND chosen for cascade attention performance

#### Memory Access Patterns
```cuda
// Optimized access pattern for cascade attention
for (int page_idx = 0; page_idx < chunk.num_pages; page_idx++) {
    for (int token_idx = 0; token_idx < page_size; token_idx++) {
        // Coalesced memory access
        float4 k_vector = kv_cache[layer][page_idx][0][head][token_idx];
        float4 v_vector = kv_cache[layer][page_idx][1][head][token_idx];
    }
}
```

### 5.4 Content-Addressable Storage

#### SHA256 Hashing Strategy
```python
def compute_chunk_hash(content: str) -> str:
    # Normalize content for consistent hashing
    normalized = content.strip().replace('\r\n', '\n')
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
```

#### Deduplication Benefits
- **Memory Savings**: 53.3% reduction for shared system prompts
- **Computation Savings**: One-time prefill for repeated content
- **Consistency**: Identical content guaranteed to produce identical results

### 5.5 Memory Pool Management

#### Dynamic Resizing Strategy
```python
class MemoryPool:
    def expand_if_needed(self, required_pages):
        if self.free_pages_count < required_pages:
            if self.can_evict_chunks():
                self.evict_lru_chunks(required_pages)
            else:
                raise OutOfMemoryError("Cannot allocate required pages")
```

#### Memory Pressure Handling
1. **Soft Pressure**: LRU eviction of unused chunks
2. **Hard Pressure**: Forced eviction with user notification
3. **Critical Pressure**: Request rejection with memory expansion suggestion

## 6. Performance Characteristics

### 6.1 Current Performance Metrics

#### Baseline Performance
- **Single Token**: ~29.4 tok/s (batch size 1)
- **Batch Processing**: 237 tok/s (batch size 8)
- **Chunk Attention Capability**: 2,938 tok/s theoretical maximum

#### Optimization Impact
- **Fused Kernels**: 2.64x speedup (when compatible)
- **Cascade Attention**: 53.3% memory savings
- **Chunk Reuse**: Near-zero marginal cost for repeated content

### 6.2 Performance Analysis by Component

#### Kernel Performance
| Operation | Standard | Fused | Speedup |
|-----------|----------|-------|---------|
| RMSNorm+QKV | 0.22s | 0.083s | 2.64x |
| Attention | Variable | Optimized | Model-dependent |
| MLP | Standard | Planned | TBD |

#### Memory Performance
- **Page Allocation**: O(1) allocation from pre-allocated pool
- **Chunk Lookup**: O(1) hash table access
- **Memory Bandwidth**: Optimized through fused operations

#### Cascade Attention Performance
```
Chunk Combination Cost: O(total_tokens) with O(1) per-token overhead
Memory Access: Sequential, cache-friendly patterns
Compute Efficiency: 95%+ utilization through online algorithms
```

### 6.3 Scaling Characteristics

#### Model Size Scaling
- **Memory**: Linear with parameter count
- **Compute**: Linear with sequence length
- **KV Cache**: Linear with context length

#### Batch Size Scaling  
- **Throughput**: Near-linear up to memory limits
- **Latency**: Minimal increase for reasonable batch sizes
- **Memory**: Linear increase in KV cache requirements

#### Context Length Scaling
- **Standard Attention**: O(n²) memory and compute
- **Cascade Attention**: O(n) memory with chunking, O(n²) compute
- **Memory Savings**: Proportional to chunk reuse ratio

### 6.4 Performance Comparison

#### vs. Standard PyTorch
- **Throughput**: 2.64x improvement with fused kernels
- **Memory**: 53% reduction with chunk reuse
- **Latency**: Comparable for single requests

#### vs. vLLM
- **Focus**: Single-GPU optimization vs. distributed scaling
- **Memory**: Similar paged attention concept
- **Features**: Chunk-based context management unique to Braidinfer

#### vs. llama.cpp
- **Performance**: Competitive for full-precision inference
- **Quantization**: llama.cpp advantage (4-bit/8-bit support)
- **Integration**: Better Python/PyTorch ecosystem integration

### 6.5 Performance Optimization Roadmap

#### Immediate Targets (Q1)
- **MLP Fusion**: Target 400+ tok/s
- **Output Fusion**: Reduce kernel launch overhead
- **Memory Layout**: Further optimize cache access patterns

#### Medium-term Goals (Q2-Q3)
- **Quantization**: INT8/INT4 support for memory-bound scenarios
- **Multi-Model**: Efficient model switching and batching
- **Advanced Fusion**: Full layer fusion (single kernel per layer)

#### Long-term Vision (Q4+)
- **TensorRT Integration**: Production deployment optimization
- **Adaptive Optimization**: Model-specific kernel selection
- **Hardware Acceleration**: Tensor Core utilization improvement

### 6.6 Performance Monitoring

#### Key Metrics
- **Tokens per Second**: Primary throughput metric
- **Memory Utilization**: Page allocation efficiency
- **Kernel Efficiency**: GPU utilization percentages
- **Cache Hit Rate**: Chunk reuse effectiveness

#### Monitoring Infrastructure
```python
class PerformanceMonitor:
    def track_inference(self, request_id, metrics):
        # Token generation rate
        # Memory allocation patterns
        # Kernel execution times
        # Cache hit/miss ratios
```

### 6.7 Performance Tuning Guidelines

#### Model Selection
- **Compatible Models**: Choose models with stable weight distributions
- **Size Optimization**: Balance capability vs. performance requirements
- **Architecture**: Prefer standard LLaMA-based architectures

#### Configuration Optimization
```python
optimal_config = {
    'batch_size': 8,  # Sweet spot for throughput
    'page_size': 16,  # Balance memory vs. compute
    'use_fused_kernels': True,  # When compatible
    'enable_cascade_attention': True,  # Always beneficial
}
```

#### Hardware Considerations
- **GPU Memory**: Plan for 2-3x model size for KV cache
- **Memory Bandwidth**: Critical for attention operations
- **Compute Capability**: Required for optimal kernel performance

## Conclusion

Braidinfer's architecture represents a sophisticated approach to single-GPU LLM inference optimization. The combination of chunk-based context management, cascade attention, and intelligent kernel fusion creates a system that balances performance, memory efficiency, and numerical stability.

The architecture's modular design enables continuous optimization while maintaining compatibility across diverse model architectures. Future development will focus on expanding quantization support, enhancing kernel fusion capabilities, and improving automated optimization strategies.

Key architectural strengths:
- **Chunk-based context management** for efficient memory reuse
- **Cascade attention** for scalable long-context processing  
- **Intelligent kernel fusion** with automatic compatibility detection
- **Sophisticated memory management** with paged allocation
- **Mathematical correctness** through careful numerical precision handling

This architecture provides a solid foundation for high-performance local LLM inference while maintaining the flexibility needed for future enhancements and model support.