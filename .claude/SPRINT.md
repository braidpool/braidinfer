# SPRINT: Breaking Through the Performance Wall - Mastering Fused Kernels

## Sprint Goal

To correct our fundamental misunderstanding of GPU kernel performance and prove that custom fusion is the correct path to achieving high-performance inference. This sprint is dedicated to fixing the slow `fused_rmsnorm_qkv` kernel and making it **faster** than the unfused PyTorch baseline.

---

## Root Cause Analysis & The Path Forward

Our previous sprints have hit a wall. We concluded that custom kernels are slower than PyTorch/cuBLAS. This conclusion is incorrect. We are not hitting a limitation of the technology, but a gap in our implementation skills. Our kernels are slow because they are implemented with naive, CPU-style logic.

**The ONLY task for this sprint is to fix this.** We will not pivot to other tasks. We will learn to write a performant kernel correctly.

**The Core Principles We Will Implement:**
1.  **Tiling:** We will process matrices in small blocks (tiles) that fit into the GPU's fastest memory.
2.  **Shared Memory:** We will use shared memory to stage these tiles, minimizing slow round-trips to global VRAM.
3.  **Tensor Cores (`tl.dot`):** We will use the `tl.dot` instruction to explicitly command the GPU to use its dedicated, ultra-fast matrix multiplication hardware.

---

## Sprint Tasks

### 1. Mandatory Reading & Education (1 Day)

-   **Objective:** Understand the theory before writing code.
-   **Action Items:**
    1.  Read the Triton documentation on matrix multiplication from start to finish: [https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    2.  Read the Triton documentation on fused attention, paying close attention to how it handles tiling and `tl.dot`: [https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### 2. Re-implement the `fused_rmsnorm_qkv` Kernel CORRECTLY (3 Days)

-   **Objective:** Rewrite the existing slow kernel using the patterns from the Triton documentation.
-   **Location:** `nanovllm/kernels/fused_rmsnorm_qkv.py`
-   **Action Items:**
    1.  **Delete the existing, slow kernel code.** Start fresh to avoid patching a broken design.
    2.  Implement a new `fused_rmsnorm_qkv_kernel`.
    3.  **RMSNorm Pass:** The first part of the kernel should compute the RMS Norm. This can be done efficiently using warp-level primitives to reduce the sum of squares.
    4.  **Tiled GEMM for Q, K, V:**
        -   The main body of the kernel will be a loop over the input sequence dimension.
        -   Inside the loop, load **tiles** of the normalized input and the Q, K, and V weight matrices into **shared memory**.
        -   Use `tl.dot` to multiply the normalized input tile with the weight tiles to produce tiles of the Q, K, and V output.
        -   Store the resulting output tiles back to global memory.
    5.  Ensure the kernel is numerically identical to the PyTorch version.

### 3. Benchmark and Prove the Speedup (1 Day)

-   **Objective:** Validate that the correct implementation is faster and document the success.
-   **Action Items:**
    1.  Run the `benchmark_fusion.py` test.
    2.  The new, correct kernel **must be faster** than the PyTorch baseline. There is no ambiguity here. If it is not, the implementation is still incorrect.
    3.  Create a new document `.claude/SPRINT_FUSION_SUCCESS.md`.
    4.  In this document, post the new performance numbers and write a paragraph explaining how tiling, shared memory, and `tl.dot` were the keys to unlocking the performance that was previously thought to be impossible.

---

## Success Criteria

-   **Non-Negotiable:** The new `fused_rmsnorm_qkv_kernel` is measurably and significantly **FASTER** than the baseline PyTorch implementation for a batch size of 1. A slowdown is not an option and indicates a failed sprint.
-   **Target:** Achieve a >1.5x speedup on the RMSNorm+QKV operation, resulting in a noticeable improvement in end-to-end tok/s.
-   **Stretch:** The intern can confidently explain the role of each optimization (tiling, shared memory, `tl.dot`) and is ready to apply this pattern to other fusions like the MLP block.
