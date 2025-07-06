# Sprint Goal: Fix `chat.py` with Fused Kernels

## Situation

Excellent work on the fused RMSNorm + QKV kernel. The performance uplift demonstrated in the benchmarks is significant and proves the value of this approach.

However, as noted, the kernel is not yet working correctly when integrated into the full model. The `chat.py` script produces "gibberish" when the `--custom-kernels` flag is enabled. This is a classic integration challenge. The goal of this sprint is to bridge that gap.

**Objective:** Enable the fused kernel in `chat.py` and get correct, streaming output.

## Debugging and Action Plan

The problem is almost certainly not in the kernel's logic itself (which is numerically accurate in isolation), but in how it interacts with the rest of the model. We need to find the first point of numerical divergence.

### 1. Isolate the Problem with a Test Script

Debugging an interactive script is difficult. Let's create a repeatable test case.

- **Action:** Create a new script, e.g., `tests/debug_fused_chat.py`.
- **Content:** This script should:
    1. Load the model twice: once with `use_custom_kernels=True` and once with `False`.
    2. Use a fixed, simple input prompt (e.g., "Hello").
    3. Perform a single generation step (i.e., predict the next token).
    4. The goal is to compare the logits produced by both models.

### 2. Pinpoint the Divergence

We will compare the outputs of each transformer layer to find where the calculation goes wrong.

- **Action:** Use PyTorch's forward hooks to capture the output of each `Qwen2DecoderLayer`.

```python
# Example of using hooks to capture outputs
outputs_pytorch = {}
outputs_triton = {}

def get_hook(storage_dict, layer_name):
    def hook(model, input, output):
        storage_dict[layer_name] = output[0].detach().cpu()
    return hook

# Register hooks on each layer for both models
for i in range(num_layers):
    model_pytorch.model.layers[i].register_forward_hook(get_hook(outputs_pytorch, f'layer_{i}'))
    model_triton.model.layers[i].register_forward_hook(get_hook(outputs_triton, f'layer_{i}'))

# Run both models
# ...

# Compare outputs
for i in range(num_layers):
    print(f"Comparing layer {i}...")
    are_close = torch.allclose(outputs_pytorch[f'layer_{i}'], outputs_triton[f'layer_{i}'], atol=1e-3, rtol=1e-3)
    if not are_close:
        print(f"Divergence detected at layer {i}!")
        # Further analysis here
        break
```

### 3. Inspect the Faulty Layer

Once you identify the first layer where the outputs diverge, zoom in on the inputs and outputs of your fused kernel.

- **Action:** Within that layer, compare the following tensors between the PyTorch and Triton versions:
    1.  **The `hidden_states` input** to the RMSNorm/QKV block. Are they *identical*?
    2.  **The `q`, `k`, and `v` tensors** produced by your kernel. How much do they differ from the PyTorch version? Use `torch.allclose` and also print the tensors to inspect them manually if needed.

### 4. Common Pitfalls to Investigate

- **Tensor Memory Layout (`.is_contiguous()`):** Your Triton kernel might be stricter about memory layout than PyTorch's default operations. Ensure the `hidden_states` tensor and the weight matrices are in the format your kernel expects. You may need to use `.contiguous()` before passing a tensor to the kernel.
- **Weight Transposition:** Your report mentions `weight_tile.trans()`. This is a critical area. Are you 100% certain that the Q, K, and V weight matrices are loaded and handled correctly? It's easy to have a mix-up here (e.g., `(D, H)` vs `(H, D)`). Double-check the matrix dimensions and whether a transpose is needed before or within the kernel.
- **Data Types (`dtype`):** Verify that all `dtype` conversions are correct. Your kernel uses mixed precision, so ensure the inputs are cast correctly and the outputs are returned in the expected format.

### 5. Final Steps

1.  Once you find and fix the discrepancy, your debug script should show that the logits from both models are nearly identical.
2.  Verify the fix by running `python chat.py --custom-kernels`. The output should now be coherent.
3.  Remove the `(experimental - currently produces gibberish)` warning from the `argparse` help text in `chat.py`.
4.  Change the default value for `use_custom_kernels` in `FastChat` to `True` to enable the optimization by default.

This systematic approach will isolate the bug and get your high-performance kernel fully operational.
