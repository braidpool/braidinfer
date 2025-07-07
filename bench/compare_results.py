#!/usr/bin/env python3
"""Compare standard vs custom kernel benchmark results."""

import json

# Load results
with open("results_standard.json") as f:
    standard = json.load(f)

with open("results_custom_kernels.json") as f:
    custom = json.load(f)

print("=== Performance Comparison: Standard vs Custom Kernels ===\n")
print("Batch Size | Standard (tok/s) | Custom (tok/s) | Difference")
print("-" * 60)

for s, c in zip(standard, custom):
    batch = s["batch_size"]
    std_tps = s["throughput_tokens_per_sec"]
    cus_tps = c["throughput_tokens_per_sec"]
    diff = cus_tps - std_tps
    pct = (diff / std_tps) * 100
    
    print(f"{batch:^10} | {std_tps:^16.2f} | {cus_tps:^14.2f} | {diff:+7.2f} ({pct:+.1f}%)")

print("\nConclusion:")
avg_std = sum(s["throughput_tokens_per_sec"] for s in standard) / len(standard)
avg_cus = sum(c["throughput_tokens_per_sec"] for c in custom) / len(custom)
avg_diff = ((avg_cus - avg_std) / avg_std) * 100

print(f"Average throughput - Standard: {avg_std:.2f} tok/s, Custom: {avg_cus:.2f} tok/s")
print(f"Average difference: {avg_diff:+.1f}%")