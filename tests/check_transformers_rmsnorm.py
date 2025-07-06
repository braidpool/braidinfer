#!/usr/bin/env python3
"""Check the exact transformers RMSNorm implementation."""

import os
import inspect
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

# Get the source code
print("Qwen3RMSNorm source code:")
print("="*60)
print(inspect.getsource(Qwen3RMSNorm))

# Check if it inherits from another class
print("\n" + "="*60)
print("Base classes:", Qwen3RMSNorm.__bases__)

# Check the forward method specifically
print("\n" + "="*60)
print("Forward method:")
print(inspect.getsource(Qwen3RMSNorm.forward))