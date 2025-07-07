import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoModel
from pathlib import Path


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, model_name_or_path: str):
    """Load model weights from either a local path or HuggingFace model ID."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Check if it's a local directory
    if os.path.isdir(model_name_or_path):
        weight_files = glob(os.path.join(model_name_or_path, "*.safetensors"))
        if not weight_files:
            # Try .bin files as fallback
            weight_files = glob(os.path.join(model_name_or_path, "*.bin"))
            if weight_files:
                # Load from .bin files using torch.load
                for file in weight_files:
                    state_dict = torch.load(file, map_location="cpu")
                    for name, param in state_dict.items():
                        # Handle GPT-2 naming
                        mapped_name = name
                        if hasattr(model, 'transformer') and not name.startswith('transformer.'):
                            if name.startswith('wte.') or name.startswith('wpe.') or name.startswith('h.') or name.startswith('ln_f.'):
                                mapped_name = f'transformer.{name}'
                        
                        if mapped_name in model.state_dict():
                            model.state_dict()[mapped_name].copy_(param)
                        else:
                            print(f"Warning: Skipping weight {name}")
                return
    else:
        # It's a HuggingFace model ID - use transformers to get the cached path
        from transformers.utils import cached_file
        try:
            # Try to get the safetensors file from cache
            weight_file = cached_file(model_name_or_path, "model.safetensors", _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
            weight_files = [weight_file]
        except:
            # Fallback to pytorch_model.bin
            try:
                weight_file = cached_file(model_name_or_path, "pytorch_model.bin", _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
                state_dict = torch.load(weight_file, map_location="cpu")
                for name, param in state_dict.items():
                    # Handle GPT-2 naming
                    mapped_name = name
                    if hasattr(model, 'transformer') and not name.startswith('transformer.'):
                        if name.startswith('wte.') or name.startswith('wpe.') or name.startswith('h.') or name.startswith('ln_f.'):
                            mapped_name = f'transformer.{name}'
                    
                    if mapped_name in model.state_dict():
                        model.state_dict()[mapped_name].copy_(param)
                    else:
                        print(f"Warning: Skipping weight {name}")
                return
            except:
                raise ValueError(f"Could not find model weights for {model_name_or_path}")
    
    # Load from safetensors files
    for file in weight_files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # Handle model-specific weight name mappings
                    mapped_name = weight_name
                    
                    # GPT-2 specific mappings
                    if hasattr(model, 'transformer') and not weight_name.startswith('transformer.'):
                        # HuggingFace GPT-2 doesn't prefix with 'transformer.'
                        if weight_name.startswith('wte.') or weight_name.startswith('wpe.') or weight_name.startswith('h.') or weight_name.startswith('ln_f.'):
                            mapped_name = f'transformer.{weight_name}'
                    
                    try:
                        param = model.get_parameter(mapped_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except AttributeError:
                        # Skip weights that don't exist in our model
                        print(f"Warning: Skipping weight {weight_name} -> {mapped_name}")
