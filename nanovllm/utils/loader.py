import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                tensor = f.get_tensor(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                        except AttributeError:
                            print(f"Warning: Model has no parameter named '{weight_name}'. Skipping.")
                            break
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError as e:
                        print(f"Warning: Could not load parameter '{weight_name}' ({e}). Skipping.")
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
