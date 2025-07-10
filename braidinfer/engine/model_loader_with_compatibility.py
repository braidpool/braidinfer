"""
Model loading with kernel compatibility checking.

This is an example of how to integrate compatibility checking into the model loader.
"""

import torch
import logging
from typing import Optional

from braidinfer.config import Config
from braidinfer.models.qwen3 import Qwen3ForCausalLM
from braidinfer.models.gpt2 import GPT2ForCausalLM
from braidinfer.layers.sampler import Sampler
from braidinfer.utils.kernel_compatibility import check_model_compatibility
from braidinfer.engine.model_loader import ModelLoader as BaseModelLoader

logger = logging.getLogger(__name__)


class ModelLoaderWithCompatibility(BaseModelLoader):
    """Model loader with integrated kernel compatibility checking."""
    
    @staticmethod
    def load_model(config: Config) -> torch.nn.Module:
        """Load model with compatibility checking for custom kernels.
        
        Args:
            config: Configuration object
            
        Returns:
            Loaded model instance with appropriate kernel configuration
        """
        hf_config = config.hf_config
        model_type = hf_config.model_type
        
        # Get requested kernel setting
        requested_custom_kernels = getattr(config, 'use_custom_kernels', False)
        force_custom_kernels = getattr(config, 'force_custom_kernels', False)
        
        # First, load model with standard kernels for analysis
        if model_type == "qwen3":
            model = Qwen3ForCausalLM(hf_config, use_custom_kernels=False)
        elif model_type == "gpt2":
            model = GPT2ForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # If custom kernels requested, check compatibility
        use_custom_kernels = False
        if requested_custom_kernels:
            if force_custom_kernels:
                logger.warning(
                    "Forcing custom kernels without compatibility check. "
                    "This may produce incorrect output!"
                )
                use_custom_kernels = True
            else:
                # Run compatibility check
                can_use, result = check_model_compatibility(
                    model,
                    model_name=f"{model_type}-{config.model}",
                    use_custom_kernels=True
                )
                
                if can_use:
                    use_custom_kernels = True
                    logger.info("Model is compatible with custom kernels")
                else:
                    logger.warning(
                        f"Model incompatible with custom kernels: {result.reason}. "
                        "Falling back to standard kernels."
                    )
                    # Show recommendations
                    if result and result.recommendations:
                        for rec in result.recommendations[:2]:
                            logger.info(f"Recommendation: {rec}")
        
        # Reload model with determined kernel setting if needed
        if use_custom_kernels and model_type == "qwen3":
            # Need to recreate model with custom kernels
            del model
            torch.cuda.empty_cache()
            model = Qwen3ForCausalLM(hf_config, use_custom_kernels=True)
            logger.info("Reloaded model with custom kernels enabled")
        
        # Store compatibility info on the model
        model._use_custom_kernels = use_custom_kernels
        model._requested_custom_kernels = requested_custom_kernels
        
        # Continue with rest of initialization from base class
        # (cascade attention setup, weight loading, etc.)
        
        return model


# Example of how to patch the existing ModelLoader
def patch_model_loader():
    """Patch the existing ModelLoader to add compatibility checking."""
    import braidinfer.engine.model_loader as model_loader_module
    
    # Store original
    original_load_model = model_loader_module.ModelLoader.load_model
    
    def load_model_with_compatibility(config: Config) -> torch.nn.Module:
        """Wrapper that adds compatibility checking."""
        # Get settings
        requested_custom_kernels = getattr(config, 'use_custom_kernels', False)
        force_custom_kernels = getattr(config, 'force_custom_kernels', False)
        
        # Load model
        model = original_load_model(config)
        
        # If custom kernels were requested but model doesn't support them
        if requested_custom_kernels and hasattr(model, '_use_custom_kernels'):
            actual_custom_kernels = model._use_custom_kernels
            if not actual_custom_kernels and not force_custom_kernels:
                # Run compatibility check to get detailed reason
                _, result = check_model_compatibility(model, use_custom_kernels=True)
                if result:
                    logger.warning(
                        f"Custom kernels disabled due to compatibility: {result.reason}"
                    )
        
        return model
    
    # Replace method
    model_loader_module.ModelLoader.load_model = staticmethod(load_model_with_compatibility)