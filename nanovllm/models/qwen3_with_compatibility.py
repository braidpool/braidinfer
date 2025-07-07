"""
Example of Qwen3 model with integrated compatibility checking.

This shows how to integrate the compatibility checker into model initialization.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM as BaseQwen3ForCausalLM
from nanovllm.utils.kernel_compatibility import check_model_compatibility

logger = logging.getLogger(__name__)


class Qwen3ForCausalLM(BaseQwen3ForCausalLM):
    """Qwen3 model with automatic kernel compatibility checking."""
    
    def __init__(
        self,
        config: Qwen3Config,
        use_custom_kernels: bool = False,
        force_custom_kernels: bool = False,
        compatibility_check: bool = True
    ):
        """
        Initialize Qwen3 model with compatibility checking.
        
        Args:
            config: Model configuration
            use_custom_kernels: Whether to use custom kernels (subject to compatibility)
            force_custom_kernels: Force custom kernels even if incompatible (for testing)
            compatibility_check: Whether to run compatibility check
        """
        # Store original request
        requested_custom_kernels = use_custom_kernels
        
        # Initialize with standard kernels first
        super().__init__(config, use_custom_kernels=False)
        
        # Run compatibility check if custom kernels requested
        if requested_custom_kernels and compatibility_check and not force_custom_kernels:
            can_use, result = check_model_compatibility(
                self,
                model_name=f"Qwen3-{getattr(config, 'model_type', 'unknown')}",
                use_custom_kernels=True
            )
            
            if not can_use:
                logger.error(
                    "Model is incompatible with fused kernels. "
                    "Use force_custom_kernels=True to override (may produce incorrect output)."
                )
                use_custom_kernels = False
            else:
                use_custom_kernels = True
                
            # Store compatibility result
            self._kernel_compatibility = result
        elif force_custom_kernels:
            logger.warning(
                "Forcing custom kernels without compatibility check. "
                "This may produce incorrect output!"
            )
            use_custom_kernels = True
            self._kernel_compatibility = None
        else:
            use_custom_kernels = requested_custom_kernels
            self._kernel_compatibility = None
        
        # Reinitialize with determined kernel setting if needed
        if use_custom_kernels != False:  # Only reinit if we need custom kernels
            self.__init_subclass__()  # Reset the class
            super().__init__(config, use_custom_kernels=use_custom_kernels)
        
        self.use_custom_kernels = use_custom_kernels
    
    def get_compatibility_report(self) -> Optional[str]:
        """Get the compatibility report if available."""
        if hasattr(self, '_kernel_compatibility') and self._kernel_compatibility:
            from nanovllm.utils.kernel_compatibility import generate_compatibility_report
            return generate_compatibility_report(
                self._kernel_compatibility,
                self.__class__.__name__
            )
        return None


def create_model_with_compatibility_check(
    config: Qwen3Config,
    use_custom_kernels: bool = False,
    **kwargs
) -> Qwen3ForCausalLM:
    """
    Factory function to create a Qwen3 model with compatibility checking.
    
    This is the recommended way to create models with custom kernels.
    """
    model = Qwen3ForCausalLM(
        config,
        use_custom_kernels=use_custom_kernels,
        compatibility_check=True,
        **kwargs
    )
    
    # Log the result
    if use_custom_kernels and not model.use_custom_kernels:
        logger.info(
            "Custom kernels were requested but disabled due to compatibility issues. "
            "Run 'python -m nanovllm.utils.check_compatibility_cli <model>' for details."
        )
    
    return model