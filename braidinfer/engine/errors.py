"""
Error handling and custom exceptions for nano-vllm.
"""

from typing import Optional, Any
import traceback


class NanoVLLMError(Exception):
    """Base exception for nano-vllm errors."""
    pass


class ModelLoadError(NanoVLLMError):
    """Error during model loading."""
    pass


class InferenceError(NanoVLLMError):
    """Error during inference."""
    pass


class MemoryError(NanoVLLMError):
    """Error related to memory allocation."""
    pass


class DistributedError(NanoVLLMError):
    """Error in distributed communication."""
    pass


class ConfigurationError(NanoVLLMError):
    """Error in configuration."""
    pass


class ErrorContext:
    """Context manager for error handling with detailed information."""
    
    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.context = kwargs
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Enhance the exception with context
            if isinstance(exc_val, NanoVLLMError):
                # Already our exception, add context
                exc_val.operation = self.operation
                exc_val.context = self.context
            else:
                # Wrap in our exception
                error_msg = f"Error during {self.operation}: {str(exc_val)}"
                if self.context:
                    error_msg += f"\nContext: {self.context}"
                    
                # Map to appropriate exception type
                if "cuda" in str(exc_val).lower() or "memory" in str(exc_val).lower():
                    new_exc = MemoryError(error_msg)
                elif "distributed" in str(exc_val).lower() or "nccl" in str(exc_val).lower():
                    new_exc = DistributedError(error_msg)
                else:
                    new_exc = InferenceError(error_msg)
                    
                new_exc.__cause__ = exc_val
                raise new_exc from exc_val
        return False


def handle_inference_error(func):
    """Decorator for handling errors in inference methods."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the full traceback for debugging
            traceback.print_exc()
            
            # Re-raise our exceptions, wrap others
            if isinstance(e, NanoVLLMError):
                raise
            else:
                raise InferenceError(f"Inference failed in {func.__name__}: {str(e)}") from e
    return wrapper