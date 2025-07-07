#!/usr/bin/env python3
"""
Command-line tool for checking model compatibility with fused kernels.

Usage:
    python -m nanovllm.utils.check_compatibility_cli <model_path>
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm import LLM
from nanovllm.utils.kernel_compatibility import (
    FusedKernelCompatibilityChecker,
    generate_compatibility_report
)


def main():
    parser = argparse.ArgumentParser(
        description="Check if a model is compatible with fused kernels"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed layer-by-layer analysis"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the default weight ratio threshold"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export report to file"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        # Try as HuggingFace model ID
        try:
            config = AutoConfig.from_pretrained(args.model_path)
            model_name = args.model_path
        except Exception:
            print(f"Error: Model path '{args.model_path}' not found", file=sys.stderr)
            sys.exit(1)
    else:
        model_path = str(model_path)
        model_name = model_path.name if hasattr(model_path, 'name') else os.path.basename(model_path)
    
    print(f"Checking compatibility for: {model_name}")
    print("=" * 50)
    
    try:
        # Load model
        print("Loading model...")
        llm = LLM(
            model_path if isinstance(model_path, str) else args.model_path,
            enforce_eager=True,
            model_kwargs={"use_custom_kernels": False}
        )
        
        # Get the actual model
        model = None
        
        # LLM inherits from LLMEngine which has model_runner
        if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model'):
            model = llm.model_runner.model
            print(f"Found model: {type(model).__name__}")
        else:
            # Fallback: search in attributes
            for attr_name in ['model', 'model_runner']:
                if hasattr(llm, attr_name):
                    attr = getattr(llm, attr_name)
                    if hasattr(attr, 'model'):
                        model = attr.model
                        break
        
        if model is None:
            print("Error: Could not access model for analysis", file=sys.stderr)
            sys.exit(1)
        
        # Run compatibility check
        print("Analyzing model weights...")
        checker = FusedKernelCompatibilityChecker()
        
        # Override threshold if specified
        if args.threshold is not None:
            checker.MAX_SAFE_WEIGHT_RATIO = args.threshold
            checker.WARNING_WEIGHT_RATIO = args.threshold * 0.75
        
        result = checker.check_model(model, model_name)
        
        # Generate report
        report = generate_compatibility_report(result, model_name)
        
        # Print report
        print("\n" + report)
        
        # Verbose output
        if args.verbose:
            print("\n" + "="*50)
            print("Detailed Layer Analysis:")
            print("="*50)
            
            for metric in result.layer_metrics:
                print(f"\nLayer {metric.layer_idx}:")
                print(f"  K norm: max={metric.max_k_weight:.3f}, "
                      f"median={metric.median_k_weight:.3f}, "
                      f"ratio={metric.k_weight_ratio:.1f}")
                print(f"  Q norm: max={metric.max_q_weight:.3f}, "
                      f"median={metric.median_q_weight:.3f}, "
                      f"ratio={metric.q_weight_ratio:.1f}")
                
                if metric.has_extreme_weights:
                    print("  ⚠️  EXTREME WEIGHTS DETECTED")
        
        # Export if requested
        if args.export:
            with open(args.export, 'w') as f:
                f.write(report)
                if args.verbose:
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("Detailed Layer Analysis:\n")
                    f.write("="*50 + "\n")
                    
                    for metric in result.layer_metrics:
                        f.write(f"\nLayer {metric.layer_idx}:\n")
                        f.write(f"  K norm: max={metric.max_k_weight:.3f}, "
                              f"median={metric.median_k_weight:.3f}, "
                              f"ratio={metric.k_weight_ratio:.1f}\n")
                        f.write(f"  Q norm: max={metric.max_q_weight:.3f}, "
                              f"median={metric.median_q_weight:.3f}, "
                              f"ratio={metric.q_weight_ratio:.1f}\n")
                        
                        if metric.has_extreme_weights:
                            f.write("  ⚠️  EXTREME WEIGHTS DETECTED\n")
            
            print(f"\nReport exported to: {args.export}")
        
        # Exit code based on status
        if result.status == "INCOMPATIBLE":
            sys.exit(2)  # Incompatible
        elif result.status == "WARNING":
            sys.exit(1)  # Warning
        else:
            sys.exit(0)  # Compatible
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()