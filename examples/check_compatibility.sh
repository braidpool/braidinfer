#!/bin/bash
# Example script showing how to use the compatibility checker in automation

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

echo "Checking kernel compatibility for: $MODEL_PATH"
echo "=================================================="

# Run compatibility check
python -m nanovllm.utils.check_compatibility_cli "$MODEL_PATH"
EXIT_CODE=$?

# Handle exit codes
case $EXIT_CODE in
    0)
        echo -e "\n✓ Model is COMPATIBLE with fused kernels"
        echo "You can safely use Braidinfer."
        ;;
    1)
        echo -e "\n⚠ Model has WARNINGS for fused kernels"
        echo "Test thoroughly before using in production"
        ;;
    2)
        echo -e "\n✗ Model is INCOMPATIBLE with fused kernels"
        echo "Use standard kernels only"
        ;;
    *)
        echo -e "\n❌ Error checking compatibility (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
        ;;
esac

# Example of using the result in a Python script
cat << EOF

# Example Python usage based on compatibility:
from nanovllm import LLM

llm = LLM(
    "$MODEL_PATH",
    model_kwargs={
        
    }
)
EOF