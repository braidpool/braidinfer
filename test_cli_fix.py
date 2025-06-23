#!/usr/bin/env python3
"""Actually test if the CLI inference is working"""

import subprocess
import time

# Test cases
test_cases = [
    {
        "name": "Basic inference without args",
        "input": "Hello world\n/infer\nexit\n",
        "expected": ["Added chunk:", "Running inference", "Generated"]
    },
    {
        "name": "Inference with query",
        "input": "The sky is blue.\n/infer Why?\nexit\n",
        "expected": ["Added chunk:", "Running inference", "new tokens"]
    }
]

for test in test_cases:
    print(f"\n=== Testing: {test['name']} ===")
    
    # Run CLI with test input
    process = subprocess.Popen(
        ["python", "cli.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Set a shorter timeout for testing
    try:
        output, _ = process.communicate(input=test["input"], timeout=20)
        
        # Check for expected outputs
        success = True
        for expected in test["expected"]:
            if expected in output:
                print(f"✓ Found: {expected}")
            else:
                print(f"✗ Missing: {expected}")
                success = False
        
        # Check for errors
        if "Failed to run inference" in output:
            print("✗ ERROR: Inference failed")
            success = False
            # Print the error details
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if "Failed to run inference" in line:
                    print(f"  Error: {line}")
                    # Print next few lines for context
                    for j in range(i+1, min(i+5, len(lines))):
                        if lines[j].strip():
                            print(f"  {lines[j]}")
        
        if success:
            print("✓ Test PASSED")
        else:
            print("✗ Test FAILED")
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("✗ Test TIMEOUT - process took too long")
    except Exception as e:
        print(f"✗ Test ERROR: {e}")

print("\n=== Test Summary ===")
print("If tests are failing, check the output above for details.")