# USER_SUGGESTIONS.md - User Feedback and Suggestions

## Recent User Feedback

### Model Architecture Handling
- "Do not 'fix' the qwen3.py file. Instead create a new model file in nanovllm/models for each model or distinct model architecture."
- Created gpt2.py for GPT-2 architecture support

### HuggingFace Cache Support
- "Modify this script to use these models in the location that huggingface-cli places them" (~/.cache/huggingface/hub/)
- Implemented support for loading from HF cache

### Qwen3 Model Behavior
- "The qwen model doesn't work with greedy decoding. You should use temperature=0.6."
- Qwen3 uses `<think>` blocks for reasoning, accepts `/no_think` to disable

### Testing Practices (Critical)
- "Write new rules for yourself in CLAUDE.md so that you will not litter my repo with tests"
- "Every time you perform a test, you turn it into a python unittest"
- Tests must go in tests/ directory as proper unit tests
- No temporary test scripts in the repository

### Performance Expectations
- Expected performance: ~200+ tok/s
- Current performance: ~23 tok/s (regression from previous)
- "Performance is even worse than the last commit"

### WrapperManager Investigation
- User confirmed single wrapper approach is correct after reviewing FlashInfer docs
- "We removed the wrapper manager because you said the FlashInfer docs, tests, and examples used only one wrapper across all layers"

### Code Quality
- Keep the repository clean
- No debug scripts lying around
- Proper git practices (no wildcards in git add)
- Delete irrelevant files (completed)

## Implementation Notes
- User is on Arch Linux
- Single GPU setup (local inference focus)
- FlashInfer is the authoritative implementation for cascade attention
- Must conform exactly to FlashInfer's API and expectations

## Priority Order
1. Fix performance regression (23 tok/s → 200+ tok/s)
2. Sprint review and cleanup
3. Maintain proper testing practices
4. Keep repository clean