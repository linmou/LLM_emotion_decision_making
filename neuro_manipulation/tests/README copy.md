# Emotion Game Experiment Test Suite

This test suite is designed to validate the data processing and alignment functionality of the Emotion Game Experiment implementation.

## Test Structure

### `test_emotion_game_experiment.py`

#### Key Test Cases:

1. `test_post_process_batch_alignment`
   - Tests if input-output alignments are maintained during batch processing
   - Validates correct mapping of:
     - Scenarios
     - Descriptions
     - Prompts
     - Decisions
     - Rationales
     - Categories (option_ids)

2. `test_post_process_batch_with_repeats`
   - Tests handling of repeated experiments
   - Validates:
     - Correct repeat count
     - Proper repeat numbering
     - Data consistency across repeats
     - Correct mapping of repeated outputs

## Running Tests

To run the tests:

```bash
python -m unittest tests/test_emotion_game_experiment.py
```

## Debugging Notes

The test suite specifically targets potential issues in:
1. Batch size handling
2. Repeat experiment handling
3. Data alignment between inputs and outputs
4. Index management for batched and repeated data

If tests fail, check:
1. Index calculations in _post_process_batch
2. Batch size modulo operations
3. ThreadPoolExecutor result ordering
4. Data structure alignment between batches and control outputs 