# Anger Activation Word Probability Demo

## Overview

This demonstration script shows how anger emotion activation affects the probability of generating the word "angry" in different contexts using the representation engineering pipeline.

## Purpose

The demo validates that:
1. Emotion manipulation (anger activation) has measurable effects on model output probabilities
2. The sequence probability measurement system works correctly with vLLM models
3. Higher anger activation intensities correlate with increased probability of anger-related words

## Key Components

### AngerActivationDemo Class

Main class that orchestrates the entire demonstration:

- **Model Setup**: Loads Qwen2.5-0.5B-Instruct model with emotion readers
- **Activation Generation**: Creates anger emotion activations at different intensities  
- **Probability Measurement**: Uses SequenceProbVLLMHook to measure word probabilities
- **Visualization**: Creates plots showing the relationship between activation intensity and probability

### Core Methods

#### `__init__(model_path)`
- Initializes the demo with specified model
- Loads emotion readers for all emotions
- Sets up rep-control pipeline for vLLM inference
- Prepares target word tokenization

#### `create_test_prompts()`
- Returns list of prompts designed to elicit emotional responses
- Examples: "The person felt very", "His reaction was quite", etc.

#### `get_anger_activations(intensity)`
- Generates anger emotion activations for given intensity
- Uses pre-trained emotion direction vectors
- Returns activation tensors for all model layers

#### `measure_word_probability(prompts, activations)`
- Measures probability of target word ("angry") given prompts and activations
- Uses SequenceProbVLLMHook for accurate probability calculation
- Returns list of probabilities for each prompt

#### `run_probability_analysis()`
- Runs complete analysis across multiple anger intensities (0.0, 0.5, 1.0, 1.5, 2.0)
- Compares baseline (no activation) vs. anger activation conditions
- Returns comprehensive results dictionary

#### `analyze_single_sequence_detailed(prompt)`
- Provides detailed analysis of single prompt with step-by-step logging
- Tests multiple target words ("angry", "happy", "sad", "calm")
- Shows generated text and probability breakdowns

#### `create_visualization(results)`
- Creates matplotlib plot showing intensity vs. probability relationship
- Saves visualization to `logs/anger_demo_results/anger_probability_impact.png`
- Annotates plot with actual probability values

## Usage

### Basic Usage

```python
from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo

# Initialize demo with default model
demo = AngerActivationDemo()

# Run detailed analysis on single sequence
detailed_results = demo.analyze_single_sequence_detailed("The person felt very")

# Run full probability analysis
results = demo.run_probability_analysis()

# Create visualization
plot_path = demo.create_visualization(results)
```

### Custom Model Usage

```python
# Use custom model path
demo = AngerActivationDemo("/path/to/your/model")
```

### Command Line Usage

```bash
cd neuro_manipulation/repe
python sequence_prob_demo.py
```

## Output

The demo produces:

1. **Console Logs**: Detailed progress and probability measurements
2. **Visualization**: Plot showing anger intensity vs. word probability
3. **Summary Statistics**: Baseline vs. maximum intensity comparison

### Expected Results

A successful demo should show:
- Increasing probability of "angry" word as anger activation intensity increases
- Clear correlation between emotion manipulation and model behavior
- Measurable differences between baseline and activated conditions

### Sample Output

```
✅ Model: /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct
✅ Target word: 'angry'
✅ Target tokens: [26921]
✅ Test prompts: 10 prompts
✅ Visualization saved: logs/anger_demo_results/anger_probability_impact.png

📊 KEY FINDINGS:
   Baseline probability: 0.001234
   Max intensity probability: 0.003456
   Change ratio: 2.80x
   🔥 Strong positive correlation: Anger activation increases 'angry' probability!
```

## Dependencies

- PyTorch
- vLLM
- transformers  
- matplotlib
- numpy
- pandas

## Technical Details

### Model Requirements

- Supports any model compatible with the representation engineering pipeline
- Requires pre-trained emotion direction vectors
- Works with both HuggingFace and vLLM model loading

### Memory Usage

- First loads model in HuggingFace format for emotion reader extraction
- Switches to vLLM for efficient inference
- Manages memory by deleting HuggingFace model after emotion reader loading

### Threading

- Uses ThreadPoolExecutor for parallel post-processing
- Implements pipeline queue for memory-efficient batch processing
- Thread-safe logging with proper naming

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or use smaller model
2. **Model Loading Error**: Check model path and permissions
3. **Hook Registration Error**: Ensure vLLM model is properly initialized

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration

This demo integrates with:

- `SequenceProbVLLMHook`: For probability measurement
- `emotion_game_experiment.py`: For emotion activation patterns
- `rep_control_pipeline`: For controlled text generation
- `load_emotion_readers`: For emotion direction vectors

## Performance

- Typical runtime: 2-5 minutes for full analysis
- Memory usage: ~8GB GPU memory for Qwen2.5-0.5B
- CPU usage: Moderate during post-processing phases

## Validation

The demo serves as validation that:

1. Emotion manipulation affects model probabilities in expected directions
2. Sequence probability measurement is accurate
3. The entire rep-engineering pipeline functions correctly
4. vLLM integration works seamlessly with emotion control 