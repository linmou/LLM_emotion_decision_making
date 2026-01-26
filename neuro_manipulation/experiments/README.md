# Emotion Game Experiment

This module contains the implementation of emotion-based game theory experiments using representation engineering.

## Key Components

### EmotionGameExperiment

The main class that handles running experiments with different emotions and intensities in game theory scenarios.

#### Features:
- Configurable emotion and intensity settings
- Support for multiple game types
- Statistical analysis of results
- Customizable model and tokenizer setup
- Parallel processing for batch inference
- Comprehensive logging and result storage

### Usage

```python
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config
from games.game_configs import get_game_config
from constants import GameNames

# Setup configurations
exp_config = get_exp_config('path_to_config.yaml')
game_name = GameNames.from_string(exp_config['experiment']['game']['name'])
model_name = exp_config['experiment']['llm']['model_name']
repe_eng_config = get_repe_eng_config(model_name)
game_config = get_game_config(game_name)

# Create experiment instance
experiment = EmotionGameExperiment(
    repe_eng_config,
    exp_config,
    game_config,
    batch_size=4,
    repeat=1
)

# Run sanity check
df, stats = experiment.run_sanity_check()
```

### What the Sanity Check Does

1. Takes a small sample (10 examples) from your dataset
2. Tests with one emotion and one intensity value
3. Includes a neutral condition for comparison
4. Saves results in a separate directory
5. Returns both the results DataFrame and statistical analysis

### Output

The sanity check creates a separate directory with:
- JSON file with raw results
- CSV file with processed results
- Statistical analysis results
- Configuration file used for the check

### Benefits

- Quick validation of experiment setup
- Early detection of configuration issues
- Sample size small enough for manual review
- Preserves original experiment settings
- Helps identify potential data processing issues

## Best Practices

1. Always run a sanity check before launching a full experiment
2. Review the generated outputs manually
3. Verify the statistical analysis makes sense
4. Check the emotion and intensity effects are as expected
5. Validate the model's response format 