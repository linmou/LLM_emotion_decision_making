# Public Release: Emotion Experiment Engine
<!-- Updated: 2026-01-25 | Commit: 4575bc12b777d1de784605ec64b1b248994ce482 -->

This repository is the public snapshot of `emotion_experiment_engine` and its
core dependencies (`neuro_manipulation`, `games`, and shared utilities). It is
scoped for experimentation and evaluation only.

## Scope
- Included: `emotion_experiment_engine`, `neuro_manipulation`, `games`, and shared utilities.
- Excluded: `data_creation`, datasets, caches, logs, and large artifacts.

## Layout
- `emotion_experiment_engine/`: benchmark orchestration, datasets, evaluation helpers
- `neuro_manipulation/`: prompt formats, RepE pipelines, vLLM hooks
- `games/`: game theory scenarios and payoff matrices
- `constants.py`, `statistical_engine.py`, `merge_data_samples.py`: shared utilities

## Setup
```bash
pip install -r requirements.txt
```

## API Keys
Fill in `api_configs.py` with your own credentials before running any
LLM-backed evaluation.

## Usage
See `emotion_experiment_engine/README.md` for benchmark and runner details.
