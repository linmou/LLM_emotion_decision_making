# Public Release: Emotion Experiment Engine
<!-- Updated: 2026-05-11 | Commit: pending -->

This repository is the public snapshot of `emotion_experiment_engine` and its
core dependencies (`neuro_manipulation`, `games`, and shared utilities). It is
scoped for experimentation and evaluation only.

## Scope
- Included: `emotion_experiment_engine`, `neuro_manipulation`, `games`, public game datasets, emotion stimulus data, and shared utilities.
- Excluded: `data_creation`, caches, logs, generated results, and model weights.
- Large VLM stimulus images under `multimodal_crow_envnt/emotion_envent/` are stored with Git LFS.

## Layout
- `emotion_experiment_engine/`: benchmark orchestration, datasets, evaluation helpers
- `neuro_manipulation/`: prompt formats, RepE pipelines, vLLM hooks
- `games/`: game theory scenarios and payoff matrices
- `dataset/`: game theory scenario JSON files used by the public configs
- `data/stimulus/crowd-enVent_textlike/`: text emotion stimulus used by LM configs
- `multimodal_crow_envnt/emotion_envent/`: VLM emotion stimulus JSON and images
- `constants.py`, `statistical_engine.py`, `merge_data_samples.py`: shared utilities

## Setup
```bash
git lfs install
git lfs pull
pip install -r requirements.txt
```

The four game-theory configs use `${USER_HOME}/huggingface_models/...` for local
model paths. Set `USER_HOME` before running them, for example:

```bash
export USER_HOME="$HOME"
```

## API Keys
Fill in `api_configs.py` with your own credentials before running any
LLM-backed evaluation.

## Usage
See `emotion_experiment_engine/README.md` for benchmark and runner details.

Current public game-theory configs:

- `config/new_game_theory_config.yaml`: LM sweep over `game_theory`
- `config/new_game_theory_decision_config.yaml`: LM sweep over `game_theory_decision`
- `config/vlm_mm_game_theory_300.yaml`: VLM sweep over `game_theory`
- `config/vlm_mm_game_theory_decision_300.yaml`: VLM sweep over `game_theory_decision`

Each config covers the 9 public game task variants:
`Prisoners_Dilemma`, `Stag_Hunt`, `Escalation_Game`,
`Trust_Game_Trustor`, `Trust_Game_Trustee`, `Ultimatum_Game_Proposer`,
`Ultimatum_Game_Responder`, `Beauty_Contest`, and `Sealed_Auction`.
