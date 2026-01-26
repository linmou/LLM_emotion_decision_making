"""Tests for persisted augmented choice bins in GameTheoryDataset.

Responsible for: emotion_experiment_engine/datasets/games.py persisted-bin loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset
from games.payoff_matrices import PayoffLeaf, PayoffMatrix
from games.prisoner_delimma import PrisonerDilemmaScenario, PrisionerDelimmaDecision


def _cfg(tmp_path: Path, *, augmentation_config: dict | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=tmp_path / "pd.json",
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=augmentation_config,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_games_dataset_uses_persisted_augmented_bins_when_flag_set(
    tmp_path: Path, monkeypatch
) -> None:
    """GameTheoryDataset should use `augmented_bins` when augmentation_config requests it."""
    scenarios = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
            "augmented_bins": {
                "bins_4": [
                    {"behavior": "cooperate", "text": "Cooperate fully"},
                    {"behavior": "interpolated_1", "text": "Mostly cooperate, minor delay"},
                    {"behavior": "interpolated_2", "text": "Mostly defect, minimal upgrade"},
                    {"behavior": "defect", "text": "Defect fully"},
                ]
            },
        }
    ]

    data_path = tmp_path / "pd.json"
    data_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(data_path),
            "shuffle_options": False,
        },
    )

    ds = GameTheoryDataset(config=_cfg(tmp_path, augmentation_config={"use_augmented_bins": "bins_4"}))
    assert len(ds.items) == 1
    opts = ds.items[0].metadata["options"]  # type: ignore[index]
    assert len(opts) == 4
    assert [o["behavior"] for o in opts] == [
        "cooperate",
        "interpolated_1",
        "interpolated_2",
        "defect",
    ]


def test_games_dataset_ignores_persisted_augmented_bins_without_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """GameTheoryDataset should stick to scenario.get_behavior_choices() unless enabled."""
    scenarios = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
            "augmented_bins": {
                "bins_4": [
                    {"behavior": "cooperate", "text": "Cooperate fully"},
                    {"behavior": "interpolated_1", "text": "Mostly cooperate, minor delay"},
                    {"behavior": "interpolated_2", "text": "Mostly defect, minimal upgrade"},
                    {"behavior": "defect", "text": "Defect fully"},
                ]
            },
        }
    ]

    data_path = tmp_path / "pd.json"
    data_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(data_path),
        },
    )

    ds = GameTheoryDataset(config=_cfg(tmp_path, augmentation_config=None))
    opts = ds.items[0].metadata["options"]  # type: ignore[index]
    assert len(opts) == 2


def test_games_dataset_uses_augmented_options_field_when_configured(
    tmp_path: Path, monkeypatch
) -> None:
    """GameTheoryDataset should use `augmented_options_field` when explicitly enabled."""
    scenarios = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
            "augmented_options_v1": [
                "Cooperate fully",
                "Mostly cooperate, but I might delay a bit",
                "Mostly defect, but I'm still considering cooperating",
                "Defect fully",
            ],
        }
    ]

    data_path = tmp_path / "pd.json"
    data_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(data_path),
        },
    )

    ds = GameTheoryDataset(
        config=_cfg(
            tmp_path,
            augmentation_config={
                "use_augmented_options": True,
                "augmented_options_field": "augmented_options_v1",
                "augmented_options_bins": 4,
                "shuffle_options": False,
            },
        )
    )
    assert len(ds.items) == 1
    behaviors = [o["behavior"] for o in ds.items[0].metadata["options"]]  # type: ignore[index]
    assert behaviors.count("cooperate") == 1
    assert behaviors.count("defect") == 1
    assert behaviors.count("interpolated") == 2


def test_games_dataset_errors_when_augmented_options_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing `augmented_options_field` triggers a clear error when `use_augmented_options` is enabled."""
    scenarios = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
        }
    ]

    data_path = tmp_path / "pd.json"
    data_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(data_path),
        },
    )

    with pytest.raises(ValueError, match="Missing augmented options field"):
        GameTheoryDataset(
            config=_cfg(
                tmp_path,
                augmentation_config={
                    "use_augmented_options": True,
                    "augmented_options_field": "augmented_options_v1",
                    "augmented_options_bins": 4,
                },
            )
        )


def test_games_dataset_errors_when_augmented_options_length_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    """A mismatch between declared bins and actual list length should reject the dataset."""
    scenarios = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
            "augmented_options_v1": [
                "Cooperate fully",
                "Defect fully",
            ],
        }
    ]

    data_path = tmp_path / "pd.json"
    data_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(data_path),
        },
    )

    with pytest.raises(ValueError, match="Expected 4 augmented options"):
        GameTheoryDataset(
            config=_cfg(
                tmp_path,
                augmentation_config={
                    "use_augmented_options": True,
                    "augmented_options_field": "augmented_options_v1",
                    "augmented_options_bins": 4,
                },
            )
        )


def test_games_dataset_prefers_config_data_path_over_game_config(
    tmp_path: Path, monkeypatch
) -> None:
    """BenchmarkConfig.data_path should override get_game_config()['data_path']."""
    augmented = [
        {
            "scenario": "Test scenario",
            "description": "Desc",
            "participants": [{"name": "You"}, {"name": "Bob"}],
            "behavior_choices": {
                "cooperate": "Cooperate fully",
                "defect": "Defect fully",
            },
            "payoff_matrix_description": {},
            "payoff_description": "",
            "game_name": "Prisoners_Dilemma",
            "augmented_options_v1": [
                "Cooperate fully",
                "Mostly cooperate, but I might delay a bit",
                "Mostly defect, but I'm still considering cooperating",
                "Defect fully",
            ],
        }
    ]
    augmented_path = tmp_path / "aug.json"
    augmented_path.write_text(__import__("json").dumps(augmented), encoding="utf-8")

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(__import__("json").dumps(augmented[:0]), encoding="utf-8")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _task: {
            "game_name": "Prisoners_Dilemma",
            "scenario_class": PrisonerDilemmaScenario,
            "decision_class": PrisionerDelimmaDecision,
            "payoff_matrix": PayoffMatrix(
                payoff_leaves=[
                    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(1, 1))
                ]
            ),
            "data_path": str(baseline_path),
            "shuffle_options": False,
        },
    )

    cfg = BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=augmented_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config={
            "use_augmented_options": True,
            "augmented_options_field": "augmented_options_v1",
            "augmented_options_bins": 4,
            "shuffle_options": False,
        },
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    ds = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)
    assert len(ds.items) == 1
    opts = ds.items[0].metadata["options"]  # type: ignore[index]
    assert len(opts) == 4
