# Tests for emotion_experiment_engine/datasets/games.py
"""Unit tests for GameTheoryDataset split-metric aggregation."""

from __future__ import annotations

from typing import Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, ResultRecord
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def _stub_game_config() -> Dict[str, object]:
    return {
        "scenarios": [
            {
                "id": "pd-1",
                "event": "Do you cooperate or defect?",
                "options": [
                    {"id": 1, "text": "Cooperate"},
                    {"id": 2, "text": "Defect"},
                ],
            }
        ]
    }


def _build_records() -> List[ResultRecord]:
    return [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="Option 1: Cooperate\nOption 2: Defect",
            response="I'll cooperate",
            ground_truth=None,
            score=1.0,
            repeat_id=0,
            metadata=None,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="Option 1: Cooperate\nOption 2: Defect",
            response="Defect",
            ground_truth=None,
            score=2.0,
            repeat_id=0,
            metadata=None,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="Option 1: Cooperate\nOption 2: Defect",
            response="Defect",
            ground_truth=None,
            score=2.0,
            repeat_id=1,
            metadata=None,
        ),
        ResultRecord(
            emotion="happy",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="Option 1: Cooperate\nOption 2: Defect",
            response="I'll cooperate",
            ground_truth=None,
            score=1.0,
            repeat_id=0,
            metadata=None,
        ),
    ]


def test_game_dataset_reports_choice_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config(),
    )

    config = BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    dataset = GameTheoryDataset(
        config=config,
        prompt_wrapper=None,
        answer_wrapper=None,
    )

    metrics = dataset.compute_split_metrics(_build_records())

    assert "choice_ratio" in metrics
    choice_ratio = metrics["choice_ratio"]
    assert set(choice_ratio.keys()) == {"overall", "by_repeat"}

    overall = choice_ratio["overall"]
    overall_map = {
        (row["emotion"], row["intensity"], row["option_id"]): row["ratio"]
        for row in overall
    }
    expected_overall = {
        ("anger", 0.1, 1): 1 / 3,
        ("anger", 0.1, 2): 2 / 3,
        ("happy", 0.1, 1): 1.0,
    }
    assert set(overall_map.keys()) == set(expected_overall.keys())
    for key, expected_ratio in expected_overall.items():
        assert overall_map[key] == pytest.approx(expected_ratio)

    by_repeat = choice_ratio["by_repeat"]
    by_repeat_map = {
        (row["emotion"], row["intensity"], row["repeat_id"], row["option_id"]): row["ratio"]
        for row in by_repeat
    }
    expected_by_repeat = {
        ("anger", 0.1, 0, 1): 0.5,
        ("anger", 0.1, 0, 2): 0.5,
        ("anger", 0.1, 1, 2): 1.0,
        ("happy", 0.1, 0, 1): 1.0,
    }
    assert set(by_repeat_map.keys()) == set(expected_by_repeat.keys())
    for key, expected_ratio in expected_by_repeat.items():
        assert by_repeat_map[key] == pytest.approx(expected_ratio)
