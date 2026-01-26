# Tests for emotion_experiment_engine/experiment.py
"""Ensure EmotionExperiment persists dataset-driven choice ratio summaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from emotion_experiment_engine.data_models import (
    BenchmarkConfig,
    ExperimentConfig,
    ResultRecord,
)
from emotion_experiment_engine.experiment import EmotionExperiment


class _ChoiceRatioDataset:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.calls: int = 0

    def compute_split_metrics(self, records: List[ResultRecord]) -> Dict[str, Any]:
        self.calls += 1
        return self.payload


@pytest.fixture()
def tmp_benchmark_config(tmp_path: Path) -> BenchmarkConfig:
    data_path = tmp_path / "games.jsonl"
    data_path.write_text("[]", encoding="utf-8")
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def _build_results() -> List[ResultRecord]:
    # Include minimal metadata so raw_results.json contains per-item options
    # and can be used to trace decisions back to the underlying choices.
    options = [
        {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
        {"id": 2, "text": "Defect", "behavior": "defect"},
    ]
    meta: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }
    return [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,
            repeat_id=0,
            metadata=meta,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-2",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=1,
            metadata=meta,
        ),
    ]


def test_save_results_writes_choice_ratio_csv(tmp_path: Path, tmp_benchmark_config: BenchmarkConfig) -> None:
    payload = {
        "choice_ratio": {
            "overall": [
                {"emotion": "anger", "intensity": 0.1, "option_id": 1, "ratio": 0.4},
                {"emotion": "anger", "intensity": 0.1, "option_id": 2, "ratio": 0.6},
            ],
            "by_repeat": [
                {
                    "emotion": "anger",
                    "intensity": 0.1,
                    "repeat_id": 0,
                    "option_id": 1,
                    "ratio": 1.0,
                },
                {
                    "emotion": "anger",
                    "intensity": 0.1,
                    "repeat_id": 1,
                    "option_id": 2,
                    "ratio": 1.0,
                },
            ],
        }
    }

    dataset = _ChoiceRatioDataset(payload)

    experiment_config = ExperimentConfig(
        model_path="/dev/null",
        emotions=["anger"],
        intensities=[0.1],
        benchmark=tmp_benchmark_config,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config=None,
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
        defer_evaluation=False,
    )

    experiment = EmotionExperiment.__new__(EmotionExperiment)
    experiment.config = experiment_config
    experiment.logger = logging.getLogger("test-choice-ratio")
    experiment.logger.addHandler(logging.NullHandler())
    experiment.output_dir = tmp_path
    experiment.dataset = dataset
    experiment._save_experiment_config = lambda: None

    df = experiment._save_results(_build_results())
    assert not df.empty
    assert dataset.calls == 1

    split_path = tmp_path / "split_metrics.json"
    choice_ratio_path = tmp_path / "summary_choice_ratio.csv"
    choice_ratio_repeat_path = tmp_path / "summary_choice_ratio_by_repeat.csv"

    assert split_path.exists()
    persisted_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert persisted_payload == payload

    assert choice_ratio_path.exists()
    overall_df = pd.read_csv(choice_ratio_path)
    assert set(overall_df.columns) == {"emotion", "intensity", "option_id", "ratio"}
    assert len(overall_df) == 2
    overall_ratios = overall_df.sort_values("option_id")["ratio"].tolist()
    assert overall_ratios == pytest.approx([0.4, 0.6])

    assert choice_ratio_repeat_path.exists()
    repeat_df = pd.read_csv(choice_ratio_repeat_path)
    assert set(repeat_df.columns) == {"emotion", "intensity", "repeat_id", "option_id", "ratio"}
    assert len(repeat_df) == 2
    repeat_ratios = repeat_df.sort_values(["repeat_id", "option_id"])["ratio"].tolist()
    assert repeat_ratios == pytest.approx([1.0, 1.0])


def test_save_results_writes_behavior_ratio_csv(tmp_path: Path, tmp_benchmark_config: BenchmarkConfig) -> None:
    """Ensure EmotionExperiment persists behavior-level choice ratio summaries."""
    payload = {
        "choice_ratio": {
            "overall": [
                {"emotion": "anger", "intensity": 0.1, "option_id": 1, "ratio": 0.4},
                {"emotion": "anger", "intensity": 0.1, "option_id": 2, "ratio": 0.6},
            ],
            "by_repeat": [],
        },
        "behavior_choice_ratio": {
            "overall": [
                {"emotion": "anger", "intensity": 0.1, "behavior": "cooperate", "ratio": 0.4},
                {"emotion": "anger", "intensity": 0.1, "behavior": "defect", "ratio": 0.6},
            ],
            "by_repeat": [
                {"emotion": "anger", "intensity": 0.1, "repeat_id": 0, "behavior": "cooperate", "ratio": 1.0},
                {"emotion": "anger", "intensity": 0.1, "repeat_id": 1, "behavior": "defect", "ratio": 1.0},
            ],
        },
    }

    dataset = _ChoiceRatioDataset(payload)

    experiment_config = ExperimentConfig(
        model_path="/dev/null",
        emotions=["anger"],
        intensities=[0.1],
        benchmark=tmp_benchmark_config,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config=None,
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
        defer_evaluation=False,
    )

    experiment = EmotionExperiment.__new__(EmotionExperiment)
    experiment.config = experiment_config
    experiment.logger = logging.getLogger("test-behavior-ratio")
    experiment.logger.addHandler(logging.NullHandler())
    experiment.output_dir = tmp_path
    experiment.dataset = dataset
    experiment._save_experiment_config = lambda: None

    df = experiment._save_results(_build_results())
    assert not df.empty
    assert dataset.calls == 1

    behavior_ratio_path = tmp_path / "summary_behavior_ratio.csv"
    assert behavior_ratio_path.exists()

    behavior_df = pd.read_csv(behavior_ratio_path)
    assert set(behavior_df.columns) == {"emotion", "intensity", "behavior", "ratio"}
    assert len(behavior_df) == 2
    behavior_df = behavior_df.sort_values("behavior")
    ratios = behavior_df["ratio"].tolist()
    assert ratios == pytest.approx([0.4, 0.6])

    behavior_ratio_repeat_path = tmp_path / "summary_behavior_ratio_by_repeat.csv"
    assert behavior_ratio_repeat_path.exists()
    repeat_df = pd.read_csv(behavior_ratio_repeat_path)
    assert set(repeat_df.columns) == {"emotion", "intensity", "repeat_id", "behavior", "ratio"}
    assert len(repeat_df) == 2
    repeat_df = repeat_df.sort_values(["repeat_id", "behavior"])
    repeat_ratios = repeat_df["ratio"].tolist()
    assert repeat_ratios == pytest.approx([1.0, 1.0])

    # raw_results.json should retain the enriched metadata, including options.
    raw_path = tmp_path / "raw_results.json"
    assert raw_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw
    first = raw[0]
    md = first.get("metadata") or {}
    item_md = md.get("item_metadata") or {}
    opts = item_md.get("options")
    assert isinstance(opts, list) and opts
    assert {opt["behavior"] for opt in opts} == {"cooperate", "defect"}


def test_save_results_adds_chosen_behavior_column(tmp_path: Path, tmp_benchmark_config: BenchmarkConfig) -> None:
    """Test for emotion_experiment_engine/experiment.py: chosen_behavior saved in detailed_results.csv."""

    dataset = _ChoiceRatioDataset(payload={})

    experiment_config = ExperimentConfig(
        model_path="/dev/null",
        emotions=["anger"],
        intensities=[0.1],
        benchmark=tmp_benchmark_config,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config=None,
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
        defer_evaluation=False,
    )

    experiment = EmotionExperiment.__new__(EmotionExperiment)
    experiment.config = experiment_config
    experiment.logger = logging.getLogger("test-chosen-behavior")
    experiment.logger.addHandler(logging.NullHandler())
    experiment.output_dir = tmp_path
    experiment.dataset = dataset
    experiment._save_experiment_config = lambda: None

    df = experiment._save_results(_build_results())
    assert "chosen_behavior" in df.columns
    assert df["chosen_behavior"].tolist() == ["cooperate", "defect"]

    detailed_df = pd.read_csv(tmp_path / "detailed_results.csv")
    assert "chosen_behavior" in detailed_df.columns
    assert detailed_df["chosen_behavior"].tolist() == ["cooperate", "defect"]


def test_save_results_behavior_ratio_renamed_and_logged(
    tmp_path: Path,
    tmp_benchmark_config: BenchmarkConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test for emotion_experiment_engine/experiment.py: behavior ratios use unknown bucket + are shown in summary log."""

    payload = {
        "behavior_choice_ratio": {
            "overall": [
                {"emotion": "anger", "intensity": 0.1, "behavior_label": "cooperate", "ratio": 0.5},
                {"emotion": "anger", "intensity": 0.1, "behavior_label": "unknown", "ratio": 0.5},
            ],
            "by_repeat": [
                {"emotion": "anger", "intensity": 0.1, "repeat_id": 0, "behavior_label": "cooperate", "ratio": 1.0},
                {"emotion": "anger", "intensity": 0.1, "repeat_id": 1, "behavior_label": "unknown", "ratio": 1.0},
            ],
        }
    }
    dataset = _ChoiceRatioDataset(payload)

    experiment_config = ExperimentConfig(
        model_path="/dev/null",
        emotions=["anger"],
        intensities=[0.1],
        benchmark=tmp_benchmark_config,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config=None,
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
        defer_evaluation=False,
    )

    experiment = EmotionExperiment.__new__(EmotionExperiment)
    experiment.config = experiment_config
    experiment.output_dir = tmp_path
    experiment.dataset = dataset
    experiment._save_experiment_config = lambda: None

    logger = logging.getLogger("test-behavior-summary-log")
    logger.handlers.clear()
    logger.propagate = True
    experiment.logger = logger

    caplog.set_level(logging.INFO)
    experiment._save_results(_build_results())

    behavior_ratio_path = tmp_path / "summary_behavior_ratio.csv"
    assert behavior_ratio_path.exists()
    behavior_df = pd.read_csv(behavior_ratio_path)
    assert set(behavior_df.columns) == {"emotion", "intensity", "behavior", "ratio"}
    assert "unknown" in set(behavior_df["behavior"].astype(str))

    # Shuffled games: score mean is meaningless; log the behavior ratio summary instead.
    assert "=== EXPERIMENT RESULTS SUMMARY ===" in caplog.text
    assert "unknown" in caplog.text
