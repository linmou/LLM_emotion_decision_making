"""Tests for behavior-level choice ratios in GameTheoryDataset.

Responsible file: emotion_experiment_engine/datasets/games.py
Purpose: verify that compute_split_metrics returns both id-based and
behavior-based choice ratios, and that invalid/missing behavior categories are
rejected explicitly (FR-007).
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, ResultRecord
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def _stub_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory",
        # Use a real game type so get_game_config(GameNames.from_string(...)) succeeds.
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


def _build_records_with_metadata() -> List[ResultRecord]:
    """Construct records with per-item behavior categories in metadata."""
    options = [
        {"id": 1, "text": "Choose A", "behavior": "cat_a"},
        {"id": 2, "text": "Choose B", "behavior": "cat_b"},
    ]
    meta: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }
    return [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-1",
            task_name="DummyGame",
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
            item_id="item-2",
            task_name="DummyGame",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=0,
            metadata=meta,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-3",
            task_name="DummyGame",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=1,
            metadata=meta,
        ),
    ]


def test_compute_split_metrics_returns_behavior_choice_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """[US2] compute_split_metrics should expose both id-based and behavior-based choice ratios."""
    # Stub out data loading so we don't depend on real game configs.
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )

    cfg = _stub_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)
    records = _build_records_with_metadata()

    metrics = dataset.compute_split_metrics(records)

    assert "choice_ratio" in metrics
    assert "behavior_choice_ratio" in metrics

    choice_ratio = metrics["choice_ratio"]
    behavior_ratio = metrics["behavior_choice_ratio"]

    # Existing id-based payload format is preserved.
    assert set(choice_ratio.keys()) == {"overall", "by_repeat"}

    overall_beh = behavior_ratio.get("overall")
    assert isinstance(overall_beh, list)
    # Expect counts: cat_a:1, cat_b:2 for anger,0.1
    by_key: Dict[tuple, float] = {
        (row["emotion"], row["intensity"], row["behavior_label"]): row["ratio"]
        for row in overall_beh
    }
    assert by_key[("anger", 0.1, "cat_a")] == pytest.approx(1 / 3)
    assert by_key[("anger", 0.1, "cat_b")] == pytest.approx(2 / 3)


def test_behavior_choice_ratio_missing_behavior_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """[FR-007] Missing behavior category for a chosen option must raise a clear error."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )

    cfg = _stub_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    # Build records where options lack a behavior category for id=2
    options = [
        {"id": 1, "text": "Choose A", "behavior": "cat_a"},
        {"id": 2, "text": "Choose B"},  # missing behavior
    ]
    meta: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }
    bad_records = [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-1",
            task_name="DummyGame",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=0,
            metadata=meta,
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        dataset.compute_split_metrics(bad_records)

    msg = str(excinfo.value)
    assert "behavior category" in msg or "behavior" in msg


def test_unmapped_option_id_counts_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unmappable option_id should count under an explicit unknown behavior bucket."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )

    cfg = _stub_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    options = [
        {"id": 1, "text": "Choose A", "behavior": "cat_a"},
        {"id": 2, "text": "Choose B", "behavior": "cat_b"},
    ]
    meta: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }
    records = [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-1",
            task_name="DummyGame",
            prompt="",
            response="",
            ground_truth=None,
            score=3.0,  # no matching option id in options
            repeat_id=0,
            metadata=meta,
        )
    ]

    metrics = dataset.compute_split_metrics(records)

    choice_overall = metrics["choice_ratio"]["overall"]
    behavior_overall = metrics["behavior_choice_ratio"]["overall"]

    assert len(choice_overall) == 1
    assert choice_overall[0]["option_id"] == 3
    assert choice_overall[0]["ratio"] == pytest.approx(1.0)

    assert len(behavior_overall) == 1
    assert behavior_overall[0]["behavior_label"] == "unknown"
    assert behavior_overall[0]["ratio"] == pytest.approx(1.0)


def test_choice_and_behavior_ratios_use_same_decision_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """[FR-006] When both ratio types exist, they should reflect the same decisions."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )

    cfg = _stub_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    # One record with behavior metadata, one without any item_metadata/options.
    options = [
        {"id": 1, "text": "Choose A", "behavior": "cat_a"},
        {"id": 2, "text": "Choose B", "behavior": "cat_b"},
    ]
    meta_with_opts: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }
    meta_without_opts: Dict[str, object] = {"benchmark": "game_theory"}

    records = [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,
            repeat_id=0,
            metadata=meta_with_opts,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="item-2",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=0,
            metadata=meta_without_opts,
        ),
    ]

    metrics = dataset.compute_split_metrics(records)

    choice_overall = metrics["choice_ratio"]["overall"]
    behavior_overall = metrics["behavior_choice_ratio"]["overall"]

    # The record without options metadata should be excluded from both views.
    assert len(choice_overall) == 1
    assert len(behavior_overall) == 1

    row = choice_overall[0]
    assert row["emotion"] == "anger"
    assert row["intensity"] == 0.1
    assert row["option_id"] == 1
    assert row["ratio"] == pytest.approx(1.0)

    b_row = behavior_overall[0]
    assert b_row["emotion"] == "anger"
    assert b_row["intensity"] == 0.1
    assert b_row["behavior_label"] == "cat_a"
    assert b_row["ratio"] == pytest.approx(1.0)


def test_behavior_choice_ratio_does_not_cache_options_across_emotions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: behavior ratios must use each record's own options metadata.

    Responsible file: emotion_experiment_engine/datasets/games.py
    Bug: caching options by item_id reuses a different emotion's shuffled options.
    """
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )

    cfg = _stub_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    meta_anger: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {
            "options": [
                {"id": 1, "text": "A", "behavior": "cooperate"},
                {"id": 2, "text": "B", "behavior": "defect"},
            ]
        },
    }
    meta_neutral: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {
            "options": [
                {"id": 1, "text": "A", "behavior": "defect"},
                {"id": 2, "text": "B", "behavior": "cooperate"},
            ]
        },
    }

    records = [
        ResultRecord(
            emotion="anger",
            intensity=1.0,
            item_id="item-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,  # cooperate for anger
            repeat_id=0,
            metadata=meta_anger,
        ),
        ResultRecord(
            emotion="neutral",
            intensity=0.0,
            item_id="item-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,  # defect for neutral (swapped)
            repeat_id=0,
            metadata=meta_neutral,
        ),
    ]

    metrics = dataset.compute_split_metrics(records)
    overall = metrics["behavior_choice_ratio"]["overall"]
    by_key = {(r["emotion"], float(r["intensity"]), r["behavior_label"]): r["ratio"] for r in overall}

    assert by_key[("anger", 1.0, "cooperate")] == pytest.approx(1.0)
    assert by_key[("neutral", 0.0, "defect")] == pytest.approx(1.0)
