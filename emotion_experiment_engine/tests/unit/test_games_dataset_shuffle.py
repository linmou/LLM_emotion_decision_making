"""Tests for shuffling and metadata structure in GameTheoryDataset.

Responsible file: emotion_experiment_engine/datasets/games.py
Purpose: verify that GameTheoryDataset builds options with behavior categories
and supports per-scenario shuffling without breaking existing metrics.
"""

from __future__ import annotations

import random
from typing import Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset
from games.game import BehaviorChoices, GameScenario


class _DummyChoices(BehaviorChoices):
    option_a: str
    option_b: str

    def get_choices(self) -> list[str]:  # type: ignore[override]
        return [self.option_a, self.option_b]

    def is_valid_choice(self, choice: str) -> bool:  # type: ignore[override]
        return choice in self.get_choices()

    @staticmethod
    def example() -> dict:
        return {"option_a": "Choose A", "option_b": "Choose B"}


class _DummyScenario(GameScenario):
    scenario: str
    description: str
    participants: list[Dict[str, object]]
    behavior_choices: _DummyChoices

    def get_scenario_info(self) -> dict:  # type: ignore[override]
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> _DummyChoices:  # type: ignore[override]
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:  # type: ignore[override]
        if decision == self.behavior_choices.option_a:
            return "cat_a"
        if decision == self.behavior_choices.option_b:
            return "cat_b"
        raise ValueError("Unknown decision")

    @staticmethod
    def example() -> dict:
        return {
            "scenario": "Dummy scenario",
            "description": "Dummy description",
            "participants": [{"name": "You"}, {"name": "Other"}],
            "behavior_choices": _DummyChoices.example(),
            "payoff_matrix": {},
            "game_name": "DummyGame",
        }


def _stub_game_config_with_scenario() -> Dict[str, object]:
    """Return a minimal game config using _DummyScenario."""
    example = _DummyScenario.example()
    return {
        "game_name": "DummyGame",
        "scenario_class": _DummyScenario,
        "payoff_matrix": {},
        "scenarios": [example],
    }


def _make_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory",
        task_type="DummyGame",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_dataset_attaches_behavior_and_shuffles_via_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """[US1] Ensure options include behavior categories and shuffling can be controlled via random.seed."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config_with_scenario(),
    )

    # Seed RNG and build dataset
    random.seed(123)
    cfg = _make_benchmark_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    assert dataset.items, "Expected at least one BenchmarkItem"
    item = dataset.items[0]
    options: List[Dict[str, object]] = item.metadata["options"]  # type: ignore[index]

    # Behavior categories should be attached and ids should be 1-based contiguous
    assert options, "Expected non-empty options list"
    ids = [opt["id"] for opt in options]
    assert ids == list(range(1, len(options) + 1))
    for opt in options:
        assert "text" in opt
        assert "behavior" in opt
        assert isinstance(opt["behavior"], str)

    # The set of texts must match the underlying behavior choices
    base_choices = _DummyScenario(**_DummyScenario.example()).get_behavior_choices().get_choices()
    assert set(opt["text"] for opt in options) == set(base_choices)


def test_dataset_shuffles_order_when_seed_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """[US1] With different random seeds, behavior categories can appear at different option indices."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config_with_scenario(),
    )

    cfg = _make_benchmark_config()

    seen_mappings = set()
    for seed in range(5):
        random.seed(seed)
        ds = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)
        opts = ds.items[0].metadata["options"]  # type: ignore[index]
        mapping = tuple(sorted((opt["behavior"], opt["id"]) for opt in opts))  # type: ignore[index]
        seen_mappings.add(mapping)
        if len(seen_mappings) >= 2:
            break

    # Expect at least two distinct behavior→index mappings across seeds.
    assert len(seen_mappings) >= 2


def test_round_trip_options_and_decision_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """[US3] Round-trip options metadata and a simulated decision via ResultRecord-like fields."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config_with_scenario(),
    )

    random.seed(0)
    cfg = _make_benchmark_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)
    assert dataset.items, "Expected at least one BenchmarkItem"

    item = dataset.items[0]
    options: List[Dict[str, object]] = item.metadata["options"]  # type: ignore[index]
    # Pick one option and simulate a decision choosing its id.
    chosen = options[0]
    chosen_id = int(chosen["id"])
    chosen_behavior = str(chosen["behavior"])

    # Build the metadata shape used in ResultRecord for logging.
    meta: Dict[str, object] = {
        "benchmark": "game_theory",
        "item_metadata": {"options": options},
    }

    # Simulate "DecisionRecord-like" fields: item_id, score (option id), and metadata.
    # In real runs this is a ResultRecord, but for this test we only need the mapping logic.
    reconstructed_behavior = None
    for opt in meta["item_metadata"]["options"]:  # type: ignore[index]
        opt_id = int(opt["id"])  # type: ignore[index]
        if opt_id == chosen_id:
            reconstructed_behavior = str(opt["behavior"])  # type: ignore[index]
            break

    assert reconstructed_behavior is not None
    assert reconstructed_behavior == chosen_behavior


def test_dataset_preserves_scenario_text_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GameTheoryDataset exposes scenario contents in item.metadata for raw_results.json auditing."""
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config_with_scenario(),
    )

    cfg = _make_benchmark_config()
    dataset = GameTheoryDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    item = dataset.items[0]
    assert item.metadata.get("scenario") == "Dummy scenario"
    assert item.metadata.get("description") == "Dummy description"
