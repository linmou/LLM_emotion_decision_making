"""Tests for disabling LLM fallback in GameTheoryDataset evaluation.

Responsible file: emotion_experiment_engine/datasets/games.py
Purpose: ensure that when LLM quota is unavailable, we can skip fallback calls
and mark unsolved decisions as option_id=-1 without making any API calls.
"""

from __future__ import annotations

from typing import Dict

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def _stub_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config={"client": "openai"},
    )


def test_evaluate_response_returns_minus_one_when_llm_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_LLM_JUDGE", "1")
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )
    dataset = GameTheoryDataset(config=_stub_config(), prompt_wrapper=None, answer_wrapper=None)

    def _fail_if_called():
        raise AssertionError("LLM fallback should not be invoked")

    monkeypatch.setattr(dataset, "_ensure_llm_client", _fail_if_called)

    prompt = "\n".join(
        [
            "Scenario: test",
            "Option 1. Cooperate",
            "Option 2. Defect",
        ]
    )
    response = '{"decision":"Not an option"}'
    score = dataset.evaluate_response(response, None, "Prisoners_Dilemma", prompt)
    assert score == -1.0


def test_evaluate_response_returns_minus_one_when_choice_unparseable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responsible file: emotion_experiment_engine/datasets/games.py
    Purpose: unparseable choices should be counted as option_id=-1 (not NaN) so ratios include failures.
    """

    monkeypatch.delenv("DISABLE_LLM_JUDGE", raising=False)
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )
    dataset = GameTheoryDataset(config=_stub_config(), prompt_wrapper=None, answer_wrapper=None)

    monkeypatch.setattr(dataset, "_fallback_option_via_llm", lambda response, options: None)

    prompt = "\n".join(
        [
            "Scenario: test",
            "Option 1. Cooperate",
            "Option 2. Defect",
        ]
    )
    response = '{"decision":"Not an option"}'
    score = dataset.evaluate_response(response, None, "Prisoners_Dilemma", prompt)
    assert score == -1.0


def test_evaluate_response_extracts_option_text_without_decision_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responsible file: emotion_experiment_engine/datasets/games.py
    Purpose: rule-based extraction should match option content in free-form responses before relying on option ids.
    """

    monkeypatch.setenv("DISABLE_LLM_JUDGE", "1")
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [],
    )
    dataset = GameTheoryDataset(config=_stub_config(), prompt_wrapper=None, answer_wrapper=None)

    prompt = "\n".join(
        [
            "Scenario: test",
            "Option 1. Cooperate",
            "Option 2. Defect",
        ]
    )
    response = "After serious consideration, I will Cooperate."
    score = dataset.evaluate_response(response, None, "Prisoners_Dilemma", prompt)
    assert score == 1.0
