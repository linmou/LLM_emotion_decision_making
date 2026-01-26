# Tests for emotion_experiment_engine/game_prompt_wrapper.py
"""Unit tests for the game benchmark prompt wrapper adapter."""

from typing import Any, List

import pytest

from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper
from emotion_experiment_engine.game_prompt_wrapper import GameCompletionOptionIdPromptWrapper
from emotion_experiment_engine.game_prompt_wrapper import GameDecisionPromptWrapper


class _DummyPromptFormat:
    """Minimal prompt format stub."""

    def __init__(self) -> None:
        self.records: List[Any] = []

    def build(self, system_prompt: str, user_messages: List[str], enable_thinking: bool = False):
        self.records.append((system_prompt, user_messages, enable_thinking))
        return f"PROMPT::{enable_thinking}"


@pytest.fixture(autouse=True)
def patch_game_config(monkeypatch):
    class _FakeDecision:
        @staticmethod
        def example() -> str:
            return "{\"decision\": \"Option 1\"}"

    monkeypatch.setattr(
        "emotion_experiment_engine.game_prompt_wrapper.get_game_config",
        lambda task_type: {"decision_class": _FakeDecision},
    )


@pytest.fixture()
def prompt_wrapper(monkeypatch) -> GameBenchmarkPromptWrapper:
    captured = {}

    class _FakeGameReactWrapper:
        def __init__(self, prompt_format, response_format):
            captured["init"] = (prompt_format, response_format)

        def __call__(self, event, options, user_messages, enable_thinking=False):
            captured["call"] = {
                "event": event,
                "options": options,
                "user_messages": user_messages,
                "enable_thinking": enable_thinking,
            }
            return "react-output"

    monkeypatch.setattr(
        "emotion_experiment_engine.game_prompt_wrapper.GameReactPromptWrapper",
        _FakeGameReactWrapper,
    )

    wrapper = GameBenchmarkPromptWrapper(_DummyPromptFormat(), "Prisoners_Dilemma")
    wrapper._captured = captured  # type: ignore[attr-defined]
    return wrapper


def test_wrapper_builds_prompt(prompt_wrapper: GameBenchmarkPromptWrapper):
    prompt = prompt_wrapper(
        context="irrelevant",
        question="Describe the payoff matrix",
        user_messages=["Choose wisely."],
        enable_thinking=True,
        augmentation_config=None,
        answer=None,
        emotion="anger",
        options=[
            {"id": 1, "text": "Cooperate"},
            {"id": 2, "text": "Defect"},
        ],
    )

    assert prompt == "react-output"

    captured = prompt_wrapper._captured  # type: ignore[attr-defined]
    assert captured["call"]["event"] == "Describe the payoff matrix"

    option_texts = captured["call"]["options"]
    assert option_texts == ["Cooperate", "Defect"]
    assert captured["call"]["user_messages"] == ["Choose wisely."]
    assert captured["call"]["enable_thinking"] is True


def test_decision_wrapper_includes_decide_now_instruction(patch_game_config):
    fmt = _DummyPromptFormat()
    wrapper = GameDecisionPromptWrapper(fmt, "Prisoners_Dilemma")

    prompt = wrapper(
        context=None,
        question="You must pick.",
        user_messages="State your choice.",
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=[{"id": 1, "text": "Cooperate"}, {"id": 2, "text": "Defect"}],
    )

    assert fmt.records, "prompt_format.build should be invoked"
    system_prompt, user_msgs, enable_thinking = fmt.records[0]
    assert enable_thinking is False
    assert user_msgs == ["State your choice."]
    assert "You must make a decision now" in system_prompt
    assert '{"decision":' in system_prompt
    assert "Option 1." in system_prompt
    assert "Option 2." in system_prompt
    assert prompt  # final prompt text should be non-empty


def test_completion_wrapper_ignores_chat_template_and_demands_single_token_choice(patch_game_config):
    """Base-model wrapper should emit a raw completion prompt that forces a single numeric token."""

    fmt = _DummyPromptFormat()
    wrapper = GameCompletionOptionIdPromptWrapper(fmt, "Prisoners_Dilemma")

    prompt = wrapper(
        context=None,
        question="Scenario: Prisoners_Dilemma\nYou and Bob are deciding.",
        user_messages=None,
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=[
            {"id": 1, "text": "Cooperate"},
            {"id": 2, "text": "Defect"},
            {"id": 3, "text": "Stay silent"},
        ],
    )

    assert fmt.records == [], "completion wrapper must not call prompt_format.build"
    assert "Option 1." in prompt
    assert "Option 2." in prompt
    assert "you choose option" not in prompt
    assert "Output exactly one character: 1, 2, or 3." in prompt
    assert prompt.strip().endswith("Answer:")


def test_completion_wrapper_adapts_numeric_range_for_many_options(patch_game_config):
    fmt = _DummyPromptFormat()
    wrapper = GameCompletionOptionIdPromptWrapper(fmt, "Prisoners_Dilemma")

    options = [{"id": idx, "text": f"Choice {idx}"} for idx in range(1, 13)]
    prompt = wrapper(
        context=None,
        question="Scenario: Prisoners_Dilemma\nPick one.",
        user_messages=None,
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=options,
    )

    assert "Output only the option number (1-12)." in prompt
    assert prompt.strip().endswith("Answer:")
