# Tests for emotion_experiment_engine/datasets/games.py
"""Unit tests for game theory dataset evaluation logic."""

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem
from emotion_experiment_engine.datasets.games import (
    GameTheoryCompletionOptionIdDataset,
    GameTheoryDataset,
)


@pytest.fixture()
def benchmark_config() -> BenchmarkConfig:
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
        llm_eval_config=None,
    )


@pytest.fixture()
def dataset(monkeypatch, benchmark_config) -> GameTheoryDataset:
    monkeypatch.setattr(
        GameTheoryDataset,
        "_load_and_parse_data",
        lambda self: [
            BenchmarkItem(
                id="pd-1",
                input_text="Prisoners dilemma event",
                context=None,
                ground_truth=None,
                metadata={
                    "options": [
                        {"id": 1, "text": "Cooperate"},
                        {"id": 2, "text": "Defect"},
                    ]
                },
            )
        ],
    )

    return GameTheoryDataset(
        config=benchmark_config,
        prompt_wrapper=None,
        answer_wrapper=None,
    )


@pytest.fixture()
def completion_dataset(monkeypatch, benchmark_config) -> GameTheoryCompletionOptionIdDataset:
    monkeypatch.setattr(
        GameTheoryCompletionOptionIdDataset,
        "_load_and_parse_data",
        lambda self: [
            BenchmarkItem(
                id="pd-1",
                input_text="Prisoners dilemma event",
                context=None,
                ground_truth=None,
                metadata={
                    "options": [
                        {"id": 1, "text": "Cooperate"},
                        {"id": 2, "text": "Defect"},
                    ]
                },
            )
        ],
    )

    return GameTheoryCompletionOptionIdDataset(
        config=benchmark_config,
        prompt_wrapper=None,
        answer_wrapper=None,
    )


def test_evaluate_response_regex_only(dataset: GameTheoryDataset):
    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
    )
    response = '{"analysis": "...", "decision": "defect"}'

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)


def test_evaluate_response_llm_fallback(monkeypatch, dataset: GameTheoryDataset):
    captured = {"called": False}

    def _fake_fallback(*args, **kwargs):
        captured["called"] = True
        return 1

    monkeypatch.setattr(GameTheoryDataset, "_fallback_option_via_llm", _fake_fallback)

    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
    )
    response = "No decision present"

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(1.0)
    assert captured["called"] is True


def test_evaluate_response_parses_option_number_json(monkeypatch, dataset: GameTheoryDataset):
    """If the model returns a JSON decision like 'Option 1', use it directly without LLM."""

    def _fail_fallback(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("LLM fallback should not be invoked for JSON decision")

    monkeypatch.setattr(GameTheoryDataset, "_fallback_option_via_llm", _fail_fallback)

    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
    )
    response = '{"decision": "Option 1"}'

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(1.0)


def test_evaluate_response_uses_gemini_client(monkeypatch, dataset: GameTheoryDataset):
    """Gemini client should go through evaluation_utils.llm_evaluate_response, not OpenAI."""

    captured = {}

    def _fake_eval(system_prompt: str, query: str, llm_eval_config: dict):
        captured["system_prompt"] = system_prompt
        captured["query"] = query
        captured["config"] = llm_eval_config
        return {"option_id": 2}

    def _fail_openai_client():
        raise AssertionError("OpenAI client should not be used for gemini")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.evaluation_utils.llm_evaluate_response",
        _fake_eval,
    )
    monkeypatch.setattr(
        GameTheoryDataset,
        "_ensure_llm_client",
        lambda self: _fail_openai_client(),
    )

    dataset.llm_eval_config = {"client": "gemini", "model": "gemini-pro"}

    prompt = (
        "Scenario: Trust game\n"
        "Option 1. Return money\n"
        "Option 2. Keep money\n"
    )
    response = "No explicit decision"

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Trust_Game_Trustee",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)
    assert "gemini" in captured["config"]["client"]
    assert "Available options" in captured["query"]
    assert "Option 1" in captured["query"] or "Option 2" in captured["query"]


def test_completion_dataset_parses_leading_digit_without_json(monkeypatch, completion_dataset: GameTheoryCompletionOptionIdDataset):
    """Completion dataset should parse a bare option id without calling LLM fallback."""

    def _fail_fallback(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("LLM fallback should not be invoked for leading-digit choice")

    monkeypatch.setattr(GameTheoryCompletionOptionIdDataset, "_fallback_option_via_llm", _fail_fallback)

    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
        'After serious consideration, you choose option "'
    )
    response = ' 2"'

    choice = completion_dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)


def test_evaluate_response_ambiguous_truncated_text_falls_back(monkeypatch, completion_dataset: GameTheoryCompletionOptionIdDataset):
    """When a truncated response matches multiple options, avoid guessing and fall back."""

    captured = {"called": False}

    def _fake_fallback(*args, **kwargs):
        captured["called"] = True
        return 2

    monkeypatch.setattr(GameTheoryCompletionOptionIdDataset, "_fallback_option_via_llm", _fake_fallback)

    prompt = (
        "Scenario: Ambiguous options\n"
        "Option 1. Alpha beta\n"
        "Option 2. Alpha gamma\n"
    )
    response = "Alpha"  # matches both options; should not pick arbitrarily

    choice = completion_dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Escalation_Game",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)
    assert captured["called"] is True


def test_evaluate_response_parses_plain_option_prefix_without_json(monkeypatch, dataset: GameTheoryDataset):
    """Base LMs may emit 'Option 2: ...' without any JSON wrapper."""

    captured = {"called": False}

    def _fallback(*args, **kwargs):
        captured["called"] = True
        return 2

    monkeypatch.setattr(GameTheoryDataset, "_fallback_option_via_llm", _fallback)

    prompt = (
        "Scenario: Escalation\n"
        "Option 1. Keep the current coffee selection and menu.\n"
        "Option 2. Upgrade to premium coffee beans and introduce specialty drinks.\n"
        'After serious consideration, you choose option "'
    )
    response = "Option 2: Upgrade to premium coffee beans"

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Escalation_Game",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)
    assert captured["called"] is True


def test_completion_dataset_parses_plain_option_prefix_without_json(monkeypatch, completion_dataset: GameTheoryCompletionOptionIdDataset):
    """Completion dataset should parse 'Option 2: ...' without calling LLM fallback."""

    def _fail_fallback(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("LLM fallback should not be invoked for plain Option-N responses")

    monkeypatch.setattr(GameTheoryCompletionOptionIdDataset, "_fallback_option_via_llm", _fail_fallback)

    prompt = (
        "Scenario: Escalation\n"
        "Option 1. Keep the current coffee selection and menu.\n"
        "Option 2. Upgrade to premium coffee beans and introduce specialty drinks.\n"
        'After serious consideration, you choose option "'
    )
    response = "Option 2: Upgrade to premium coffee beans"

    choice = completion_dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Escalation_Game",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)


def test_evaluate_response_parses_plain_option_text_without_json(monkeypatch, dataset: GameTheoryDataset):
    """Base LMs may emit only the option text; match it against the presented option list."""

    captured = {"called": False}

    def _fallback(*args, **kwargs):
        captured["called"] = True
        return 2

    monkeypatch.setattr(GameTheoryDataset, "_fallback_option_via_llm", _fallback)

    prompt = (
        "Scenario: Escalation\n"
        "Option 1. Keep the current coffee selection and menu.\n"
        "Option 2. Upgrade to premium coffee beans and introduce specialty drinks.\n"
        'After serious consideration, you choose option "'
    )
    response = "Upgrade to premium coffee beans"

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Escalation_Game",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)
    assert captured["called"] is True


def test_completion_dataset_parses_plain_option_text_without_json(monkeypatch, completion_dataset: GameTheoryCompletionOptionIdDataset):
    """Completion dataset should match plain option text without calling LLM fallback."""

    def _fail_fallback(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("LLM fallback should not be invoked for plain option-text responses")

    monkeypatch.setattr(GameTheoryCompletionOptionIdDataset, "_fallback_option_via_llm", _fail_fallback)

    prompt = (
        "Scenario: Escalation\n"
        "Option 1. Keep the current coffee selection and menu.\n"
        "Option 2. Upgrade to premium coffee beans and introduce specialty drinks.\n"
        'After serious consideration, you choose option "'
    )
    response = "Upgrade to premium coffee beans"

    choice = completion_dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Escalation_Game",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)


def test_completion_dataset_strips_speaker_prefix_before_parsing(monkeypatch, completion_dataset: GameTheoryCompletionOptionIdDataset):
    """Some base checkpoints emit chat-style prefixes like 'Human: 1'."""

    def _fail_fallback(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("LLM fallback should not be invoked for speaker-prefixed digits")

    monkeypatch.setattr(GameTheoryCompletionOptionIdDataset, "_fallback_option_via_llm", _fail_fallback)

    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
        "Output exactly one character: 1 or 2."
    )
    response = "Human: 2"

    choice = completion_dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)
