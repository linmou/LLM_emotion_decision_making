# Tests for game benchmark integration
"""Integration checks for registry wiring and dataset parity."""

from __future__ import annotations

import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset
from games.game_configs import get_game_config
from games.trust_game import TrustGameTrustorScenario
from games.trust_game import TrustGameTrusteeScenario
from games.ultimatum_game import UltimatumGameProposerScenario
from games.ultimatum_game import UltimatumGameResponderScenario
from neuro_manipulation.datasets.game_scenario_dataset import GameScenarioDataset


class _DummyPromptFormat:
    """Minimal prompt format stub for registry tests."""

    def __init__(self) -> None:
        self.calls = []

    def build(self, *args, **kwargs):  # pragma: no cover - passthrough stub
        self.calls.append((args, kwargs))
        return ""


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
def prompt_format() -> _DummyPromptFormat:
    return _DummyPromptFormat()


def _mock_game_config():
    return {
        "scenarios": [
            {
                "id": "pd-1",
                "event": "Cooperate or defect?",
                "options": [
                    {"id": 1, "text": "Cooperate"},
                    {"id": 2, "text": "Defect"},
                ],
            }
        ]
    }


def _ensure_stubbed_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    if "vllm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=object))
    if "openai" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "openai",
            types.SimpleNamespace(OpenAI=lambda **_: None, AzureOpenAI=lambda **_: None),
        )


@pytest.fixture()
def patched_registry_game_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _mock_game_config(),
    )


def test_registry_creates_components(
    patched_registry_game_config, benchmark_config, prompt_format
):
    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="game_theory",
        task_type="Prisoners_Dilemma",
        config=benchmark_config,
        prompt_format=prompt_format,
    )

    from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper

    assert callable(prompt_wrapper)
    assert callable(answer_wrapper)
    assert isinstance(dataset, GameTheoryDataset)
    assert isinstance(dataset.prompt_wrapper.func.__self__, GameBenchmarkPromptWrapper)

    batch_item = dataset[0]
    assert set(batch_item.keys()) == {"item", "prompt", "ground_truth"}
    collated = dataset.collate_fn([batch_item])
    assert set(collated.keys()) == {"prompts", "items", "ground_truths"}
    assert collated["prompts"][0]


def test_diplomacy_escalation_game_theory_dataset(prompt_format) -> None:
    # Test for emotion_experiment_engine/datasets/games.py: ensure diplomacy escalation data loads via game_theory benchmark.
    config = BenchmarkConfig(
        name="game_theory",
        task_type="Diplomacy_Escalation_Game",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="game_theory",
        task_type="Diplomacy_Escalation_Game",
        config=config,
        prompt_format=prompt_format,
    )

    assert callable(prompt_wrapper)
    assert callable(answer_wrapper)
    assert isinstance(dataset, GameTheoryDataset)
    assert (
        Path(dataset.config.data_path).name
        == "diplomacy_Escalation_Game_all_data_samples.json"
    )
    first_item = dataset[0]["item"]
    assert first_item.metadata["options"]
    assert len(first_item.metadata["options"]) == 2


@pytest.mark.parametrize(
    "task_type, extra",
    [
        ("Trust_Game_Trustee", {"previous_actions_length": 1, "previous_trust_level": 0}),
        ("Ultimatum_Game_Responder", {"previous_actions_length": 1, "previous_offer_level": 0}),
        ("Escalation_Game", {"previous_actions_length": 0}),
    ],
)
def test_dataset_parity(monkeypatch: pytest.MonkeyPatch, task_type: str, extra: dict[str, int]) -> None:
    _ensure_stubbed_dependencies(monkeypatch)

    game_config = deepcopy(get_game_config(task_type))
    game_config.update(extra)

    def _prompt(event: str, options: tuple[str, ...]) -> str:
        return f"{event}\n{options}"

    legacy_dataset = GameScenarioDataset(game_config, prompt_wrapper=_prompt)

    benchmark_config = BenchmarkConfig(
        name="game_theory",
        task_type=task_type,
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=extra,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    new_dataset = GameTheoryDataset(
        config=benchmark_config,
        prompt_wrapper=None,
        answer_wrapper=None,
    )

    assert len(new_dataset) == len(legacy_dataset) > 0

    for idx, item in enumerate(new_dataset):
        legacy_scenario = legacy_dataset.data[idx]
        legacy_choices = legacy_scenario.get_behavior_choices().get_choices()
        new_choices = [opt["text"] for opt in item["item"].metadata["options"]]
        assert legacy_choices == new_choices

        expected_text = GameTheoryDataset._format_scenario(legacy_scenario)
        assert item["item"].input_text == expected_text

    raw_name = Path(legacy_dataset.game_config["data_path"]).name.split(".")[0]
    assert Path(new_dataset.config.data_path).name.startswith(raw_name)


def test_trustee_previous_actions_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test for games/trust_game.TrustGameTrusteeScenario formatting of previous actions in dataset items.
    scenario_record = {
        "id": "trust-1",
        "scenario": "Trust payoff mid round",
        "description": "Casey considers Alex's investment move.",
        "participants": [
            {"role": "Trustor", "name": "Alex", "profile": "Investor"},
            {"role": "Trustee", "name": "Casey", "profile": "Caretaker"},
        ],
        "trustor_behavior_choices": {
            "trust_none": "Do not invest",
            "trust_low": "Invest a smaller amount",
            "trust_high": "Invest everything",
        },
        "trustee_behavior_choices": {
            "return_none": "Keep all funds",
            "return_medium": "Return half of funds",
            "return_high": "Return full amount",
        },
        "previous_actions_length": 1,
        "previous_trust_level": 0,
    }

    stub_config = {
        "scenario_class": TrustGameTrusteeScenario,
        "payoff_matrix": {},
        "scenarios": [scenario_record],
    }

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _: stub_config,
    )

    config = BenchmarkConfig(
        name="game_theory",
        task_type="Trust_Game_Trustee",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    dataset = GameTheoryDataset(config=config, prompt_wrapper=None, answer_wrapper=None)
    item = dataset[0]["item"]
    assert item.metadata is not None

    previous_actions = item.metadata.get("previous_actions")
    assert previous_actions, "Expected previous actions metadata to be populated"
    assert previous_actions[0][0] == "Alex"
    assert "Invest a smaller amount" in previous_actions[0][1]


def test_trustor_previous_actions_not_bound_method(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test for games/trust_game.TrustGameTrustorScenario: previous_actions renders as a value, not a bound method.
    scenario_record = {
        "id": "trustor-1",
        "scenario": "Trust payoff opening round",
        "description": "Alex decides how much to invest with Casey.",
        "participants": [
            {"role": "Trustor", "name": "Alex", "profile": "Investor"},
            {"role": "Trustee", "name": "Casey", "profile": "Caretaker"},
        ],
        "trustor_behavior_choices": {
            "trust_none": "Do not invest",
            "trust_low": "Invest a smaller amount",
            "trust_high": "Invest everything",
        },
        "trustee_behavior_choices": {
            "return_none": "Keep all funds",
            "return_medium": "Return half of funds",
            "return_high": "Return full amount",
        },
        "previous_actions_length": 0,
    }

    stub_config = {
        "scenario_class": TrustGameTrustorScenario,
        "payoff_matrix": {},
        "scenarios": [scenario_record],
    }

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda _: stub_config,
    )

    config = BenchmarkConfig(
        name="game_theory",
        task_type="Trust_Game_Trustor",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    dataset = GameTheoryDataset(config=config, prompt_wrapper=None, answer_wrapper=None)
    item = dataset[0]["item"]

    assert "<bound method" not in item.input_text
    assert "Previous Actions:" not in item.input_text


def test_ultimatum_previous_actions_is_property_contract() -> None:
    # Test for games/ultimatum_game.py: `previous_actions` must be a property, not a callable method.
    proposer = UltimatumGameProposerScenario(
        scenario="Task split",
        description="Alex proposes a split to Casey.",
        participants=[
            {"role": "Proposer", "name": "Alex", "profile": "Manager"},
            {"role": "Responder", "name": "Casey", "profile": "Member"},
        ],
        proposer_behavior_choices={
            "offer_low": "Offer 10%",
            "offer_medium": "Offer 40%",
            "offer_high": "Offer 50%",
        },
        responder_behavior_choices={"accept": "Accept", "reject": "Reject"},
        previous_actions_length=0,
        payoff_matrix={},
    )
    assert not callable(proposer.previous_actions)
    assert proposer.previous_actions == []
    assert "Previous Actions:" in str(proposer)

    responder = UltimatumGameResponderScenario(
        scenario="Task split",
        description="Alex proposes a split to Casey.",
        participants=[
            {"role": "Proposer", "name": "Alex", "profile": "Manager"},
            {"role": "Responder", "name": "Casey", "profile": "Member"},
        ],
        proposer_behavior_choices={
            "offer_low": "Offer 10%",
            "offer_medium": "Offer 40%",
            "offer_high": "Offer 50%",
        },
        responder_behavior_choices={"accept": "Accept", "reject": "Reject"},
        previous_actions_length=1,
        previous_offer_level=1,
        payoff_matrix={},
    )
    assert not callable(responder.previous_actions)
    assert responder.previous_actions == [("Alex", "Offer 40%")]
    assert "Previous Actions:" in str(responder)


@pytest.mark.parametrize(
    "scenario_cls, scenario_payload",
    [
        (
            TrustGameTrusteeScenario,
            {
                "scenario": "Fallback trust role",
                "description": "Casey considers Alex's investment move.",
                "participants": [
                    {"name": "Alex", "profile": "Investor"},
                    {"name": "Casey", "profile": "Caretaker", "role": "Steward"},
                ],
                "trustor_behavior_choices": {
                    "trust_none": "Do not invest",
                    "trust_low": "Invest a smaller amount",
                    "trust_high": "Invest everything",
                },
                "trustee_behavior_choices": {
                    "return_none": "Keep all funds",
                    "return_medium": "Return half of funds",
                    "return_high": "Return full amount",
                },
                "previous_actions_length": 1,
                "previous_trust_level": 0,
                "payoff_matrix": {},
            },
        ),
        (
            UltimatumGameResponderScenario,
            {
                "scenario": "Fallback ultimatum",
                "description": "Responder evaluates offer.",
                "participants": [
                    {"name": "Alex", "profile": "Investor"},
                    {"name": "Casey", "profile": "Caretaker", "role": "Observer"},
                ],
                "proposer_behavior_choices": {
                    "offer_low": "Offer 10%",
                    "offer_medium": "Offer 40%",
                    "offer_high": "Offer 50%",
                },
                "responder_behavior_choices": {
                    "accept": "Accept",
                    "reject": "Reject",
                },
                "previous_actions_length": 1,
                "previous_offer_level": 0,
                "payoff_matrix": {},
            },
        ),
    ],
)
def test_missing_roles_raise_validation_error(scenario_cls, scenario_payload) -> None:
    with pytest.raises(ValueError, match="Missing participant role"):
        scenario_cls(**scenario_payload)
