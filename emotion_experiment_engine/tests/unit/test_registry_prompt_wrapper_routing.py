"""
Unit tests for benchmark_component_registry prompt wrapper routing.

Covers representative benchmark/task combos and asserts that the registry
instantiates the expected prompt wrapper class directly (without calling the
fallback factory) and wires the partial correctly.

Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: Validate direct class-based routing in BENCHMARK_SPECS.
"""

import unittest
from unittest.mock import patch
from pathlib import Path

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig

# Target wrapper classes to validate types
from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
from emotion_experiment_engine.truthfulqa_prompt_wrapper import TruthfulQAPromptWrapper
from emotion_experiment_engine.memory_prompt_wrapper import (
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
)
from emotion_experiment_engine.game_prompt_wrapper import (
    GameCompletionOptionIdPromptWrapper,
    GameDecisionPromptWrapper,
)


class DummyPromptFormat:
    """Minimal prompt format with a build() API used by wrappers in tests."""

    def build(self, system_prompt: str, user_messages, **kwargs) -> str:
        return f"{system_prompt}\n{user_messages}"


def _extract_wrapper_instance_from_partial(prompt_wrapper_partial):
    """Helper to get the underlying wrapper instance from the partial."""
    bound_call = prompt_wrapper_partial.func  # bound method __call__
    return getattr(bound_call, "__self__", None)


class TestRegistryPromptWrapperRouting(unittest.TestCase):
    """Red phase: expect failures until registry sets explicit classes."""

    def setUp(self):
        self.prompt_format = DummyPromptFormat()
        # Avoid dataset IO by mocking dataset creation
        self._dataset_patcher = patch(
            "emotion_experiment_engine.benchmark_component_registry.create_dataset_from_config",
            return_value=object(),
        )
        self._dataset_patcher.start()

    def tearDown(self):
        self._dataset_patcher.stop()

    def _mk_cfg(self, name: str, task_type: str) -> BenchmarkConfig:
        # Construct minimal valid config; paths only used by datasets, not invoked here
        return BenchmarkConfig(
            name=name,
            task_type=task_type,
            data_path=Path("/dev/null"),
            base_data_dir="/tmp",  # not used in these tests
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_mtbench101_cm_uses_mtbench_wrapper_direct(self, mock_factory):
        cfg = self._mk_cfg("mtbench101", "CM")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, MTBench101PromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_truthfulqa_uses_truthfulqa_wrapper_direct(self, mock_factory):
        # TruthfulQA uses mc1/mc2 task types
        import tempfile, json

        # Create a minimal valid TruthfulQA JSONL file
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        Path(path).write_text(json.dumps({
            "question": "What is 2+2?",
            "options": ["3", "4"],
            "answers": ["4"]
        }) + "\n", encoding="utf-8")

        cfg = self._mk_cfg("truthfulqa", "mc1")
        cfg.data_path = Path(path)

        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, TruthfulQAPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_infinitebench_passkey_uses_passkey_wrapper_direct(self, mock_factory):
        cfg = self._mk_cfg("infinitebench", "passkey")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, PasskeyPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_longbench_retrieval_uses_retrieval_wrapper_direct(self, mock_factory):
        cfg = self._mk_cfg("longbench", "passage_retrieval_en")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, LongbenchRetrievalPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_longbench_qa_uses_longcontextqa_wrapper_direct(self, mock_factory):
        cfg = self._mk_cfg("longbench", "hotpotqa")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, LongContextQAPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_locomo_uses_conversationalqa_wrapper_direct(self, mock_factory):
        cfg = self._mk_cfg("locomo", "locomo")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_game_theory_decision_uses_decision_wrapper(self, mock_factory):
        cfg = self._mk_cfg("game_theory_decision", "Prisoners_Dilemma")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, GameDecisionPromptWrapper)
        mock_factory.assert_not_called()

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_game_theory_completion_option_id_uses_completion_wrapper(self, mock_factory):
        cfg = self._mk_cfg("game_theory_completion_option_id", "Prisoners_Dilemma")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, GameCompletionOptionIdPromptWrapper)
        mock_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
