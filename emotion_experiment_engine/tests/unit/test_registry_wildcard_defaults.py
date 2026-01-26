"""
Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: Ensure wildcard defaults allow concise BENCHMARK_SPECS by mapping
         all tasks under a benchmark (e.g., mtbench101) to the same spec
         without enumerating each task, and that the direct wrapper path
         is used (factory not called).
"""

import unittest
from unittest.mock import patch
from pathlib import Path

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper


class DummyPromptFormat:
    def build(self, system_prompt: str, user_messages, **kwargs) -> str:
        return f"{system_prompt}\n{user_messages}"


def _extract_wrapper_instance_from_partial(prompt_wrapper_partial):
    bound_call = prompt_wrapper_partial.func
    return getattr(bound_call, "__self__", None)


class TestWildcardDefaults(unittest.TestCase):
    """Red phase: expect failure until wildcard fallback is implemented."""

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

    @patch("emotion_experiment_engine.benchmark_component_registry.get_benchmark_prompt_wrapper")
    def test_mtbench101_unknown_task_uses_default_wrapper(self, mock_factory):
        # Use a task type that is not explicitly listed in the registry
        cfg = BenchmarkConfig(
            name="mtbench101",
            task_type="ZZ",
            data_path=Path("/dev/null"),
            base_data_dir="/tmp",
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )

        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, MTBench101PromptWrapper)
        mock_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

