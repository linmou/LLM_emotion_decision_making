"""
Regression suite validating the synchronous BaseBenchmarkDataset.evaluate_batch implementation.
Replaces the legacy asyncio-specific tests that targeted the old async pipeline.
"""

import math
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem
from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset


class SyntheticDataset(BaseBenchmarkDataset):
    """Lightweight dataset that simulates LLM judging without external services"""

    LLM_EVAL_CONFIG = {"model": "gpt-4o-mini", "temperature": 0.0}

    def __init__(self, items: List[BenchmarkItem], eval_workers: int | None = None):
        config = BenchmarkConfig(
            name="synthetic",
            task_type="qa",
            data_path=None,
            base_data_dir=None,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        self._items_source = items
        super().__init__(
            config=config,
            prompt_wrapper=None,
            max_context_length=None,
            tokenizer=None,
            truncation_strategy="right",
            answer_wrapper=None,
        )
        if eval_workers is not None:
            self.eval_workers = eval_workers

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        return self._items_source

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        if response == "EX":
            raise RuntimeError("forced error")
        return 1.0 if response.strip().lower() == str(ground_truth).strip().lower() else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]


class TestEvaluateBatchComprehensive(unittest.TestCase):
    """Updated regression tests for the thread-pooled evaluate_batch implementation"""

    def setUp(self) -> None:
        self.items = [
            BenchmarkItem(id="1", input_text="Q1", context=None, ground_truth="foo", metadata=None),
            BenchmarkItem(id="2", input_text="Q2", context=None, ground_truth="bar", metadata=None),
            BenchmarkItem(id="3", input_text="Q3", context=None, ground_truth="baz", metadata=None),
        ]
        self.dataset = SyntheticDataset(self.items)

    def test_evaluate_batch_returns_scores_for_each_input(self) -> None:
        responses = ["foo", "wrong", "baz"]
        ground_truths = [item.ground_truth for item in self.items]
        prompts = ["prompt"] * len(self.items)
        task_names = ["qa"] * len(self.items)

        scores = self.dataset.evaluate_batch(responses, ground_truths, task_names, prompts)

        self.assertEqual(scores, [1.0, 0.0, 1.0])
        self.assertEqual(self.dataset._last_eval_errors, [None, None, None])

    def test_evaluate_batch_respects_eval_workers_override(self) -> None:
        dataset = SyntheticDataset(self.items, eval_workers=3)
        responses = ["foo"] * len(self.items)
        ground_truths = [item.ground_truth for item in self.items]
        prompts = ["prompt"] * len(self.items)
        task_names = ["qa"] * len(self.items)

        created: List[int] = []

        def factory(*args, **kwargs):
            executor = RecordingExecutor(*args, **kwargs)
            created.append(executor.max_workers)
            return executor

        with patch(
            "concurrent.futures.ThreadPoolExecutor",
            side_effect=factory,
        ):
            dataset.evaluate_batch(responses, ground_truths, task_names, prompts)

        self.assertEqual(created, [3])

    def test_evaluate_batch_captures_exceptions(self) -> None:
        responses = ["foo", "EX", "bad"]
        ground_truths = [item.ground_truth for item in self.items]
        prompts = ["prompt"] * len(self.items)
        task_names = ["qa"] * len(self.items)

        with patch(
            "concurrent.futures.ThreadPoolExecutor",
            side_effect=lambda *args, **kwargs: RecordingExecutor(*args, **kwargs),
        ):
            scores = self.dataset.evaluate_batch(responses, ground_truths, task_names, prompts)

        self.assertEqual(scores[0], 1.0)
        self.assertTrue(math.isnan(scores[1]))
        self.assertEqual(scores[2], 0.0)
        self.assertEqual(self.dataset._last_eval_errors[1], "forced error")


class RecordingExecutor:
    """Utility class reused across tests to emulate ThreadPoolExecutor"""

    def __init__(self, *args, **kwargs):
        self.max_workers = kwargs.get("max_workers")
        if self.max_workers is None and args:
            self.max_workers = args[0]

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def submit(self, fn, *args, **kwargs):
        try:
            value = fn(*args, **kwargs)
            return ImmediateFuture(value=value)
        except Exception as exc:  # pragma: no cover - delegated to caller handling
            return ImmediateFuture(error=exc)

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


class ImmediateFuture:
    def __init__(self, value: Any = None, error: Exception | None = None):
        self._value = value
        self._error = error

    def result(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._value


if __name__ == "__main__":
    unittest.main()
