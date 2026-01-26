"""
Tests for the synchronous LLM evaluation helpers and dataset batch evaluation pipeline.
These tests codify the September 2025 "LLM Evaluation System Update" contract.
"""

import math
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from .. import evaluation_utils
from ..data_models import BenchmarkConfig, BenchmarkItem
from ..datasets.base import BaseBenchmarkDataset


class DummyDataset(BaseBenchmarkDataset):
    """Minimal dataset that exercises BaseBenchmarkDataset.evaluate_batch"""

    LLM_EVAL_CONFIG = {"model": "gpt-4o-mini", "temperature": 0.0}

    def __init__(self, eval_workers: int | None = None):
        config = BenchmarkConfig(
            name="dummy",
            task_type="unit",
            data_path=None,
            base_data_dir=None,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
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
        return [
            BenchmarkItem(
                id="1",
                input_text="Q1",
                context=None,
                ground_truth="GT1",
                metadata=None,
            )
        ]

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        if response == "boom":
            raise RuntimeError("evaluation error")
        return 1.0 if response == ground_truth else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]


class TestLLMEvaluateResponse(unittest.TestCase):
    """Unit tests for evaluation_utils.llm_evaluate_response"""

    def setUp(self) -> None:
        evaluation_utils._global_client = None  # type: ignore[attr-defined]
        if hasattr(evaluation_utils, "_global_gemini_model"):
            evaluation_utils._global_gemini_model = None  # type: ignore[attr-defined]
        if hasattr(evaluation_utils, "_global_gemini_model_name"):
            evaluation_utils._global_gemini_model_name = None  # type: ignore[attr-defined]

    @patch("emotion_experiment_engine.evaluation_utils.openai.OpenAI")
    def test_llm_evaluate_response_parses_json(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"score": 0.75, "explanation": "Looks correct"}'
                    )
                )
            ]
        )

        result = evaluation_utils.llm_evaluate_response(
            system_prompt="Judge",
            query="Evaluate this answer",
            llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.1},
        )

        self.assertEqual(result["score"], 0.75)
        self.assertIn("explanation", result)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_kwargs["temperature"], 0.1)
        self.assertEqual(
            call_kwargs["response_format"], {"type": "json_object"}
        )

    @patch("emotion_experiment_engine.evaluation_utils.openai.OpenAI")
    def test_llm_evaluate_response_reuses_global_client(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="{}"))]
        )

        evaluation_utils.llm_evaluate_response(
            system_prompt="Judge",
            query="First",
            llm_eval_config={"model": "gpt-4o-mini"},
        )
        evaluation_utils.llm_evaluate_response(
            system_prompt="Judge",
            query="Second",
            llm_eval_config={"model": "gpt-4o-mini"},
        )

        mock_openai.assert_called_once()
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("emotion_experiment_engine.evaluation_utils.openai.OpenAI")
    def test_llm_evaluate_response_raises_runtime_error(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ValueError("boom")

        with self.assertRaises(RuntimeError) as exc:
            evaluation_utils.llm_evaluate_response(
                system_prompt="Judge",
                query="bad",
                llm_eval_config={"model": "gpt-4o-mini"},
            )
        self.assertIn("LLM evaluation failed", str(exc.exception))

    def test_llm_evaluate_response_supports_gemini_client(self) -> None:
        # Stub google.generativeai module to avoid real dependency calls
        import sys
        import types

        fake_genai = types.SimpleNamespace()
        model_instance = MagicMock()
        model_instance.generate_content.return_value = MagicMock(
            text='{"score": 0.5}'
        )
        fake_genai.GenerativeModel = MagicMock(return_value=model_instance)
        fake_genai.configure = MagicMock()

        sys.modules["google"] = types.ModuleType("google")
        sys.modules["google.generativeai"] = fake_genai

        with patch("emotion_experiment_engine.evaluation_utils.openai.OpenAI") as mock_openai:
            result = evaluation_utils.llm_evaluate_response(
                system_prompt="Judge",
                query="Evaluate",
                llm_eval_config={"client": "gemini", "model": "gemini-test"},
            )

        self.assertAlmostEqual(result["score"], 0.5)
        fake_genai.configure.assert_called_once()
        fake_genai.GenerativeModel.assert_called_once_with("gemini-test")
        model_instance.generate_content.assert_called_once()
        mock_openai.assert_not_called()

    def test_llm_evaluate_response_strips_code_fences_from_llm_output(self) -> None:
        import sys
        import types

        fake_genai = types.SimpleNamespace()
        model_instance = MagicMock()
        model_instance.generate_content.return_value = MagicMock(
            text="```json\\n{\"score\": 0.9}\\n```"
        )
        fake_genai.GenerativeModel = MagicMock(return_value=model_instance)
        fake_genai.configure = MagicMock()

        sys.modules["google"] = types.ModuleType("google")
        sys.modules["google.generativeai"] = fake_genai

        result = evaluation_utils.llm_evaluate_response(
            system_prompt="Judge",
            query="Evaluate",
            llm_eval_config={"client": "gemini", "model": "gemini-test"},
        )

        self.assertAlmostEqual(result["score"], 0.9)


class ImmediateFuture:
    """Simple future used by RecordingExecutor to emulate thread pool results"""

    def __init__(self, value: Any = None, error: Exception | None = None):
        self._value = value
        self._error = error

    def result(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._value


class RecordingExecutor:
    """ThreadPoolExecutor stand-in that records requested max_workers"""

    def __init__(self, *args, **kwargs):
        max_workers = kwargs.get("max_workers")
        if max_workers is None and args:
            max_workers = args[0]
        self.max_workers = max_workers
        self.submitted: List[ImmediateFuture] = []

    def __enter__(self):
        return self

    def submit(self, fn, *args, **kwargs):
        try:
            value = fn(*args, **kwargs)
            future = ImmediateFuture(value=value)
        except Exception as exc:  # pragma: no cover - delegated to caller handling
            future = ImmediateFuture(error=exc)
        self.submitted.append(future)
        return future

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDatasetEvaluateBatch(unittest.TestCase):
    """Tests for BaseBenchmarkDataset.evaluate_batch"""

    def test_default_worker_cap_is_64(self) -> None:
        dataset = DummyDataset()
        count = 128
        responses = [f"gt{i}" for i in range(count)]
        ground_truths = list(responses)
        task_names = ["unit"] * count
        prompts = ["prompt"] * count

        created: List[RecordingExecutor] = []

        def factory(*args, **kwargs):
            executor = RecordingExecutor(*args, **kwargs)
            created.append(executor)
            return executor

        with patch(
            "concurrent.futures.ThreadPoolExecutor",
            side_effect=factory,
        ):
            scores = dataset.evaluate_batch(responses, ground_truths, task_names, prompts)

        self.assertEqual(scores, [1.0] * count)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].max_workers, 64)

    def test_evaluate_batch_captures_errors(self) -> None:
        dataset = DummyDataset()
        responses = ["GT1", "boom", "other"]
        ground_truths = ["GT1", "GT1", "GT1"]
        task_names = ["unit", "unit", "unit"]
        prompts = ["p1", "p2", "p3"]

        with patch(
            "concurrent.futures.ThreadPoolExecutor",
            side_effect=lambda *args, **kwargs: RecordingExecutor(*args, **kwargs),
        ):
            scores = dataset.evaluate_batch(responses, ground_truths, task_names, prompts)

        self.assertEqual(len(scores), 3)
        self.assertTrue(math.isnan(scores[1]))
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[2], 0.0)
        self.assertEqual(dataset._last_eval_errors[1], "evaluation error")


if __name__ == "__main__":
    unittest.main()
