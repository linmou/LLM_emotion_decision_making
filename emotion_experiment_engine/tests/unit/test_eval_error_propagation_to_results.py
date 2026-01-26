"""
Unit test: ensure evaluation errors from dataset.evaluate_batch are propagated
into EmotionExperiment results and persisted by _save_results, and that
emotion_experiment_engine.evaluate_saved reuses dataset-level batch scoring
during deferred replay.

Covers: emotion_experiment_engine/experiment.py
- _post_process_batch should read dataset._last_eval_errors and set ResultRecord.error
- _save_results should include an 'error' column in detailed_results.csv/DataFrame
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock, patch
import importlib
import sys

import math
import pandas as pd
import pytest

from emotion_experiment_engine import evaluate_saved
from emotion_experiment_engine.data_models import BenchmarkItem
from emotion_experiment_engine.tests.test_utils import (
    create_mock_experiment_config,
)


class DummyDataset:
    """Minimal dataset stub exposing evaluate_batch and collate_fn-like expectations."""

    def __init__(self) -> None:
        self._last_eval_errors: List[Any] = []

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        # Simulate one error and one success
        self._last_eval_errors = ["mock eval error", None]
        # Return NaN for the errored item, and 1.0 for the successful item
        return [float("nan"), 1.0]


class _RecordingDataset:
    """Stub dataset that records which evaluation API is used."""

    def __init__(self) -> None:
        self.batch_calls: List[Dict[str, Sequence[str]]] = []
        self.response_calls: List[Dict[str, str]] = []
        self._last_eval_errors: List[str | None] = []

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        self.batch_calls.append(
            {
                "responses": tuple(responses),
                "ground_truths": tuple(map(str, ground_truths)),
                "task_names": tuple(task_names),
                "prompts": tuple(prompts),
            }
        )
        self._last_eval_errors = ["err-one", None]
        return [0.25, 0.75]

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        self.response_calls.append(
            {
                "response": response,
                "ground_truth": str(ground_truth),
                "task_name": task_name,
                "prompt": prompt,
            }
        )
        return 1.0


@pytest.fixture()
def deferred_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "model_path": "dummy",
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmark": {
            "name": "dummy",
            "task_type": "dummy_task",
            "data_path": None,
            "base_data_dir": None,
            "sample_limit": None,
            "augmentation_config": None,
            "enable_auto_truncation": False,
            "truncation_strategy": "right",
            "preserve_ratio": 0.8,
            "llm_eval_config": None,
        },
        "output_dir": str(run),
        "batch_size": 2,
        "generation_config": None,
        "repe_eng_config": None,
        "max_evaluation_workers": 4,
        "pipeline_queue_size": 2,
        "defer_evaluation": True,
    }
    (run / "experiment_config.json").write_text(json.dumps(manifest), encoding="utf-8")

    raw_rows = [
        {
            "emotion": "anger",
            "intensity": 1.0,
            "item_id": "item-1",
            "task_name": "dummy_task",
            "prompt": "prompt-1",
            "response": "response-1",
            "ground_truth": "gt-1",
            "repeat_id": 0,
            "metadata": None,
        },
        {
            "emotion": "anger",
            "intensity": 1.0,
            "item_id": "item-2",
            "task_name": "dummy_task",
            "prompt": "prompt-2",
            "response": "response-2",
            "ground_truth": "gt-2",
            "repeat_id": 0,
            "metadata": None,
        },
    ]
    (run / "raw_results.json").write_text(json.dumps(raw_rows), encoding="utf-8")
    return run


def test_evaluate_saved_prefers_batch_api(monkeypatch: pytest.MonkeyPatch, deferred_run_dir: Path) -> None:
    dataset = _RecordingDataset()

    def fake_factory(*_: Any, **__: Any) -> _RecordingDataset:
        return dataset

    def fake_save_results(self: Any, records: List[Any]) -> pd.DataFrame:
        df_rows = [{"item_id": r.item_id, "score": r.score, "error": r.error} for r in records]
        return pd.DataFrame(df_rows)

    monkeypatch.setattr(evaluate_saved, "create_dataset_from_config", fake_factory)
    monkeypatch.setattr(evaluate_saved.EmotionExperiment, "_save_results", fake_save_results, raising=False)

    df = evaluate_saved.evaluate_saved_run(deferred_run_dir, max_workers=3)

    # Batch path should be used exactly once
    assert len(dataset.batch_calls) == 1
    assert dataset.response_calls == []

    # Scores and errors should reflect the batch outputs
    assert list(df["score"]) == [0.25, 0.75]
    assert list(df["error"]) == ["err-one", None]

    stored = json.loads((deferred_run_dir / "raw_results.json").read_text(encoding="utf-8"))
    assert [row.get("error") for row in stored] == ["err-one", None]


def test_evaluate_saved_logs_progress_and_errors(
    monkeypatch: pytest.MonkeyPatch, deferred_run_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class _ChunkedDataset(_RecordingDataset):
        def __init__(self) -> None:
            super().__init__()
            self._call = 0
            self.offline_eval_chunk_size = 1

        def evaluate_batch(
            self,
            responses: List[str],
            ground_truths: List[Any],
            task_names: List[str],
            prompts: List[str],
        ) -> List[float]:
            if self._call == 0:
                self._last_eval_errors = ["chunk-error"]
                result = [float("nan")]
            else:
                self._last_eval_errors = [None]
                result = [0.5]
            self._call += 1
            self.batch_calls.append(
                {
                    "responses": tuple(responses),
                    "ground_truths": tuple(map(str, ground_truths)),
                    "task_names": tuple(task_names),
                    "prompts": tuple(prompts),
                }
            )
            return result

    dataset = _ChunkedDataset()

    def fake_factory(*_: Any, **__: Any) -> _ChunkedDataset:
        return dataset

    def fake_save_results(self: Any, records: List[Any]) -> pd.DataFrame:
        df_rows = [{"item_id": r.item_id, "score": r.score, "error": r.error} for r in records]
        return pd.DataFrame(df_rows)

    monkeypatch.setattr(evaluate_saved, "create_dataset_from_config", fake_factory)
    monkeypatch.setattr(evaluate_saved.EmotionExperiment, "_save_results", fake_save_results, raising=False)

    with caplog.at_level(logging.INFO):
        df = evaluate_saved.evaluate_saved_run(deferred_run_dir, max_workers=2)

    # Ensure progress logs were emitted for each chunk and summary includes error count
    progress_logs = [rec.message for rec in caplog.records if "Deferred evaluation progress" in rec.message]
    assert len(dataset.batch_calls) == 2
    assert any(msg.startswith("[") and "]" in msg for msg in progress_logs)
    assert any("1/2" in msg for msg in progress_logs)
    assert any("2/2" in msg for msg in progress_logs)
    assert any(
        rec.message.startswith("[") and "Deferred evaluation completed" in rec.message and "errors=1" in rec.message
        for rec in caplog.records
    )

    # DataFrame reflects chunked scoring
    assert list(df["error"]) == ["chunk-error", None]


def test_eval_error_is_propagated_to_results(tmp_path: Path):
    # Arrange minimal experiment
    # Stub out the repe_eng_config import to avoid pydantic dependency chain
    fake_repe_module = type(sys)("neuro_manipulation.configs.experiment_config")
    def _fake_get_repe_eng_config(*args, **kwargs):
        return {}
    fake_repe_module.get_repe_eng_config = _fake_get_repe_eng_config  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.configs.experiment_config"] = fake_repe_module

    # Also stub the utils module to avoid importing torch and heavy deps
    fake_utils_module = type(sys)("neuro_manipulation.utils")
    def _fake_load_tokenizer_only(*args, **kwargs):
        return (MagicMock(), None)
    fake_utils_module.load_tokenizer_only = _fake_load_tokenizer_only  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.utils"] = fake_utils_module

    # Stub other heavy neuro_manipulation modules to avoid torch deps on import
    fake_mld_module = type(sys)("neuro_manipulation.model_layer_detector")
    class _FakeMLD:
        @staticmethod
        def num_layers(model=None):
            return 4
    fake_mld_module.ModelLayerDetector = _FakeMLD  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.model_layer_detector"] = fake_mld_module

    fake_mu_module = type(sys)("neuro_manipulation.model_utils")
    def _fake_setup_model_and_tokenizer(*args, **kwargs):
        return (MagicMock(), MagicMock(), "chat", MagicMock())
    def _fake_load_emotion_readers(*args, **kwargs):
        return {"anger": MagicMock(), "neutral": MagicMock()}
    fake_mu_module.setup_model_and_tokenizer = _fake_setup_model_and_tokenizer  # type: ignore[attr-defined]
    fake_mu_module.load_emotion_readers = _fake_load_emotion_readers  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.model_utils"] = fake_mu_module

    fake_pipelines_module = type(sys)("neuro_manipulation.repe.pipelines")
    def _fake_get_pipeline(*args, **kwargs):
        return MagicMock()
    fake_pipelines_module.get_pipeline = _fake_get_pipeline  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.repe.pipelines"] = fake_pipelines_module

    # Stub the benchmark registry to avoid importing real datasets (which require torch)
    fake_registry_module = type(sys)("emotion_experiment_engine.benchmark_component_registry")
    def _fake_create_benchmark_components(**kwargs):
        # Return (prompt_wrapper, answer_wrapper, dataset)
        return (MagicMock(), MagicMock(), [1, 2])
    fake_registry_module.create_benchmark_components = _fake_create_benchmark_components  # type: ignore[attr-defined]
    sys.modules["emotion_experiment_engine.benchmark_component_registry"] = fake_registry_module

    # Import after stubbing to avoid import-time errors
    experiment_module = importlib.import_module("emotion_experiment_engine.experiment")
    EmotionExperiment = experiment_module.EmotionExperiment

    config = create_mock_experiment_config("passkey", 2)
    config.output_dir = str(tmp_path)

    exp = EmotionExperiment(config, dry_run=True)
    exp.is_vllm = False  # ensure non-vLLM branch in post-process
    exp.dataset = DummyDataset()  # inject our dummy dataset
    exp.cur_emotion = "anger"
    exp.cur_intensity = 1.0

    # Build a minimal batch with two items
    batch = {
        "prompts": ["P1", "P2"],
        "items": [
            BenchmarkItem(id="a", input_text="q1", context=None, ground_truth="gt1", metadata=None),
            BenchmarkItem(id="b", input_text="q2", context=None, ground_truth="gt2", metadata=None),
        ],
        "ground_truths": ["gt1", "gt2"],
    }
    control_outputs = [
        [{"generated_text": "resp1"}],
        [{"generated_text": "resp2"}],
    ]

    # Act: post-process batch to produce ResultRecord entries
    results = exp._post_process_batch(batch, control_outputs, batch_idx=0)

    # Assert: first item has error and NaN score; second has no error and score 1.0
    assert len(results) == 2
    assert results[0].error == "mock eval error"
    assert math.isnan(results[0].score if results[0].score is not None else float("nan"))
    assert results[1].error is None
    assert results[1].score == 1.0

    # Act: save results and load DataFrame
    df = exp._save_results(results)

    # Assert: error column exists and values match
    assert isinstance(df, pd.DataFrame)
    assert "error" in df.columns
    row0 = df[df["item_id"] == "a"].iloc[0]
    row1 = df[df["item_id"] == "b"].iloc[0]
    assert row0["error"] == "mock eval error"
    assert pd.isna(row1["error"]) or row1["error"] in (None, "")
