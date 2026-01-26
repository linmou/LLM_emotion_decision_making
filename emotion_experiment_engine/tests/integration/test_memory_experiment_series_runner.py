#!/usr/bin/env python3
"""
Comprehensive test suite for emotion_experiment_series_runner.py

Covers:
- BenchmarkConfig creation and pattern expansion behavior
- Error handling paths in the series runner
- Import coverage and integration behavior with mocks

Uses package imports; skips tests if runner unavailable in env.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from emotion_experiment_engine.data_models import BenchmarkConfig

try:
    from emotion_experiment_engine.emotion_experiment_series_runner import (
        MemoryExperimentSeriesRunner,
        ExperimentStatus,
    )
    RUNNER_AVAILABLE = True
except Exception as e:
    MemoryExperimentSeriesRunner = None  # type: ignore
    RUNNER_AVAILABLE = False
    print(f"Warning: MemoryExperimentSeriesRunner unavailable: {e}")


class TestMemoryExperimentSeriesRunner(unittest.TestCase):
    """Comprehensive test suite for MemoryExperimentSeriesRunner"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"

        self.test_config = {
            "models": ["test_model_1", "test_model_2"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [
                {
                    "name": "test_benchmark_1",
                    "task_type": "test_task",
                    "sample_limit": 5,
                    "augmentation_config": None,
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8,
                    "llm_eval_config": {"model": "gpt-4o-mini", "temperature": 0.1},
                },
                {
                    "name": "test_benchmark_2",
                    "task_type": "another_task",
                    "sample_limit": 5,
                    "augmentation_config": None,
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8,
                    "llm_eval_config": {"model": "gpt-4o-mini", "temperature": 0.1},
                },
            ],
            "output_dir": str(Path(self.temp_dir) / "results"),
            "base_data_dir": str(Path(self.temp_dir) / "data"),
            "loading_config": {
                "model_path": "/data/models/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.8,
                "tensor_parallel_size": 1,
                "max_model_len": 1024,
                "enforce_eager": True,
                "quantization": None,
                "trust_remote_code": True,
                "dtype": "float16",
                "seed": 42,
                "disable_custom_all_reduce": False,
                "additional_vllm_kwargs": {},
            },
        }

        with open(self.config_file, "w") as f:
            yaml.dump(self.test_config, f)

        self.minimal_config = {
            "models": ["/test/model"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [],
            "loading_config": self.test_config["loading_config"],
        }

        if RUNNER_AVAILABLE:
            # Runner expects a config path; create a minimal YAML
            self.minimal_cfg_file = Path(self.temp_dir) / "minimal.yaml"
            with open(self.minimal_cfg_file, "w") as f:
                yaml.dump(self.minimal_config, f)
            self.runner = MemoryExperimentSeriesRunner(str(self.minimal_cfg_file))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_emotion_experiment_series_runner_import(self):
        try:
            import emotion_experiment_engine.emotion_experiment_series_runner  # noqa: F401
            self.assertTrue(True)
        except ImportError as e:
            if "vllm" in str(e).lower():
                self.skipTest("Skipping due to vLLM dependency")
            else:
                self.fail(f"Import failed: {e}")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_runner_initialization(self):
        self.assertIsNotNone(self.runner)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_literal_task_type(self):
        benchmarks = [{"name": "test_bench", "task_type": "literal_task", "sample_limit": 100}]
        expanded = self.runner.expand_benchmark_configs(benchmarks)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["task_type"], "literal_task")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_pattern_task_type_now_works(self):
        benchmarks = [
            {
                "name": "test_bench",
                "task_type": ".*test.*",
                "sample_limit": 100,
                "base_data_dir": self.temp_dir,
                "augmentation_config": None,
                "enable_auto_truncation": False,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            }
        ]
        for task in ["test_task1", "test_task2"]:
            (Path(self.temp_dir) / f"test_bench_{task}.jsonl").write_text("{}\n")

        expanded = self.runner.expand_benchmark_configs(benchmarks)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0]["task_type"], "test_task1")
        self.assertEqual(expanded[1]["task_type"], "test_task2")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_is_pattern_task_type_detection(self):
        self.assertTrue(self.runner._is_pattern_task_type(".*"))
        self.assertTrue(self.runner._is_pattern_task_type("test.*"))
        self.assertTrue(self.runner._is_pattern_task_type(".*qa.*"))
        self.assertTrue(self.runner._is_pattern_task_type("[abc]+"))
        self.assertFalse(self.runner._is_pattern_task_type("literal_task"))
        self.assertFalse(self.runner._is_pattern_task_type("passkey"))
        self.assertFalse(self.runner._is_pattern_task_type("narrativeqa"))
        self.assertFalse(self.runner._is_pattern_task_type(""))

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_experiment_series_continues_after_failure(self):
        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence"
        ) as mock_check_model, patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment"
        ) as mock_run_single:
            mock_check_model.return_value = "/resolved/model"
            attempted, failed, succeeded = [], [], []

            def _run(benchmark_config, model_name, exp_id):
                attempted.append(exp_id)
                status = ExperimentStatus.COMPLETED
                if exp_id == "test_benchmark_1_test_task_test_model_1":
                    failed.append(exp_id)
                    status = ExperimentStatus.FAILED
                    result = False
                else:
                    succeeded.append(exp_id)
                    result = True

                runner.report.update_experiment(exp_id, status=status)
                return result

            mock_run_single.side_effect = _run
            runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=False)
            runner.run_experiment_series()

            expected_total = 4
            self.assertEqual(len(attempted), expected_total)
            self.assertEqual(len(failed), 1)
            self.assertEqual(len(succeeded), expected_total - 1)
            summary = runner.report.get_summary()
            self.assertEqual(summary["total"], expected_total)
            self.assertEqual(summary["failed"], 1)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_resolved_model_path_passed_to_run_single_experiment(self):
        resolved_path = "/mirror/facebook/MobileLLM-R1-950M"

        temp_config_path = Path(self.temp_dir) / "single_run.yaml"
        single_config = {
            "models": ["facebook/MobileLLM-R1-950M"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [
                {
                    "name": "trustllm",
                    "task_type": "stereotype_recognition",
                    "data_path": str(Path(self.temp_dir) / "dummy.json"),
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8,
                }
            ],
            "output_dir": str(Path(self.temp_dir) / "results_single"),
            "loading_config": self.test_config["loading_config"],
        }

        with open(temp_config_path, "w") as f:
            yaml.dump(single_config, f)

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
            return_value=resolved_path,
        ) as mock_check_model:
            captured_model_name = {}

            def _capture_run(benchmark_config, model_name, exp_id):
                captured_model_name["value"] = model_name
                return True

            with patch(
                "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
                side_effect=_capture_run,
            ):
                runner = MemoryExperimentSeriesRunner(str(temp_config_path), dry_run=False)
                runner.run_experiment_series()

        mock_check_model.assert_called()
        self.assertEqual(captured_model_name.get("value"), resolved_path)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_dry_run_errors_bubble_up(self):
        # Test for emotion_experiment_engine.emotion_experiment_series_runner.dry_run_series error propagation when setup fails.
        runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=True)

        with patch.object(runner, "setup_experiment", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "Config 1 failed"):
                runner.dry_run_series()

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_run_single_experiment_releases_resources_on_success(self):
        runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=False)
        benchmark = self.test_config["benchmarks"][0]

        with patch(
            "emotion_experiment_engine.experiment.EmotionExperiment",
            create=True,
        ) as mock_experiment_cls, patch.object(
            runner, "_clean_cuda_memory"
        ) as mock_clean_cuda:
            mock_experiment = mock_experiment_cls.return_value
            mock_experiment.run_experiment.return_value = None
            mock_experiment.close = MagicMock(name="close")

            result = runner.run_single_experiment(benchmark, "/fake/model", "exp-success")

            self.assertTrue(result)
            mock_experiment.run_experiment.assert_called_once()
            mock_experiment.close.assert_called_once()
            mock_clean_cuda.assert_called_once()

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_run_single_experiment_releases_resources_on_failure(self):
        runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=False)
        benchmark = self.test_config["benchmarks"][0]

        with patch(
            "emotion_experiment_engine.experiment.EmotionExperiment",
            create=True,
        ) as mock_experiment_cls, patch.object(
            runner, "_clean_cuda_memory"
        ) as mock_clean_cuda:
            mock_experiment = mock_experiment_cls.return_value
            mock_experiment.run_experiment.side_effect = RuntimeError("boom")
            mock_experiment.close = MagicMock(name="close")

            result = runner.run_single_experiment(benchmark, "/fake/model", "exp-fail")

            self.assertFalse(result)
            mock_experiment.run_experiment.assert_called_once()
            mock_experiment.close.assert_called_once()
            mock_clean_cuda.assert_called_once()

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_check_model_existence_abs_path_nonexistent(self):
        # Ensure a non-existent absolute path returns None (clear failure)
        bogus = os.path.join(self.temp_dir, "this_path_should_not_exist_123456")
        if os.path.exists(bogus):
            # Extremely unlikely, but ensure non-existence
            import shutil
            shutil.rmtree(bogus, ignore_errors=True)
        self.assertFalse(os.path.exists(bogus))
        resolved = self.runner._check_model_existence(bogus)
        self.assertIsNone(resolved)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_check_model_existence_abs_path_existing_dir(self):
        # Existing absolute path should resolve to its realpath and be returned
        existing = os.path.join(self.temp_dir, "local_model_dir")
        os.makedirs(existing, exist_ok=True)
        # Minimal HF folder requirement
        with open(os.path.join(existing, "config.json"), "w") as f:
            f.write("{}")
        resolved = self.runner._check_model_existence(existing)
        self.assertEqual(resolved, os.path.realpath(existing))

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_check_model_existence_abs_path_existing_dir_without_config(self):
        # Existing absolute path without config.json should be rejected
        existing = os.path.join(self.temp_dir, "bad_local_model_dir")
        os.makedirs(existing, exist_ok=True)
        # Ensure config.json does not exist
        cfg = os.path.join(existing, "config.json")
        if os.path.exists(cfg):
            os.remove(cfg)
        resolved = self.runner._check_model_existence(existing)
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
