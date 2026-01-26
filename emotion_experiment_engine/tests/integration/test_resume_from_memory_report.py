#!/usr/bin/env python3
"""
Tests for continuing an experiment series from a saved MemoryExperimentReport
and for persisting the experiment series config inside the report file.

Responsible file: emotion_experiment_engine/emotion_experiment_series_runner.py

This suite verifies two behaviors:
1) The report JSON persists a `series_config` snapshot of the original config.
2) A new runner can resume from a saved report JSON by executing only the
   experiments that are still marked as pending in the report.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import yaml

from emotion_experiment_engine.emotion_experiment_series_runner import (
    MemoryExperimentSeriesRunner,
)


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def _basic_config(tmpdir: str, num_benchmarks: int = 3) -> dict:
    benches: List[dict] = []
    for i in range(num_benchmarks):
        benches.append(
            {
                "name": f"bench_{i}",
                "task_type": f"task_{i}",
                "sample_limit": 5,
                "enable_auto_truncation": False,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            }
        )

    return {
        "models": ["dummy/model"],
        "emotions": ["anger", "happiness"],
        "intensities": [0.5, 1.0],
        "benchmarks": benches,
        "output_dir": str(Path(tmpdir) / "results"),
        "loading_config": {
            "model_path": "dummy/model",
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


@pytest.mark.integration
def test_report_includes_series_config_and_resume_from_report_executes_pendings():
    tmpdir = tempfile.mkdtemp()
    cfg_path = Path(tmpdir) / "series.yaml"
    config = _basic_config(tmpdir, num_benchmarks=3)
    _write_yaml(cfg_path, config)

    # Phase 1: run once to generate a report and all experiments
    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls_phase1 = []

        def _run_first(benchmark_config, model_name, exp_id):
            calls_phase1.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_first,
        ):
            runner = MemoryExperimentSeriesRunner(str(cfg_path), series_name="t_series", dry_run=False)
            runner.run_experiment_series()

    # Verify report exists and contains a series_config snapshot
    report_path = runner.report.report_file
    assert report_path.exists(), "Report file should exist after run"

    with open(report_path, "r") as f:
        data = json.load(f)

    assert "series_config" in data, "Report should persist the series_config"
    series_cfg = data["series_config"]
    # Minimal sanity checks on captured config snapshot
    for k in ["models", "emotions", "intensities", "benchmarks", "loading_config", "output_dir"]:
        assert k in series_cfg, f"series_config should include '{k}'"

    # Mutate the report to mark some experiments as pending again
    # Choose the first two experiments to resume
    all_exps = list(data["experiments"].values())
    assert len(all_exps) >= 3, "Expected at least 3 experiments from config"
    for exp in all_exps[:2]:
        exp["status"] = "pending"
        exp["end_time"] = None
        exp["time_cost_seconds"] = None
        exp["error"] = None

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)

    # Phase 2: resume from the saved report and ensure only pendings run
    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls_phase2 = []

        def _run_pending_only(benchmark_config, model_name, exp_id):
            calls_phase2.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_pending_only,
        ):
            # Create a new runner that resumes purely from the report path
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="t_series",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    # We should have executed exactly the 2 pending experiments
    assert len(calls_phase2) == 2, f"Expected 2 pending runs, got {len(calls_phase2)}"

    # Final summary should show no pending experiments
    summary = resumed.report.get_summary()
    assert summary["pending"] == 0
