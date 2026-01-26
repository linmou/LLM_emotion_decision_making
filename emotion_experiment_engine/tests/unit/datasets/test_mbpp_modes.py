"""Unit tests for MBPPDataset modes.

Requires MbppPlus jsonl to be available locally. Skips if data is missing.
"""

# Responsible: emotion_experiment_engine/datasets/mbpp.py
# Purpose: Validate default/plus/* loading and EvalPlus-based evaluation

import os
from pathlib import Path

import pytest

from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components
from emotion_experiment_engine.data_models import BenchmarkConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MBPP_PLUS = (
    REPO_ROOT / "../evalplus/datasets/MbppPlus-v0.2.0.jsonl"
).resolve()

PLUS_PATH_STR = os.environ.get("MBPP_PLUS_GZ")
PLUS_PATH = Path(PLUS_PATH_STR).resolve() if PLUS_PATH_STR else DEFAULT_MBPP_PLUS


@pytest.mark.skipif(not PLUS_PATH.exists(), reason="MbppPlus dataset not found")
def test_mbpp_default_mode_uses_base_only():
    cfg = BenchmarkConfig(
        name="mbpp",
        task_type="default",
        data_path=PLUS_PATH,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    prompt_format = type("DummyPF", (), {"build": lambda self, system_prompt, user_messages, **_: "".join(user_messages or [])})()
    _, _, dataset = create_benchmark_components(
        benchmark_name="mbpp",
        task_type="default",
        config=cfg,
        prompt_format=prompt_format,
    )
    record = dataset[0]
    item = record["item"]
    assert item.metadata["mode"] == "default"
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]
    try:
        ok = dataset.evaluate_response(canonical, gt, "default", record["prompt"])
        assert ok == pytest.approx(1.0)
        bad = f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
        assert (
            dataset.evaluate_response(bad, gt, "default", record["prompt"])
            == pytest.approx(0.0)
        )
    except RuntimeError as exc:
        if "tree_sitter_python" in str(exc):
            pytest.skip("tree_sitter_python missing; MBPP evaluation requires it")
        raise


@pytest.mark.skipif(not PLUS_PATH.exists(), reason="MbppPlus dataset not found")
def test_mbpp_plus_mode_runs_full_tests():
    cfg = BenchmarkConfig(
        name="mbpp",
        task_type="plus",
        data_path=PLUS_PATH,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    prompt_format = type("DummyPF", (), {"build": lambda self, system_prompt, user_messages, **_: "".join(user_messages or [])})()
    _, _, dataset = create_benchmark_components(
        benchmark_name="mbpp",
        task_type="plus",
        config=cfg,
        prompt_format=prompt_format,
    )
    record = dataset[0]
    item = record["item"]
    assert item.metadata["mode"] == "plus"
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]
    try:
        ok = dataset.evaluate_response(canonical, gt, "plus", record["prompt"])
        assert ok == pytest.approx(1.0)
        bad = f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
        assert (
            dataset.evaluate_response(bad, gt, "plus", record["prompt"])
            == pytest.approx(0.0)
        )
    except RuntimeError as exc:
        if "tree_sitter_python" in str(exc):
            pytest.skip("tree_sitter_python missing; MBPP evaluation requires it")
        raise


@pytest.mark.skipif(not PLUS_PATH.exists(), reason="MbppPlus dataset not found")
def test_mbpp_star_mode_emits_default_and_plus():
    cfg = BenchmarkConfig(
        name="mbpp",
        task_type="*",
        data_path=PLUS_PATH,
        base_data_dir=None,
        sample_limit=4,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    prompt_format = type("DummyPF", (), {"build": lambda self, system_prompt, user_messages, **_: "".join(user_messages or [])})()
    _, _, dataset = create_benchmark_components(
        benchmark_name="mbpp",
        task_type="*",
        config=cfg,
        prompt_format=prompt_format,
    )
    ids = [dataset[i]["item"].id for i in range(len(dataset))]
    assert any(id_.endswith("::default") for id_ in ids)
    assert any(not id_.endswith("::default") for id_ in ids)
