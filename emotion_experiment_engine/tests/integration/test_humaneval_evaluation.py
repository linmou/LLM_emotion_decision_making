"""
Integration tests for HumanEvalDataset evaluation paths.

Covers default (original HumanEval) and plus (EvalPlus) modes when datasets are
available locally. Tests skip gracefully if environment variables/paths are not
set to avoid network calls.
"""

import gzip
import json
import os
from pathlib import Path

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import create_dataset_from_config

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PATH = (REPO_ROOT / "../evalplus/datasets/HumanEval.jsonl").resolve()
DEFAULT_PLUS = (
    REPO_ROOT / "../evalplus/datasets/HumanEvalPlus-v0.1.10.jsonl"
).resolve()

PLUS_PATH_ENV = os.environ.get("HUMANEVAL_PLUS_GZ")
PLUS_PATH = Path(PLUS_PATH_ENV).resolve() if PLUS_PATH_ENV else DEFAULT_PLUS


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="HumanEval original file missing")
def test_humaneval_default_canonical_vs_bad():
    cfg = BenchmarkConfig(
        name="humaneval",
        task_type="default",
        data_path=DEFAULT_PATH,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    dataset = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get("question", ""))
    record = dataset[0]
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]

    ok = dataset.evaluate_response(canonical, gt, "default", record["prompt"])
    assert ok == pytest.approx(1.0)

    bad = f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
    bad_score = dataset.evaluate_response(bad, gt, "default", record["prompt"])
    assert bad_score == pytest.approx(0.0)


@pytest.mark.skipif(not PLUS_PATH or not PLUS_PATH.exists(), reason="HUMANEVAL_PLUS_GZ not set")
def test_humaneval_plus_canonical_vs_bad():
    cfg = BenchmarkConfig(
        name="humaneval",
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
    dataset = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get("question", ""))
    record = dataset[0]
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]

    ok = dataset.evaluate_response(canonical, gt, "plus", record["prompt"])
    assert ok == pytest.approx(1.0)

    bad = f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
    bad_score = dataset.evaluate_response(bad, gt, "plus", record["prompt"])
    assert bad_score == pytest.approx(0.0)


@pytest.mark.skipif(not PLUS_PATH or not PLUS_PATH.exists(), reason="HUMANEVAL_PLUS_GZ not set")
def test_humaneval_plus_helper_exposes_evalplus_init():
    # Purpose: ensure helper wraps EvalPlus state initialization for readability.
    cfg = BenchmarkConfig(
        name="humaneval",
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
    dataset = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get("question", ""))
    record = dataset[0]
    gt = record["ground_truth"]

    assert dataset._ep_state.expected_outputs is None
    helper = getattr(dataset, "_ensure_evalplus_state")
    expected_outputs = helper(gt)
    assert dataset._ep_state.problems is not None
    assert dataset._ep_state.dataset_hash is not None
    assert dataset._ep_state.expected_outputs is not None
    assert gt["task_id"] in expected_outputs


@pytest.mark.skipif(not PLUS_PATH or not PLUS_PATH.exists(), reason="HUMANEVAL_PLUS_GZ not set")
def test_humaneval_star_emits_both_modes():
    cfg = BenchmarkConfig(
        name="humaneval",
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
    dataset = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get("question", ""))
    ids = [dataset[i]["item"].id for i in range(len(dataset))]
    assert any(id_.endswith("::default") for id_ in ids)
    assert any(not id_.endswith("::default") for id_ in ids)
