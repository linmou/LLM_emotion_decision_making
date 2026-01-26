# Purpose: Ensure HumanEvalDataset exposes helper for EvalPlus state initialization.

import json
import sys
import types
from pathlib import Path

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import create_dataset_from_config


def _install_evalplus_stubs(monkeypatch, cache_dir: Path, expected_payload: dict) -> None:
    """Install minimal evalplus stubs so helper can run without heavy deps."""
    evalplus_module = types.ModuleType("evalplus")
    data_pkg = types.ModuleType("evalplus.data")
    humaneval_module = types.ModuleType("evalplus.data.humaneval")
    evaluate_module = types.ModuleType("evalplus.evaluate")
    utils_module = types.ModuleType("evalplus.data.utils")

    def fake_get_groundtruth(problems, dataset_hash, extra):  # type: ignore[no-untyped-def]
        return expected_payload

    def fake_check_correctness(**kwargs):  # type: ignore[no-untyped-def]
        return {"base": ("pass", {}), "plus": ("pass", {})}

    humaneval_module.get_human_eval_plus_hash = lambda: "stub"  # type: ignore[attr-defined]
    evaluate_module.get_groundtruth = fake_get_groundtruth  # type: ignore[attr-defined]
    evaluate_module.check_correctness = fake_check_correctness  # type: ignore[attr-defined]
    utils_module.CACHE_DIR = str(cache_dir)

    sys.modules["evalplus"] = evalplus_module
    sys.modules["evalplus.data"] = data_pkg
    sys.modules["evalplus.data.humaneval"] = humaneval_module
    sys.modules["evalplus.evaluate"] = evaluate_module
    sys.modules["evalplus.data.utils"] = utils_module

    monkeypatch.setenv("EVALPLUS_STUB", "1")


def _write_plus_sample(tmp_path: Path) -> Path:
    row = {
        "task_id": "HumanEval/999",
        "prompt": "def add(a, b):\n    return a + b\n",
        "entry_point": "add",
        "canonical_solution": "def add(a, b):\n    return a + b\n",
        "test": "\nMETADATA = {}\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n",
        "base_input": [[[1, 2]]],
        "plus_input": [[[3, 4]]],
    }
    path = tmp_path / "humaneval_plus_sample.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_humaneval_dataset_exposes_evalplus_helper(monkeypatch, tmp_path):
    expected_payload = {"HumanEval/999": {"base": [], "plus": []}}
    _install_evalplus_stubs(monkeypatch, tmp_path, expected_payload)
    plus_path = _write_plus_sample(tmp_path)

    cfg = BenchmarkConfig(
        name="humaneval",
        task_type="plus",
        data_path=plus_path,
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

    helper = getattr(dataset, "_ensure_evalplus_state")
    result = helper(record["ground_truth"])

    assert dataset._ep_state.problems is not None
    assert dataset._ep_state.dataset_hash is not None
    assert dataset._ep_state.expected_outputs is not None
    assert result == expected_payload
