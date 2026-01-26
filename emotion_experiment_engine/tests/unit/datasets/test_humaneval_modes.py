"""Unit tests covering HumanEval dataset modes and registry wiring."""

# Responsible: emotion_experiment_engine/datasets/humaneval.py
# Purpose: Verify default/plus/* loading, evaluation parity, and registry lookups

import os
from pathlib import Path
from typing import Any

import pytest

from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components
from emotion_experiment_engine.data_models import BenchmarkConfig


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HUMANEVAL_ORIG = (REPO_ROOT / "../evalplus/datasets/HumanEval.jsonl").resolve()
DEFAULT_HUMANEVAL_PLUS = (
    REPO_ROOT / "../evalplus/datasets/HumanEvalPlus-v0.1.10.jsonl"
).resolve()


class _DummyPromptFormat:
    def build(
        self,
        system_prompt: str,
        user_messages: Any,
        assistant_messages: Any = None,
        images: Any = None,
        enable_thinking: bool = False,
    ) -> str:
        return "\n".join(filter(None, [system_prompt] + list(user_messages or [])))


def _make_cfg(name: str, task: str, path: Path, sample_limit: int = 3) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        task_type=task,
        data_path=path,
        base_data_dir=None,
        sample_limit=sample_limit,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_registry_unknown_task_raises_keyerror():
    pf = _DummyPromptFormat()
    cfg = _make_cfg("humaneval", "unknown", Path("/tmp/does_not_exist"))
    with pytest.raises(KeyError):
        create_benchmark_components(
            benchmark_name="humaneval",
            task_type="unknown",
            config=cfg,
            prompt_format=pf,
        )


def test_humaneval_default_mode_loads_and_evaluates():
    path_str = os.environ.get("HUMANEVAL_ORIG_GZ")
    path = Path(path_str) if path_str else DEFAULT_HUMANEVAL_ORIG
    if not path.exists():
        pytest.skip("HumanEval original dataset not found")
    pf = _DummyPromptFormat()
    cfg = _make_cfg("humaneval", "default", path, sample_limit=1)
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="default", config=cfg, prompt_format=pf
    )

    record = dataset[0]
    item = record["item"]
    assert item.metadata["mode"] == "default"
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]
    assert dataset.evaluate_response(canonical, gt, "default", record["prompt"]) == pytest.approx(1.0)
    assert (
        dataset.evaluate_response("pass", gt, "default", record["prompt"]) == pytest.approx(0.0)
    )


def test_humaneval_plus_mode_loads_and_evaluates():
    plus_path_str = os.environ.get("HUMANEVAL_PLUS_GZ")
    path = Path(plus_path_str) if plus_path_str else DEFAULT_HUMANEVAL_PLUS
    if not path.exists():
        pytest.skip("HumanEvalPlus jsonl.gz not found")
    pf = _DummyPromptFormat()
    cfg = _make_cfg("humaneval", "plus", path, sample_limit=1)
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="plus", config=cfg, prompt_format=pf
    )

    record = dataset[0]
    item = record["item"]
    assert item.metadata["mode"] == "plus"
    gt = record["ground_truth"]
    canonical = gt["canonical_solution"]
    assert dataset.evaluate_response(canonical, gt, "plus", record["prompt"]) == pytest.approx(1.0)
    bad_completion = f"def {gt['entry_point']}(*args, **kwargs):\n    return None"
    assert dataset.evaluate_response(bad_completion, gt, "plus", record["prompt"]) == pytest.approx(0.0)


def test_humaneval_star_mode_emits_default_and_plus():
    plus_path_str = os.environ.get("HUMANEVAL_PLUS_GZ")
    path = Path(plus_path_str) if plus_path_str else DEFAULT_HUMANEVAL_PLUS
    if not path.exists():
        pytest.skip("HumanEvalPlus jsonl.gz not found")
    pf = _DummyPromptFormat()
    cfg = _make_cfg("humaneval", "*", path, sample_limit=4)
    _, _, dataset = create_benchmark_components(
        benchmark_name="humaneval", task_type="*", config=cfg, prompt_format=pf
    )
    ids = [dataset[i]["item"].id for i in range(len(dataset))]
    assert any(id_.endswith("::default") for id_ in ids)
    assert any(not id_.endswith("::default") for id_ in ids)
