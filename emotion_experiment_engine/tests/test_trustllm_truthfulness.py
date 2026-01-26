"""
TrustLLM Truthfulness unit tests.

Responsible for: emotion_experiment_engine/datasets/trustllm_truthfulness.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(api_key="", OpenAI=object, AzureOpenAI=object)

sys.path.insert(0, str(Path("/data/home/jjl7137/TrustLLM/trustllm_pkg")))

from emotion_experiment_engine.data_models import BenchmarkConfig, ResultRecord
from emotion_experiment_engine.datasets.trustllm_truthfulness import TrustLLMTruthfulnessDataset


def _cfg(task: str, data_path: Path | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="trustllm_truthfulness",
        task_type=task,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )


def _write_json(tmp_path: Path, name: str, payload: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_external(tmp_path: Path):
    rows = [
        {"prompt": "Claim", "answer": "SUPPORT", "source": "climate"}
    ]
    path = _write_json(tmp_path, "external.json", rows)
    ds = TrustLLMTruthfulnessDataset(config=_cfg("external", path), prompt_wrapper=None)
    assert len(ds) == 1
    assert ds[0]["ground_truth"] == "support"


def test_loader_rejects_unknown(tmp_path: Path):
    path = _write_json(tmp_path, "external.json", [])
    try:
        TrustLLMTruthfulnessDataset(config=_cfg("unknown", path), prompt_wrapper=None)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_eval_external_support():
    ds = TrustLLMTruthfulnessDataset(config=_cfg("external", None), prompt_wrapper=None)
    score = ds.evaluate_response(
        "Answer: Support",
        ground_truth="support",
        task_name="external",
        prompt="p",
    )
    assert score == 1.0


def _record(task: str, gt: str, response: str, score: float, metadata: dict | None = None) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{task}_{gt}",
        task_name=task,
        prompt="p",
        response=response,
        ground_truth=gt,
        score=score,
        repeat_id=0,
        metadata=metadata,
    )


def test_external_split_metrics_parity():
    from trustllm.task.truthfulness import TruthfulnessEval

    ds = TrustLLMTruthfulnessDataset(config=_cfg("external", None), prompt_wrapper=None)
    recs = [
        _record("external", "support", "Answer: support", 1.0, metadata={"source": "climate", "answer": "SUPPORT"}),
        _record("external", "refute", "Answer: refute", 1.0, metadata={"source": "climate", "answer": "REFUTE"}),
        _record("external", "support", "Answer: refute", 0.0, metadata={"source": "covid", "answer": "SUPPORT"}),
        _record("external", "refute", "Answer: refute", 1.0, metadata={"source": "covid", "answer": "REFUTE"}),
    ]
    metrics = ds.compute_split_metrics(recs)["external"]

    trust = TruthfulnessEval()
    for src, score in metrics["per_source"].items():
        trust_data = [
            {
                "answer": (r.metadata or {}).get("answer", r.ground_truth.upper()),
                "source": (r.metadata or {}).get("source", ""),
                "res": r.response,
            }
            for r in recs
            if (r.metadata or {}).get("source", "") == src
        ]
        trust_score = trust.eval_single_source(trust_data, src)
        assert abs(score - trust_score) < 1e-9


def test_hallucination_split_metrics():
    from trustllm.task.truthfulness import TruthfulnessEval

    ds = TrustLLMTruthfulnessDataset(config=_cfg("hallucination", None), prompt_wrapper=None)
    recs = [
        _record("hallucination", "yes", "yes", 1.0, metadata={"source": "halu_qa", "answer": "Yes"}),
        _record("hallucination", "no", "no", 1.0, metadata={"source": "halu_qa", "answer": "No"}),
        _record("hallucination", "A", "Answer: A", 1.0, metadata={"source": "mc", "answer": "A"}),
    ]
    metrics = ds.compute_split_metrics(recs)["hallucination"]

    trust = TruthfulnessEval()
    trust_input = [
        {
            "res": r.response,
            "answer": (r.metadata or {}).get("answer", r.ground_truth),
            "source": (r.metadata or {}).get("source", ""),
        }
        for r in recs
    ]
    qa_score = trust.eval_hallucination_single(trust_input, "halu_qa")
    assert abs(metrics["per_source"]["halu_qa"] - qa_score) < 1e-9
    mc_score = trust.eval_hallucination_mc(trust_input)
    assert abs(metrics["per_source"]["mc"] - mc_score) < 1e-9
