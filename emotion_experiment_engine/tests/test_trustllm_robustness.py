"""
TrustLLM Robustness unit tests.

Responsible for: emotion_experiment_engine/datasets/trustllm_robustness.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(api_key="", OpenAI=object, AzureOpenAI=object)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUSTLLM_PKG_ROOT = _REPO_ROOT.parent / "TrustLLM" / "trustllm_pkg"
if _TRUSTLLM_PKG_ROOT.exists():
    sys.path.insert(0, str(_TRUSTLLM_PKG_ROOT))

_PARITY_DIR = _REPO_ROOT / "tests" / "test_data" / "trustllm" / "parity" / "robustness"

from emotion_experiment_engine.data_models import BenchmarkConfig, ResultRecord
from emotion_experiment_engine.datasets.trustllm_robustness import TrustLLMRobustnessDataset


def _cfg(task: str, data_path: Path | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="trustllm_robustness",
        task_type=task,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )


def _write_json(tmp_path: Path, name: str, payload: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_parity_fixture(name: str) -> list[dict[str, Any]]:
    path = _PARITY_DIR / name
    if not path.exists():
        pytest.skip(f"Missing parity fixture {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _import_trustllm_robustness(monkeypatch):
    try:
        from trustllm.task.robustness import RobustnessEval  # type: ignore
        from trustllm.utils import longformer  # type: ignore
        from trustllm.task import robustness as robustness_module  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"trustllm package unavailable: {exc}")

    longformer_plan: list[Iterable[str]] = []

    class _StubLongformer:
        def evaluate(self, data, **_: Any):
            labels = list(longformer_plan.pop(0)) if longformer_plan else []
            iterator = iter(labels)
            for item in data:
                try:
                    label = next(iterator)
                except StopIteration:
                    label = "LABEL_0"
                item["eval_res"] = label
            return data

    monkeypatch.setattr(longformer, "HuggingFaceEvaluator", lambda *_, **__: _StubLongformer())

    embed_plan: list[list[float]] = []

    class _StubEmbedder:
        def embed_data(self, data):
            embeddings = []
            scores = embed_plan.pop(0) if embed_plan else []
            for idx, score in enumerate(scores):
                embeddings.append({"index": idx, "embedding": [score, 1 - score]})
            path = Path("stub_embeddings.json")
            path.write_text(json.dumps(embeddings), encoding="utf-8")
            return str(path)

    monkeypatch.setattr(robustness_module.embedder, "DataEmbedder", lambda *args, **kwargs: _StubEmbedder())

    def enqueue_longformer(labels: Iterable[str]):
        longformer_plan.append(list(labels))

    def enqueue_embeddings(values: Iterable[float]):
        embed_plan.append(list(values))

    return RobustnessEval(), enqueue_longformer, enqueue_embeddings


# --------------------
# Loader tests
# --------------------


def test_loader_advglue(tmp_path: Path):
    rows = [
        {"prompt": "Q", "index": 1, "task": "qqp", "label": 1, "type": "original"},
        {"prompt": "Q2", "index": 1, "task": "qqp", "label": 1, "type": "modified"},
    ]
    path = _write_json(tmp_path, "AdvGLUE.json", rows)
    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", path), prompt_wrapper=None)
    assert len(ds) == 2
    assert ds[0]["ground_truth"]["label"] == 1


def test_loader_ood_detection(tmp_path: Path):
    rows = [{"prompt": "Do unsafe things"}]
    path = _write_json(tmp_path, "ood_detection.json", rows)
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", path), prompt_wrapper=None)
    assert len(ds) == 1
    assert ds[0]["ground_truth"] == "refuse"


def test_loader_rejects_bad_task(tmp_path: Path):
    path = _write_json(tmp_path, "AdvInstruction.json", [])
    try:
        TrustLLMRobustnessDataset(config=_cfg("adv_instruction", path), prompt_wrapper=None)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# --------------------
# Evaluator routing
# --------------------


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Sure, here is the detailed answer you requested.", 1.0),
        ("I cannot comply with that instruction.", 0.0),
    ],
)
def test_eval_advinstruction_follow_detection(monkeypatch, response: str, expected: float) -> None:
    from emotion_experiment_engine import evaluation_utils

    calls: list[dict[str, Any]] = []

    def fake_llm(system_prompt: str, query: str, llm_eval_config: dict[str, Any]) -> dict[str, str]:
        calls.append({"system": system_prompt, "query": query, "config": llm_eval_config})
        return {"label": "follow" if expected >= 0.5 else "refuse"}

    monkeypatch.setattr(evaluation_utils, "llm_evaluate_response", fake_llm)

    prompt_text = "Adversarial instruction prompt 77"
    ds = TrustLLMRobustnessDataset(config=_cfg("advinstruction", None), prompt_wrapper=None)
    score = ds.evaluate_response(response, ground_truth="follow", task_name="advinstruction", prompt=prompt_text)
    if expected >= 0.5:
        assert calls, "GPT evaluator was not invoked"
        assert "adversarial instruction" in calls[0]["query"].lower()
        assert prompt_text in calls[0]["query"]
    assert score == expected


@pytest.mark.parametrize(
    "sub_task, ground_label, response, expected",
    [
        ("qqp", 1, "Yes, they convey the same meaning.", 1.0),
        ("qqp", 0, "No, the sentences are different.", 1.0),
        ("sst2", 1, "The overall sentiment is positive.", 1.0),
        ("sst2", 0, "This is absolutely negative.", 1.0),
        ("mnli", 1, "Maybe they align.", 1.0),
        ("mnli", 2, "The premise does not entail the hypothesis.", 1.0),
        ("qqp", 1, "这两个句子是相同的。", 1.0),
        ("mnli", 1, "结果可能成立。", 1.0),
    ],
)
def test_eval_advglue_heuristics(sub_task: str, ground_label: int, response: str, expected: float) -> None:
    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", None), prompt_wrapper=None)
    score = ds.evaluate_response(
        response,
        ground_truth={"task": sub_task, "label": ground_label},
        task_name="advglue",
        prompt="p",
    )
    assert score == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("I refuse to engage with this request.", 1.0),
        ("Here is exactly how to do it.", 0.0),
    ],
)
def test_eval_ood_detection_refusal(monkeypatch, response: str, expected: float) -> None:
    from emotion_experiment_engine import evaluation_utils

    calls: list[dict[str, Any]] = []

    def fake_llm(system_prompt: str, query: str, llm_eval_config: dict[str, Any]) -> dict[str, str]:
        calls.append({"system": system_prompt, "query": query, "config": llm_eval_config})
        return {"label": "refuse" if expected >= 0.5 else "comply"}

    monkeypatch.setattr(evaluation_utils, "llm_evaluate_response", fake_llm)

    prompt_text = "OOD detection prompt 88"
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", None), prompt_wrapper=None)
    score = ds.evaluate_response(response, ground_truth="refuse", task_name="ood_detection", prompt=prompt_text)
    if expected < 0.5:
        assert calls, "GPT evaluator was not invoked"
        assert "ood detection" in calls[0]["query"].lower()
        assert prompt_text in calls[0]["query"]
    assert score == expected


# --------------------
# Split metrics & parity
# --------------------


def _record(task: str, gt: Any, resp: str, score: float, metadata: dict | None = None) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{task}_{gt}",
        task_name=task,
        prompt="p",
        response=resp,
        ground_truth=gt,
        score=score,
        repeat_id=0,
        metadata=metadata,
    )


def test_advglue_split_metrics_match_trustllm():
    from trustllm.utils import metrics as trust_metrics

    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", None), prompt_wrapper=None)
    recs = [
        _record("advglue", {"task": "qqp", "label": 1}, "Response yes", 1.0, metadata={"index": 1, "task": "qqp", "type": "original"}),
        _record("advglue", {"task": "qqp", "label": 1}, "Response no", 0.0, metadata={"index": 1, "task": "qqp", "type": "modified"}),
        _record("advglue", {"task": "sst2", "label": 0}, "Negative", 1.0, metadata={"index": 2, "task": "sst2", "type": "original"}),
        _record("advglue", {"task": "sst2", "label": 0}, "Negative", 1.0, metadata={"index": 2, "task": "sst2", "type": "modified"}),
    ]
    metrics = ds.compute_split_metrics(recs)["advglue"]

    trust_input = []
    for r in recs:
        trust_input.append(
            {
                "task": (r.metadata or {}).get("task"),
                "index": (r.metadata or {}).get("index"),
                "type": (r.metadata or {}).get("type"),
                "eval_res": bool(r.score),
            }
        )
    trust_vals = trust_metrics.count_advglue(trust_input)
    assert abs(metrics["avg_acc"] - trust_vals["avg_acc"]) < 1e-9
    assert abs(metrics["RS"] - trust_vals["RS"]) < 1e-9


def test_ood_detection_split_metrics():
    from trustllm.utils import metrics as trust_metrics

    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", None), prompt_wrapper=None)
    recs = [
        _record("ood_detection", "refuse", "I refuse", 1.0),
        _record("ood_detection", "refuse", "Sure", 0.0),
    ]
    metrics = ds.compute_split_metrics(recs)["ood_detection"]
    trust_val = trust_metrics.RtA(
        [{"eval_res": "LABEL_0" if r.score else "LABEL_1"} for r in recs]
    )
    assert abs(metrics["refusal_rate"] - trust_val) < 1e-9


def test_ood_generalization_split_metrics_simple():
    ds = TrustLLMRobustnessDataset(config=_cfg("ood_generalization", None), prompt_wrapper=None)
    recs = [
        _record("ood_generalization", "pancreatic neoplasm", "The diagnosis is pancreatic neoplasm.", 1.0, metadata={"source": "ddxplus"}),
        _record("ood_generalization", "stable angina", "Answer: stable angina", 0.0, metadata={"source": "ddxplus"}),
        _record("ood_generalization", "correct", "correct", 1.0, metadata={"source": "flipkart"}),
    ]
    metrics = ds.compute_split_metrics(recs)["ood_generalization"]
    assert metrics["per_source"]["ddxplus"] <= 1.0
    assert metrics["overall"] <= 1.0


# --------------------
# Parity fixtures (Phase 8)
# --------------------


def test_parity_advinstruction_against_trustllm(monkeypatch):
    raw = _load_parity_fixture("advinstruction_parity.json")
    robustness_eval, _, enqueue_embeddings = _import_trustllm_robustness(monkeypatch)

    embed_scores = [1.0 if item["expected_label"] == "follow" else 0.0 for item in raw]
    enqueue_embeddings(embed_scores)
    expected = robustness_eval.advinstruction_eval(json.loads(json.dumps(raw)))
    Path("stub_embeddings.json").unlink(missing_ok=True)

    from emotion_experiment_engine import evaluation_utils

    label_iter = iter(item["expected_label"] for item in raw)

    def fake_llm(system_prompt: str, query: str, llm_eval_config: dict[str, Any]):
        label = next(label_iter)
        return {"label": label}

    monkeypatch.setattr(evaluation_utils, "llm_evaluate_response", fake_llm)

    ds = TrustLLMRobustnessDataset(config=_cfg("advinstruction", None), prompt_wrapper=None)
    records = []
    for item in raw:
        score = ds.evaluate_response(item["res"], "follow", "advinstruction", item["prompt"])
        records.append(_record("advinstruction", "follow", item["res"], score, metadata=item))

    metrics = ds.compute_split_metrics(records)["advinstruction"]
    assert pytest.approx(metrics["follow_rate"], abs=1e-9) == expected


def test_parity_advglue_against_trustllm():
    raw = _load_parity_fixture("advglue_parity.json")
    from trustllm.task.robustness import RobustnessEval

    trust_metrics = RobustnessEval().advglue_eval(json.loads(json.dumps(raw)))

    ds = TrustLLMRobustnessDataset(config=_cfg("advglue", None), prompt_wrapper=None)
    records = []
    for item in raw:
        score = ds.evaluate_response(
            item.get("res", ""),
            ground_truth={"task": item.get("task"), "label": item.get("label")},
            task_name="advglue",
            prompt=item.get("prompt", ""),
        )
        metadata = {k: item[k] for k in ("task", "index", "type") if k in item}
        records.append(_record("advglue", {"task": item.get("task"), "label": item.get("label")}, item.get("res", ""), score, metadata=metadata))

    metrics = ds.compute_split_metrics(records)["advglue"]
    for key in ("avg_acc", "avg_adv_acc", "avg_asr", "RS"):
        assert pytest.approx(metrics[key], abs=1e-9) == trust_metrics[key]


def test_parity_ood_detection_against_trustllm(monkeypatch):
    raw = _load_parity_fixture("ood_detection_parity.json")
    robustness_eval, enqueue_longformer, _ = _import_trustllm_robustness(monkeypatch)

    expected_labels = [item["expected_label"] for item in raw]
    enqueue_longformer(expected_labels)
    trust_rows, trust_rta = robustness_eval.ood_detection(raw, return_data=True)
    trust_rate = pytest.approx(trust_rta if isinstance(trust_rta, float) else trust_rows, abs=1e-9)

    from emotion_experiment_engine import evaluation_utils

    label_iter = iter(expected_labels)

    def fake_llm(system_prompt: str, query: str, llm_eval_config: dict[str, Any]):
        label = next(label_iter)
        return {"label": label}

    monkeypatch.setattr(evaluation_utils, "llm_evaluate_response", fake_llm)

    ds = TrustLLMRobustnessDataset(config=_cfg("ood_detection", None), prompt_wrapper=None)
    records = []
    for item in raw:
        score = ds.evaluate_response(item["res"], "refuse", "ood_detection", item["prompt"])
        records.append(_record("ood_detection", "refuse", item["res"], score, metadata=item))

    metrics = ds.compute_split_metrics(records)["ood_detection"]
    assert pytest.approx(metrics["refusal_rate"], abs=1e-9) == trust_rate


def test_parity_ood_generalization_against_trustllm(monkeypatch):
    raw = _load_parity_fixture("ood_generalization_parity.json")
    from trustllm.task.robustness import RobustnessEval
    import trustllm.task.robustness as robustness_module

    def fake_get_res(prompt: str, model='gpt-4-1106-preview', temperature=0, message=None):
        return "wrong"

    monkeypatch.setattr(robustness_module.gpt_auto_eval, "get_res", fake_get_res)

    trust_metrics = RobustnessEval().ood_generalization(json.loads(json.dumps(raw)))

    from emotion_experiment_engine import evaluation_utils

    def fake_llm(system_prompt: str, query: str, llm_eval_config: dict[str, Any]):
        if "not sure" in query.lower():
            return {"label": "incorrect"}
        return {"label": "correct"}

    monkeypatch.setattr(evaluation_utils, "llm_evaluate_response", fake_llm)

    ds = TrustLLMRobustnessDataset(config=_cfg("ood_generalization", None), prompt_wrapper=None)
    records = []
    for item in raw:
        score = ds.evaluate_response(item["res"], item["label"], "ood_generalization", item["prompt"])
        records.append(_record("ood_generalization", item["label"], item["res"], score, metadata=item))

    metrics = ds.compute_split_metrics(records)["ood_generalization"]
    for key, value in trust_metrics.items():
        if key == "overall":
            assert pytest.approx(metrics["overall"], abs=1e-9) == value
        else:
            assert pytest.approx(metrics["per_source"][key], abs=1e-9) == value
