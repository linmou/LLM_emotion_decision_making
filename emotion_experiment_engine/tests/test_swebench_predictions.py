"""
Responsible file: emotion_experiment_engine/experiment.py
Purpose: Ensure SWE-bench runs emit predictions JSONL and annotate ResultRecord metadata.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig
from emotion_experiment_engine.experiment import EmotionExperiment

os.environ.setdefault("OMP_NUM_THREADS", "1")


class _DummyTokenizer:
    vocab_size = 4
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    is_fast = False

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("Dummy tokenizer should not be invoked in this test")


class _DummyPromptFormat:
    def __init__(self, tokenizer: _DummyTokenizer):
        self.tokenizer = tokenizer

    def build(self, system_prompt: str, user_messages, enable_thinking: bool = False) -> str:
        if isinstance(user_messages, (list, tuple)):
            user = "\n".join(str(m) for m in user_messages)
        else:
            user = str(user_messages)
        return f"{system_prompt}\n{user}"


class _FakeDataset:
    def __init__(self, rows: List[Dict[str, str]]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._rows[idx]


def _build_configs(tmp_path: Path) -> Tuple[ExperimentConfig, List[Dict[str, str]]]:
    dataset_dir = tmp_path / "cache" / "datasets" / "swebench_dummy"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"instance_id": "inst_0", "text_inputs": "<SWE PROMPT>"}]
    with open(dataset_dir / "data.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp)

    bench_cfg = BenchmarkConfig(
        name="swebench",
        task_type="patch",
        data_path=dataset_dir,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    experiment_cfg = ExperimentConfig(
        model_path="dummy-model",
        emotions=["anger"],
        intensities=[0.5],
        benchmark=bench_cfg,
        output_dir=str(tmp_path / "results"),
        batch_size=1,
        generation_config={
            "temperature": 0.0,
            "max_new_tokens": 16,
            "do_sample": False,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
        defer_evaluation=True,
    )

    return experiment_cfg, rows


@pytest.mark.order(3)
def test_swebench_post_process_writes_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    experiment_cfg, rows = _build_configs(tmp_path)

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.swebench.load_from_disk",
        lambda path: _FakeDataset(rows),
    )
    monkeypatch.setattr(
        "neuro_manipulation.utils.load_tokenizer_only",
        lambda *args, **kwargs: (_DummyTokenizer(), None),
    )
    monkeypatch.setattr(
        "neuro_manipulation.prompt_formats.PromptFormat",
        _DummyPromptFormat,
    )

    experiment = EmotionExperiment(experiment_cfg, dry_run=True)
    dataset = experiment.emotion_datasets["anger"]

    experiment.cur_emotion = "anger"
    experiment.cur_intensity = 0.5
    experiment.cur_repeat = 0
    experiment.dataset = dataset

    sample = dataset.__getitem__(0)
    prompt_text = sample["prompt"]
    batch = {
        "prompts": [prompt_text],
        "items": [sample["item"]],
        "ground_truths": [None],
    }
    control_outputs = [[{"generated_text": f"{prompt_text}\n---\nPATCH"}]]

    results = experiment._post_process_batch(batch, control_outputs, batch_idx=0)
    experiment.dataset.flush_predictions(Path(experiment.output_dir))

    assert len(results) == 1
    result = results[0]
    metadata = result.metadata or {}
    assert "predictions_path" in metadata
    assert "run_id" in metadata

    predictions_path = Path(metadata["predictions_path"])
    assert predictions_path.exists()

    data = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    assert data == [
        {
            "instance_id": sample["item"].id,
            "model_patch": "---\nPATCH",
        }
    ]
