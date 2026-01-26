"""
Responsible files:
- emotion_experiment_engine/datasets/swebench.py
- emotion_experiment_engine/experiment.py

Purpose: Verify the prompts fed into the LLM match SWE-bench's reference text_inputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from datasets import load_from_disk

from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig
from emotion_experiment_engine.datasets.swebench import SWEbenchDataset
from emotion_experiment_engine.experiment import EmotionExperiment

DATASET_DIR = Path("cache/datasets/SWE-bench_Lite_text_inputs_dataset")
ORIG_DIR = Path("cache/datasets/SWE-bench__SWE-bench_Lite__style-3__fs-bm25__k-20__mcc-32768-llama")

pytestmark = pytest.mark.skipif(
    not DATASET_DIR.exists() or not ORIG_DIR.exists(),
    reason="SWE-bench caches are required for prompt parity tests.",
)


def _load_original_texts() -> List[str]:
    dataset = load_from_disk(str(ORIG_DIR))
    if isinstance(dataset, dict):
        dataset = dataset["test"]
    return [row["text"] for row in dataset]


def _benchmark_config(sample_limit: Optional[int] = None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="swebench",
        task_type="patch",
        data_path=DATASET_DIR,
        base_data_dir=None,
        sample_limit=sample_limit,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_swebench_dataset_prompt_parity():
    dataset = SWEbenchDataset(_benchmark_config(), prompt_wrapper=None)
    original_texts = _load_original_texts()

    for idx in range(min(10, len(dataset))):
        example = dataset[idx]
        assert example["prompt"] == original_texts[idx]


class _DummyTokenizer:
    vocab_size = 10
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class _DummyPromptFormat:
    def __init__(self, tokenizer: _DummyTokenizer):
        self.tokenizer = tokenizer

    def build(self, system_prompt: str, user_messages, enable_thinking: bool = False) -> str:
        if isinstance(user_messages, (list, tuple)):
            user = "\n".join(str(m) for m in user_messages)
        else:
            user = str(user_messages)
        return f"{system_prompt}\n{user}"


def test_emotion_experiment_dry_run_prompts_match_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    original_texts = _load_original_texts()

    monkeypatch.setattr(
        "neuro_manipulation.utils.load_tokenizer_only",
        lambda *args, **kwargs: (_DummyTokenizer(), None),
    )
    monkeypatch.setattr(
        "neuro_manipulation.prompt_formats.PromptFormat",
        _DummyPromptFormat,
    )

    experiment_cfg = ExperimentConfig(
        model_path="dummy-model",
        emotions=["anger"],
        intensities=[0.5],
        benchmark=_benchmark_config(sample_limit=5),
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

    experiment = EmotionExperiment(experiment_cfg, dry_run=True)
    dataset = experiment.emotion_datasets["anger"]

    prompts = [dataset[idx]["prompt"] for idx in range(len(dataset))]
    assert prompts == original_texts[: len(prompts)]
