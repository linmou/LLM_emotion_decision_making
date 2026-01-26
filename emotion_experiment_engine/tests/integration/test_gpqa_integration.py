#!/usr/bin/env python3
"""
Integration test for GPQA dataset using a real tokenizer and PromptFormat.
Validates full pipeline: CSV -> dataset -> prompt formatting -> evaluation.
"""

import csv
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from neuro_manipulation.prompt_formats import PromptFormat
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components

_LOCAL_QWEN_PATH = Path("/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct")


def _resolve_qwen_model(preferred: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    """Prefer the local snapshot when available (offline env)."""
    if _LOCAL_QWEN_PATH.exists():
        return str(_LOCAL_QWEN_PATH)
    return preferred


def _make_csv(rows=1) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".csv")
    path = Path(path_str)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "Record ID",
            ]
        )
        for i in range(rows):
            writer.writerow(
                [
                    f"Which option is correct for Q{i}?",
                    f"Correct{i}",
                    f"WrongA{i}",
                    f"WrongB{i}",
                    f"WrongC{i}",
                    f"rec_{i}",
                ]
            )
    return path


def test_gpqa_prompt_with_real_prompt_format():
    data_path = _make_csv(rows=2)
    try:
        # Real tokenizer and prompt format (consistent with other integration tests)
        tokenizer = AutoTokenizer.from_pretrained(_resolve_qwen_model())
        prompt_format = PromptFormat(tokenizer)

        config = BenchmarkConfig(
            name="gpqa",
            task_type="main",
            data_path=data_path,
            base_data_dir=str(data_path.parent),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

        prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="main",
            config=config,
            prompt_format=prompt_format,
        )

        item = dataset[0]
        prompt = item["prompt"]
        # Structure checks for GPQA raw prompt format
        # Accept either "Choices:" (GPQA style) or "Options:" (legacy) for robustness
        assert ("Choices:" in prompt) or ("Options:" in prompt)
        # Lettered options are expected in GPQA prompts
        assert "(A)" in prompt and "(B)" in prompt
        # Many GPQA prompts include a CoT preamble
        assert "Let's think" in prompt

        # Evaluate correctness
        score = dataset.evaluate_response(
            "Correct0", item["ground_truth"], config.task_type, prompt
        )
        assert score == 1.0
    finally:
        try:
            data_path.unlink()
        except Exception:
            pass
