"""
Unit tests for Fantom benchmark integration.

Responsible files:
- emotion_experiment_engine/benchmark_component_registry.py
- emotion_experiment_engine/datasets/fantom.py (to be implemented)

Purpose:
- Red phase: ensure registry routing and dataset parsing/evaluation behavior
  are defined for Fantom. Tests will initially fail until implementation.
"""

import sys
import types
import unittest
from typing import Optional
from pathlib import Path
from unittest.mock import patch

# Stub heavy deps before importing target modules, but do NOT mask real modules if available.
try:
    import neuro_manipulation.prompt_formats as _pf_mod  # type: ignore  # noqa: F401
except Exception:
    dummy_prompt_formats = types.ModuleType("neuro_manipulation.prompt_formats")

    class _DummyPromptFormat:
        def build(self, system_prompt: str, user_messages, **kwargs) -> str:
            return f"{system_prompt}\n{user_messages}"

    dummy_prompt_formats.PromptFormat = _DummyPromptFormat
    sys.modules["neuro_manipulation.prompt_formats"] = dummy_prompt_formats

try:
    import neuro_manipulation.prompt_wrapper as _pw_mod  # type: ignore  # noqa: F401
except Exception:
    dummy_prompt_wrapper = types.ModuleType("neuro_manipulation.prompt_wrapper")

    class _DummyWrapper:
        def __init__(self, prompt_format):
            self.prompt_format = prompt_format

        def __call__(self, *args, **kwargs):
            return self.prompt_format.build("sys", ["user"])  # minimal

    dummy_prompt_wrapper.PromptWrapper = _DummyWrapper
    sys.modules["neuro_manipulation.prompt_wrapper"] = dummy_prompt_wrapper

# Stub torch.utils.data.Dataset only when torch import is unavailable
try:
    import torch  # type: ignore  # noqa: F401
except Exception:
    dummy_torch = types.ModuleType("torch")
    dummy_utils = types.ModuleType("torch.utils")
    dummy_utils_data = types.ModuleType("torch.utils.data")

    class _DummyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    dummy_utils_data.Dataset = _DummyDataset
    dummy_utils.data = dummy_utils_data
    dummy_torch.utils = dummy_utils
    sys.modules["torch"] = dummy_torch
    sys.modules["torch.utils"] = dummy_utils
    sys.modules["torch.utils.data"] = dummy_utils_data

# Stub heavy dataset submodules to avoid importing dependencies
def _stub_ds_module(qualname: str, cls_name: str):
    mod = types.ModuleType(qualname)
    # Create a minimal base to avoid importing the real BaseBenchmarkDataset here
    class _Base:
        def __init__(self, *args, **kwargs):
            pass

    # Create a placeholder class with the expected name
    Placeholder = type(cls_name, (_Base,), {})
    setattr(mod, cls_name, Placeholder)
    sys.modules[qualname] = mod

_stub_ds_module("emotion_experiment_engine.datasets.infinitebench", "InfiniteBenchDataset")
_stub_ds_module("emotion_experiment_engine.datasets.longbench", "LongBenchDataset")
_stub_ds_module("emotion_experiment_engine.datasets.locomo", "LoCoMoDataset")
_stub_ds_module("emotion_experiment_engine.datasets.mtbench101", "MTBench101Dataset")
_stub_ds_module("emotion_experiment_engine.datasets.truthfulqa", "TruthfulQADataset")

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.fantom_prompt_wrapper import FantomPromptWrapper


class DummyPromptFormat(_DummyPromptFormat):
    pass


def _extract_wrapper_instance_from_partial(prompt_wrapper_partial):
    bound_call = prompt_wrapper_partial.func  # bound method __call__
    return getattr(bound_call, "__self__", None)


class TestFantomRegistryAndDataset(unittest.TestCase):
    """Red phase: expect failures until Fantom is wired up and implemented."""

    def setUp(self):
        self.prompt_format = DummyPromptFormat()

    def _mk_cfg(self, task_type: str, path: Optional[Path] = None) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            name="fantom",
            task_type=task_type,
            data_path=path,
            base_data_dir="data/fantom",
            sample_limit=3,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        if path is None:
            cfg.data_path = cfg.get_data_path()
        return cfg

    @patch(
        "emotion_experiment_engine.benchmark_component_registry.create_dataset_from_config",
        return_value=object(),
    )
    def test_registry_uses_fantom_wrapper_direct(self, _mock_ds):
        cfg = self._mk_cfg("short_answerability_binary_inaccessible")
        prompt_partial, _, _ = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        wrapper = _extract_wrapper_instance_from_partial(prompt_partial)
        self.assertIsInstance(wrapper, FantomPromptWrapper)

    def test_dataset_parsing_and_eval_binary(self):
        # Expect dataset to read JSONL under data/fantom
        cfg = self._mk_cfg("short_answerability_binary_inaccessible")
        # Building components triggers dataset construction
        prompt_partial, answer_partial, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )

        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("ground_truth", sample)

        # Evaluate simple binary correctness from plain text
        gt = sample["ground_truth"]  # either "yes" or "no"
        # Feed a matching response (case-insensitive)
        score = dataset.evaluate_response(gt.upper(), gt, cfg.task_type, sample["prompt"])  # type: ignore[arg-type]
        self.assertEqual(score, 1.0)

        wrong = "yes" if isinstance(gt, str) and gt.lower() == "no" else "no"
        score_wrong = dataset.evaluate_response(wrong, gt, cfg.task_type, sample["prompt"])  # type: ignore[arg-type]
        self.assertEqual(score_wrong, 0.0)

        # Evaluate JSON-formatted response
        json_resp = {"reational": "short reasoning", "answer": gt}
        score_json = dataset.evaluate_response(str(json_resp), gt, cfg.task_type, sample["prompt"])  # type: ignore[arg-type]
        self.assertEqual(score_json, 1.0)

        # JSON with alternative key and letter+content
        # Build a prompt with Options including Yes/No by fetching a fresh item
        sample2 = dataset[0]
        alt_json = {"predict answer": "A. Yes"}
        score_alt = dataset.evaluate_response(str(alt_json), "yes", cfg.task_type, sample2["prompt"])  # type: ignore[arg-type]
        self.assertEqual(score_alt, 1.0)

        # Letter with dot only should map via fallback A/1=Yes, B/2=No
        letter_json = {"predict answer": "B."}
        score_letter_only = dataset.evaluate_response(str(letter_json), "no", cfg.task_type, sample2["prompt"])  # type: ignore[arg-type]
        self.assertEqual(score_letter_only, 1.0)

    def test_dataset_parsing_and_eval_choice(self):
        cfg = self._mk_cfg("short_belief_choice_inaccessible")
        prompt_partial, answer_partial, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=self.prompt_format,
        )
        self.assertGreater(len(dataset), 0)
        item = dataset[0]
        # Validate Fantom prompt structure: question+options in user, schema in system
        prompt_text = item["prompt"]
        self.assertIn("Question:", prompt_text)
        self.assertIn("Options:", prompt_text)
        # JSON schema keys must be present in the system instruction
        self.assertIn("'reational'", prompt_text)
        self.assertIn("'answer'", prompt_text)
        options = item["item"].metadata.get("options")  # type: ignore[attr-defined]
        self.assertIsInstance(options, list)
        self.assertGreaterEqual(len(options), 2)

        # Evaluate correct index and wrong index
        gt_idx = item["item"].metadata.get("correct_index")  # type: ignore[attr-defined]
        correct_text = options[gt_idx]
        # Accept the exact option content (not letter/index)
        score_text = dataset.evaluate_response(correct_text, [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_text, 1.0)

        # JSON text response
        json_text = {"reational": "brief", "answer": correct_text}
        score_json_text = dataset.evaluate_response(str(json_text), [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_json_text, 1.0)

        # Letter-only response should map via options
        gt_idx = item["item"].metadata.get("correct_index")  # type: ignore[attr-defined]
        letter = chr(ord('A') + gt_idx)
        score_letter = dataset.evaluate_response(letter, [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_letter, 1.0)

        # Letter with content
        resp_with_content = f"{letter}. {correct_text}"
        score_letter_content = dataset.evaluate_response(resp_with_content, [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_letter_content, 1.0)

        # Numeric index (1-based)
        num_index = str(gt_idx + 1)
        score_num = dataset.evaluate_response(num_index, [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_num, 1.0)

        # JSON index fields
        json_idx = {"reational": "r", "answer_index": [gt_idx + 1]}
        score_json_idx = dataset.evaluate_response(str(json_idx), [correct_text], cfg.task_type, item["prompt"])  # type: ignore[list-item, arg-type]
        self.assertEqual(score_json_idx, 1.0)


if __name__ == "__main__":
    unittest.main()
