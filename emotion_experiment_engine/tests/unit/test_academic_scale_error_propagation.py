"""
Unit tests for error handling in (emotion_check, academic_scale).

Verifies that non-matching option text raises an error instead of returning neutral fallback,
and that single-quoted dict-like responses are parsed correctly.
"""

import unittest
from pathlib import Path

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig


class DummyPromptFormat:
    def build(self, system_text: str, user_messages, enable_thinking: bool = False) -> str:
        return system_text


class TestAcademicScaleErrorHandling(unittest.TestCase):
    def setUp(self):
        self.data_file = Path("data/emotion_scales/emotion_check_academic_scales.jsonl")
        if not self.data_file.exists():
            self.skipTest("Missing academic scales data file")

    def test_unmatched_option_raises(self):
        cfg = BenchmarkConfig(
            name="emotion_check",
            task_type="academic_scale",
            data_path=self.data_file,
            base_data_dir=str(self.data_file.parent),
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

        prompt_format = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=prompt_format,
            emotion="anger",
        )

        sample = dataset[0]
        gt = sample["ground_truth"]
        with self.assertRaises(ValueError):
            dataset.evaluate_response("THIS DOES NOT MATCH ANY OPTION", gt, "academic_scale")

    def test_single_quote_jsonish_response_is_parsed(self):
        cfg = BenchmarkConfig(
            name="emotion_check",
            task_type="academic_scale",
            data_path=self.data_file,
            base_data_dir=str(self.data_file.parent),
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

        prompt_format = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=prompt_format,
            emotion="happiness",
        )

        sample = dataset[0]
        gt = sample["ground_truth"]
        opts = gt.get("options") or []
        # Choose a mid-to-high anchor text if available, else a sensible default
        choice_text = None
        for cand in ["Quite a bit", "Moderately", "Agree", "Extremely"]:
            for o in opts:
                if cand.lower() in str(o).lower():
                    choice_text = cand
                    break
            if choice_text:
                break
        if not choice_text and opts:
            import re
            choice_text = re.sub(r"^\s*\d+\s*[.=)\-:]\s*", "", str(opts[-1])).strip()

        # Simulate model returning Python-dict-like string with the chosen text
        resp = "{'response': '" + (choice_text or "Agree") + "'}"
        score = dataset.evaluate_response(resp, gt, "academic_scale")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

