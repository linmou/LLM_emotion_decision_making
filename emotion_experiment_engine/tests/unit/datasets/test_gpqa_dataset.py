"""
Responsible file: emotion_experiment_engine/datasets/gpqa.py
Purpose: Validate GPQA dataset parsing, evaluation, and registry integration.

This test suite follows TDD (Red-Green-Refactor):
- Red: Add failing tests for a new GPQA benchmark integration
- Green: Implement minimal dataset + wrapper + registry to satisfy tests
- Refactor: Clean up once tests pass, then run regression tests
"""

import csv
import tempfile
import unittest
from pathlib import Path

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)


def _make_temp_gpqa_csv(rows: int = 2) -> Path:
    """Create a minimal GPQA-format CSV and return its path."""
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
                "Subdomain",
                "High-level domain",
                "Record ID",
            ]
        )
        for i in range(rows):
            writer.writerow(
                [
                    f"What is the answer to Q{i}?",
                    f"Correct{i}",
                    f"WrongA{i}",
                    f"WrongB{i}",
                    f"WrongC{i}",
                    "Physics",
                    "Science",
                    f"rec_{i}",
                ]
            )
    return path


class TestGPQADataset(unittest.TestCase):
    """Tests for GPQA dataset integration and behavior."""

    def setUp(self):
        self.temp_csv = _make_temp_gpqa_csv(rows=3)
        self.config = BenchmarkConfig(
            name="gpqa",
            task_type="main",  # supports: main, extended, diamond; treated as MC1
            data_path=self.temp_csv,
            base_data_dir="data/GPQA",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

    def tearDown(self):
        try:
            self.temp_csv.unlink()
        except Exception:
            pass

    def test_dataset_created_via_registry(self):
        """Factory should build GPQA components and dataset from registry."""
        prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="main",
            config=self.config,
            prompt_format=None,  # we won't build prompts in this test
        )

        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 3)

        item0 = dataset.items[0]
        self.assertEqual(item0.input_text, "What is the answer to Q0?")
        self.assertEqual(item0.context, "What is the answer to Q0?")
        self.assertEqual(item0.ground_truth, ["Correct0"])  # single correct answer
        self.assertIn("options", item0.metadata)
        self.assertEqual(len(item0.metadata["options"]), 4)

    def test_evaluation_logic(self):
        """Evaluate response: exact-text match (case-insensitive) yields 1.0 else 0.0."""
        _, _, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="main",
            config=self.config,
            prompt_format=None,
        )

        gt = ["Correct1"]
        # exact match
        self.assertEqual(dataset.evaluate_response("Correct1", gt, "main", ""), 1.0)
        # case-insensitive match
        self.assertEqual(dataset.evaluate_response("correct1", gt, "main", ""), 1.0)
        # wrong answer
        self.assertEqual(dataset.evaluate_response("WrongA1", gt, "main", ""), 0.0)

    def test_registry_uses_truthfulqa_style_prompt_wrapper(self):
        """Ensure registry assigns a multiple-choice prompt wrapper for GPQA."""
        prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="diamond",  # other split should also work
            config=self.config,
            prompt_format=None,
        )

        # Just ensure prompt_wrapper is callable and accepts 'options'
        # We won't call it here (no real PromptFormat), but signature should exist.
        self.assertTrue(callable(prompt_wrapper))

    def test_getitem_parity_with_csv_row(self):
        """__getitem__ should reflect CSV content exactly (question/options/id)."""
        class _StubPromptFormat:
            def build(self, **kwargs):
                # Minimal stub; content not used for parity of item fields
                return "<stub>"

        _, _, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="main",
            config=self.config,
            prompt_format=_StubPromptFormat(),
        )

        # Check every row
        for i in range(3):
            item_dict = dataset.__getitem__(i)
            item = item_dict["item"]

            # Question/context parity
            expected_q = f"What is the answer to Q{i}?"
            self.assertEqual(item.input_text, expected_q)
            self.assertEqual(item.context, expected_q)

            # ID parity (uses Record ID column)
            self.assertEqual(item.id, f"rec_{i}")

            # Options parity in order: [correct, wrong1, wrong2, wrong3]
            opts = item.metadata.get("options", [])
            self.assertEqual(
                opts,
                [f"Correct{i}", f"WrongA{i}", f"WrongB{i}", f"WrongC{i}"]
            )


if __name__ == "__main__":
    unittest.main()
