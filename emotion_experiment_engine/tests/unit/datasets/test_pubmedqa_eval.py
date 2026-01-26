"""
Responsible file: emotion_experiment_engine/datasets/pubmed_qa.py
Purpose: Validate PubMedQADataset evaluation logic independently of data loading.

Red phase: Import will fail until PubMedQADataset is implemented. When present,
we subclass it to bypass network/data loading by overriding _load_and_parse_data.
"""

import unittest
from typing import List

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem


class TestPubMedQAEval(unittest.TestCase):
    def _make_cfg(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            name="pubmed_qa",
            task_type="pqa_labeled",
            data_path=None,
            base_data_dir="data/PubMedQA",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

    def test_eval_yes_no_maybe(self):
        # Import here (will fail in Red phase until dataset exists)
        from emotion_experiment_engine.datasets.pubmed_qa import PubMedQADataset

        class _NoLoadPubMedQADataset(PubMedQADataset):
            def _load_and_parse_data(self) -> List[BenchmarkItem]:
                return []

        ds = _NoLoadPubMedQADataset(
            config=self._make_cfg(),
            prompt_wrapper=None,
            tokenizer=None,
            max_context_length=None,
            truncation_strategy="right",
            answer_wrapper=None,
        )

        for y in ["yes", "Yes", " YES "]:
            self.assertEqual(ds.evaluate_response(y, "yes", "pqa_labeled", ""), 1.0)
        for n in ["no", "No", " NO "]:
            self.assertEqual(ds.evaluate_response(n, "no", "pqa_labeled", ""), 1.0)
        for m in ["maybe", "Maybe", " MAYBE "]:
            self.assertEqual(ds.evaluate_response(m, "maybe", "pqa_labeled", ""), 1.0)

        # Wrong label
        self.assertEqual(ds.evaluate_response("yes", "no", "pqa_labeled", ""), 0.0)
        self.assertEqual(ds.evaluate_response("no", "maybe", "pqa_labeled", ""), 0.0)
        self.assertEqual(ds.evaluate_response("maybe", "yes", "pqa_labeled", ""), 0.0)

    def test_eval_letter_coded_and_explanatory(self):
        from emotion_experiment_engine.datasets.pubmed_qa import PubMedQADataset

        class _NoLoadPubMedQADataset(PubMedQADataset):
            def _load_and_parse_data(self):
                return []

        ds = _NoLoadPubMedQADataset(
            config=self._make_cfg(),
            prompt_wrapper=None,
            tokenizer=None,
            max_context_length=None,
            truncation_strategy="right",
            answer_wrapper=None,
        )

        # Letter-coded forms
        self.assertEqual(ds.evaluate_response("A. yes", "yes", "pqa_labeled", ""), 1.0)
        self.assertEqual(ds.evaluate_response("B) no", "no", "pqa_labeled", ""), 1.0)
        self.assertEqual(ds.evaluate_response("C: maybe", "maybe", "pqa_labeled", ""), 1.0)
        # Explanatory sentence containing letter choice
        self.assertEqual(ds.evaluate_response("The correct answer is A. yes.", "yes", "pqa_labeled", ""), 1.0)
        self.assertEqual(ds.evaluate_response("Answer: b.", "no", "pqa_labeled", ""), 1.0)
        self.assertEqual(ds.evaluate_response("I choose C", "maybe", "pqa_labeled", ""), 1.0)
        self.assertEqual(
            ds.evaluate_response('{"answers": "yes"}', "yes", "pqa_labeled", ""), 1.0
        )

    def test_context_dict_is_flattened(self):
        from emotion_experiment_engine.datasets.pubmed_qa import PubMedQADataset

        class _MockContextDataset(PubMedQADataset):
            def _load_and_parse_data(self):
                rec = {
                    "question": "Sample question?",
                    "context": {
                        "contexts": ["Sentence one.", "Sentence two."],
                        "labels": ["BACKGROUND", "RESULTS"],
                        "meshes": ["A", "B"],
                    },
                    "final_decision": "yes",
                    "pubid": "mock1",
                }
                ctx_text, ctx_meta = self._extract_context_text(rec["context"])
                metadata = {
                    "options": ["yes", "no", "maybe"],
                    "split": self.split,
                }
                if ctx_meta:
                    metadata["context_metadata"] = ctx_meta
                return [
                    BenchmarkItem(
                        id="mock1",
                        input_text=rec["question"],
                        context=ctx_text,
                        ground_truth="yes",
                        metadata=metadata,
                    )
                ]

        ds = _MockContextDataset(
            config=self._make_cfg(),
            prompt_wrapper=None,
            tokenizer=None,
            max_context_length=None,
            truncation_strategy="right",
            answer_wrapper=None,
        )

        item = ds.items[0]
        self.assertEqual(item.context, "Sentence one. Sentence two.")
        self.assertIn("context_metadata", item.metadata)
        self.assertEqual(item.metadata["context_metadata"].get("labels"), ["BACKGROUND", "RESULTS"])
        self.assertNotIn("labels", item.context)


if __name__ == "__main__":
    unittest.main()
