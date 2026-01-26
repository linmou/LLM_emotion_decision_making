"""Merged FANToM dataset evaluation tests.

This consolidates legacy suites:
- test_fantom_eval_dispatch.py
- test_fantom_all_tasks.py
- test_fantom_batch_eval.py

Responsible files under test:
- emotion_experiment_engine/datasets/fantom.py
- scripts/convert_fantom.py (data conversion for FANToM tasks)

Purpose:
- Cover evaluation dispatch and behavior for binary, choice, list, fact/gen tasks
- Ensure embedding-dependent batch evaluation raises/falls back correctly
- Confirm the embedding cache deduplicates across batched inputs
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import List

import numpy as np

# Lightweight stubs to avoid heavy deps during import collection (fallback only)
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

# Stub heavy dataset submodules to avoid importing full dependencies via registry
def _stub_ds_module(qualname: str, cls_name: str):
    mod = types.ModuleType(qualname)
    class _Base:
        pass
    Placeholder = type(cls_name, (_Base,), {})
    setattr(mod, cls_name, Placeholder)
    sys.modules[qualname] = mod

_stub_ds_module("emotion_experiment_engine.datasets.infinitebench", "InfiniteBenchDataset")
_stub_ds_module("emotion_experiment_engine.datasets.longbench", "LongBenchDataset")
_stub_ds_module("emotion_experiment_engine.datasets.locomo", "LoCoMoDataset")
_stub_ds_module("emotion_experiment_engine.datasets.mtbench101", "MTBench101Dataset")
_stub_ds_module("emotion_experiment_engine.datasets.truthfulqa", "TruthfulQADataset")

# Avoid importing openai from evaluation_utils during incidental imports
sys.modules.setdefault("openai", types.ModuleType("openai"))

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.fantom import FantomDataset


# ---- From former test_fantom_eval_dispatch.py ----

def _cfg(task: str) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        name="fantom",
        task_type=task,
        data_path=None,
        base_data_dir="data/fantom",
        sample_limit=3,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    cfg.data_path = cfg.get_data_path()
    return cfg


class TestFantomEvalDispatch(unittest.TestCase):
    def _build(self, task: str):
        class _PF:
            def build(self, system_prompt: str, user_messages, **kwargs):
                return f"{system_prompt}\n{user_messages}"

        cfg = _cfg(task)
        return create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=_PF(),
        )

    def test_binary_synonyms_and_fail(self):
        _, _, ds = self._build("short_infoaccessibility_binary_inaccessible")
        ex = ds[0]
        gt = ex["ground_truth"]
        # map synonyms: true/false, knows/does not know
        if gt == "yes":
            self.assertEqual(
                ds.evaluate_response("True", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                1.0,
            )
            self.assertEqual(
                ds.evaluate_response("no", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                0.0,
            )
        else:
            self.assertEqual(
                ds.evaluate_response("does not know", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                1.0,
            )
            self.assertEqual(
                ds.evaluate_response("yes", gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]),
                0.0,
            )

    def test_choice_letter_mapping_pass_and_fail(self):
        _, _, ds = self._build("short_belief_choice_inaccessible")
        ex = ds[0]
        md = ex["item"].metadata  # type: ignore[attr-defined]
        correct_idx = md.get("correct_index")
        # Build letter answers
        correct_letter = chr(ord("a") + int(correct_idx))
        wrong_letter = chr(ord("a") + ((int(correct_idx) + 1) % len(md.get("options", []))))
        self.assertEqual(
            ds.evaluate_response(f"({correct_letter})", ex["ground_truth"], "short_belief_choice_inaccessible", ex["prompt"]),
            1.0,
        )
        self.assertEqual(
            ds.evaluate_response(f"{wrong_letter}.", ex["ground_truth"], "short_belief_choice_inaccessible", ex["prompt"]),
            0.0,
        )

    def test_list_json_and_missing_element(self):
        _, _, ds = self._build("short_answerability_list_inaccessible")
        ex = ds[0]
        gt = ex["ground_truth"]
        # JSON form should pass
        resp = {"answer": gt}
        self.assertEqual(
            ds.evaluate_response(str(resp), gt, "short_answerability_list_inaccessible", ex["prompt"]),
            1.0,
        )
        # Missing element should fail (not equal set)
        if len(gt) > 1:
            bad = gt[:-1]
        else:
            bad = ["nonexistent"]
        self.assertEqual(
            ds.evaluate_response(
                ", ".join(bad), gt, "short_answerability_list_inaccessible", ex["prompt"]
            ),
            0.0,
        )

        # If response includes a wrong item, it should fail even if all correct are present
        wrong = ex["item"].metadata.get("wrong_answer")  # type: ignore[attr-defined]
        if wrong and isinstance(wrong, list) and wrong:
            combined = gt + [wrong[0]]
            resp2 = {"answer": combined}
            self.assertEqual(
                ds.evaluate_response(str(resp2), gt, "short_answerability_list_inaccessible", ex["prompt"]),
                0.0,
            )

    def test_fact_and_gen_eval(self):
        # fact
        _, _, ds_fact = self._build("short_fact")
        exf = ds_fact[0]
        gt_f = exf["ground_truth"]
        self.assertEqual(
            ds_fact.evaluate_response(gt_f, gt_f, "short_fact", exf["prompt"]), 1.0
        )
        self.assertLess(
            ds_fact.evaluate_response(gt_f + " extra", gt_f, "short_fact", exf["prompt"]),
            1.0,
        )

        # gen: should prefer similarity to correct_answer over wrong_answer
        _, _, ds_gen = self._build("short_belief_gen_inaccessible")
        exg = ds_gen[0]
        gt_g = exg["ground_truth"]
        wrong = exg["item"].metadata.get("wrong_answer")  # type: ignore[attr-defined]
        # Force a deterministic lightweight embedder by monkeypatching _get_embedder
        class _Dummy:
            def encode(self, s):
                # Very simple bag length as 1-dim vector to avoid heavy deps
                return __import__('numpy').array([len(str(s).split())], dtype=float)
        ds_gen._embedder = _Dummy()  # type: ignore[attr-defined]

        # Exact correct should be 1.0
        self.assertEqual(
            ds_gen.evaluate_response(gt_g, gt_g, "short_belief_gen_inaccessible", exg["prompt"]),
            1.0,
        )
        if wrong:
            # A response identical to wrong_answer should be scored 0.0
            self.assertEqual(
                ds_gen.evaluate_response(wrong, gt_g, "short_belief_gen_inaccessible", exg["prompt"]),
                0.0,
            )


# ---- From former test_fantom_all_tasks.py ----

class TestFantomAllTasks(unittest.TestCase):
    def _cfg(self, task: str) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            name="fantom",
            task_type=task,
            data_path=None,
            base_data_dir="data/fantom",
            sample_limit=3,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        cfg.data_path = cfg.get_data_path()
        return cfg

    def _build(self, task: str):
        # Minimal prompt_format stub using Fantom wrapper contract indirectly
        class _PF:
            def build(self, system_prompt: str, user_messages, **kwargs):
                return f"{system_prompt}\n{user_messages}"

        cfg = self._cfg(task)
        return create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=_PF(),
        )

    def test_answerability_list_inaccessible(self):
        _, _, ds = self._build("short_answerability_list_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, list)
        # JSON list answer should match
        resp = {"reational": "r", "answer": gt}
        self.assertEqual(ds.evaluate_response(str(resp), gt, "short_answerability_list_inaccessible", ex["prompt"]), 1.0)
        # Comma-separated case-insensitive order-insensitive
        s = ", ".join(gt[::-1])
        self.assertEqual(ds.evaluate_response(s, gt, "short_answerability_list_inaccessible", ex["prompt"]), 1.0)

    def test_infoaccessibility_binary_inaccessible(self):
        _, _, ds = self._build("short_infoaccessibility_binary_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIn(gt, ("yes", "no"))
        self.assertEqual(ds.evaluate_response(gt.upper(), gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]), 1.0)

    def test_infoaccessibility_list_inaccessible(self):
        _, _, ds = self._build("short_infoaccessibility_list_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, list)
        # Allow JSON list
        resp = {"reational": "r", "answer": gt}
        self.assertEqual(ds.evaluate_response(str(resp), gt, "short_infoaccessibility_list_inaccessible", ex["prompt"]), 1.0)

    def test_list_accepts_json_with_answer_as_scalar_string(self):
        """
        emotion_experiment_engine/datasets/fantom.py: ensure _eval_list accepts
        a JSON object where 'answer' is a comma-separated string of items.

        This mirrors real outputs observed in results/fantom_qwen3/no-thinking-nogen
        where the model returns {"reational": "...", "answer": "A, B, C"}
        instead of a JSON array.
        """
        _, _, ds = self._build("short_infoaccessibility_list_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        # Construct a scalar string joining ground-truth items with commas
        scalar = ", ".join(gt)
        resp = {"reational": "r", "answer": scalar}
        self.assertEqual(
            ds.evaluate_response(str(resp), gt, "short_infoaccessibility_list_inaccessible", ex["prompt"]),
            1.0,
        )

    def test_belief_choice_accessible(self):
        _, _, ds = self._build("short_belief_choice_accessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        opts = ex["item"].metadata.get("options")  # type: ignore[attr-defined]
        gt_text = ex["ground_truth"][0]
        self.assertIn(gt_text, opts)
        # Exact text
        self.assertEqual(ds.evaluate_response(gt_text, [gt_text], "short_belief_choice_accessible", ex["prompt"]), 1.0)

    def test_belief_gen_inaccessible(self):
        _, _, ds = self._build("short_belief_gen_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, str)
        # Exact normalized match
        self.assertEqual(ds.evaluate_response(gt, gt, "short_belief_gen_inaccessible", ex["prompt"]), 1.0)

    def test_fact(self):
        _, _, ds = self._build("short_fact")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, str)
        self.assertEqual(ds.evaluate_response(gt, gt, "short_fact", ex["prompt"]), 1.0)


def _make_temp_fantom_file(samples: List[dict]) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    with tmp.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample) + "\n")
    return tmp


class _DummyEmbedder:
    """Tiny deterministic sentence-transformers-like stub with normalize support."""

    def __init__(self):
        self._vocab = {
            "apple": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "banana": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "car": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "bike": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def encode(self, texts: List[str], normalize_embeddings: bool = True):  # type: ignore[override]
        rows = []
        for text in texts:
            key = (text or "").strip().lower()
            vec = self._vocab.get(key, np.zeros(4, dtype=np.float32))
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm:
                    vec = vec / norm
            rows.append(vec)
        return np.stack(rows, axis=0)


class _CountingEmbedder(_DummyEmbedder):
    """Embedder stub that counts how many strings were actually encoded."""

    def __init__(self):
        super().__init__()
        self.total_encoded = 0

    def encode(self, texts: List[str], normalize_embeddings: bool = True):  # type: ignore[override]
        self.total_encoded += len(texts)
        return super().encode(texts, normalize_embeddings)


class TestFantomBatchEval(unittest.TestCase):
    def _make_cfg(self, tmp: Path) -> BenchmarkConfig:
        return BenchmarkConfig(
            name="fantom",
            task_type="short_belief_gen_inaccessible",
            data_path=tmp,
            base_data_dir=None,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

    def test_eval_batch_raises_without_embedder(self):
        tmp = _make_temp_fantom_file(
            [
                {
                    "context": "ctx",
                    "question": "q1",
                    "correct_answer": "apple",
                    "wrong_answer": "banana",
                }
            ]
        )

        cfg = self._make_cfg(tmp)
        ds = FantomDataset(cfg, prompt_wrapper=None)
        batch = ds.collate_fn([ds[0]])
        prompts = batch["prompts"]
        gts = batch["ground_truths"]
        tasks = [cfg.task_type]

        def _boom():
            raise RuntimeError("embedder unavailable")

        ds._get_embedder = _boom  # type: ignore[attr-defined]

        with self.assertRaisesRegex(RuntimeError, "embedder"):
            ds.evaluate_batch(["dummy"], gts, tasks, prompts)

    def test_vectorized_batch_scoring(self):
        tmp = _make_temp_fantom_file(
            [
                {
                    "context": "ctx1",
                    "question": "q1",
                    "correct_answer": "apple",
                    "wrong_answer": "banana",
                },
                {
                    "context": "ctx2",
                    "question": "q2",
                    "correct_answer": "car",
                    "wrong_answer": "bike",
                },
            ]
        )

        cfg = self._make_cfg(tmp)
        ds = FantomDataset(cfg, prompt_wrapper=None)
        batch = ds.collate_fn([ds[0], ds[1]])
        prompts = batch["prompts"]
        gts = batch["ground_truths"]
        tasks = [cfg.task_type, cfg.task_type]

        ds._get_embedder = lambda: _DummyEmbedder()  # type: ignore[attr-defined]

        scores = ds.evaluate_batch(["apple", "bike"], gts, tasks, prompts)
        self.assertEqual(scores, [1.0, 0.0])

    def test_embedding_cache_deduplicates(self):
        samples = [
            {
                "context": "ctx1",
                "question": "q1",
                "correct_answer": "apple",
                "wrong_answer": "banana",
            },
            {
                "context": "ctx2",
                "question": "q2",
                "correct_answer": "apple",
                "wrong_answer": "banana",
            },
            {
                "context": "ctx3",
                "question": "q3",
                "correct_answer": "apple",
                "wrong_answer": "banana",
            },
        ]
        tmp = _make_temp_fantom_file(samples)

        cfg = self._make_cfg(tmp)
        ds = FantomDataset(cfg, prompt_wrapper=None)
        batch = ds.collate_fn([ds[0], ds[1], ds[2]])
        prompts = batch["prompts"]
        gts = batch["ground_truths"]
        tasks = [cfg.task_type] * 3

        embedder = _CountingEmbedder()
        ds._get_embedder = lambda: embedder  # type: ignore[attr-defined]

        ds.evaluate_batch(["apple", "apple", "apple"], gts, tasks, prompts)

        self.assertEqual(embedder.total_encoded, 2)


if __name__ == "__main__":
    unittest.main()
