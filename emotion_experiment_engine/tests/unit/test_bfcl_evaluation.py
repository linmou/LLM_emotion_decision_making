"""
Tests: AST-based evaluation for BFCL live_simple and live_multiple
Responsible files:
- emotion_experiment_engine/datasets/bfcl.py (evaluate_response implementation)
"""

import json
import unittest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)


class DummyPromptFormat:
    def __init__(self) -> None:
        self.last_system_prompt = None
        self.last_user_messages = None
        self.last_enable_thinking = None

    def build(self, system_prompt, user_messages, assistant_messages=None, enable_thinking=False):
        self.last_system_prompt = system_prompt
        self.last_user_messages = user_messages
        self.last_enable_thinking = enable_thinking
        return f"SYSTEM:{system_prompt}\nUSER:{user_messages[0] if isinstance(user_messages, list) and user_messages else ''}"


class TestBFCLEvaluation(unittest.TestCase):
    def _mk_cfg(self, task_type: str, base_dir: str = "data/BFCL") -> BenchmarkConfig:
        return BenchmarkConfig(
            name="bfcl",
            task_type=task_type,
            data_path=None,
            base_data_dir=base_dir,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

    def test_live_simple_accepts_optional_omission(self):
        cfg = self._mk_cfg("live_simple")
        pf = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
        )

        batch = dataset[0]
        gt = batch["ground_truth"]

        # Unit is optional ("" allowed), omit it
        resp = json.dumps([
            {"get_current_weather": {"location": "Tel Aviv, Israel"}}
        ])
        score = dataset.evaluate_response(resp, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score, 1.0)

        # Include unit explicitly as fahrenheit (should also pass)
        resp2 = json.dumps([
            {"get_current_weather": {"location": "Tel Aviv, Israel", "unit": "fahrenheit"}}
        ])
        score2 = dataset.evaluate_response(resp2, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score2, 1.0)

    def test_live_simple_wrong_param_or_extra_keys(self):
        cfg = self._mk_cfg("live_simple")
        pf = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
        )
        batch = dataset[0]
        gt = batch["ground_truth"]

        # Wrong enum value
        resp = json.dumps([
            {"get_current_weather": {"location": "Tel Aviv, Israel", "unit": "celsius"}}
        ])
        score = dataset.evaluate_response(resp, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score, 0.0)

        # Extra key not allowed
        resp2 = json.dumps([
            {"get_current_weather": {"location": "Tel Aviv, Israel", "foo": "bar"}}
        ])
        score2 = dataset.evaluate_response(resp2, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score2, 0.0)

    def test_live_multiple_order_and_params(self):
        cfg = self._mk_cfg("live_multiple")
        pf = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
        )
        batch = dataset[0]
        gt = batch["ground_truth"]

        # Correct order and params
        resp = json.dumps([
            {"book_flight": {"destination": "San Francisco"}},
            {"get_current_weather": {"location": "Palo Alto, CA", "unit": "fahrenheit"}},
        ])
        score = dataset.evaluate_response(resp, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score, 1.0)

        # Wrong order
        resp_wrong_order = json.dumps([
            {"get_current_weather": {"location": "Palo Alto, CA", "unit": "fahrenheit"}},
            {"book_flight": {"destination": "San Francisco"}},
        ])
        score_wrong = dataset.evaluate_response(resp_wrong_order, gt, cfg.task_type, batch["prompt"])
        self.assertEqual(score_wrong, 0.0)

    # Multi-turn is not supported in this repo's dataset/evaluator.


if __name__ == "__main__":
    unittest.main()
