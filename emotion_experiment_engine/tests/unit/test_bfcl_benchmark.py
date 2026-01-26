"""
Tests for BFCL benchmark integration (dataset + prompt wrapper + registry wiring)

Responsible files:
- emotion_experiment_engine/datasets/bfcl.py
- emotion_experiment_engine/bfcl_prompt_wrapper.py
- emotion_experiment_engine/dataset_factory.py (registry)
- emotion_experiment_engine/benchmark_prompt_wrapper.py (factory wiring)
- emotion_experiment_engine/benchmark_component_registry.py (spec mapping)

These tests validate that:
- BFCL dataset loads minimal JSONL for live_simple and live_multiple
- Prompt construction includes tool schema and strict JSON-only instructions
- Registry-based factories return the correct classes/partials
"""

from pathlib import Path
import json
import unittest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import (
    create_dataset_from_config,
)
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.benchmark_prompt_wrapper import (
    get_benchmark_prompt_wrapper,
)


class DummyPromptFormat:
    """Minimal PromptFormat stub to capture build() args for assertions."""

    def __init__(self) -> None:
        self.last_system_prompt = None
        self.last_user_messages = None
        self.last_enable_thinking = None

    def build(self, system_prompt, user_messages, assistant_messages=None, enable_thinking=False):
        self.last_system_prompt = system_prompt
        self.last_user_messages = user_messages
        self.last_enable_thinking = enable_thinking
        # Return a simple combined string so callers can verify non-empty
        return f"SYSTEM:{system_prompt}\nUSER:{user_messages[0] if isinstance(user_messages, list) and user_messages else ''}"


class TestBFCLBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Ensure minimal sample data exists under data/BFCL
        cls.data_dir = Path("data/BFCL")
        cls.data_dir.mkdir(parents=True, exist_ok=True)

        # live_simple sample (single tool)
        simple_path = cls.data_dir / "bfcl_live_simple.jsonl"
        if not simple_path.exists():
            record = {
                "id": "live_simple_1",
                "question": [
                    {
                        "role": "user",
                        "content": "What is the weather in Tel Aviv? Use Fahrenheit.",
                    }
                ],
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "fahrenheit",
                            },
                        },
                        "required": ["location"],
                    },
                },
                # Inline ground truth to fit our dataset interface
                "ground_truth": [
                    {
                        "get_current_weather": {
                            "location": ["Tel Aviv, Israel"],
                            "unit": ["fahrenheit", ""],
                        }
                    }
                ],
            }
            with simple_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        # live_multiple sample (two tools)
        multi_path = cls.data_dir / "bfcl_live_multiple.jsonl"
        if not multi_path.exists():
            record = {
                "id": "live_multiple_1",
                "question": [
                    {
                        "role": "user",
                        "content": "Book a flight to San Francisco and get weather in Palo Alto, Fahrenheit.",
                    }
                ],
                "functions": [
                    {
                        "name": "book_flight",
                        "parameters": {
                            "type": "object",
                            "properties": {"destination": {"type": "string"}},
                            "required": ["destination"],
                        },
                    },
                    {
                        "name": "get_current_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                    },
                ],
                "ground_truth": [
                    {"book_flight": {"destination": ["San Francisco"]}},
                    {
                        "get_current_weather": {
                            "location": ["Palo Alto, CA"],
                            "unit": ["fahrenheit"],
                        }
                    },
                ],
            }
            with multi_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def _mk_cfg(self, task_type: str) -> BenchmarkConfig:
        return BenchmarkConfig(
            name="bfcl",
            task_type=task_type,
            data_path=None,
            base_data_dir=str(self.data_dir),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

    def test_prompt_wrapper_factory_returns_bfcl(self):
        pf = DummyPromptFormat()
        wrapper = get_benchmark_prompt_wrapper("bfcl", "live_simple", pf)
        # Build a tiny prompt to verify integration path; context carries tool schema
        tools_ctx = json.dumps({
            "name": "get_current_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        })
        out = wrapper(
            context=tools_ctx,
            question="Weather in Tel Aviv",
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config=None,
            answer=None,
            emotion=None,
            options=None,
        )
        self.assertIsInstance(out, str)
        # System prompt should contain tool schema and strict JSON instruction
        self.assertIn("get_current_weather", pf.last_system_prompt)
        self.assertIn("JSON", pf.last_system_prompt)
        self.assertIn("only", pf.last_system_prompt.lower())
        # Stronger formatting constraints added
        self.assertIn("Begin response with '['", pf.last_system_prompt)
        self.assertIn("no code fences", pf.last_system_prompt.lower())
        self.assertIn("Do NOT include wrapper keys like 'name' or 'parameters'", pf.last_system_prompt)

    def test_create_components_and_dataset_live_simple(self):
        cfg = self._mk_cfg("live_simple")
        pf = DummyPromptFormat()
        prompt_partial, answer_partial, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
            emotion=None,
            enable_thinking=False,
        )
        # Access first item and ensure prompt includes function name and JSON-only instruction
        batch = dataset[0]
        self.assertIn("get_current_weather", batch["prompt"])  # tool name present
        self.assertIn("JSON", pf.last_system_prompt)
        self.assertGreaterEqual(len(dataset), 1)

    def test_create_components_and_dataset_live_multiple(self):
        cfg = self._mk_cfg("live_multiple")
        pf = DummyPromptFormat()
        prompt_partial, answer_partial, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
            emotion=None,
            enable_thinking=False,
        )
        batch = dataset[0]
        # Both tool names should appear in the constructed prompt/context
        self.assertIn("book_flight", batch["prompt"])  # multi-tool name present
        self.assertIn("get_current_weather", batch["prompt"])  # second tool
        self.assertIn("JSON", pf.last_system_prompt)
        self.assertIn("Example:", pf.last_system_prompt)

    def test_nested_question_is_extracted_and_used_as_user_turn(self):
        # Create a temporary nested-question sample file
        nested_path = self.data_dir / "bfcl_live_simple_nested.jsonl"
        record = {
            "id": "live_simple_nested_1",
            # BFCL sometimes stores question as a nested list of turns; ensure we can read it
            "question": [[
                {
                    "role": "user",
                    "content": "update my latte to a large size with coconut milk, extra sweet",
                }
            ]],
            "function": {
                "name": "ChaDri.change_drink",
                "parameters": {
                    "type": "dict",
                    "required": ["new_preferences"],
                    "properties": {
                        "new_preferences": {
                            "type": "dict",
                            "properties": {
                                "size": {"type": "string"}
                            }
                        }
                    }
                },
            },
            "ground_truth": [
                {"ChaDri.change_drink": {"new_preferences": [""]}}
            ],
        }
        with nested_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Build config that points directly to this file
        cfg = BenchmarkConfig(
            name="bfcl",
            task_type="live_simple",
            data_path=nested_path,
            base_data_dir=str(self.data_dir),
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

        pf = DummyPromptFormat()
        prompt_partial, answer_partial, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=pf,
            emotion=None,
            enable_thinking=False,
        )

        # Access first item to trigger prompt construction via wrapper
        batch = dataset[0]
        # Verify the question text was extracted and used as the user message
        self.assertIsInstance(pf.last_user_messages, list)
        self.assertGreaterEqual(len(pf.last_user_messages), 1)
        self.assertIn("latte", pf.last_user_messages[0])
        # Ensure we are not using the default placeholder as the user turn
        self.assertNotIn("Please provide your answer.", pf.last_user_messages[0])
        # And the system should not redundantly include a "User request:" prefix anymore
        self.assertNotIn("User request:", pf.last_system_prompt or "")


if __name__ == "__main__":
    unittest.main()
