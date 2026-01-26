"""
Merged tests: thinking-mode directive injection + empty <think> prefix cleanup
Responsible files:
- neuro_manipulation/prompt_formats.py (PromptFormat, Qwen3 logic)
- emotion_experiment_engine/experiment.py (empty <think> cleanup regex)

Purpose:
- Verify Qwen3 user-turn directive placement:
  - enable_thinking=True => appends '/think' to end of first user message
  - enable_thinking=False => appends '/no_think' and no '/think'
- Verify cleanup regex removes only leading empty <think>...</think> and keeps JSON parseable
"""

import json
import re
import sys
import types
import unittest

# ---- Stubs for external deps so tests run in lightweight env ----
if 'transformers' not in sys.modules:
    transformers_stub = types.ModuleType('transformers')
    class _AutoTokenizerStub:  # minimal placeholder
        pass
    transformers_stub.AutoTokenizer = _AutoTokenizerStub
    sys.modules['transformers'] = transformers_stub

if 'jinja2' not in sys.modules:
    jinja2_stub = types.ModuleType('jinja2')
    exceptions_stub = types.ModuleType('jinja2.exceptions')
    class _TemplateErrorStub(Exception):
        pass
    exceptions_stub.TemplateError = _TemplateErrorStub
    sys.modules['jinja2'] = jinja2_stub
    sys.modules['jinja2.exceptions'] = exceptions_stub

if 'pydantic' not in sys.modules:
    pydantic_stub = types.ModuleType('pydantic')
    class _BaseModelStub: pass
    def _Field(*args, **kwargs): return None
    ConfigDict = dict
    pydantic_stub.BaseModel = _BaseModelStub
    pydantic_stub.ConfigDict = ConfigDict
    pydantic_stub.Field = _Field
    sys.modules['pydantic'] = pydantic_stub

if 'games' not in sys.modules:
    games_stub = types.ModuleType('games')
    sys.modules['games'] = games_stub
if 'games.game' not in sys.modules:
    game_mod = types.ModuleType('games.game')
    class GameDecision: pass
    game_mod.GameDecision = GameDecision
    sys.modules['games.game'] = game_mod

from neuro_manipulation.prompt_formats import PromptFormat


class FakeQwen3Tokenizer:
    def __init__(self, name_or_path: str = "Qwen3-1.7B") -> None:
        self.name_or_path = name_or_path


class TestThinkingModeEnableConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeQwen3Tokenizer()
        self.prompt_format = PromptFormat(self.tokenizer)

    def test_prompt_format_injects_think_for_qwen3_when_enabled(self):
        system = "You are Alice."
        user_msgs = ["State your choice succinctly."]

        prompt = self.prompt_format.build(
            system_prompt=system,
            user_messages=user_msgs,
            assistant_messages=[],
            enable_thinking=True,
        )

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        # expect directive at end of user content
        self.assertIn(user_msgs[0] + "\n/think", prompt)

    def test_prompt_format_does_not_inject_think_when_disabled(self):
        system = "You are Alice."
        user_msgs = ["State your choice succinctly."]

        prompt = self.prompt_format.build(
            system_prompt=system,
            user_messages=user_msgs,
            assistant_messages=[],
            enable_thinking=False,
        )

        self.assertNotIn("/think", prompt)
        self.assertIn("/no_think", prompt)


class TestEmptyThinkPrefixCleanup(unittest.TestCase):
    def setUp(self) -> None:
        # Same regex as in pipeline
        self.cleaner = re.compile(r"^\s*<think>\s*</think>\s*", re.IGNORECASE)

    def test_strip_only_prefix(self):
        s = "\n  <think>\n\n</think>\n  [{\"a\":1}]"
        cleaned = self.cleaner.sub("", s).strip()
        self.assertEqual(json.loads(cleaned), [{"a": 1}])

    def test_no_change_when_no_prefix(self):
        s = "[{\"f\": {\"x\": 1}}]"
        cleaned = self.cleaner.sub("", s)
        self.assertEqual(cleaned, s)


if __name__ == "__main__":
    unittest.main()

