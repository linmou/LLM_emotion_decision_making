# Responsible: emotion_experiment_engine/humaneval_prompt_wrapper.py
# Purpose: Validate default chat-style formatting for HumanEval wrapper.

from emotion_experiment_engine.humaneval_prompt_wrapper import HumanEvalPromptWrapper


class _DummyPromptFormat:
    def build(self, system_prompt, user_messages, assistant_messages, images=None, enable_thinking=False):
        parts = []
        if system_prompt:
            parts.append(f"SYSTEM:{system_prompt}")
        for m in user_messages or []:
            parts.append(f"USER:{m}")
        return "\n".join(parts)


def test_humaneval_wrapper_default_is_chat():
    fmt = _DummyPromptFormat()
    w = HumanEvalPromptWrapper(fmt)
    prompt = "def add(a, b):\n    \"\"\"Add two numbers\"\"\"\n    pass\n"
    out = w(context="", question=prompt, options=None)
    assert isinstance(out, str)
    assert "USER:" in out and "def add(" in out
    assert out != prompt


def test_humaneval_wrapper_ignores_augmentation_config():
    fmt = _DummyPromptFormat()
    w = HumanEvalPromptWrapper(fmt)
    prompt = "def mul(a, b):\n    pass\n"
    out = w(context="", question=prompt, options=None, augmentation_config={"humaneval_mode": "raw"})
    assert isinstance(out, str)
    assert "USER:" in out and "def mul" in out
