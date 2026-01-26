"""
Responsible files: emotion_experiment_engine/benchmark_prompt_wrapper.py
Purpose: Ensure prompt wrapper routing returns MemoryPromptWrapper for
unknown/new benchmark 'pubmed_qa' with task 'pqa_labeled'.

Note: This likely passes before implementation (fallback behavior), but
guards routing expectations for this benchmark.
"""

import unittest

from emotion_experiment_engine.benchmark_prompt_wrapper import (
    get_benchmark_prompt_wrapper,
)
from emotion_experiment_engine.pubmedqa_prompt_wrapper import PubMedQAPromptWrapper


class _RecordingPromptFormat:
    def __init__(self):
        self.last_args = None
        self.last_kwargs = None

    def build(self, *args, **kwargs):  # pragma: no cover
        self.last_args = args
        self.last_kwargs = kwargs
        return "<stub>"


class TestPromptWrapperPubMedQA(unittest.TestCase):
    def test_wrapper_is_pubmedqa_prompt_wrapper(self):
        wrapper = get_benchmark_prompt_wrapper(
            "pubmed_qa", "pqa_labeled", prompt_format=_RecordingPromptFormat()
        )
        self.assertIsInstance(wrapper, PubMedQAPromptWrapper)

    def test_cot_instruction_appended(self):
        prompt_format = _RecordingPromptFormat()
        wrapper = get_benchmark_prompt_wrapper(
            "pubmed_qa", "pqa_labeled", prompt_format=prompt_format
        )
        wrapper(
            context="Context text",
            question="Is this helpful?",
            user_messages="Answer please.",
            augmentation_config={"method": "cot"},
        )
        system_prompt, user_messages = prompt_format.last_args[:2]
        self.assertIn("Context text", system_prompt)
        self.assertIn("Answer please.", "\n".join(user_messages))
        self.assertIn("Before answering, briefly reason", "\n".join(user_messages))


if __name__ == "__main__":
    unittest.main()
