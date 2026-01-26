"""
Test: Emotion Check prompt formatting
Responsible files:
- emotion_experiment_engine/memory_prompt_wrapper.py (EmotionCheckPromptWrapper)
- emotion_experiment_engine/benchmark_prompt_wrapper.py or registry wiring

Purpose:
Ensure the emotion_check benchmark formats prompts as:
- System: sets persona to an average American
- User: contains the question text (moved out of system)
"""

import unittest

from emotion_experiment_engine.benchmark_prompt_wrapper import (
    get_benchmark_prompt_wrapper,
)


class DummyPromptFormat:
    """Minimal PromptFormat stub to capture build() args."""

    def __init__(self) -> None:
        self.last_system_prompt = None
        self.last_user_messages = None
        self.last_enable_thinking = None

    def build(self, system_prompt, user_messages, assistant_messages=None, enable_thinking=False):
        # Record inputs for assertions
        self.last_system_prompt = system_prompt
        self.last_user_messages = user_messages
        self.last_enable_thinking = enable_thinking
        return f"SYSTEM:{system_prompt}\nUSER:{user_messages[0]}"


class TestEmotionCheckPromptFormat(unittest.TestCase):
    def test_emotion_check_moves_question_to_user(self):
        prompt_format = DummyPromptFormat()
        wrapper = get_benchmark_prompt_wrapper(
            "emotion_check", "emotion_check", prompt_format
        )

        question = "I flare up quickly but get over it quickly."
        out = wrapper(
            context=None,
            question=question,
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config=None,
            answer=None,
            emotion=None,
            options=None,
        )

        # Ensure build() was called and outputs produced
        self.assertIsInstance(out, str)

        # System prompt should set persona and NOT include the question
        self.assertIn("average American", prompt_format.last_system_prompt)
        self.assertNotIn(question, prompt_format.last_system_prompt)

        # User message should contain the question
        self.assertIsInstance(prompt_format.last_user_messages, list)
        self.assertEqual(len(prompt_format.last_user_messages), 1)
        self.assertIn(question, prompt_format.last_user_messages[0])

        # Should instruct JSON response format
        self.assertIn("Respond in JSON", prompt_format.last_user_messages[0])
        self.assertIn("'rational'", prompt_format.last_user_messages[0])
        self.assertIn("'choice'", prompt_format.last_user_messages[0])

    def test_academic_scale_adds_likert_and_json(self):
        prompt_format = DummyPromptFormat()
        wrapper = get_benchmark_prompt_wrapper(
            "emotion_check", "academic_scale", prompt_format
        )

        question = "I flare up quickly but get over it quickly."
        out = wrapper(
            context=None,
            question=question,
            user_messages="",
            enable_thinking=False,
            augmentation_config=None,
            answer=None,
            emotion="anger",
            options=None,
        )

        self.assertIsInstance(out, str)
        # Persona and anti-refusal
        self.assertIn("average American", prompt_format.last_system_prompt)
        # Likert anchors present as options 1..5
        um = prompt_format.last_user_messages[0]
        self.assertIn("1 = Strongly Disagree", um)
        self.assertIn("5 = Strongly Agree", um)
        # JSON instruction present
        self.assertIn("Respond in JSON", um)
        self.assertIn("'rational'", um)
        self.assertIn("'choice'", um)
        # Should discourage mentioning current feelings
        self.assertIn("Evaluate the statement", um)
        self.assertIn("Do not mention your current", um)
        # No explicit emotion hint for academic_scale
        self.assertNotIn("You currently feel", um)

    def test_numeric_options_when_provided(self):
        prompt_format = DummyPromptFormat()
        wrapper = get_benchmark_prompt_wrapper(
            "emotion_check", "emotion_check", prompt_format
        )

        question = "Which fits you best?"
        options = ["Option A", "Option B", "Option C"]
        out = wrapper(
            context=None,
            question=question,
            user_messages="",
            enable_thinking=False,
            augmentation_config=None,
            answer=None,
            emotion=None,
            options=options,
        )

        self.assertIsInstance(out, str)
        um = prompt_format.last_user_messages[0]
        # Options should be numeric in user message
        self.assertIn("1. Option A", um)
        self.assertIn("2. Option B", um)
        self.assertIn("3. Option C", um)
        # JSON instruction present
        self.assertIn("Respond in JSON", um)


if __name__ == "__main__":
    unittest.main()
